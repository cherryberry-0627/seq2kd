#!/usr/bin/env python3
"""后三个优先级基础试点：蒸馏、严格泛化分层、CDR-H3突变闭环。"""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import torch
from minimal_model.io import read_dataset
from minimal_model.cdr import load_cdr_table
from minimal_model.candidates import build_antigen_library
from minimal_model.knn_ensemble import matrix, standardize, predict
from minimal_model.esm_features import ESM2FeatureExtractor
import torch.nn.functional as F
from minimal_model.stage3 import train_stage3_model, evaluate_stage3, predict_stage3, DEFAULT_CONFIG
from minimal_model.metrics import macro_f1, mae

TARGETS=['T01','T02','T03','T04','T05']; AFF=['medium','strong','weak']; AA='ACDEFGHIKLMNPQRSTVWY'
def cfg(epochs=30):
 c=dict(DEFAULT_CONFIG); c.update({'max_vh_length':512,'batch_size':64,'epochs':epochs,'use_cdr':True,'use_chunk_attention':False,'use_cross_attention':False,'freeze_encoder':False,'amp':True,'tf32':True,'device':'auto','seed':2026}); return c
def identity(a,b): return sum(x==y for x,y in zip(a,b))/max(len(a),len(b))
def esm_score(extractor, sequences, batch_size=64):
    scores=[]
    with torch.no_grad():
        for start in range(0,len(sequences),batch_size):
            current=[s.upper() for s in sequences[start:start+batch_size]]
            _, strings, tokens=extractor.batch_converter([(str(i),s) for i,s in enumerate(current)])
            result=extractor.model(tokens.to(extractor.device),repr_layers=[],return_contacts=False)
            logp=F.log_softmax(result['logits'].float(),-1)
            for i,s in enumerate(strings):
                residue=tokens[i,1:len(s)+1].to(logp.device)
                scores.append(float(logp[i,1:len(s)+1,residue].mean().cpu()))
    return scores
def eval_group(pred,val):
 yt=np.array([s['target_id'] for s in val]); ya=np.array([s['affinity_class'] for s in val]); yz=np.array([s['log_kd_nm'] for s in val],float); f=macro_f1(yt,pred['target'],TARGETS); af=macro_f1(ya,pred['affinity'],AFF); e=mae(yz,pred['z']); return {'n':len(val),'target_f1':f,'affinity_f1':af,'log_kd_mae':e,'combined':.3*f+.35*af+.35*max(0,1-e/2)}
def main(root):
 train=read_dataset(root/'data/train.csv'); val=read_dataset(root/'data/val.csv'); lib=build_antigen_library(train,val); cdr=load_cdr_table(root/'data/cdr_annotations.csv'); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); out={}
 # Formal train-only KNN soft targets via leave-one-out, then baseline M2 distillation support check.
 vhtr,vhval=standardize(matrix(train,'vh'),matrix(val,'vh')); y=np.array([TARGETS.index(s['target_id']) for s in train]); a=np.array([AFF.index(s['affinity_class']) for s in train]); z=np.array([s['log_kd_nm'] for s in train],float)
 knn_target=[]
 for i in range(len(train)):
  ref=np.delete(vhtr,i,0); ts,_,_=predict(vhtr[i:i+1],ref,np.delete(y,i),np.delete(a,i),np.delete(z,i),3,device); knn_target.append(ts[0])
 out['distillation_setup']={'n_leave_one_out':len(knn_target),'temperature':1.0,'target_weight':.15,'affinity_weight':.10,'regression_weight':.10}
 # Strict generalization bins: nearest train identity measured on val, no threshold fitting.
 identities=np.array([max(identity(s['vh_sequence_aa'],t['vh_sequence_aa']) for t in train) for s in val])
 bins={'low_lt_0.6':identities<.6,'mid_0.6_0.8':(identities>=.6)&(identities<.8),'high_ge_0.8':identities>=.8}; out['identity_bins']={k:{'n':int(v.sum()),'mean_identity':float(identities[v].mean()) if v.any() else None} for k,v in bins.items()}
 # Train M2 once for strict split diagnostics.
 start=time.perf_counter(); bundle=train_stage3_model(train,lib,cdr,cfg()); pred=predict_stage3(bundle,val); out['m2_seconds']=time.perf_counter()-start
 true_idx=np.array([lib['target_to_index'][s['target_id']] for s in val])
 rank=np.argmax(pred['rank_score'],1); aff=np.argmax(pred['affinity_logits'][np.arange(len(val)),true_idx],1); zz=pred['log_kd_pred'][np.arange(len(val)),true_idx]
 full={'target':np.array(TARGETS)[rank],'affinity':np.array(AFF)[aff],'z':zz}
 out['strict_generalization']={}
 for name,mask in bins.items():
  if mask.any(): out['strict_generalization'][name]=eval_group({key:value[mask] for key,value in full.items()},[s for s,m in zip(val,mask) if m])
 # CDR-H3 single substitution closure: generate, score ESM2 likelihood, similarity and current model/KNN proxy.
 ext=ESM2FeatureExtractor(device='auto',amp=True); candidates=[]
 for s in train[:500]:
  entry=cdr.get(s['sample_id'],{}); span=entry.get('spans',{}).get('h3')
  if span is None: continue
  start_i,end_i=span; parent=s['vh_sequence_aa']
  for pos in range(start_i,end_i):
   for aa in AA:
    if aa==parent[pos]: continue
    seq=parent[:pos]+aa+parent[pos+1:]; item=dict(s); item['sample_id']=f"pilot_{s['sample_id']}_{pos}_{aa}"; item['parent_sample_id']=s['sample_id']; item['vh_sequence_aa']=seq; item['mutation_position']=pos; item['mutation']=parent[pos]+'>'+aa; candidates.append(item)
 candidates=candidates[:10000]; ext.attach([candidates],batch_size=64); candidate_scores=esm_score(ext,[s['vh_sequence_aa'] for s in candidates],batch_size=64); parent_score_map={s['sample_id']:score for s,score in zip(train[:500],esm_score(ext,[s['vh_sequence_aa'] for s in train[:500]],batch_size=64))}; novel=[]; trainseq=[s['vh_sequence_aa'] for s in train]
 for s,score in zip(candidates,candidate_scores):
  if score < parent_score_map.get(s['parent_sample_id'],score) - 1.5: continue
  # A single-point mutant is expected to be >0.98 identical to its parent;
  # exclude the parent from novelty matching and use a stricter near-duplicate threshold.
  others=[t['vh_sequence_aa'] for t in train if t['sample_id']!=s['parent_sample_id']]
  if others and max(identity(s['vh_sequence_aa'],t) for t in others)>=.995: continue
  novel.append(s)
 out['cdr_h3_mutation_pilot']={'generated':len(candidates),'esm_similarity_pass':len(novel),'unique_positions':len(set((s['parent_sample_id'],s['mutation_position']) for s in novel))}
 print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__': p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent);main(p.parse_args().root)
