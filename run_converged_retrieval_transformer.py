#!/usr/bin/env python3
"""收敛主线：VH-only KNN + CDR Stage3 + 可选 ESM2 + 自适应任务融合。"""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np, torch
from minimal_model.io import read_dataset
from minimal_model.candidates import build_antigen_library
from minimal_model.cdr import load_cdr_table
from minimal_model.esm_features import ESM2FeatureExtractor
from minimal_model.knn_ensemble import matrix,standardize,predict
from minimal_model.metrics import macro_f1,mae
from minimal_model.stage3 import train_stage3_model,evaluate_stage3,predict_stage3,DEFAULT_CONFIG
TARGETS=['T01','T02','T03','T04','T05']; AFF=['medium','strong','weak']
def cfg(esm=False):
 c=dict(DEFAULT_CONFIG); c.update({'max_vh_length':512,'batch_size':64,'epochs':30,'use_cdr':True,'use_chunk_attention':False,'use_cross_attention':False,'freeze_encoder':False,'amp':True,'tf32':True,'device':'auto','seed':2026})
 if esm: c.update({'esm_dim':320,'esm_projection_size':64,'esm_target_only':True})
 return c
def entropy(p): return -(p*np.log(np.maximum(p,1e-9))).sum(1)
def metrics(ts,ass,z,val):
 yt=np.array([s['target_id'] for s in val]); ya=np.array([s['affinity_class'] for s in val]); yz=np.array([s['log_kd_nm'] for s in val],float); tp=np.array(TARGETS)[ts.argmax(1)]; ap=np.array(AFF)[ass.argmax(1)]; f=macro_f1(yt,tp,TARGETS); af=macro_f1(ya,ap,AFF); e=mae(yz,z); return {'target_f1':f,'affinity_f1':af,'log_kd_mae':e,'combined':.3*f+.35*af+.35*max(0,1-e/2),'target_acc':float(np.mean(tp==yt)),'affinity_acc':float(np.mean(ap==ya))}
def main(root):
 train=read_dataset(root/'data/train.csv'); val=read_dataset(root/'data/val.csv'); lib=build_antigen_library(train,val); cdr=load_cdr_table(root/'data/cdr_annotations.csv'); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); out={}
 vhtr,vhv=standardize(matrix(train,'vh'),matrix(val,'vh')); pairtr,pairv=standardize(matrix(train,'pair'),matrix(val,'pair')); yt=np.array([TARGETS.index(s['target_id']) for s in train]); ya=np.array([AFF.index(s['affinity_class']) for s in train]); yz=np.array([s['log_kd_nm'] for s in train],float)
 kt,_,_=predict(vhv,vhtr,yt,ya,yz,3,device); _,ka,kz=predict(pairv,pairtr,yt,ya,yz,3,device)
 for name,use_esm in [('m2',False),('m2_esm2',True)]:
  if use_esm:
   ext=ESM2FeatureExtractor(device='auto',amp=True); ext.attach([train,val],64)
  start=time.perf_counter(); b=train_stage3_model(train,lib,cdr,cfg(use_esm)); p=predict_stage3(b,val); m=evaluate_stage3(b,val); out[name]={'metrics':{'target_f1':m['target_macro_f1_ranking'],'affinity_f1':m['positive_affinity_macro_f1'],'log_kd_mae':m['positive_log_kd_mae'],'combined':m['combined_score_ranking']},'seconds':time.perf_counter()-start}
  idx=np.array([lib['target_to_index'][s['target_id']] for s in val]); trt=torch.softmax(torch.tensor(p['target_logits']),1).numpy(); tra=torch.softmax(torch.tensor(p['affinity_logits'][np.arange(len(val)),idx]),1).numpy(); trz=p['log_kd_pred'][np.arange(len(val)),idx]
  # Confidence-aware gate: KNN target/affinity entropy, fixed predeclared weighting (not val-tuned).
  gate_t=(1-entropy(kt)/np.log(5))[:,None]; gate_a=(1-entropy(ka)/np.log(3))[:,None]
  fused=metrics(gate_t*kt+(1-gate_t)*trt,gate_a*ka+(1-gate_a)*tra,gate_a[:,0]*kz+(1-gate_a[:,0])*trz,val); out[name]['adaptive_fusion']=fused
  identities=np.array([max(sum(a==b for a,b in zip(s['vh_sequence_aa'],t['vh_sequence_aa']))/max(len(s['vh_sequence_aa']),len(t['vh_sequence_aa'])) for t in train) for s in val]); out[name]['identity_bins']={}
  for label,mask in {'low':identities<.6,'mid':(identities>=.6)&(identities<.8),'high':identities>=.8}.items():
   if mask.any(): out[name]['identity_bins'][label]={'n':int(mask.sum()),'mean_identity':float(identities[mask].mean()),'target_f1':macro_f1(np.array([s['target_id'] for s,m in zip(val,mask) if m]),np.array(TARGETS)[(gate_t*kt+(1-gate_t)*trt).argmax(1)][mask],TARGETS)}
 print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__': p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent);main(p.parse_args().root)
