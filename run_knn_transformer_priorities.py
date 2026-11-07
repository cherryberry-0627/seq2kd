#!/usr/bin/env python3
"""优先级 C/D/E：Transformer+2mer、KNN后融合、KNN蒸馏准备。"""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np, torch
from minimal_model.io import read_dataset
from minimal_model.candidates import build_antigen_library
from minimal_model.cdr import load_cdr_table
from minimal_model.knn_ensemble import matrix,standardize,predict,distill_loss
from minimal_model.stage3 import train_stage3_model,evaluate_stage3,predict_stage3,DEFAULT_CONFIG
from minimal_model.metrics import macro_f1,mae
TARGETS=['T01','T02','T03','T04','T05']; AFF=['medium','strong','weak']
def cfg(kmer=False):
 c=dict(DEFAULT_CONFIG); c.update({'max_vh_length':512,'batch_size':64,'epochs':30,'use_cdr':True,'use_chunk_attention':False,'use_cross_attention':False,'freeze_encoder':False,'amp':True,'tf32':True,'device':'auto','seed':2026})
 if kmer: c.update({'kmer_dim':428,'kmer_projection_size':64})
 return c
def attach_kmer(ds):
 for s in ds: s['kmer_features']=matrix([s])[0].astype(np.float32)
def metric_arrays(pred,val):
 yt=np.array([s['target_id'] for s in val]); ya=np.array([s['affinity_class'] for s in val]); yz=np.array([s['log_kd_nm'] for s in val],float); idx=np.array([TARGETS.index(s['target_id']) for s in val]);
 tp=np.array(TARGETS)[pred['target_score'].argmax(1)]; ap=np.array(AFF)[pred['affinity_score'].argmax(1)]; z=pred['z']; f=macro_f1(yt,tp,TARGETS); af=macro_f1(ya,ap,AFF); e=mae(yz,z); return {'target_macro_f1':f,'affinity_macro_f1':af,'log_kd_mae':e,'combined_score':.3*f+.35*af+.35*max(0,1-e/2),'target_accuracy':float(np.mean(tp==yt)),'affinity_accuracy':float(np.mean(ap==ya))}
def add_knn_soft_labels(train, k=3, device=None):
    device=device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vh=matrix(train,'vh'); pair=matrix(train,'pair'); vh,_=standardize(vh,vh); pair,_=standardize(pair,pair)
    yt=np.array([TARGETS.index(s['target_id']) for s in train]); ya=np.array([AFF.index(s['affinity_class']) for s in train]); yz=np.array([s['log_kd_nm'] for s in train],float)
    for i,sample in enumerate(train):
        ref=np.delete(vh,i,0); rt=np.delete(yt,i); ra=np.delete(ya,i); rz=np.delete(yz,i)
        ts,_,_=predict(vh[i:i+1],ref,rt,ra,rz,k,device)
        pair_ref=np.delete(pair,i,0); _,ass,z=predict(pair[i:i+1],pair_ref,rt,ra,rz,k,device)
        sample['knn_target_soft']=ts[0]; sample['knn_affinity_soft']=ass[0]; sample['knn_log_kd']=float(z[0])

def main(root):
 train=read_dataset(root/'data/train.csv'); val=read_dataset(root/'data/val.csv'); lib=build_antigen_library(train,val); cdr=load_cdr_table(root/'data/cdr_annotations.csv'); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 xtr,xv=standardize(matrix(train),matrix(val)); ytar=np.array([TARGETS.index(s['target_id']) for s in train]); yaff=np.array([AFF.index(s['affinity_class']) for s in train]); yz=np.array([s['log_kd_nm'] for s in train],float); ts,as_,z= predict(xv,xtr,ytar,yaff,yz,3,device)
 out={}
 start=time.perf_counter(); base=train_stage3_model(train,lib,cdr,cfg()); bp=predict_stage3(base,val); bm=evaluate_stage3(base,val); out['m2']={'target_macro_f1':bm['target_macro_f1_ranking'],'affinity_macro_f1':bm['positive_affinity_macro_f1'],'log_kd_mae':bm['positive_log_kd_mae'],'combined_score':bm['combined_score_ranking'],'seconds':time.perf_counter()-start}
 # C: kmer branch
 attach_kmer(train); attach_kmer(val); start=time.perf_counter(); kb=train_stage3_model(train,lib,cdr,cfg(True)); km=evaluate_stage3(kb,val); out['transformer_2mer']={'target_macro_f1':km['target_macro_f1_ranking'],'affinity_macro_f1':km['positive_affinity_macro_f1'],'log_kd_mae':km['positive_log_kd_mae'],'combined_score':km['combined_score_ranking'],'seconds':time.perf_counter()-start}
 # E: leave-one-out KNN soft-target distillation on train only.
 add_knn_soft_labels(train,3,device)
 dc=cfg(); dc.update({'lambda_knn_target':0.15,'lambda_knn_affinity':0.10,'lambda_knn_regression':0.10,'knn_temperature':1.0})
 start=time.perf_counter(); db=train_stage3_model(train,lib,cdr,dc); dm=evaluate_stage3(db,val); out['knn_distillation']={'target_macro_f1':dm['target_macro_f1_ranking'],'affinity_macro_f1':dm['positive_affinity_macro_f1'],'log_kd_mae':dm['positive_log_kd_mae'],'combined_score':dm['combined_score_ranking'],'seconds':time.perf_counter()-start}
 # D: use correctly shaped independent VH and VH+antigen KNN features
 vtr,vv=standardize(matrix(train),matrix(val)); atr,av=standardize(np.stack([np.concatenate([matrix([s])[0],matrix([s], 'pair')[0]]) for s in train]),np.stack([np.concatenate([matrix([s])[0],matrix([s], 'pair')[0]]) for s in val]))
 kt,ka,kz=predict(vv,vtr,ytar,yaff,yz,3,device); _unused,kaa,kzz=predict(av,atr,ytar,yaff,yz,3,device)
 p= predict_stage3(base,val); idx=np.array([lib['target_to_index'][s['target_id']] for s in val]); tlog=torch.softmax(torch.tensor(p['target_logits']),1).numpy(); alog=torch.softmax(torch.tensor(p['affinity_logits'][np.arange(len(val)),idx]),1).numpy(); zl=p['log_kd_pred'][np.arange(len(val)),idx]
 for alpha in [.5,.75]: out[f'late_fusion_{alpha}']=metric_arrays({'target_score':alpha*kt+(1-alpha)*tlog,'affinity_score':alpha*kaa+(1-alpha)*alog,'z':alpha*kzz+(1-alpha)*zl},val)
 print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__': p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent);main(p.parse_args().root)
