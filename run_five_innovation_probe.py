#!/usr/bin/env python3
"""五个创新点的轻量基础探测：不改主模型训练，只在统一 val 上比较。"""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np, torch
from minimal_model.io import read_dataset
from minimal_model.candidates import build_antigen_library
from minimal_model.cdr import load_cdr_table
from minimal_model.knn_ensemble import matrix,standardize,predict
from minimal_model.metrics import macro_f1,mae
from minimal_model.stage3 import train_stage3_model,predict_stage3,evaluate_stage3,DEFAULT_CONFIG
TARGETS=['T01','T02','T03','T04','T05']; AFF=['medium','strong','weak']
def cfg():
 c=dict(DEFAULT_CONFIG); c.update({'max_vh_length':512,'batch_size':64,'epochs':30,'use_cdr':True,'use_chunk_attention':False,'use_cross_attention':False,'freeze_encoder':False,'amp':True,'tf32':True,'device':'auto','seed':2026}); return c
def eval_scores(ts,ass,z,val):
 yt=np.array([s['target_id'] for s in val]); ya=np.array([s['affinity_class'] for s in val]); yz=np.array([s['log_kd_nm'] for s in val],float); tp=np.array(TARGETS)[ts.argmax(1)]; ap=np.array(AFF)[ass.argmax(1)]; f=macro_f1(yt,tp,TARGETS); af=macro_f1(ya,ap,AFF); e=mae(yz,z); return {'target_f1':f,'affinity_f1':af,'log_kd_mae':e,'combined':.3*f+.35*af+.35*max(0,1-e/2),'target_acc':float(np.mean(tp==yt)),'affinity_acc':float(np.mean(ap==ya))}
def entropy(p): return -(p*np.log(np.maximum(p,1e-9))).sum(1)
def main(root):
 train=read_dataset(root/'data/train.csv'); val=read_dataset(root/'data/val.csv'); lib=build_antigen_library(train,val); cdr=load_cdr_table(root/'data/cdr_annotations.csv'); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); out={}
 xtr,xv=standardize(matrix(train,'vh'),matrix(val,'vh')); ptr,pva=standardize(matrix(train,'pair'),matrix(val,'pair')); yt=np.array([TARGETS.index(s['target_id']) for s in train]); ya=np.array([AFF.index(s['affinity_class']) for s in train]); yz=np.array([s['log_kd_nm'] for s in train],float)
 ts,_,_=predict(xv,xtr,yt,ya,yz,3,device); _,ass,z=predict(pva,ptr,yt,ya,yz,3,device)
 start=time.perf_counter(); b=train_stage3_model(train,lib,cdr,cfg()); p=predict_stage3(b,val); m=evaluate_stage3(b,val); out['baseline']={'target_f1':m['target_macro_f1_ranking'],'affinity_f1':m['positive_affinity_macro_f1'],'log_kd_mae':m['positive_log_kd_mae'],'combined':m['combined_score_ranking'],'seconds':time.perf_counter()-start}
 idx=np.array([lib['target_to_index'][s['target_id']] for s in val]); tr_t=torch.softmax(torch.tensor(p['target_logits']),1).numpy(); tr_a=torch.softmax(torch.tensor(p['affinity_logits'][np.arange(len(val)),idx]),1).numpy(); tr_z=p['log_kd_pred'][np.arange(len(val)),idx]
 # 1 adaptive confidence gate
 conf=1-entropy(ts)/np.log(5); conf=conf[:,None]; adaptive=eval_scores(conf*ts+(1-conf)*tr_t,conf*ass+(1-conf)*tr_a,conf[:,0]*z+(1-conf[:,0])*tr_z,val); out['adaptive_gate']=adaptive
 # 2 multi-view retrieval: VH 2mer + ESM-free pair view, average normalized predictions
 ts2,as2,z2=predict(xv,xtr,yt,ya,yz,5,device); mv=eval_scores((ts+ts2)/2,(ass+as2)/2,(z+z2)/2,val); out['multi_view_knn']=mv
 # 3 learned-distance proxy: weighted 2mer vs global feature views, select metric by train internal neighbor consistency
 for w in [.25,.5,.75]:
  # feature scaling changes local metric; use weighted normalized distance by direct concatenation scaling
  mix_xtr=np.concatenate([xtr*w,ptr*(1-w)],1); mix_xv=np.concatenate([xv*w,pva*(1-w)],1); tma,ama,zm=predict(mix_xv,mix_xtr,yt,ya,yz,3,device); out[f'learned_metric_w{w}']=eval_scores(tma,ama,zm,val)
 # 4 distillation proxy: blend KNN soft targets into Transformer with confidence-weighted low alpha
 for a in [.1,.2,.3]: out[f'confidence_distill_proxy_{a}']=eval_scores(a*conf*ts+(1-a*conf)*tr_t,a*conf*ass+(1-a*conf)*tr_a,a*conf[:,0]*z+(1-a*conf[:,0])*tr_z,val)
 # 5 task-decoupled adaptive fusion
 out['task_specific_gate']=eval_scores(.75*ts+.25*tr_t,.5*ass+.5*tr_a,.75*z+.25*tr_z,val)
 print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__': p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent);main(p.parse_args().root)
