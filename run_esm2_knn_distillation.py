#!/usr/bin/env python3
"""M2 / M2+ESM2 / KNN蒸馏 / ESM2+KNN蒸馏四组正式对比。"""
from __future__ import annotations
import argparse,json,time
from pathlib import Path
import numpy as np, torch
from minimal_model.io import read_dataset
from minimal_model.candidates import build_antigen_library
from minimal_model.cdr import load_cdr_table
from minimal_model.esm_features import ESM2FeatureExtractor
from minimal_model.knn_ensemble import matrix,standardize,predict
from minimal_model.stage3 import train_stage3_model,evaluate_stage3,DEFAULT_CONFIG
TARGETS=['T01','T02','T03','T04','T05']; AFF=['medium','strong','weak']
def cfg(esm=False,distill=False):
 c=dict(DEFAULT_CONFIG); c.update({'max_vh_length':512,'batch_size':64,'epochs':30,'use_cdr':True,'use_chunk_attention':False,'use_cross_attention':False,'freeze_encoder':False,'amp':True,'tf32':True,'device':'auto','seed':2026})
 if esm: c.update({'esm_dim':320,'esm_projection_size':64})
 if distill: c.update({'lambda_knn_target':.15,'lambda_knn_affinity':.10,'lambda_knn_regression':.10,'knn_temperature':1.0})
 return c
def add_loo(train,device,k=3):
 x,_=standardize(matrix(train,'vh'),matrix(train,'vh')); p,_=standardize(matrix(train,'pair'),matrix(train,'pair')); y=np.array([TARGETS.index(s['target_id']) for s in train]); a=np.array([AFF.index(s['affinity_class']) for s in train]); z=np.array([s['log_kd_nm'] for s in train],float)
 for i,s in enumerate(train):
  ref=np.delete(x,i,0); rt=np.delete(y,i); ra=np.delete(a,i); rz=np.delete(z,i); ts,_,_=predict(x[i:i+1],ref,rt,ra,rz,k,device); refp=np.delete(p,i,0); _,ass,zz=predict(p[i:i+1],refp,rt,ra,rz,k,device); s['knn_target_soft']=ts[0]; s['knn_affinity_soft']=ass[0]; s['knn_log_kd']=float(zz[0])
def main(root):
 train=read_dataset(root/'data/train.csv'); val=read_dataset(root/'data/val.csv'); lib=build_antigen_library(train,val); cdr=load_cdr_table(root/'data/cdr_annotations.csv'); device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'); ext=ESM2FeatureExtractor(device='auto',amp=True); ext.attach([train,val],64); add_loo(train,device,3); out={}
 for name,esm,distill in [('m2',False,False),('m2_esm2',True,False),('m2_knn_distill',False,True),('m2_esm2_knn_distill',True,True)]:
  start=time.perf_counter(); b=train_stage3_model(train,lib,cdr,cfg(esm,distill)); m=evaluate_stage3(b,val); out[name]={'target_f1':m['target_macro_f1_ranking'],'affinity_f1':m['positive_affinity_macro_f1'],'log_kd_mae':m['positive_log_kd_mae'],'combined':m['combined_score_ranking'],'seconds':time.perf_counter()-start}
 print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__': p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent);main(p.parse_args().root)
