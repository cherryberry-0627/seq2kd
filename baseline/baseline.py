#!/usr/bin/env python3
"""NumPy-only baseline for the public train/validation data."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import numpy as np

AA = 'ACDEFGHIKLMNPQRSTVWY'
HYDRO = set('AILMFWVY')
POS = set('KRH')
NEG = set('DE')

def seq_features(seq: str) -> np.ndarray:
    seq = seq.strip().upper(); n = max(1, len(seq))
    counts = np.array([seq.count(a) / n for a in AA], dtype=float)
    extras = np.array([len(seq)/1000, sum(a in HYDRO for a in seq)/n,
        sum(a in POS for a in seq)/n, sum(a in NEG for a in seq)/n,
        (sum(a in POS for a in seq)-sum(a in NEG for a in seq))/n,
        seq.count('C')/n, seq.count('G')/n, seq.count('P')/n])
    return np.concatenate([counts, extras])

def features(rows, mode):
    out=[]
    for r in rows:
        vh=seq_features(r['vh_sequence_aa'])
        out.append(vh if mode=='target' else np.concatenate([vh,seq_features(r['antigen_sequence'])]))
    return np.asarray(out,float)

def read(path):
    with open(path,encoding='utf-8',newline='') as f:return list(csv.DictReader(f))

def standardize(a,b):
    mu=a.mean(0); sd=a.std(0); sd[sd<1e-8]=1
    return (a-mu)/sd,(b-mu)/sd

def centroid(x,y,z,classes):
    c=np.vstack([x[np.asarray(y)==k].mean(0) for k in classes])
    return np.asarray(classes)[np.argmin(((z[:,None,:]-c[None,:,:])**2).sum(2),1)]

def macro_f1(y,p,classes):
    vals=[]
    for c in classes:
        tp=((y==c)&(p==c)).sum(); fp=((y!=c)&(p==c)).sum(); fn=((y==c)&(p!=c)).sum()
        pr=tp/(tp+fp) if tp+fp else 0; re=tp/(tp+fn) if tp+fn else 0
        vals.append(2*pr*re/(pr+re) if pr+re else 0)
    return float(np.mean(vals))

def ridge(x,y,z,alpha=10.):
    xt=np.c_[np.ones(len(x)),x]; ze=np.c_[np.ones(len(z)),z]
    reg=np.eye(xt.shape[1])*alpha; reg[0,0]=0
    return ze@np.linalg.solve(xt.T@xt+reg,xt.T@y)

def main(root):
    train=read(root/'data/train.csv'); val=read(root/'data/val.csv')
    out={}
    xt,xv=standardize(features(train,'target'),features(val,'target'))
    yt=np.array([r['target_id'] for r in train]); yv=np.array([r['target_id'] for r in val])
    classes=sorted(set(yt)); pv=centroid(xt,yt,xv,classes)
    out['target_task']={'val_macro_f1':macro_f1(yv,pv,classes),'classes':classes}
    xt,xv=standardize(features(train,'affinity'),features(val,'affinity'))
    classes=['medium','strong','weak']; at=np.array([r['affinity_class'] for r in train]); av=np.array([r['affinity_class'] for r in val])
    ap=centroid(xt,at,xv,classes); logt=np.array([float(r['log_kd_nm']) for r in train]); logv=np.array([float(r['log_kd_nm']) for r in val])
    pred=ridge(xt,logt,xv); mae=float(np.mean(np.abs(logv-pred)))
    out['affinity_task']={'val_macro_f1':macro_f1(av,ap,classes),'val_mae_log10_kd':mae,'classes':classes}
    out['combined_score']={'val':0.30*out['target_task']['val_macro_f1']+0.35*out['affinity_task']['val_macro_f1']+0.35*max(0,1-mae/2)}
    print(json.dumps(out,ensure_ascii=False,indent=2))
    (root/'results').mkdir(parents=True, exist_ok=True)
    (root/'results/baseline_metrics.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]); main(ap.parse_args().root)
