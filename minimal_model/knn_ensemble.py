"""可复用的 2-mer KNN、后融合和蒸馏 soft targets。"""
from __future__ import annotations
from collections import Counter
import numpy as np
import torch

AA="ACDEFGHIKLMNPQRSTVWY"; HYDRO=set("AILMFWVY"); POS=set("KRH"); NEG=set("DE")
TARGET_IDS=np.array(["T01","T02","T03","T04","T05"]); AFFINITY_IDS=np.array(["medium","strong","weak"])

def seq_features(seq):
    seq=seq.upper(); n=max(1,len(seq)); counts=np.array([seq.count(a) for a in AA],float)/n
    extra=np.array([len(seq)/1000,sum(a in HYDRO for a in seq)/n,sum(a in POS for a in seq)/n,sum(a in NEG for a in seq)/n,(sum(a in POS for a in seq)-sum(a in NEG for a in seq))/n,seq.count('C')/n,seq.count('G')/n,seq.count('P')/n])
    pairs=Counter(seq[i:i+2] for i in range(len(seq)-1)); total=max(1,len(seq)-1)
    return np.concatenate([counts,extra,np.array([pairs[a+b]/total for a in AA for b in AA])])

def matrix(dataset, mode='vh'):
    return np.stack([seq_features(s['vh_sequence_aa']) if mode == 'vh' else np.concatenate([seq_features(s['vh_sequence_aa']), seq_features(s['antigen_sequence'])]) for s in dataset])
def standardize(train,val):
    mean=train.mean(0); scale=train.std(0); scale[scale<1e-8]=1; return (train-mean)/scale,(val-mean)/scale

def predict(query,reference,target,affinity,log_kd,k=3,device=None):
    device=device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    q=torch.as_tensor(query,dtype=torch.float32,device=device); r=torch.as_tensor(reference,dtype=torch.float32,device=device)
    dist=torch.cdist(q,r); values,indices=dist.topk(min(k,len(r)),largest=False); weights=1/(values+1e-6)
    target=torch.as_tensor(target,device=device); affinity=torch.as_tensor(affinity,device=device); z=torch.as_tensor(log_kd,dtype=torch.float32,device=device)
    ts=torch.stack([(weights*(target[indices]==c).float()).sum(1) for c in range(5)],1); ts=ts/ts.sum(1,keepdim=True).clamp_min(1e-9)
    ass=torch.stack([(weights*(affinity[indices]==c).float()).sum(1) for c in range(3)],1); ass=ass/ass.sum(1,keepdim=True).clamp_min(1e-9)
    zp=(weights*z[indices]).sum(1)/weights.sum(1)
    return ts.cpu().numpy(),ass.cpu().numpy(),zp.cpu().numpy()

def distill_loss(student_target_logits,student_affinity_logits,student_z,teacher_target,teacher_affinity,teacher_z,temperature=1.0):
    """KNN soft target 蒸馏损失；调用者可与真实标签损失加权。"""
    import torch.nn.functional as F
    target_loss=F.kl_div(F.log_softmax(student_target_logits/temperature,-1),torch.as_tensor(teacher_target,dtype=torch.float32,device=student_target_logits.device),reduction='batchmean')*temperature**2
    affinity_loss=F.kl_div(F.log_softmax(student_affinity_logits/temperature,-1),torch.as_tensor(teacher_affinity,dtype=torch.float32,device=student_affinity_logits.device),reduction='batchmean')*temperature**2
    regression_loss=F.smooth_l1_loss(student_z,torch.as_tensor(teacher_z,dtype=torch.float32,device=student_z.device))
    return target_loss,affinity_loss,regression_loss
