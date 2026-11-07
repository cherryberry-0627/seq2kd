#!/usr/bin/env python3
"""R1: violation-gated deterministic hard-ranking budget; rank head only."""
from __future__ import annotations
import argparse, copy, json, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from minimal_model.io import read_dataset
from minimal_model.candidates import build_antigen_library
from minimal_model.cdr import load_cdr_table
from minimal_model.stage3 import DEFAULT_CONFIG, collate_stage3, evaluate_stage3, train_stage3_model
from minimal_model.training import batch_slices


def cfg():
 c=dict(DEFAULT_CONFIG); c.update({'max_vh_length':512,'batch_size':64,'epochs':30,'use_cdr':True,'use_chunk_attention':False,'use_cross_attention':False,'freeze_encoder':False,'amp':True,'tf32':True,'device':'auto','seed':2026}); return c

def finetune(base, train, mode='discrete', base_lambda=.05, margin=.0):
 model=copy.deepcopy(base.model).to(base.device); device=base.device; model.train()
 for p in model.parameters(): p.requires_grad_(False)
 for p in model.rank_head.parameters(): p.requires_grad_(True)
 optimizer=torch.optim.AdamW(model.rank_head.parameters(),lr=2e-3,weight_decay=1e-4)
 amp=device.type=='cuda'; scaler=torch.cuda.amp.GradScaler(enabled=amp); history=[]; active_total=0; seen=0
 for epoch in range(10):
  total=0.; n=0
  for ids in batch_slices(len(train),64,True,2026+epoch):
   samples=[train[int(i)] for i in ids]; batch=collate_stage3(samples,base.tokenizer,base.antigen_tokens,base.library['target_to_index'],base.cdr_table,device=device)
   optimizer.zero_grad(set_to_none=True)
   with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=amp):
    out=model.forward_candidates(batch); scores=out['rank_score']; target=batch['target_index']; rows=torch.arange(len(target),device=device)
    target_score=scores[rows,target]; masked=scores.clone(); masked[rows,target]=torch.finfo(scores.dtype).min; hard=masked.max(1).values
    violation=hard-target_score+margin; active=(violation>0).float()
    lam=base_lambda*active if mode=='discrete' else base_lambda*torch.sigmoid(violation/0.25)
    per_sample=F.softplus(violation)*lam; denom=lam.sum().clamp_min(1e-6); loss=per_sample.sum()/denom
   if float(lam.sum())>0:
    scaler.scale(loss).backward(); scaler.step(optimizer); scaler.update()
   active_total+=int(active.sum().detach().cpu()); seen+=len(samples); total+=float(loss.detach().cpu())*len(samples); n+=len(samples)
  history.append({'epoch':epoch+1,'loss':total/max(n,1),'active_rate':active_total/max(seen,1)})
 result=copy.copy(base); result.model=model; result.history=history; result.active_rate=active_total/max(seen,1); return result

def main(root):
 train=read_dataset(root/'data/train.csv'); val=read_dataset(root/'data/val.csv'); lib=build_antigen_library(train,val); cdr=load_cdr_table(root/'data/cdr_annotations.csv'); c=cfg(); start=time.perf_counter(); base=train_stage3_model(train,lib,cdr,c); out={}
 bm=evaluate_stage3(base,val); out['baseline']={'target_f1_vh_only':bm['target_macro_f1_vh_only'],'target_f1_ranking':bm['target_macro_f1_ranking'],'target_acc_ranking':bm['target_accuracy_ranking'],'affinity_f1':bm['positive_affinity_macro_f1'],'log_kd_mae':bm['positive_log_kd_mae'],'mrr':bm['ranking_mrr'],'combined':bm['combined_score_ranking']}
 for name,mode in [('r1_discrete','discrete'),('r1_continuous','continuous')]:
  b=finetune(base,train,mode,.05,0.0); m=evaluate_stage3(b,val); out[name]={'target_f1_vh_only':m['target_macro_f1_vh_only'],'target_f1_ranking':m['target_macro_f1_ranking'],'target_acc_ranking':m['target_accuracy_ranking'],'affinity_f1':m['positive_affinity_macro_f1'],'log_kd_mae':m['positive_log_kd_mae'],'mrr':m['ranking_mrr'],'combined':m['combined_score_ranking'],'active_rate':b.active_rate,'history_tail':b.history[-3:]}
 out['seconds']=time.perf_counter()-start; print(json.dumps(out,ensure_ascii=False,indent=2))
if __name__=='__main__': p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent); main(p.parse_args().root)
