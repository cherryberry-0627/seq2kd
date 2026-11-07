#!/usr/bin/env python3
"""CDR-only sequence VAE: framework fixed, ESM2/quality/similarity filtered."""
from __future__ import annotations
import argparse, copy, json, time
from pathlib import Path
import numpy as np
from minimal_model.candidates import build_antigen_library
from minimal_model.cdr import load_cdr_table
from minimal_model.esm_features import ESM2FeatureExtractor
from minimal_model.io import read_dataset
from minimal_model.cdr_sequence_vae import fit_cdr_vae, sample_cdr_regions
from minimal_model.augmentation import build_synthetic_cdr_table, merge_cdr_tables
from minimal_model.pairs import assert_disjoint_vh
from minimal_model.stage3 import DEFAULT_CONFIG, train_stage3_model, evaluate_stage3, predict_stage3

AA=set('ACDEFGHIKLMNPQRSTVWY')
def region(sample, table, names):
    spans=table[sample['sample_id']]['spans']; pieces=[]; valid=True
    for n in names:
        s=spans.get(n)
        if s is None: valid=False; break
        pieces.append(sample['vh_sequence_aa'][s[0]:s[1]])
    return ''.join(pieces) if valid else None

def replace_region(parent, table_entry, names, generated):
    seq=parent['vh_sequence_aa']; spans=table_entry['spans']; out=seq; offset=0
    pos=0
    for n in names:
        s=spans[n]; old=seq[s[0]:s[1]]
        new=generated[pos:pos+len(old)]; pos+=len(old)
        a=s[0]+offset; b=s[1]+offset; out=out[:a]+new+out[b:]; offset += len(new)-len(old)
    return out

def simple_quality(seq):
    """ProtParam/developability gate with Biopython when available and safe fallback."""
    if not seq or set(seq)-AA: return False
    try:
        from Bio.SeqUtils.ProtParam import ProteinAnalysis
        analysis=ProteinAnalysis(seq)
        return (analysis.instability_index() < 70 and -2.0 < analysis.gravy() < 1.5
                and 4.0 < analysis.isoelectric_point() < 12.0)
    except ImportError:
        hyd=sum(a in 'AILMFWVY' for a in seq)/len(seq)
        charge=(sum(a in 'KRH' for a in seq)-sum(a in 'DE' for a in seq))/len(seq)
        return .05 < hyd < .75 and -0.35 < charge < .45

def identity(a,b):
    n=min(len(a),len(b)); return sum(x==y for x,y in zip(a[:n],b[:n]))/max(len(a),len(b))

def config(epochs=10):
    c=dict(DEFAULT_CONFIG); c.update({'max_vh_length':512,'batch_size':64,'epochs':epochs,'use_cdr':True,'use_chunk_attention':False,'use_cross_attention':False,'freeze_encoder':False,'esm_dim':320,'esm_projection_size':64,'amp':True,'tf32':True,'device':'auto','seed':2026}); return c

def main(root, epochs=30, vae_epochs=50):
    train=read_dataset(root/'data/train.csv'); val=read_dataset(root/'data/val.csv'); assert_disjoint_vh(train,val)
    library=build_antigen_library(train,val); cdr=load_cdr_table(root/'data/cdr_annotations.csv'); ext=ESM2FeatureExtractor(device='auto',amp=True)
    ext.attach([train,val],batch_size=64); base=train_stage3_model(train,library,cdr,config(epochs)); base_metrics=evaluate_stage3(base,val); base_metrics['final_train_loss']=base.history[-1]['loss']; base_metrics['final_train_epoch']=base.history[-1]['epoch']
    print('=== baseline ==='); print(json.dumps(base_metrics,ensure_ascii=False,indent=2))
    target=np.array([library['target_to_index'][s['target_id']] for s in train]); aff=np.array([{'medium':0,'strong':1,'weak':2}[s['affinity_class']] for s in train])
    for names,label in [(('h3',),'cdr_h3'),(('h2','h3'),'cdr_h2_h3')]:
        valid=[(s,region(s,cdr,names)) for s in train]; valid=[x for x in valid if x[1] is not None]
        parent=[x[0] for x in valid]; regions=[x[1] for x in valid]; ti=np.array([library['target_to_index'][s['target_id']] for s in parent]); ai=np.array([{'medium':0,'strong':1,'weak':2}[s['affinity_class']] for s in parent])
        maxlen=max(map(len,regions)); vae=fit_cdr_vae(regions,ti,ai,maxlen,epochs=vae_epochs,batch_size=256,device=str(base.device),amp=True)
        spans=[cdr[s['sample_id']]['spans'] for s in parent]; rows=sample_cdr_regions(vae,regions,ti,ai,[s['sample_id'] for s in parent],spans,label,device=str(base.device))
        generated=[]
        for row,p in zip(rows,parent):
            item=dict(p); item.update(row); item['vh_sequence_aa']=replace_region(p,cdr[p['sample_id']],names,row['generated_region']); generated.append(item)
        ext.attach([generated],batch_size=64)
        # quality, ESM2 embedding cosine/score proxy, and train nearest identity filters
        trainseq=[s['vh_sequence_aa'] for s in train]; kept=[]; stats={'generated':len(generated),'quality':0,'novel':0}
        parent_scores=ext.score([p['vh_sequence_aa'] for p in parent],batch_size=64)
        generated_scores=ext.score([row['vh_sequence_aa'] for row in generated],batch_size=64)
        stats['esm_pass']=0
        for row,p,parent_score,generated_score in zip(generated,parent,parent_scores,generated_scores):
            # Keep generated sequences whose ESM2 average residue likelihood is not
            # more than 1.5 nats below the parent sequence.
            if generated_score < parent_score - 1.5: continue
            stats['esm_pass']+=1
            if not simple_quality(row['vh_sequence_aa']): continue
            stats['quality']+=1
            if max(identity(row['vh_sequence_aa'],x) for x in trainseq) >= .98: continue
            stats['novel']+=1; kept.append(row)
        # teacher ranking/affinity gate
        cdr_aug=merge_cdr_tables(cdr,build_synthetic_cdr_table(kept,cdr)); tb=copy.copy(base); tb.cdr_table=cdr_aug; pred=predict_stage3(tb,kept); accepted=[]
        for i,row in enumerate(kept):
            ti0=library['target_to_index'][row['target_id']]; order=pred['rank_score'][i].argsort()[::-1]; ai0={'medium':0,'strong':1,'weak':2}[row['affinity_class']]
            if int(order[0])==ti0 and int(pred['affinity_logits'][i,ti0].argmax())==ai0: accepted.append(row)
        cdr_final=merge_cdr_tables(cdr,build_synthetic_cdr_table(accepted,cdr)); start=time.perf_counter(); b=train_stage3_model(train+accepted,library,cdr_final,config(epochs)); m=evaluate_stage3(b,val); m['generation_seconds']=time.perf_counter()-start; m['final_train_loss']=b.history[-1]['loss']; m['final_train_epoch']=b.history[-1]['epoch']
        print('===',label,'==='); print(json.dumps({'filter':stats,'teacher_accepted':len(accepted),**m},ensure_ascii=False,indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent); p.add_argument('--epochs',type=int,default=30); p.add_argument('--vae-epochs',type=int,default=50); args=p.parse_args(); main(args.root,args.epochs,args.vae_epochs)
