#!/usr/bin/env python3
"""训练 VH-prior CFG：VH-only base + VH-antigen conditional ranking。"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from minimal_model.candidates import build_antigen_library
from minimal_model.cfg_vh_prior import DEFAULT_CONFIG, evaluate_vh_prior, train_vh_prior_model
from minimal_model.io import read_dataset

def main(root: Path):
    train=read_dataset(root/'data/train.csv'); val=read_dataset(root/'data/val.csv')
    library=build_antigen_library(train,val)
    config=dict(DEFAULT_CONFIG, epochs=10, batch_size=16, freeze_encoder=True,
                lambda_prior=0.5, lambda_rank=0.5, lambda_reg=1.0,
                lambda_affinity=1.0, seed=2026, guidance_scale=1.0)
    t=time.perf_counter(); bundle=train_vh_prior_model(train,library,config)
    print('device:',bundle.config['device_used'],'train_seconds:',time.perf_counter()-t)
    print('config:',json.dumps(bundle.config,ensure_ascii=False,sort_keys=True))
    print('final_train:',json.dumps(bundle.history[-1],ensure_ascii=False))
    for scale in (0.0,0.5,1.0,1.5,2.0,3.0):
        metrics=evaluate_vh_prior(bundle,val,guidance_scale=scale)
        print(json.dumps(metrics,ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    main(p.parse_args().root)
