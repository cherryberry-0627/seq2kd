"""VH-prior CFG：VH 是始终可见的 environment，抗原是可选 condition。

base score: VH-only prior logits [B,5]
cond score: VH + candidate antigen pair rank logits [B,5]
guided score = base + scale * (cond - base)

CFG 只用于候选 target ranking；log-Kd 和 affinity 使用 conditional pair 分支。
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from .candidates import TARGET_IDS, AFFINITY_IDS, collate_candidates
from .metrics import macro_f1, mae
from .tokenizer import build_tokenizer
from .training import AFFINITY_TO_ID, ID_TO_AFFINITY, ID_TO_TARGET, batch_slices, resolve_device
from .transformer import TransformerEncoder

DEFAULT_CONFIG = {
    "hidden_size": 64, "num_heads": 4, "ff_size": 128, "num_layers": 2,
    "fusion_size": 96, "dropout": 0.1, "max_vh_length": 192,
    "antigen_chunk_length": 192, "antigen_overlap": 32, "batch_size": 16,
    "epochs": 10, "learning_rate": 2e-3, "weight_decay": 1e-4,
    "grad_clip": 1.0, "seed": 2026, "device": "auto", "freeze_encoder": True,
    "lambda_prior": 0.5, "lambda_rank": 0.5, "lambda_reg": 1.0,
    "lambda_affinity": 1.0, "guidance_scale": 1.0,
}

class VHPriorCFGModel(nn.Module):
    """共享 VH/抗原编码器，输出 VH prior 与 VH-antigen conditional scores。"""
    def __init__(self, tokenizer, hidden_size=64, num_heads=4, ff_size=128,
                 num_layers=2, fusion_size=96, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.encoder_vh = TransformerEncoder(tokenizer.vocab_size, tokenizer.max_vh_length,
            hidden_size, num_heads, ff_size, num_layers, dropout)
        self.encoder_antigen = TransformerEncoder(tokenizer.vocab_size, tokenizer.antigen_chunk_length,
            hidden_size, num_heads, ff_size, num_layers, dropout)
        self.prior_head = nn.Linear(hidden_size, 5)
        self.fusion = nn.Sequential(nn.Linear(hidden_size * 4, fusion_size), nn.GELU(), nn.Dropout(dropout))
        self.rank_head = nn.Linear(fusion_size, 1)
        self.regression_head = nn.Linear(fusion_size, 1)
        self.affinity_head = nn.Linear(fusion_size, 3)

    def encode_vh(self, batch):
        return self.encoder_vh.forward(batch["vh_input_ids"], batch["vh_attention_mask"])

    def encode_antigen(self, batch):
        ids, masks = batch["antigen_input_ids"], batch["antigen_attention_mask"]
        bsz, candidates, chunks, length = ids.shape
        flat_ids = ids.reshape(bsz * candidates * chunks, length)
        flat_masks = masks.reshape(bsz * candidates * chunks, length).clone()
        invalid = ~flat_masks.any(dim=-1)
        flat_masks[invalid, 0] = True
        encoded = self.encoder_antigen.forward(flat_ids, flat_masks)
        encoded = torch.nan_to_num(encoded.reshape(bsz, candidates, chunks, self.hidden_size), nan=0.0)
        chunk_mask = batch["antigen_chunk_mask"].unsqueeze(-1).to(encoded.dtype)
        return (encoded * chunk_mask).sum(dim=2) / chunk_mask.sum(dim=2).clamp_min(1.0)

    def forward_prior(self, vh):
        return self.prior_head(vh)

    def forward_conditional(self, vh, antigen):
        vh = vh.unsqueeze(1).expand(-1, antigen.shape[1], -1)
        fused = torch.cat([vh, antigen, (vh - antigen).abs(), vh * antigen], dim=-1)
        pair = self.fusion(fused)
        return {
            "rank_score": self.rank_head(pair).squeeze(-1),
            "log_kd_pred": self.regression_head(pair).squeeze(-1),
            "affinity_logits": self.affinity_head(pair),
        }

@dataclass
class VHPriorBundle:
    model: nn.Module
    tokenizer: object
    antigen_tokens: dict
    library: dict
    config: dict
    device: torch.device
    history: list

def train_vh_prior_model(train_dataset, library, config=None):
    cfg = {**DEFAULT_CONFIG, **(config or {})}
    torch.manual_seed(cfg["seed"])
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(cfg["seed"])
    device = resolve_device(cfg["device"])
    tokenizer = build_tokenizer(max_vh_length=cfg["max_vh_length"], antigen_chunk_length=cfg["antigen_chunk_length"], antigen_overlap=cfg["antigen_overlap"])
    antigen_tokens = {k: v.to(device) for k, v in tokenizer.tokenize_antigen_chunks(library["sequences"]).items()}
    model = VHPriorCFGModel(tokenizer, cfg["hidden_size"], cfg["num_heads"], cfg["ff_size"], cfg["num_layers"], cfg["fusion_size"], cfg["dropout"]).to(device)
    if cfg["freeze_encoder"]:
        for p in list(model.encoder_vh.parameters()) + list(model.encoder_antigen.parameters()): p.requires_grad_(False)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"])
    history=[]
    for epoch in range(cfg["epochs"]):
        model.train()
        if cfg["freeze_encoder"]: model.encoder_vh.eval(); model.encoder_antigen.eval()
        totals={"loss":0.,"prior_loss":0.,"rank_loss":0.,"regression_loss":0.,"affinity_loss":0.}; total_n=0
        for indices in batch_slices(len(train_dataset), cfg["batch_size"], True, cfg["seed"]+epoch):
            samples=[train_dataset[int(i)] for i in indices]
            batch=collate_candidates(samples, tokenizer, antigen_tokens, library["target_to_index"], device)
            optimizer.zero_grad(set_to_none=True)
            vh=model.encode_vh(batch); antigen=model.encode_antigen(batch)
            prior_logits=model.forward_prior(vh); cond=model.forward_conditional(vh, antigen)
            index=batch["target_index"]; rows=torch.arange(len(samples),device=device)
            prior_loss=F.cross_entropy(prior_logits,index)
            rank_loss=F.cross_entropy(cond["rank_score"],index)
            z=cond["log_kd_pred"][rows,index]
            regression_loss=F.huber_loss(z,batch["log_kd"],delta=1.0)
            cls=cond["affinity_logits"][rows,index]
            affinity_loss=F.cross_entropy(cls,batch["affinity_labels"])
            loss=(cfg["lambda_prior"]*prior_loss+cfg["lambda_rank"]*rank_loss+cfg["lambda_reg"]*regression_loss+cfg["lambda_affinity"]*affinity_loss)
            loss.backward()
            if cfg["grad_clip"] is not None: nn.utils.clip_grad_norm_(model.parameters(),cfg["grad_clip"])
            optimizer.step()
            n=len(samples); total_n+=n
            for key,val in {"loss":loss,"prior_loss":prior_loss,"rank_loss":rank_loss,"regression_loss":regression_loss,"affinity_loss":affinity_loss}.items(): totals[key]+=float(val.detach().cpu())*n
        history.append({"epoch":epoch+1,**{k:v/max(total_n,1) for k,v in totals.items()}})
    cfg=dict(cfg); cfg["device_used"]=str(device)
    return VHPriorBundle(model,tokenizer,antigen_tokens,library,cfg,device,history)

def predict_vh_prior(bundle,dataset,guidance_scale=None):
    model,device=bundle.model,bundle.device; scale=float(bundle.config["guidance_scale"] if guidance_scale is None else guidance_scale)
    model.eval(); prior_all=[]; cond_rank_all=[]; cond_z_all=[]; cond_cls_all=[]
    with torch.no_grad():
        for indices in batch_slices(len(dataset),bundle.config["batch_size"],False,0):
            samples=[dataset[int(i)] for i in indices]
            batch=collate_candidates(samples,bundle.tokenizer,bundle.antigen_tokens,bundle.library["target_to_index"],device)
            vh=model.encode_vh(batch); antigen=model.encode_antigen(batch)
            prior=model.forward_prior(vh); cond=model.forward_conditional(vh,antigen)
            guided=prior+scale*(cond["rank_score"]-prior)
            prior_all.append(prior); cond_rank_all.append(guided); cond_z_all.append(cond["log_kd_pred"]); cond_cls_all.append(cond["affinity_logits"])
    return {"prior_logits":torch.cat(prior_all).cpu().numpy(),"guided_rank_score":torch.cat(cond_rank_all).cpu().numpy(),"conditional_rank_score":torch.cat(cond_rank_all).cpu().numpy(),"conditional_log_kd":torch.cat(cond_z_all).cpu().numpy(),"conditional_affinity_logits":torch.cat(cond_cls_all).cpu().numpy()}

def evaluate_vh_prior(bundle,dataset,guidance_scale=None):
    pred=predict_vh_prior(bundle,dataset,guidance_scale); true=np.array([s["target_id"] for s in dataset]); aff_true=np.array([s["affinity_class"] for s in dataset]); log_true=np.array([s["log_kd_nm"] for s in dataset],float); index=np.array([bundle.library["target_to_index"][s["target_id"]] for s in dataset]); rows=np.arange(len(dataset))
    prior_pred=ID_TO_TARGET[pred["prior_logits"].argmax(1)]; guided_pred=ID_TO_TARGET[pred["guided_rank_score"].argmax(1)]; z=pred["conditional_log_kd"][rows,index]; aff=ID_TO_AFFINITY[pred["conditional_affinity_logits"][rows,index].argmax(1)]
    return {"target_macro_f1_vh_only":macro_f1(true,prior_pred,list(TARGET_IDS)),"target_accuracy_vh_only":float(np.mean(prior_pred==true)),"target_macro_f1_guided":macro_f1(true,guided_pred,list(TARGET_IDS)),"target_accuracy_guided":float(np.mean(guided_pred==true)),"positive_affinity_macro_f1":macro_f1(aff_true,aff,list(AFFINITY_IDS)),"positive_log_kd_mae":mae(log_true,z),"guidance_scale":float(bundle.config["guidance_scale"] if guidance_scale is None else guidance_scale),"n_samples":len(dataset)}
