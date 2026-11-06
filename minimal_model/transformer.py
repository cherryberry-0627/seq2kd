"""PyTorch Transformer 编码器、长抗原池化、融合层和预测头。"""
from __future__ import annotations
import torch
from torch import nn


class TransformerEncoder(nn.Module):
    """字符 embedding + positional embedding + PyTorch TransformerEncoder。"""
    def __init__(self, vocab_size: int, max_length: int, hidden_size: int = 64,
                 num_heads: int = 4, ff_size: int = 128, num_layers: int = 2, dropout: float = .1):
        super().__init__()
        if hidden_size % num_heads:
            raise ValueError("hidden_size must be divisible by num_heads")
        self.embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.position = nn.Embedding(max_length, hidden_size)
        layer = nn.TransformerEncoderLayer(d_model=hidden_size, nhead=num_heads,
            dim_feedforward=ff_size, dropout=dropout, activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(hidden_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        bsz, length = input_ids.shape
        positions = torch.arange(length, device=input_ids.device).unsqueeze(0).expand(bsz, length)
        x = self.embedding(input_ids) + self.position(positions)
        x = self.encoder(x, src_key_padding_mask=~attention_mask.bool())
        mask = attention_mask.unsqueeze(-1).to(x.dtype)
        return self.norm((x * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0))


class TransformerModel(nn.Module):
    """VH/抗原双编码器，可选融合 ProtParam 数值特征。"""
    def __init__(self, tokenizer, hidden_size=64, num_heads=4, ff_size=128,
                 num_layers=2, fusion_size=96, dropout=.1, protparam_dim=0,
                 protparam_projection_size=32):
        super().__init__()
        self.tokenizer = tokenizer
        self.hidden_size = hidden_size
        self.protparam_dim = protparam_dim
        self.encoder_vh = TransformerEncoder(tokenizer.vocab_size, tokenizer.max_vh_length,
            hidden_size, num_heads, ff_size, num_layers, dropout)
        self.encoder_antigen = TransformerEncoder(tokenizer.vocab_size, tokenizer.antigen_chunk_length,
            hidden_size, num_heads, ff_size, num_layers, dropout)
        if protparam_dim:
            self.protparam_projection = nn.Sequential(
                nn.Linear(protparam_dim, protparam_projection_size), nn.LayerNorm(protparam_projection_size), nn.GELU())
            self.vh_protparam_projection = nn.Sequential(
                nn.Linear(protparam_dim // 2, protparam_projection_size), nn.LayerNorm(protparam_projection_size), nn.GELU())
            target_input_size = hidden_size + protparam_projection_size
            fusion_input_size = hidden_size * 4 + protparam_projection_size
        else:
            self.protparam_projection = None
            self.vh_protparam_projection = None
            target_input_size = hidden_size
            fusion_input_size = hidden_size * 4
        self.fusion = nn.Sequential(nn.Linear(fusion_input_size, fusion_size), nn.GELU(), nn.Dropout(dropout))
        self.target_head = nn.Linear(target_input_size, 5)
        self.affinity_head = nn.Linear(fusion_size, 3)
        self.regression_head = nn.Linear(fusion_size, 1)

    def encode_batch(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        vh = self.encoder_vh(batch["vh_input_ids"], batch["vh_attention_mask"])
        ids = batch["antigen_input_ids"]
        masks = batch["antigen_attention_mask"]
        bsz, chunks, length = ids.shape
        flat_ids = ids.reshape(bsz * chunks, length)
        flat_masks = masks.reshape(bsz * chunks, length).clone()
        # Padded chunk rows have an all-False mask; PyTorch attention would
        # produce NaN gradients for an all-masked row. Keep one harmless token
        # valid for those rows, then exclude the whole chunk via chunk_mask.
        invalid_chunks = ~flat_masks.any(dim=-1)
        flat_masks[invalid_chunks, 0] = True
        ant = self.encoder_antigen(flat_ids, flat_masks)
        ant = torch.nan_to_num(ant.reshape(bsz, chunks, self.hidden_size), nan=0.0)
        chunk_mask = batch["antigen_chunk_mask"].unsqueeze(-1).to(ant.dtype)
        antigen = (ant * chunk_mask).sum(dim=1) / chunk_mask.sum(dim=1).clamp_min(1.0)
        return vh, antigen

    def forward(self, batch: dict[str, torch.Tensor], task: str = "both") -> dict[str, torch.Tensor]:
        vh, antigen = self.encode_batch(batch)
        protparam = batch.get("protparam_features")
        if self.protparam_dim:
            if protparam is None:
                raise ValueError("protparam_features are required when protparam_dim is enabled")
            protparam = protparam.to(vh.dtype)
            vh_extra = self.vh_protparam_projection(protparam[:, :self.protparam_dim // 2])
            target_input = torch.cat([vh, vh_extra], dim=-1)
        else:
            target_input = vh
        output = {"vh_embedding": vh, "antigen_embedding": antigen, "target_logits": self.target_head(target_input)}
        if task != "target":
            fused = torch.cat([vh, antigen, torch.abs(vh - antigen), vh * antigen], dim=-1)
            if self.protparam_dim:
                fused = torch.cat([fused, self.protparam_projection(protparam)], dim=-1)
            hidden = self.fusion(fused)
            output.update({"fusion": hidden, "affinity_logits": self.affinity_head(hidden),
                           "log_kd_pred": self.regression_head(hidden).squeeze(-1)})
        return output
