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
    """VH/抗原双编码器，多任务 target、affinity、log-Kd 输出。"""
    def __init__(self, tokenizer, hidden_size=64, num_heads=4, ff_size=128,
                 num_layers=2, fusion_size=96, dropout=.1):
        super().__init__()
        self.tokenizer = tokenizer
        self.hidden_size = hidden_size
        self.encoder_vh = TransformerEncoder(tokenizer.vocab_size, tokenizer.max_vh_length,
            hidden_size, num_heads, ff_size, num_layers, dropout)
        self.encoder_antigen = TransformerEncoder(tokenizer.vocab_size, tokenizer.antigen_chunk_length,
            hidden_size, num_heads, ff_size, num_layers, dropout)
        self.fusion = nn.Sequential(nn.Linear(hidden_size * 4, fusion_size), nn.GELU(), nn.Dropout(dropout))
        self.target_head = nn.Linear(hidden_size, 5)
        self.affinity_head = nn.Linear(fusion_size, 3)
        self.regression_head = nn.Linear(fusion_size, 1)

    def encode_batch(self, batch: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        vh = self.encoder_vh(batch["vh_input_ids"], batch["vh_attention_mask"])
        ids = batch["antigen_input_ids"]
        masks = batch["antigen_attention_mask"]
        bsz, chunks, length = ids.shape
        ant = self.encoder_antigen(ids.reshape(bsz * chunks, length), masks.reshape(bsz * chunks, length))
        ant = torch.nan_to_num(ant.reshape(bsz, chunks, self.hidden_size), nan=0.0)
        chunk_mask = batch["antigen_chunk_mask"].unsqueeze(-1).to(ant.dtype)
        antigen = (ant * chunk_mask).sum(dim=1) / chunk_mask.sum(dim=1).clamp_min(1.0)
        return vh, antigen

    def forward(self, batch: dict[str, torch.Tensor], task: str = "both") -> dict[str, torch.Tensor]:
        vh, antigen = self.encode_batch(batch)
        output = {"vh_embedding": vh, "antigen_embedding": antigen, "target_logits": self.target_head(vh)}
        if task != "target":
            fused = torch.cat([vh, antigen, torch.abs(vh - antigen), vh * antigen], dim=-1)
            hidden = self.fusion(fused)
            output.update({"fusion": hidden, "affinity_logits": self.affinity_head(hidden),
                           "log_kd_pred": self.regression_head(hidden).squeeze(-1)})
        return output
