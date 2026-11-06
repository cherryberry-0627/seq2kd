"""PyTorch 原生氨基酸 tokenizer 与长抗原分块。"""
from __future__ import annotations
from typing import Any
import torch

AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"
PAD_ID, BOS_ID, EOS_ID, UNK_ID = 0, 1, 2, 3
AA_TO_ID = {aa: i + 4 for i, aa in enumerate(AMINO_ACIDS)}

class AminoAcidTokenizer:
    """字符级 tokenizer；输出 torch.LongTensor，不依赖 transformers。"""
    def __init__(self, max_vh_length: int = 192, antigen_chunk_length: int = 192, antigen_overlap: int = 32):
        if antigen_overlap >= antigen_chunk_length - 2:
            raise ValueError("antigen_overlap must be smaller than the effective chunk length")
        self.max_vh_length = max_vh_length
        self.antigen_chunk_length = antigen_chunk_length
        self.antigen_overlap = antigen_overlap
        self.vocab_size = len(AA_TO_ID) + 4

    def encode(self, sequence: str, max_length: int) -> tuple[list[int], list[int]]:
        seq = sequence.strip().upper()
        usable = max(0, max_length - 2)
        if len(seq) > usable:
            left = usable // 2
            seq = seq[:left] + seq[-(usable - left):]
        ids = [BOS_ID] + [AA_TO_ID.get(aa, UNK_ID) for aa in seq] + [EOS_ID]
        ids += [PAD_ID] * (max_length - len(ids))
        mask = [1] * min(len(seq) + 2, max_length) + [0] * max(0, max_length - len(seq) - 2)
        return ids[:max_length], mask[:max_length]

    def tokenize_vh(self, sequences: list[str]) -> dict[str, torch.Tensor]:
        pairs = [self.encode(s, self.max_vh_length) for s in sequences]
        return {"input_ids": torch.tensor([p[0] for p in pairs], dtype=torch.long),
                "attention_mask": torch.tensor([p[1] for p in pairs], dtype=torch.bool)}

    def tokenize_antigen_chunks(self, sequences: list[str]) -> dict[str, torch.Tensor]:
        effective = self.antigen_chunk_length - 2
        stride = effective - self.antigen_overlap
        encoded_all, counts = [], []
        for sequence in sequences:
            seq = sequence.strip().upper()
            starts = list(range(0, max(1, len(seq) - effective + 1), stride))
            last = max(0, len(seq) - effective)
            if starts[-1] != last:
                starts.append(last)
            chunks = [seq[start:start + effective] for start in starts]
            encoded = [self.encode(chunk, self.antigen_chunk_length) for chunk in chunks]
            encoded_all.append(encoded)
            counts.append(len(encoded))
        max_chunks = max(counts)
        input_ids = torch.full((len(sequences), max_chunks, self.antigen_chunk_length), PAD_ID, dtype=torch.long)
        attention = torch.zeros_like(input_ids, dtype=torch.bool)
        chunk_mask = torch.zeros((len(sequences), max_chunks), dtype=torch.bool)
        for i, chunks in enumerate(encoded_all):
            input_ids[i, :len(chunks)] = torch.tensor([x[0] for x in chunks], dtype=torch.long)
            attention[i, :len(chunks)] = torch.tensor([x[1] for x in chunks], dtype=torch.bool)
            chunk_mask[i, :len(chunks)] = True
        return {"input_ids": input_ids, "attention_mask": attention, "chunk_mask": chunk_mask}

def build_tokenizer(model_name: str = "aa-char-transformer", **kwargs) -> AminoAcidTokenizer:
    return AminoAcidTokenizer(**kwargs)

def collate_batch(samples: list[dict[str, Any]], tokenizer: AminoAcidTokenizer) -> dict[str, Any]:
    vh = tokenizer.tokenize_vh([s["vh_sequence_aa"] for s in samples])
    antigen = tokenizer.tokenize_antigen_chunks([s["antigen_sequence"] for s in samples])
    target = [s.get("target_id") for s in samples]
    affinity = [s.get("affinity_class") for s in samples]
    log_kd = [s.get("log_kd_nm") for s in samples]
    return {"vh_input_ids": vh["input_ids"], "vh_attention_mask": vh["attention_mask"],
            "antigen_input_ids": antigen["input_ids"], "antigen_attention_mask": antigen["attention_mask"],
            "antigen_chunk_mask": antigen["chunk_mask"],
            "target_labels": target if all(x is not None for x in target) else None,
            "affinity_labels": affinity if all(x is not None for x in affinity) else None,
            "log_kd": torch.tensor(log_kd, dtype=torch.float32) if all(x is not None for x in log_kd) else None,
            "protparam_features": torch.tensor(
                [s["protparam_features"] for s in samples], dtype=torch.float32
            ) if all("protparam_features" in s for s in samples) else None,
            "sample_ids": [s["sample_id"] for s in samples]}
