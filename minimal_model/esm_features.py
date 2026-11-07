"""冻结 ESM-2 小模型的 VH 序列 embedding 提取器。

默认使用 esm2_t6_8M_UR50D；模型只做推理，不在 val 上拟合任何参数。
"""
from __future__ import annotations
from typing import Iterable
import torch

class ESM2FeatureExtractor:
    def __init__(self, model_name: str = "esm2_t6_8M_UR50D", device: str = "auto", layer: int | None = None,
                 amp: bool = True, amp_dtype: str = "float16"):
        import esm
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        loader = getattr(esm.pretrained, model_name)
        self.model, self.alphabet = loader()
        self.model = self.model.eval().to(self.device)
        self.batch_converter = self.alphabet.get_batch_converter()
        self.layer = layer or getattr(self.model, "num_layers", 6)
        self.embedding_dim = int(self.model.embed_dim)
        self.amp = bool(amp and self.device.type == "cuda")
        self.amp_dtype = torch.float16 if amp_dtype == "float16" else torch.bfloat16

    @torch.no_grad()
    def encode(self, sequences: list[str], batch_size: int = 32) -> list[torch.Tensor]:
        outputs: list[torch.Tensor] = []
        for start in range(0, len(sequences), batch_size):
            batch_sequences = sequences[start:start + batch_size]
            labels, strings, tokens = self.batch_converter([
                (str(i), sequence.strip().upper())
                for i, sequence in enumerate(batch_sequences)
            ])
            tokens = tokens.to(self.device)
            with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.amp):
                result = self.model(tokens, repr_layers=[self.layer], return_contacts=False)
            representations = result["representations"][self.layer]
            for i, sequence in enumerate(strings):
                length = len(sequence)
                outputs.append(representations[i, 1:length + 1].mean(dim=0).float().cpu())
        return outputs

    @torch.no_grad()
    def score(self, sequences: list[str], batch_size: int = 32) -> list[float]:
        """返回每个序列的平均 residue log-likelihood（ESM2 pseudo-likelihood）。"""
        scores = []
        for start in range(0, len(sequences), batch_size):
            current = [s.strip().upper() for s in sequences[start:start + batch_size]]
            _, strings, tokens = self.batch_converter([(str(i), s) for i, s in enumerate(current)])
            tokens = tokens.to(self.device)
            with torch.autocast(device_type=self.device.type, dtype=self.amp_dtype, enabled=self.amp):
                logits = self.model(tokens, repr_layers=[], return_contacts=False)["logits"]
            log_probs = torch.log_softmax(logits.float(), dim=-1)
            for i, sequence in enumerate(strings):
                residue = tokens[i, 1:len(sequence) + 1]
                scores.append(float(log_probs[i, :len(sequence), residue].mean().cpu()))
        return scores

    def attach(self, datasets: Iterable[list[dict]], batch_size: int = 32) -> None:
        """按序列去重后将 CPU NumPy-compatible embedding 挂到多个 Dataset。"""
        all_samples = [sample for dataset in datasets for sample in dataset]
        unique = list(dict.fromkeys(sample["vh_sequence_aa"] for sample in all_samples))
        embeddings = self.encode(unique, batch_size=batch_size)
        mapping = {sequence: embedding.numpy() for sequence, embedding in zip(unique, embeddings)}
        for dataset in datasets:
            for sample in dataset:
                sample["esm_vh_embedding"] = mapping[sample["vh_sequence_aa"]]
