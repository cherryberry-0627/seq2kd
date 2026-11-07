"""条件 latent VAE：在冻结 Transformer 的 VH pooled 表示上生成连续增强样本。"""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

TARGET_DIM = 5
AFFINITY_DIM = 3

class ConditionalLatentVAE(nn.Module):
    def __init__(self, hidden_size: int, latent_size: int = 32, condition_size: int = 16):
        super().__init__()
        self.hidden_size = hidden_size
        self.latent_size = latent_size
        self.target_embedding = nn.Embedding(TARGET_DIM, condition_size)
        self.affinity_embedding = nn.Embedding(AFFINITY_DIM, condition_size)
        condition_dim = condition_size * 2
        self.encoder = nn.Sequential(
            nn.Linear(hidden_size + condition_dim, 128), nn.GELU(),
            nn.Linear(128, 2 * latent_size))
        self.decoder = nn.Sequential(
            nn.Linear(latent_size + condition_dim, 128), nn.GELU(),
            nn.Linear(128, hidden_size))

    def condition(self, target: torch.Tensor, affinity: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.target_embedding(target), self.affinity_embedding(affinity)], dim=-1)

    def encode(self, embedding: torch.Tensor, target: torch.Tensor, affinity: torch.Tensor):
        stats = self.encoder(torch.cat([embedding, self.condition(target, affinity)], dim=-1))
        mean, logvar = stats.chunk(2, dim=-1)
        return mean, logvar

    def reparameterize(self, mean, logvar):
        return mean + torch.exp(0.5 * logvar).clamp(max=10.0) * torch.randn_like(mean)

    def decode(self, latent: torch.Tensor, target: torch.Tensor, affinity: torch.Tensor):
        return self.decoder(torch.cat([latent, self.condition(target, affinity)], dim=-1))

    def forward(self, embedding, target, affinity):
        mean, logvar = self.encode(embedding, target, affinity)
        latent = self.reparameterize(mean, logvar)
        reconstruction = self.decode(latent, target, affinity)
        return reconstruction, mean, logvar


def fit_latent_vae(embeddings: np.ndarray, target: np.ndarray, affinity: np.ndarray,
                   latent_size=32, epochs=80, batch_size=128, learning_rate=2e-3,
                   beta=0.01, seed=2026, device="cpu") -> ConditionalLatentVAE:
    torch.manual_seed(seed)
    model = ConditionalLatentVAE(embeddings.shape[1], latent_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    x = torch.as_tensor(embeddings, dtype=torch.float32, device=device)
    t = torch.as_tensor(target, dtype=torch.long, device=device)
    a = torch.as_tensor(affinity, dtype=torch.long, device=device)
    generator = np.random.default_rng(seed)
    model.train()
    for epoch in range(epochs):
        order = generator.permutation(len(x))
        for start in range(0, len(order), batch_size):
            idx = torch.as_tensor(order[start:start + batch_size], device=device)
            reconstruction, mean, logvar = model(x[idx], t[idx], a[idx])
            reconstruction_loss = F.smooth_l1_loss(reconstruction, x[idx])
            kl = -0.5 * (1 + logvar - mean.square() - logvar.exp()).mean()
            loss = reconstruction_loss + beta * kl
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
    model.eval()
    return model


def sample_esm_embedding_augmentations(model: ConditionalLatentVAE, embeddings: np.ndarray,
                                        target: np.ndarray, affinity: np.ndarray,
                                        sample_ids: list[str], samples_per_parent=1,
                                        noise_scale=0.35, weight=0.15, seed=2026,
                                        device="cuda") -> list[dict]:
    """在 ESM2 embedding 空间采样，输出 esm_vh_embedding。"""
    rng = np.random.default_rng(seed)
    x = torch.as_tensor(embeddings, dtype=torch.float32, device=device)
    t = torch.as_tensor(target, dtype=torch.long, device=device)
    a = torch.as_tensor(affinity, dtype=torch.long, device=device)
    output = []
    with torch.no_grad():
        mean, logvar = model.encode(x, t, a)
        for i, sample_id in enumerate(sample_ids):
            for j in range(samples_per_parent):
                latent = mean[i:i + 1] + noise_scale * torch.exp(0.5 * logvar[i:i + 1]).clamp(max=10.0) * torch.randn_like(mean[i:i + 1])
                generated = model.decode(latent, t[i:i + 1], a[i:i + 1])[0].float().cpu().numpy()
                output.append({
                    "sample_id": f"aug_esmvae_{sample_id}_{j}_{rng.integers(1_000_000_000)}",
                    "parent_sample_id": sample_id,
                    "esm_vh_embedding": generated.astype(np.float32),
                    "augmentation_method": "esm_embedding_vae",
                    "sample_weight": float(weight),
                    "target_sample_weight": float(weight),
                    "affinity_sample_weight": 0.0,
                })
    return output


def sample_latent_augmentations(model: ConditionalLatentVAE, embeddings: np.ndarray,
                                 target: np.ndarray, affinity: np.ndarray,
                                 sample_ids: list[str], samples_per_parent=1,
                                 noise_scale=0.35, weight=0.15, seed=2026,
                                 device="cpu") -> list[dict]:
    """生成与真实样本同条件的 latent 表示；伪样本不写入任何结果文件。"""
    rng = np.random.default_rng(seed)
    x = torch.as_tensor(embeddings, dtype=torch.float32, device=device)
    t = torch.as_tensor(target, dtype=torch.long, device=device)
    a = torch.as_tensor(affinity, dtype=torch.long, device=device)
    output = []
    with torch.no_grad():
        mean, logvar = model.encode(x, t, a)
        for i, sample_id in enumerate(sample_ids):
            for j in range(samples_per_parent):
                latent = mean[i:i + 1] + noise_scale * torch.exp(0.5 * logvar[i:i + 1]).clamp(max=10.0) * torch.randn_like(mean[i:i + 1])
                generated = model.decode(latent, t[i:i + 1], a[i:i + 1])[0].cpu().numpy()
                output.append({
                    "sample_id": f"aug_vae_{sample_id}_{j}_{rng.integers(1_000_000_000)}",
                    "parent_sample_id": sample_id,
                    "latent_vh_embedding": generated.astype(np.float32),
                    "augmentation_method": "conditional_latent_vae",
                    "sample_weight": float(weight),
                    "target_sample_weight": float(weight),
                    "affinity_sample_weight": 0.0,
                })
    return output
