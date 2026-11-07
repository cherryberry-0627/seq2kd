"""只生成 CDR-H3 或 CDR-H2/H3、保持 framework 不变的条件序列 VAE。"""
from __future__ import annotations
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

AA = "ACDEFGHIKLMNPQRSTVWY"
PAD, BOS, EOS = 0, 1, 2
AA_TO_ID = {aa: i + 4 for i, aa in enumerate(AA)}
ID_TO_AA = {i + 4: aa for i, aa in enumerate(AA)}

class CDRSequenceVAE(nn.Module):
    def __init__(self, max_region_length: int, hidden=96, latent=32, cond_size=16):
        super().__init__()
        self.max_region_length = max_region_length
        self.latent = latent
        self.token = nn.Embedding(24, 32, padding_idx=PAD)
        self.target = nn.Embedding(5, cond_size)
        self.affinity = nn.Embedding(3, cond_size)
        cond = cond_size * 2
        self.encoder = nn.GRU(32 + cond, hidden, batch_first=True)
        self.stats = nn.Linear(hidden, latent * 2)
        self.init = nn.Linear(latent + cond, hidden)
        self.decoder = nn.GRU(32 + cond, hidden, batch_first=True)
        self.output = nn.Linear(hidden, 24)

    def condition(self, target, affinity):
        return torch.cat([self.target(target), self.affinity(affinity)], dim=-1)

    def encode(self, tokens, lengths, target, affinity):
        cond = self.condition(target, affinity)
        x = torch.cat([self.token(tokens), cond[:, None].expand(-1, tokens.shape[1], -1)], -1)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, hidden = self.encoder(packed)
        return self.stats(hidden[-1]).chunk(2, -1)

    def decode_teacher(self, tokens, z, target, affinity):
        cond = self.condition(target, affinity)
        x = torch.cat([self.token(tokens), cond[:, None].expand(-1, tokens.shape[1], -1)], -1)
        hidden = self.init(torch.cat([z, cond], -1)).unsqueeze(0)
        return self.output(self.decoder(x, hidden)[0])

    def forward(self, tokens, lengths, target, affinity):
        mean, logvar = self.encode(tokens, lengths, target, affinity)
        z = mean + torch.exp(.5 * logvar).clamp(max=10) * torch.randn_like(mean)
        return self.decode_teacher(tokens[:, :-1], z, target, affinity), mean, logvar


def encode_regions(regions: list[str], max_length: int):
    out = torch.full((len(regions), max_length + 1), PAD, dtype=torch.long)
    lengths = []
    for i, seq in enumerate(regions):
        seq = seq[:max_length - 1]
        ids = [BOS] + [AA_TO_ID.get(a, 3) for a in seq] + [EOS]
        out[i, :len(ids)] = torch.tensor(ids)
        lengths.append(len(ids))
    return out, torch.tensor(lengths, dtype=torch.long)


def fit_cdr_vae(regions, target, affinity, max_length, epochs=25, batch_size=256,
                beta=.01, seed=2026, device="cuda", amp=True):
    torch.manual_seed(seed)
    model = CDRSequenceVAE(max_length).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    tokens, lengths = encode_regions(regions, max_length)
    tokens, lengths = tokens.to(device), lengths.to(device)
    target = torch.as_tensor(target, dtype=torch.long, device=device)
    affinity = torch.as_tensor(affinity, dtype=torch.long, device=device)
    rng = np.random.default_rng(seed)
    use_amp = bool(amp and str(device).startswith("cuda"))
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    model.train()
    for _ in range(epochs):
        order = rng.permutation(len(regions))
        for start in range(0, len(order), batch_size):
            idx = torch.as_tensor(order[start:start + batch_size], device=device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits, mean, logvar = model(tokens[idx], lengths[idx], target[idx], affinity[idx])
                rec = F.cross_entropy(logits.reshape(-1, 24), tokens[idx, 1:].reshape(-1), ignore_index=PAD)
                kl = -.5 * (1 + logvar - mean.square() - logvar.exp()).mean()
                loss = rec + beta * kl
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update()
    model.eval()
    return model


def sample_cdr_regions(model, regions, target, affinity, sample_ids, spans, method,
                        seed=2026, weight=.15, device="cuda", temperature=.8):
    rng = np.random.default_rng(seed)
    max_length = model.max_region_length
    tokens, lengths = encode_regions(regions, max_length)
    tokens, lengths = tokens.to(device), lengths.to(device)
    target = torch.as_tensor(target, dtype=torch.long, device=device)
    affinity = torch.as_tensor(affinity, dtype=torch.long, device=device)
    output = []
    with torch.no_grad():
        mean, logvar = model.encode(tokens, lengths, target, affinity)
        for i, parent in enumerate(regions):
            z = mean[i:i+1] + torch.exp(.5 * logvar[i:i+1]).clamp(max=10) * torch.randn_like(mean[i:i+1])
            cond = model.condition(target[i:i+1], affinity[i:i+1])
            hidden = model.init(torch.cat([z, cond], -1)).unsqueeze(0)
            current = torch.full((1, 1), BOS, dtype=torch.long, device=device)
            generated = []
            for _ in range(len(parent)):
                x = torch.cat([model.token(current[:, -1:]), cond[:, None]], -1)
                state, hidden = model.decoder(x, hidden)
                logits = model.output(state[:, -1]).float()[:, 4:]
                probs = torch.softmax(logits / temperature, -1)
                token = torch.multinomial(probs, 1) + 4
                current = torch.cat([current, token], 1)
                generated.append(ID_TO_AA[int(token.item())])
            row = {"sample_id": f"aug_cdrvae_{sample_ids[i]}_{rng.integers(1_000_000_000)}",
                   "parent_sample_id": sample_ids[i], "generated_region": "".join(generated),
                   "cdr_spans": spans[i], "augmentation_method": method,
                   "sample_weight": weight, "target_sample_weight": weight,
                   "affinity_sample_weight": 0.0}
            output.append(row)
    return output
