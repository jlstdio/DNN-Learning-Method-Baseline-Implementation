import torch
import torch.nn as nn
import torch.nn.functional as F


class PhyMask(nn.Module):
    def __init__(self, encoder, input_dim, d_model=128):
        super().__init__()
        self.encoder = encoder
        self.input_proj = nn.Linear(input_dim, d_model)
        self.reconstruction_head = nn.Linear(d_model, input_dim)

    def forward(self, x_masked):
        h = self.input_proj(x_masked)
        mask = torch.ones(h.shape[0], h.shape[1], device=h.device, dtype=torch.long)
        h = self.encoder(inputs_embeds=h, attention_mask=mask).last_hidden_state
        return self.reconstruction_head(h)


def compute_physical_awareness(x, window=5):
    B, T, C = x.shape

    # Local Variance
    x_unfold = x.unfold(1, window, 1)
    local_var = x_unfold.var(dim=-1).mean(dim=-1)
    pad = T - local_var.shape[1]
    local_var = F.pad(local_var, (pad // 2, pad - pad // 2), mode='replicate')

    energy = (x ** 2).mean(dim=-1)

    # Instantaneous Frequency Change
    freq_change = (torch.diff(x, dim=1) ** 2).mean(dim=-1)
    freq_change = F.pad(freq_change, (0, 1), mode='replicate')

    # norm
    def _norm(t):
        lo = t.min(dim=1, keepdim=True).values
        hi = t.max(dim=1, keepdim=True).values
        return (t - lo) / (hi - lo + 1e-8)

    return (_norm(local_var) + _norm(energy) + _norm(freq_change)) / 3


def adaptive_mask(x, mask_ratio=0.5):
    B, T, C = x.shape

    scores = compute_physical_awareness(x)
    num_to_mask = int(T * mask_ratio)

    scores = scores + torch.rand_like(scores) * 1e-6
    _, top_idx = scores.topk(num_to_mask, dim=1)

    mask = torch.zeros(B, T, dtype=torch.bool, device=x.device)
    mask.scatter_(1, top_idx, True)
    return mask


def train_phymask(model, optimizer, x_raw, mask_ratio=0.5):
    optimizer.zero_grad()

    mask = adaptive_mask(x_raw, mask_ratio=mask_ratio)

    x_masked = x_raw.clone()
    x_masked[mask] = 0.0

    pred = model(x_masked)

    if mask.any():
        loss = F.mse_loss(pred[mask], x_raw[mask])
        loss.backward()
        optimizer.step()
        return loss.item()

    return 0.0
