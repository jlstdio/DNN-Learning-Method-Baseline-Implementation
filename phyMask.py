"""
PhyMask: 물리적 인식 기반 적응형 마스킹 (Adaptive Masking).

분산(Variance), 에너지(Energy), 주파수 변화(Frequency) 세 가지
물리적 지표를 결합한 Physical-Awareness 점수를 계산하고,
점수가 높은(= 물리적으로 중요한) 구간을 우선 마스킹하여
모델이 가장 어렵고 유익한 동적 패턴을 학습하도록 유도.
"""
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


# ─────────────── Physical-Awareness 점수 ────────────────────

def compute_physical_awareness(x, window=5):
    """
    분산 + 에너지 + 주파수 변화를 결합한 물리 인식 점수.
    반환: (B, T) 범위 [0, 1]
    """
    B, T, C = x.shape

    # 1) 국소 분산 (Local Variance)
    x_unfold = x.unfold(1, window, 1)             # (B, T-w+1, C, w)
    local_var = x_unfold.var(dim=-1).mean(dim=-1)  # (B, T-w+1)
    pad = T - local_var.shape[1]
    local_var = F.pad(local_var, (pad // 2, pad - pad // 2), mode='replicate')

    # 2) 신호 에너지 (Signal Energy)
    energy = (x ** 2).mean(dim=-1)                 # (B, T)

    # 3) 순간 주파수 변화 (Instantaneous Frequency Change)
    freq_change = (torch.diff(x, dim=1) ** 2).mean(dim=-1)  # (B, T-1)
    freq_change = F.pad(freq_change, (0, 1), mode='replicate')

    # 0-1 정규화 후 결합
    def _norm(t):
        lo = t.min(dim=1, keepdim=True).values
        hi = t.max(dim=1, keepdim=True).values
        return (t - lo) / (hi - lo + 1e-8)

    return (_norm(local_var) + _norm(energy) + _norm(freq_change)) / 3


# ─────────────────── 적응형 마스킹 ──────────────────────────

def adaptive_mask(x, mask_ratio=0.5):
    """물리 인식 점수가 높은 구간을 우선 마스킹 (vectorized)."""
    B, T, C = x.shape

    scores = compute_physical_awareness(x)            # (B, T)
    num_to_mask = int(T * mask_ratio)

    scores = scores + torch.rand_like(scores) * 1e-6  # 동점 방지
    _, top_idx = scores.topk(num_to_mask, dim=1)

    mask = torch.zeros(B, T, dtype=torch.bool, device=x.device)
    mask.scatter_(1, top_idx, True)
    return mask


# ─────────────────────────── Train Step ─────────────────────

def train_phymask(model, optimizer, x_raw, mask_ratio=0.5):
    optimizer.zero_grad()

    mask = adaptive_mask(x_raw, mask_ratio=mask_ratio)  # (B, T) True=masked

    x_masked = x_raw.clone()
    x_masked[mask] = 0.0

    pred = model(x_masked)

    if mask.any():
        loss = F.mse_loss(pred[mask], x_raw[mask])
        loss.backward()
        optimizer.step()
        return loss.item()

    return 0.0
