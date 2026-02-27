"""
TS2Vec: 시계열을 위한 계층적 대조 학습.

두 개의 View를 타임스탬프 마스킹 + 시간 크롭으로 생성하고,
Temporal Contrast (동일 타임스탬프 정렬) 와
Instance Contrast (동일 시퀀스 정렬) 를 결합하여
타임스탬프 수준 + 시퀀스 수준의 범용 표현을 학습.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TS2Vec(nn.Module):
    def __init__(self, encoder, input_dim, d_model=128):
        super().__init__()
        self.encoder = encoder
        self.input_proj = nn.Linear(input_dim, d_model)

    def forward(self, x, mask=None):
        """타임스탬프별 표현 반환. (B, T, D)"""
        h = self.input_proj(x)
        if mask is not None:
            h = h * mask.unsqueeze(-1)
        attn_mask = torch.ones(h.shape[0], h.shape[1],
                               device=h.device, dtype=torch.long)
        return self.encoder(
            inputs_embeds=h, attention_mask=attn_mask
        ).last_hidden_state


# ─────────────────────────── View 생성 ──────────────────────

def generate_views(x, mask_ratio=0.5):
    """타임스탬프 마스킹 + 시간 크롭으로 두 뷰를 생성하고
    겹치는(overlap) 구간의 시작/끝 인덱스를 반환."""
    B, T, C = x.shape
    device = x.device

    mask1 = (torch.rand(B, T, device=device) > mask_ratio).float()
    mask2 = (torch.rand(B, T, device=device) > mask_ratio).float()

    crop_len = torch.randint(T // 2, T, (1,)).item()
    max_start = T - crop_len

    start1 = torch.randint(0, max_start + 1, (1,)).item()
    start2 = torch.randint(0, max_start + 1, (1,)).item()

    overlap_start = max(start1, start2)
    overlap_end   = min(start1 + crop_len, start2 + crop_len)

    # 겹침이 없으면 강제로 맞춤
    if overlap_end <= overlap_start:
        start2 = start1
        overlap_start = start1
        overlap_end   = start1 + crop_len

    return mask1, mask2, overlap_start, overlap_end


# ─────────────────────────── 계층적 대조 손실 ───────────────

def hierarchical_contrastive_loss(z1, z2, overlap_start, overlap_end,
                                  temperature=0.5):
    """Temporal Contrast + Instance Contrast 결합."""
    B, T, D = z1.shape

    if overlap_end <= overlap_start:
        return torch.tensor(0.0, device=z1.device, requires_grad=True)

    z1_o = z1[:, overlap_start:overlap_end, :]
    z2_o = z2[:, overlap_start:overlap_end, :]
    T_o = z1_o.shape[1]

    # ── Temporal Contrast ──
    # 동일 타임스탬프 = positive, 다른 타임스탬프 = negative
    z1_t = F.normalize(z1_o, dim=-1)
    z2_t = F.normalize(z2_o, dim=-1)
    sim = torch.bmm(z1_t, z2_t.transpose(1, 2)) / temperature  # (B, T', T')
    labels_t = (torch.arange(T_o, device=z1.device)
                .unsqueeze(0).expand(B, -1).reshape(-1))
    temporal_loss = F.cross_entropy(sim.reshape(B * T_o, T_o), labels_t)

    # ── Instance Contrast ──
    # 동일 시퀀스 = positive, 배치 내 다른 시퀀스 = negative
    z1_inst = F.normalize(z1_o.mean(dim=1), dim=-1)
    z2_inst = F.normalize(z2_o.mean(dim=1), dim=-1)
    sim_inst = torch.mm(z1_inst, z2_inst.T) / temperature
    labels_i = torch.arange(B, device=z1.device)
    instance_loss = (F.cross_entropy(sim_inst, labels_i) +
                     F.cross_entropy(sim_inst.T, labels_i)) / 2

    return temporal_loss + instance_loss


# ─────────────────────────── Train Step ─────────────────────

def train_ts2vec(model, optimizer, x_raw):
    optimizer.zero_grad()

    mask1, mask2, o_start, o_end = generate_views(x_raw)

    z1 = model(x_raw, mask=mask1)
    z2 = model(x_raw, mask=mask2)

    loss = hierarchical_contrastive_loss(z1, z2, o_start, o_end)

    loss.backward()
    optimizer.step()
    return loss.item()
