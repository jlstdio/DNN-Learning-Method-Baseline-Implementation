"""
Vanilla Masked Autoencoder (MAE) for Time-Series.

시계열을 temporal patch 로 나누고, 50% 랜덤 마스킹 후
인코더는 visible 패치만 처리하여 효율적으로 학습.
경량 디코더가 마스킹된 패치의 원래 값을 복원.
MSE 손실은 마스킹된 위치에 대해서만 계산.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class VanillaMAE(nn.Module):
    def __init__(self, encoder, input_dim, d_model=128, patch_size=8):
        super().__init__()
        self.encoder = encoder
        self.d_model = d_model
        self.patch_size = patch_size
        self.input_dim = input_dim

        self.input_proj = nn.Linear(input_dim * patch_size, d_model)
        self.mask_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        # 경량 디코더
        self.decoder = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, input_dim * patch_size)
        )

    def patchify(self, x):
        """(B, T, C) → (B, N, patch_size*C)"""
        B, T, C = x.shape
        N = T // self.patch_size
        return x[:, :N * self.patch_size, :].reshape(B, N, self.patch_size * C)

    def forward(self, x, mask_ratio=0.5):
        B, T, C = x.shape
        patches = self.patchify(x)           # (B, N, P*C)
        N = patches.shape[1]

        num_masked  = int(N * mask_ratio)
        num_visible = N - num_masked

        # ── 랜덤 셔플 → visible / masked 분리 ──
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        ids_visible = ids_shuffle[:, :num_visible]
        visible_patches = torch.gather(
            patches, 1,
            ids_visible.unsqueeze(-1).expand(-1, -1, patches.shape[-1]))

        # ── 인코더: visible 패치만 처리 ──
        h_visible = self.input_proj(visible_patches)
        attn_mask = torch.ones(B, num_visible, device=x.device, dtype=torch.long)
        encoded = self.encoder(
            inputs_embeds=h_visible, attention_mask=attn_mask
        ).last_hidden_state

        # ── 디코더: mask token 과 합쳐 원래 순서 복원 ──
        mask_tokens = self.mask_token.expand(B, num_masked, -1)
        full = torch.cat([encoded, mask_tokens], dim=1)
        full = torch.gather(
            full, 1,
            ids_restore.unsqueeze(-1).expand(-1, -1, self.d_model))

        reconstructed = self.decoder(full)    # (B, N, P*C)

        # ── 마스크 이진 벡터 (1 = masked) ──
        mask_binary = torch.zeros(B, N, device=x.device)
        mask_binary.scatter_(1, ids_shuffle[:, num_visible:], 1.0)

        return reconstructed, patches, mask_binary


# ─────────────────────────── Loss ───────────────────────────

def mae_loss(reconstructed, target, mask):
    """마스킹된 패치에 대한 MSE 손실."""
    loss = (reconstructed - target) ** 2
    loss = loss.mean(dim=-1)              # 패치별 평균
    return (loss * mask).sum() / mask.sum()


# ─────────────────────────── Train Step ─────────────────────

def train_vanilla_mae(model, optimizer, x_raw, mask_ratio=0.5):
    optimizer.zero_grad()
    reconstructed, target, mask = model(x_raw, mask_ratio=mask_ratio)
    loss = mae_loss(reconstructed, target, mask)
    loss.backward()
    optimizer.step()
    return loss.item()
