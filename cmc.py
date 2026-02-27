"""
Contrastive Multiview Coding (CMC) for Time-Series.

가속도계(Acc)와 자이로스코프(Gyro)를 별도의 View로 사용하여
교차 모달리티 대조 학습을 수행.

각 View에 독립된 인코더를 두고, Projection Head를 통해
공유 잠재 공간에서 InfoNCE 손실로 상호정보(MI)를 최대화.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CMC(nn.Module):
    """
    두 개의 독립 인코더가 각각 Acc / Gyro 를 인코딩하고,
    Projection Head를 통해 공유 공간에서 InfoNCE로 대조 학습.
    """
    def __init__(self, encoder_v1, encoder_v2,
                 acc_dim, gyro_dim, d_model=128, projection_dim=64):
        super().__init__()
        self.acc_dim = acc_dim
        self.gyro_dim = gyro_dim

        self.encoder_v1 = encoder_v1
        self.encoder_v2 = encoder_v2

        self.input_proj_v1 = nn.Linear(acc_dim, d_model)
        self.input_proj_v2 = nn.Linear(gyro_dim, d_model)

        self.projection_v1 = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, projection_dim))
        self.projection_v2 = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(),
            nn.Linear(d_model, projection_dim))

    def encode_view1(self, x_acc):
        h = self.input_proj_v1(x_acc)
        mask = torch.ones(h.shape[0], h.shape[1], device=h.device, dtype=torch.long)
        features = self.encoder_v1(inputs_embeds=h, attention_mask=mask).last_hidden_state
        return self.projection_v1(features.mean(dim=1))

    def encode_view2(self, x_gyro):
        h = self.input_proj_v2(x_gyro)
        mask = torch.ones(h.shape[0], h.shape[1], device=h.device, dtype=torch.long)
        features = self.encoder_v2(inputs_embeds=h, attention_mask=mask).last_hidden_state
        return self.projection_v2(features.mean(dim=1))

    def forward(self, x):
        x_acc  = x[:, :, :self.acc_dim]
        x_gyro = x[:, :, self.acc_dim:]
        return self.encode_view1(x_acc), self.encode_view2(x_gyro)


# ─────────────────────────── Loss ───────────────────────────

def info_nce_loss(z1, z2, temperature=0.07):
    """
    교차 뷰 InfoNCE 대조 손실.
    h_θ(x) = exp(sim(z1, z2) / τ)  형태의 (k+1)-way softmax.
    """
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)

    logits = torch.matmul(z1, z2.T) / temperature
    labels = torch.arange(z1.size(0), device=z1.device)

    return (F.cross_entropy(logits, labels) +
            F.cross_entropy(logits.T, labels)) / 2


# ─────────────────────────── Train Step ─────────────────────

def train_cmc(model, optimizer, x_raw):
    optimizer.zero_grad()
    z1, z2 = model(x_raw)
    loss = info_nce_loss(z1, z2)
    loss.backward()
    optimizer.step()
    return loss.item()
