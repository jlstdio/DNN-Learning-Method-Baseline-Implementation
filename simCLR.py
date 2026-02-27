import torch
import torch.nn as nn
import torch.nn.functional as F


def _random_permutation(x, max_segments=5):
    """시간 세그먼트를 랜덤으로 셔플 (Permutation)."""
    B, T, C = x.shape
    n_segments = torch.randint(2, max_segments + 1, (1,)).item()
    seg_len = T // n_segments

    segments = []
    for i in range(n_segments):
        start = i * seg_len
        end = start + seg_len if i < n_segments - 1 else T
        segments.append(x[:, start:end, :])

    perm = torch.randperm(n_segments).tolist()
    return torch.cat([segments[p] for p in perm], dim=1)


def _random_scaling_jitter(x, scale_range=(0.8, 1.2), jitter_std=0.05):
    B, T, C = x.shape

    scale = torch.empty(B, 1, C, device=x.device).uniform_(scale_range[0], scale_range[1])
    noise = torch.randn_like(x) * jitter_std

    return x * scale + noise


def _random_time_warp(x, sigma=0.2):
    """부드러운 시간 왜곡 (Time Warping)."""
    B, T, C = x.shape
    device = x.device

    random_speed = torch.ones(T, device=device) + torch.randn(T, device=device) * sigma
    random_speed = F.relu(random_speed) + 0.1
    cumulative = torch.cumsum(random_speed, dim=0)
    cumulative = (cumulative - cumulative[0]) / (cumulative[-1] - cumulative[0] + 1e-8) * (T - 1)

    idx_floor = cumulative.long().clamp(0, T - 2)
    idx_ceil = (idx_floor + 1).clamp(0, T - 1)
    frac = (cumulative - idx_floor.float()).unsqueeze(0).unsqueeze(-1)

    return x[:, idx_floor, :] * (1 - frac) + x[:, idx_ceil, :] * frac


def augment_pipeline(x):
    # Step 1: Jittering + Scaling (노이즈 추가 + 채널별 스케일링)
    x = _random_scaling_jitter(x, scale_range=(0.8, 1.2), jitter_std=0.05)
    # Step 2: Permutation (시간 세그먼트 셔플)
    x = _random_permutation(x, max_segments=5)
    # Step 3: Time Warping (부드러운 시간 왜곡)
    x = _random_time_warp(x, sigma=0.2)
    return x


class SimCLR(nn.Module):
    def __init__(self, encoder, input_dim, d_model=128, projection_dim=64):
        super().__init__()
        self.encoder = encoder
        self.input_proj = nn.Linear(input_dim, d_model)
        self.projection_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, projection_dim)
        )

    def forward(self, x):
        h = self.input_proj(x)
        mask = torch.ones(h.shape[0], h.shape[1], device=x.device, dtype=torch.long)
        features = self.encoder(inputs_embeds=h, attention_mask=mask).last_hidden_state
        global_repr = features.mean(dim=1)
        return self.projection_head(global_repr)


def nt_xent_loss(z_i, z_j, temperature=0.5):
    batch_size = z_i.size(0)

    z_i = F.normalize(z_i, dim=1)
    z_j = F.normalize(z_j, dim=1)

    z = torch.cat([z_i, z_j], dim=0)

    sim_matrix = torch.mm(z, z.t()) / temperature
    mask_self = torch.eye(2 * batch_size, device=z.device).bool()
    sim_matrix.masked_fill_(mask_self, -1e9)

    pos_sim = torch.sum(z_i * z_j, dim=1) / temperature
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)

    loss = -pos_sim + torch.logsumexp(sim_matrix, dim=1)
    return loss.mean()


def train_simclr(model, optimizer, x_raw):
    optimizer.zero_grad()

    x_view1 = augment_pipeline(x_raw)
    x_view2 = augment_pipeline(x_raw)

    z_i = model(x_view1)
    z_j = model(x_view2)

    loss = nt_xent_loss(z_i, z_j)

    loss.backward()
    optimizer.step()
    return loss.item()