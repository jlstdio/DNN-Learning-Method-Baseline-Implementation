import torch
import torch.nn as nn
import torch.nn.functional as F


def _random_crop_resample(x, crop_ratio_range=(0.5, 1.0)):
    B, T, C = x.shape
    
    ratio = torch.empty(1).uniform_(crop_ratio_range[0], crop_ratio_range[1]).item()
    crop_len = max(int(T * ratio), 2)
    start = torch.randint(0, T - crop_len + 1, (1,)).item()

    cropped = x[:, start:start + crop_len, :]
    
    # (B, C, crop_len) → interpolate → (B, C, T) → (B, T, C)
    cropped_t = cropped.permute(0, 2, 1)
    resampled = F.interpolate(cropped_t, size=T, mode='linear', align_corners=False)
    return resampled.permute(0, 2, 1)


def _random_scaling_jitter(x, scale_range=(0.8, 1.2), jitter_std=0.05):
    B, T, C = x.shape

    scale = torch.empty(B, 1, C, device=x.device).uniform_(scale_range[0], scale_range[1])
    noise = torch.randn_like(x) * jitter_std

    return x * scale + noise


def _random_temporal_smoothing(x, max_kernel_sigma=2.0):
    B, T, C = x.shape

    sigma = torch.empty(1).uniform_(0.1, max_kernel_sigma).item()
    kernel_size = int(6 * sigma) | 1
    kernel_size = max(kernel_size, 3)
    if kernel_size > T:
        kernel_size = T if T % 2 == 1 else T - 1

    half = kernel_size // 2
    t = torch.arange(-half, half + 1, dtype=x.dtype, device=x.device)
    kernel = torch.exp(-0.5 * (t / sigma) ** 2)
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, -1).expand(C, 1, -1)

    # (B, T, C) → (B, C, T) → depthwise conv → (B, C, T) → (B, T, C)
    x_t = x.permute(0, 2, 1)
    x_t = x_t.reshape(B * C, 1, T)

    pad = kernel_size // 2
    x_smooth = F.conv1d(x_t, kernel[:1], padding=pad)
    x_smooth = x_smooth.reshape(B, C, T)

    return x_smooth.permute(0, 2, 1)


def augment_pipeline(x):
    # Step 1: Random Crop + Resample  (= Random Crop & Resize)
    x = _random_crop_resample(x, crop_ratio_range=(0.5, 1.0))
    # Step 2: Random Scaling + Jitter  (= Color Distortion)
    x = _random_scaling_jitter(x, scale_range=(0.8, 1.2), jitter_std=0.05)
    # Step 3: Temporal Gaussian Smoothing  (= Gaussian Blur)
    x = _random_temporal_smoothing(x, max_kernel_sigma=2.0)
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