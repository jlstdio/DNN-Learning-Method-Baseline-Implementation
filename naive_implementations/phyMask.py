import torch
import torch.nn as nn

class phyMask(nn.Module):
    def __init__(self, input_dim=6, d_model=64):
        super().__init__()
        self.enc = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, 4, batch_first=True), 2)
        self.proj = nn.Linear(input_dim, d_model)
        self.head = nn.Linear(d_model, input_dim)

    def forward(self, x):
        return self.head(self.enc(self.proj(x)))

def get_adaptive_mask(x, ratio=0.5):
    # Adaptive masking based on variance/physical energy 
    variance = torch.var(x, dim=-1) # [B, T]
    threshold = torch.quantile(variance, 1 - ratio)
    mask = (variance < threshold).float().unsqueeze(-1) # Mask high-variance parts 
    return mask

if __name__ == "__main__":
    model = phyMask()
    x = torch.randn(8, 100, 6)
    
    mask = get_adaptive_mask(x)
    pred = model(x * mask)
    loss = torch.mean((pred - x)**2 * (1 - mask)) # Target physical peaks 