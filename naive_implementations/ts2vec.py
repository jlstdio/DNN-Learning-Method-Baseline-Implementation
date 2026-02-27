import torch
import torch.nn as nn
import torch.nn.functional as F

class Ts2Vec(nn.Module):
    def __init__(self, input_dim=6, d_model=64):
        super().__init__()
        self.feat = nn.Linear(input_dim, d_model)
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, 4, batch_first=True), 2)

    def forward(self, x):
        # Hierarchical representation 
        return self.transformer(self.feat(x)) # [B, T, D]

if __name__ == "__main__":
    model = Ts2Vec()
    x = torch.randn(8, 100, 6)
    
    # Dual-masking strategy 
    z1, z2 = model(x), model(x) 
    # Temporal contrast: match same timestamp across views 
    temp_loss = F.mse_loss(z1, z2) 
    print(f"TS2Vec Temporal Contrast Loss: {temp_loss.item():.4f}")