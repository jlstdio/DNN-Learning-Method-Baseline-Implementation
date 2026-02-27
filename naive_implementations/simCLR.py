import torch
import torch.nn as nn
import torch.nn.functional as F

class SimCLR(nn.Module):
    def __init__(self, input_dim=6, d_model=64):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, 4, batch_first=True), 2))
        self.projection_head = nn.Sequential( #
            nn.Linear(d_model, d_model), nn.ReLU(), nn.Linear(d_model, 32))

    def forward(self, x):
        h = self.backbone(x).mean(dim=1)
        return F.normalize(self.projection_head(h), dim=1)

def nt_xent_loss(z, tau=0.1): #
    n = z.shape[0] // 2
    sim_matrix = torch.matmul(z, z.T) / tau
    labels = torch.cat([torch.arange(n, 2*n), torch.arange(n)])
    return F.cross_entropy(sim_matrix, labels)

if __name__ == "__main__":
    model = SimCLR()
    x = torch.randn(8, 100, 6) # [Batch, Time, Dim]
    # Data Augmentation: Jittering + Scaling
    x_i, x_j = x + torch.randn_like(x)*0.1, x * 1.1 
    
    z = model(torch.cat([x_i, x_j], dim=0))
    loss = nt_xent_loss(z)
    print(f"SimCLR Training Loss: {loss.item():.4f}")