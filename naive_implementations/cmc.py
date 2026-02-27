import torch
import torch.nn as nn
import torch.nn.functional as F

class Cmc(nn.Module):
    def __init__(self, input_dim, d_model=64):
        super().__init__()
        self.projector = nn.Linear(input_dim, d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=d_model, nhead=4, batch_first=True), num_layers=2)
        
    def forward(self, x):
        # x shape: [Batch, Time, Dim]
        z = self.transformer(self.projector(x))
        return F.normalize(z.mean(dim=1), dim=1) # [Batch, d_model]

def info_nce_loss(z1, z2, tau=0.07):
    # Cross-view contrastive loss
    logits = torch.matmul(z1, z2.T) / tau #
    labels = torch.arange(z1.size(0)).to(z1.device)
    return F.cross_entropy(logits, labels)

# Independent Execution
if __name__ == "__main__":
    acc_encoder = Cmc(input_dim=3) # View 1
    gyro_encoder = Cmc(input_dim=3) # View 2
    
    # Dummy HHAR/PAMAP2 batch: [Batch=8, Window=100, Acc/Gyro=3]
    acc_data, gyro_data = torch.randn(8, 100, 3), torch.randn(8, 100, 3)
    
    z_acc, z_gyro = acc_encoder(acc_data), gyro_encoder(gyro_data)
    loss = info_nce_loss(z_acc, z_gyro)
    print(f"CMC Training Loss: {loss.item():.4f}")