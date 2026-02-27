import torch
import torch.nn as nn

class VanillaMAE(nn.Module):
    def __init__(self, input_dim=6, d_model=64):
        super().__init__()
        self.encoder = nn.Linear(input_dim, d_model)
        self.decoder = nn.Linear(d_model, input_dim) # 
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, 4, batch_first=True), 2)

    def forward(self, x, mask):
        # Mask shape: [Batch, Time] (Binary 1 for visible, 0 for masked) 
        # Real MAE would only pass visible tokens to encoder 
        latent = self.transformer(self.encoder(x) * mask.unsqueeze(-1))
        return self.decoder(latent)

if __name__ == "__main__":
    model = VanillaMAE()
    x = torch.randn(8, 100, 6)
    mask = (torch.rand(8, 100) > 0.75).float() # 75% Masking Ratio 
    
    pred = model(x, mask)
    loss = torch.sum(((pred - x)**2) * (1-mask).unsqueeze(-1)) / (1-mask).sum()
    print(f"MAE Reconstruction Loss: {loss.item():.4f}")