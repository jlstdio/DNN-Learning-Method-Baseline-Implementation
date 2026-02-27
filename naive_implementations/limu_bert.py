import torch
import torch.nn as nn

class Limu_bert(nn.Module):
    def __init__(self, input_dim=6, d_model=64):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.transformer = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model, 4, batch_first=True), 4)
        self.reconstruction_head = nn.Linear(d_model, input_dim)

    def forward(self, x):
        return self.reconstruction_head(self.transformer(self.proj(x)))

if __name__ == "__main__":
    model = Limu_bert()
    x = torch.randn(8, 100, 6)
    
    # Span Masking: Masking continuous segments 
    mask = torch.ones(8, 100, 6)
    mask[:, 20:40, :] = 0 # Masked Span 
    
    pred = model(x * mask)
    loss = torch.mean((pred - x)**2 * (1 - mask)) # Loss only on masked part 
    print(f"LIMU-BERT MSM Loss: {loss.item():.4f}")