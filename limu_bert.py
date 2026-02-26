import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LIMUBert(nn.Module):
    def __init__(self, encoder, input_dim, d_model=128):
        super().__init__()
        self.encoder = encoder
        self.input_proj = nn.Linear(input_dim, d_model)
        self.reconstruction_head = nn.Linear(d_model, input_dim)

    def forward(self, x_masked):
        h = self.input_proj(x_masked)
        mask = torch.ones(h.shape[0], h.shape[1], device=h.device, dtype=torch.long)
        h = self.encoder(inputs_embeds=h, attention_mask=mask).last_hidden_state
        return self.reconstruction_head(h)

def train_limu_bert(model, optimizer, x_raw):
    optimizer.zero_grad()
    
    batch_size, seq_len, feat_dim = x_raw.shape
    mask = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=x_raw.device)
    
    p_m = 0.8
    p_r = 0.15
    p_geo = 0.2
    m_max = int(seq_len * p_r)
    
    x_masked = x_raw.clone()
    
    for i in range(batch_size):
        if torch.rand(1).item() < p_m:
            m = 0
            while m < m_max:
                s = torch.randint(0, seq_len, (1,)).item()
                if not mask[i, s]:
                    l = np.random.geometric(p_geo)
                    l = min(l, 10, m_max - m)
                    e = min(s + l, seq_len)
                    
                    mask[i, s:e] = True
                    m += (e - s)
            
            x_masked[i, mask[i]] = 0.0
    
    pred = model(x_masked)
    
    if mask.any():
        loss = F.mse_loss(pred[mask], x_raw[mask])
        loss.backward()
        optimizer.step()
        return loss.item()
    
    return 0.0