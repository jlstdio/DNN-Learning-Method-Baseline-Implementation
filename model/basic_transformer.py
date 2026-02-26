import torch
import torch.nn as nn

class BasicTransformerEncoder(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=8, num_layers=4, dim_feedforward=512):
        super().__init__()
        # 입력 센서 데이터를 d_model 차원으로 투영
        self.input_proj = nn.Linear(input_dim, d_model)
        # 위치 임베딩 (최대 500 타임스텝 가정)
        self.pos_embed = nn.Parameter(torch.zeros(1, 500, d_model))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, 
            dim_feedforward=dim_feedforward, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # x shape: (batch, seq_len, input_dim)
        h = self.input_proj(x) + self.pos_embed[:, :x.size(1), :]
        return self.transformer(h) # (batch, seq_len, d_model)