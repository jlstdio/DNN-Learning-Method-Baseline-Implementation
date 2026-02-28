from transformers import AutoModel, AutoConfig
import copy
import types
import torch
import torch.nn as nn
import torch.nn.functional as F


# ────────────────────────────────────────────────────────────
# Swin V2 Wrapper: 시퀀스 인코더 인터페이스로 래핑
# ────────────────────────────────────────────────────────────

class Swinv2SequenceWrapper(nn.Module):
    """Swin V2를 시퀀스 인코더 인터페이스로 래핑.

    모든 모델이 사용하는
        encoder(inputs_embeds=h, attention_mask=mask).last_hidden_state
    인터페이스를 그대로 제공한다.

    내부 동작:
        (B, T, d_model) → 1채널 이미지 (B, 1, S, S) → Swin V2 → 보간 → (B, T, d_model)

    d_model = _IMG_SIZE = 128 로 설정하여
    input_proj가 (B, T=128, 128)을 만들면 곧바로 128×128 정사각 이미지가 된다.
    """
    _IMG_SIZE = 128          # 내부 정사각 이미지 크기
    _WINDOW_SIZE = 4         # 모든 스테이지에서 호환 (32→16→8→4)

    def __init__(self, config):
        super().__init__()
        from transformers import Swinv2Model

        cfg = copy.deepcopy(config)
        cfg.image_size = self._IMG_SIZE
        cfg.num_channels = 1
        cfg.window_size = self._WINDOW_SIZE

        self.swin = Swinv2Model(cfg)

        # hidden_size는 이미 마지막 스테이지의 출력 차원 (e.g. 768 for tiny)
        swin_out_dim = cfg.hidden_size

        # 외부에 노출하는 d_model = _IMG_SIZE
        self.d_model = self._IMG_SIZE
        self.out_proj = nn.Linear(swin_out_dim, self.d_model)

    def forward(self, inputs_embeds=None, attention_mask=None, **kwargs):
        B, T, D = inputs_embeds.shape

        # 시퀀스 길이를 _IMG_SIZE로 맞춤 (정사각 이미지 구성)
        if T < self._IMG_SIZE:
            x = F.pad(inputs_embeds, (0, 0, 0, self._IMG_SIZE - T))
        elif T > self._IMG_SIZE:
            x = inputs_embeds[:, :self._IMG_SIZE, :]
        else:
            x = inputs_embeds

        pixel_values = x.unsqueeze(1)                     # (B, 1, S, S)
        hidden = self.swin(pixel_values=pixel_values).last_hidden_state
        hidden = self.out_proj(hidden)                    # (B, N, d_model)

        # 원래 시퀀스 길이로 보간
        if hidden.shape[1] != T:
            hidden = hidden.permute(0, 2, 1)              # (B, d_model, N)
            hidden = F.interpolate(
                hidden, size=T, mode='linear', align_corners=False)
            hidden = hidden.permute(0, 2, 1)              # (B, T, d_model)

        return types.SimpleNamespace(last_hidden_state=hidden)


# ────────────────────────────────────────────────────────────
# Public API
# ────────────────────────────────────────────────────────────

def load_model(model_id):
    """model_id로 HF 모델(랜덤 초기화)과 d_model 반환."""
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)

    # ── Swin V2: 시퀀스 래퍼 반환 ──
    if getattr(config, 'model_type', '') == 'swinv2':
        wrapper = Swinv2SequenceWrapper(config)
        return wrapper, wrapper.d_model

    d_model = getattr(config, 'd_model', None) \
           or getattr(config, 'dim', None) \
           or getattr(config, 'hidden_size', 768)

    model = AutoModel.from_config(config, trust_remote_code=True)

    return model, d_model


def model_id_to_name(model_id):
    return model_id.replace('/', '_').replace('-', '_')
