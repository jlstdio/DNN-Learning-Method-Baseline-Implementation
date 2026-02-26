from transformers import AutoModel, AutoConfig


def load_model(model_id, random_init=False):
    """model_id로 HF 모델과 d_model 반환."""
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    d_model = getattr(config, 'd_model', None) \
           or getattr(config, 'dim', None) \
           or getattr(config, 'hidden_size', 768)

    if random_init:
        model = AutoModel.from_config(config, trust_remote_code=True)
    else:
        model = AutoModel.from_pretrained(model_id, trust_remote_code=True)

    return model, d_model


def model_id_to_name(model_id):
    return model_id.replace('/', '_').replace('-', '_')
