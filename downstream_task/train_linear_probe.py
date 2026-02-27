"""
Downstream Classification — Linear Probe with a Pre-trained Checkpoint.

지정한 checkpoint(LIMUBert, SimCLR 등)에서 encoder + input_proj 가중치를 불러온 뒤
encoder와 input_proj를 **freeze**하고 classification head만 학습·평가한다.
npz 파일에서 100개 샘플로 학습, 나머지로 평가(Accuracy, Precision, Recall, F1).
결과는 CSV로 저장된다.

Usage:
    python downstream_task/train_linear_probe.py \
        --checkpoint checkpoints/limu_bert_distilbert_distilbert_base_uncased_hhar_best.pt \
        --dataset hhar

    python downstream_task/train_linear_probe.py \
        --checkpoint checkpoints/simclr_distilbert_distilbert_base_uncased_pamap2_best.pt \
        --dataset pamap2
"""

import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hf_backbones import load_model, model_id_to_name


# ---------------------------------------------------------------------------
# Dataset config
# ---------------------------------------------------------------------------
DATASET_CONFIG = {
    'hhar':  {'num_classes': 6,
              'train_npz': 'dataset/Downstream_Task/HHAR/downstream_train_data.npz',
              'test_npz':  'dataset/Downstream_Task/HHAR/downstream_test_data.npz'},
    'pamap2': {'num_classes': 12,
              'train_npz': 'dataset/Downstream_Task/PAMAP2/downstream_train_data.npz',
              'test_npz':  'dataset/Downstream_Task/PAMAP2/downstream_test_data.npz'},
}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class DownstreamClassifier(nn.Module):
    """HF Encoder backbone + classification head."""

    def __init__(self, encoder, input_dim, d_model, num_classes):
        super().__init__()
        self.encoder = encoder
        self.input_proj = nn.Linear(input_dim, d_model)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x):
        h = self.input_proj(x)                          # (B, T, d_model)
        mask = torch.ones(h.shape[0], h.shape[1],
                          device=h.device, dtype=torch.long)
        features = self.encoder(
            inputs_embeds=h, attention_mask=mask
        ).last_hidden_state                              # (B, T, d_model)
        global_repr = features.mean(dim=1)               # (B, d_model)
        return self.classifier(global_repr)              # (B, num_classes)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def load_data(base_dir, dataset_name, num_train=None, seed=42):
    """Downstream_Task 폴더에서 train/test npz를 각각 로드."""
    cfg = DATASET_CONFIG[dataset_name]
    train_data = np.load(os.path.join(base_dir, cfg['train_npz']))
    test_data  = np.load(os.path.join(base_dir, cfg['test_npz']))

    X_train, y_train = train_data['X_train'], train_data['y_train']
    X_test,  y_test  = test_data['X_test'],   test_data['y_test']

    if num_train is not None and num_train < len(X_train):
        rng = np.random.RandomState(seed)
        idx = rng.permutation(len(X_train))[:num_train]
        X_train, y_train = X_train[idx], y_train[idx]

    return X_train, y_train, X_test, y_test


def evaluate(model, dataloader, device):
    """모델 평가 → (accuracy, precision, recall, f1)."""
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_labels.extend(batch_y.numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    return (
        accuracy_score(all_labels, all_preds),
        precision_score(all_labels, all_preds, average='macro', zero_division=0),
        recall_score(all_labels, all_preds, average='macro', zero_division=0),
        f1_score(all_labels, all_preds, average='macro', zero_division=0),
    )


def load_pretrained_weights(model, checkpoint_path, device):
    """
    Checkpoint에서 encoder / input_proj 가중치만 선택적으로 로드한다.
    reconstruction_head(LIMUBert)나 projection_head(SimCLR) 등은 건너뛴다.
    """
    checkpoint = torch.load(checkpoint_path, map_location=device,
                            weights_only=False)
    pretrained_state = checkpoint['model_state_dict']
    model_state = model.state_dict()

    loaded_keys, skipped_keys = [], []
    for key, value in pretrained_state.items():
        if key in model_state and model_state[key].shape == value.shape:
            model_state[key] = value
            loaded_keys.append(key)
        else:
            skipped_keys.append(key)

    model.load_state_dict(model_state)

    print(f"  Loaded  {len(loaded_keys):3d} parameter tensors from checkpoint")
    if skipped_keys:
        print(f"  Skipped {len(skipped_keys):3d} tensors "
              f"(head / shape mismatch): {skipped_keys[:5]}{'…' if len(skipped_keys) > 5 else ''}")

    return checkpoint


def freeze_encoder(model):
    """encoder와 input_proj를 freeze하고, classifier만 학습 가능하게 한다."""
    # Freeze encoder
    for param in model.encoder.parameters():
        param.requires_grad = False
    # Freeze input_proj
    for param in model.input_proj.parameters():
        param.requires_grad = False

    # classifier는 학습 가능 (기본값 True이므로 별도 설정 불필요)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f"  Frozen params : {frozen:,}")
    print(f"  Trainable params: {trainable:,}  (classifier only)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Downstream Classification — Linear Probe (Pretrained)')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='사용할 pretrained weight 파일 경로 '
                             '(e.g. checkpoints/limu_bert_..._best.pt)')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['pamap2', 'hhar'],
                        help='downstream에 사용할 데이터셋')
    parser.add_argument('--model_id', type=str, default=None,
                        help='HuggingFace model ID (미지정 시 checkpoint에서 추출)')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Linear probe는 보통 더 높은 lr 사용')
    parser.add_argument('--num_train', type=int, default=100,
                        help='학습에 사용할 샘플 수')
    parser.add_argument('--data_dir', type=str, default='.',
                        help='프로젝트 루트 디렉토리 (dataset/ 폴더가 있는 곳)')
    parser.add_argument('--save_dir', type=str,
                        default='downstream_task/results',
                        help='결과 CSV 저장 디렉토리')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ---- peek at checkpoint metadata ----
    ckpt_meta = torch.load(args.checkpoint, map_location='cpu',
                           weights_only=False)
    ckpt_model_id = ckpt_meta.get('model_id', 'distilbert/distilbert-base-uncased')
    ckpt_input_dim = ckpt_meta.get('input_dim', None)
    ckpt_d_model = ckpt_meta.get('d_model', None)
    ckpt_dataset = ckpt_meta.get('dataset', 'unknown')

    model_id = args.model_id or ckpt_model_id
    ds_name = args.dataset

    print(f"\n{'='*60}")
    print(f"[Linear Probe · Pretrained] Downstream — {ds_name.upper()}")
    print(f"Checkpoint : {args.checkpoint}")
    print(f"  └ trained on : {ckpt_dataset}")
    print(f"  └ model_id   : {ckpt_model_id}")
    print(f"  └ input_dim  : {ckpt_input_dim},  d_model: {ckpt_d_model}")
    print(f"Model ID (used): {model_id}")
    print(f"{'='*60}")

    # ---- data ----
    X_train, y_train, X_eval, y_eval = load_data(
        args.data_dir, ds_name,
        num_train=args.num_train, seed=args.seed)

    input_dim = X_train.shape[2]
    num_classes = DATASET_CONFIG[ds_name]['num_classes']
    print(f"Train: {X_train.shape}  Eval: {X_eval.shape}  "
          f"Classes: {num_classes}")

    # input_dim 호환성 체크
    if ckpt_input_dim is not None and ckpt_input_dim != input_dim:
        print(f"\n⚠  Checkpoint input_dim({ckpt_input_dim}) ≠ "
              f"dataset input_dim({input_dim}).")
        print("   input_proj 가중치는 로드하지 않고 랜덤 초기화합니다.\n")

    # ---- model ----
    encoder, d_model = load_model(model_id)
    model = DownstreamClassifier(
        encoder, input_dim, d_model, num_classes).to(device)

    # ---- load pretrained weights ----
    load_pretrained_weights(model, args.checkpoint, device)

    # ---- freeze encoder + input_proj ----
    freeze_encoder(model)

    # ---- optimizer: classifier params only ----
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train),
                      torch.LongTensor(y_train)),
        batch_size=args.batch_size, shuffle=True, drop_last=False)
    eval_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_eval),
                      torch.LongTensor(y_eval)),
        batch_size=args.batch_size, shuffle=False)

    # ---- training (classifier only) ----
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())

        if epoch % 10 == 0 or epoch == 1:
            avg_loss = np.mean(epoch_losses)
            acc, prec, rec, f1 = evaluate(model, eval_loader, device)
            print(f"  Epoch {epoch:3d}/{args.epochs} | "
                  f"Loss {avg_loss:.4f} | "
                  f"Acc {acc:.4f}  Prec {prec:.4f}  "
                  f"Rec {rec:.4f}  F1 {f1:.4f}")

    # ---- final evaluation ----
    acc, prec, rec, f1 = evaluate(model, eval_loader, device)
    ckpt_basename = os.path.basename(args.checkpoint)

    print(f"\n  ▶ Final Results ({ds_name.upper()}):")
    print(f"    Checkpoint: {ckpt_basename}")
    print(f"    Accuracy : {acc:.4f}")
    print(f"    Precision: {prec:.4f}")
    print(f"    Recall   : {rec:.4f}")
    print(f"    F1       : {f1:.4f}")

    result = {
        'dataset': ds_name,
        'model_type': 'linear_probe',
        'checkpoint': ckpt_basename,
        'model_id': model_id,
        'pretrained_on': ckpt_dataset,
        'num_train_samples': args.num_train,
        'epochs': args.epochs,
        'accuracy': round(acc, 4),
        'precision': round(prec, 4),
        'recall': round(rec, 4),
        'f1': round(f1, 4),
    }

    # ---- save / append CSV ----
    os.makedirs(args.save_dir, exist_ok=True)
    csv_path = os.path.join(args.save_dir, 'linear_probe_results.csv')

    if os.path.exists(csv_path):
        df_existing = pd.read_csv(csv_path)
        df = pd.concat([df_existing, pd.DataFrame([result])],
                       ignore_index=True)
    else:
        df = pd.DataFrame([result])

    df.to_csv(csv_path, index=False)
    print(f"\n✔ Results saved → {csv_path}")
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
