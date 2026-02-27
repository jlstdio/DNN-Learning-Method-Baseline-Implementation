"""
Downstream Classification with a Pre-trained Checkpoint.

지정한 checkpoint(LIMUBert, SimCLR 등)에서 encoder + input_proj 가중치를 불러와
classification head를 붙여 downstream 학습·평가를 수행한다.
npz 파일에서 100개 샘플로 학습, 나머지로 평가(Accuracy, Precision, Recall, F1).
결과는 CSV로 저장된다.

Usage:
    python downstream_task/train_pretrained.py \\
        --checkpoint checkpoints/limu_bert_distilbert_distilbert_base_uncased_pretrain_hhar_best.pt \\
        --dataset hhar

    python downstream_task/train_pretrained.py \\
        --checkpoint checkpoints/simclr_distilbert_distilbert_base_uncased_pretrain_pamap2_best.pt \\
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
    'hhar':  {'num_classes': 6,  'npz_file': 'hhar_test_data.npz'},
    'pamap2': {'num_classes': 12, 'npz_file': 'pamap2_test_data.npz'},
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
def load_data(data_dir, dataset_name, num_train=100, seed=42):
    """npz에서 데이터 로드 → train(num_train개) / eval(나머지) 분리."""
    npz_path = os.path.join(data_dir, DATASET_CONFIG[dataset_name]['npz_file'])
    data = np.load(npz_path)
    X, y = data['X_test'], data['y_test']

    rng = np.random.RandomState(seed)
    indices = rng.permutation(len(X))
    train_idx = indices[:num_train]
    eval_idx = indices[num_train:]

    return X[train_idx], y[train_idx], X[eval_idx], y[eval_idx]


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


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description='Downstream Classification — Pre-trained Checkpoint')
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
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_train', type=int, default=100,
                        help='학습에 사용할 샘플 수')
    parser.add_argument('--data_dir', type=str, default='checkpoints',
                        help='npz 파일이 위치한 디렉토리')
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
    print(f"[Pretrained] Downstream Classification — {ds_name.upper()}")
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

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-2)
    criterion = nn.CrossEntropyLoss()

    train_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_train),
                      torch.LongTensor(y_train)),
        batch_size=args.batch_size, shuffle=True, drop_last=False)
    eval_loader = DataLoader(
        TensorDataset(torch.FloatTensor(X_eval),
                      torch.LongTensor(y_eval)),
        batch_size=args.batch_size, shuffle=False)

    # ---- training ----
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
        'model_type': 'pretrained',
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
    csv_path = os.path.join(args.save_dir, 'pretrained_results.csv')

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
