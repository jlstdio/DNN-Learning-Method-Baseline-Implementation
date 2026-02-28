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

DATASET_CONFIG = {
    'hhar':  {'num_classes': 6,
              'train_npz': 'dataset/Downstream_Task/HHAR/downstream_train_data.npz',
              'test_npz':  'dataset/Downstream_Task/HHAR/downstream_test_data.npz'},
    'pamap2': {'num_classes': 12,
              'train_npz': 'dataset/Downstream_Task/PAMAP2/downstream_train_data.npz',
              'test_npz':  'dataset/Downstream_Task/PAMAP2/downstream_test_data.npz'},
}


class DownstreamClassifier(nn.Module):

    def __init__(self, encoder, input_dim, d_model, num_classes):
        super().__init__()
        self.encoder = encoder
        self.input_proj = nn.Linear(input_dim, d_model)
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x):
        h = self.input_proj(x)
        mask = torch.ones(h.shape[0], h.shape[1],
                          device=h.device, dtype=torch.long)
        features = self.encoder(
            inputs_embeds=h, attention_mask=mask
        ).last_hidden_state
        global_repr = features.mean(dim=1)
        return self.classifier(global_repr)


def load_data(base_dir, dataset_name, num_train=None, seed=42):
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

    # accuracy, precision, recall, f1
    return (
        accuracy_score(all_labels, all_preds),
        precision_score(all_labels, all_preds, average='macro', zero_division=0),
        recall_score(all_labels, all_preds, average='macro', zero_division=0),
        f1_score(all_labels, all_preds, average='macro', zero_division=0),
    )


def freeze_encoder(model):
    # encoder와 input_proj를 freeze
    for param in model.encoder.parameters():
        param.requires_grad = False
    for param in model.input_proj.parameters():
        param.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)


def main():
    parser = argparse.ArgumentParser(
        description='Downstream Classification — Linear Probe (Random Init)')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['pamap2', 'hhar', 'both'])
    parser.add_argument('--model_id', type=str,
                        default='distilbert/distilbert-base-uncased',
                        help='HuggingFace model ID')
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
    datasets = ['hhar', 'pamap2'] if args.dataset == 'both' else [args.dataset]
    results = []

    for ds_name in datasets:
        print(f"\n{'='*60}")
        print(f"[Linear Probe · Random Init] Downstream — {ds_name.upper()}")
        print(f"Model: {args.model_id}")
        print(f"{'='*60}")

        X_train, y_train, X_eval, y_eval = load_data(
            args.data_dir, ds_name,
            num_train=args.num_train, seed=args.seed)

        input_dim = X_train.shape[2]
        num_classes = DATASET_CONFIG[ds_name]['num_classes']
        print(f"Train: {X_train.shape}  Eval: {X_eval.shape}  "
              f"Classes: {num_classes}")

        encoder, d_model = load_model(args.model_id)
        model = DownstreamClassifier(
            encoder, input_dim, d_model, num_classes).to(device)

        freeze_encoder(model)

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

        acc, prec, rec, f1 = evaluate(model, eval_loader, device)
        print(f"\n  ▶ Final Results ({ds_name.upper()}):")
        print(f"    Accuracy : {acc:.4f}")
        print(f"    Precision: {prec:.4f}")
        print(f"    Recall   : {rec:.4f}")
        print(f"    F1       : {f1:.4f}")

        results.append({
            'dataset': ds_name,
            'model_type': 'linear_probe_random_init',
            'model_id': args.model_id,
            'num_train_samples': args.num_train,
            'epochs': args.epochs,
            'accuracy': round(acc, 4),
            'precision': round(prec, 4),
            'recall': round(rec, 4),
            'f1': round(f1, 4),
        })

    os.makedirs(args.save_dir, exist_ok=True)
    df = pd.DataFrame(results)
    csv_path = os.path.join(args.save_dir, 'linear_probe_random_init_results.csv')
    df.to_csv(csv_path, index=False)
    print(f"\n✔ Results saved → {csv_path}")
    print(df.to_string(index=False))


if __name__ == '__main__':
    main()
