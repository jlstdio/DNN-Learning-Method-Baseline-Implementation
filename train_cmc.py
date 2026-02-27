import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from cmc import CMC, info_nce_loss, train_cmc
from prepare import load_pamap2, load_hhar
from hf_backbones import load_model, model_id_to_name


def main():
    parser = argparse.ArgumentParser(description='CMC Training with HuggingFace Models')
    parser.add_argument('--dataset', type=str, required=True, choices=['pamap2', 'hhar'])
    parser.add_argument('--model_id', type=str, required=True,
                        help='HuggingFace model ID (e.g., distilbert/distilbert-base-uncased)')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--projection_dim', type=int, default=64)
    parser.add_argument('--window_size', type=int, default=128)
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    os.makedirs(args.save_dir, exist_ok=True)

    if args.dataset == 'pamap2':
        X_train, y_train, X_test, y_test = load_pamap2(
            'dataset/PAMAP2', window_size=args.window_size)
    else:
        X_train, y_train, X_test, y_test = load_hhar(
            'dataset/HHAR', window_size=args.window_size)

    input_dim = X_train.shape[2]
    acc_dim  = input_dim // 2
    gyro_dim = input_dim - acc_dim
    # print(f"Input dim: {input_dim} (acc={acc_dim}, gyro={gyro_dim}), "f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    test_data_path = os.path.join(args.save_dir, f'{args.dataset}_test_data.npz')
    if not os.path.exists(test_data_path):
        np.savez(test_data_path, X_test=X_test, y_test=y_test)

    # two diffenrent views
    encoder_v1, d_model = load_model(args.model_id)
    encoder_v2, _       = load_model(args.model_id)

    model = CMC(encoder_v1, encoder_v2,
                acc_dim=acc_dim, gyro_dim=gyro_dim,
                d_model=d_model, projection_dim=args.projection_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_tensor = torch.FloatTensor(X_train)
    train_loader = DataLoader(TensorDataset(train_tensor),
                              batch_size=args.batch_size, shuffle=True, drop_last=True)

    model_name = model_id_to_name(args.model_id)
    save_prefix = f'cmc_{model_name}_{args.dataset}'


    print(f"\n{'='*60}")
    print(f"CMC Training: {args.model_id}")
    print(f"Dataset: {args.dataset.upper()}")
    print(f"{'='*60}")

    best_loss = float('inf')
    save_path = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_losses = []

        pbar = tqdm(train_loader, desc=f'Epoch {epoch:3d}/{args.epochs}')
        for (batch_x,) in pbar:
            batch_x = batch_x.to(device)
            loss = train_cmc(model, optimizer, batch_x)
            epoch_losses.append(loss)
            pbar.set_postfix(loss=f'{loss:.4f}')

        avg_loss = np.mean(epoch_losses)
        print(f'Epoch {epoch:3d} | Avg Loss: {avg_loss:.4f}')

        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = os.path.join(args.save_dir, f'{save_prefix}_best.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
                'model_id': args.model_id,
                'input_dim': input_dim,
                'acc_dim': acc_dim,
                'gyro_dim': gyro_dim,
                'd_model': d_model,
                'projection_dim': args.projection_dim,
                'dataset': args.dataset,
            }, save_path)
            print(f'  → Best model saved (loss: {best_loss:.4f})')

    print(f"\nTraining complete. Best loss: {best_loss:.4f}")
    if save_path:
        print(f"Model saved at: {save_path}")


if __name__ == '__main__':
    main()
