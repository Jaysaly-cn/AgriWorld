"""
Benchmarks 鈥?缁熶竴璁粌 + 璇勪及
=============================
瀵规墍鏈夊熀绾挎ā鍨嬩娇鐢ㄧ浉鍚岀殑 data split銆乴oss銆乵etrics銆?
鐢ㄦ硶:
    python benchmarks/train.py --model lstm
    python benchmarks/train.py --model transformer --epochs 100
"""

import os, sys, time, argparse, json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import agriworld.config as C
from agriworld.paths import BENCHMARK_RESULTS_DIR, MERGED_DATA_PATH
from benchmarks.data import create_dataloaders
from benchmarks.models import LSTMBaseline, TransformerBaseline, MLPBaseline


MODEL_MAP = {
    'lstm':        LSTMBaseline,
    'transformer': TransformerBaseline,
    'mlp':         MLPBaseline,
}


def train_and_eval(model_name: str, data_path: str, epochs: int = 100,
                   batch_size: int = 32, lr: float = 1e-3,
                   device: str = 'cuda'):
    # 鈹€鈹€ 鏁版嵁 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    train_loader, val_loader, val_indices, full_ds = create_dataloaders(
        data_path, batch_size=batch_size
    )
    print(f"Model: {model_name} | Samples: {len(full_ds)} ({len(train_loader.dataset)} train / {len(val_loader.dataset)} val)")

    # 鈹€鈹€ 妯″瀷 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    model_cls = MODEL_MAP[model_name]
    model = model_cls().to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {n_params:,}")

    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    criterion = nn.MSELoss()

    # 鈹€鈹€ 璁粌 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    best_val = float('inf')
    best_state = None
    history = {'train': [], 'val': []}
    t0 = time.time()

    for ep in range(epochs):
        model.train()
        train_loss = 0.0
        for seq, static, y in train_loader:
            seq, static, y = seq.to(device), static.to(device), y.to(device)
            pred = model(seq, static)
            loss = criterion(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        val_loss = 0.0
        all_preds, all_tgts = [], []
        with torch.no_grad():
            for seq, static, y in val_loader:
                seq, static, y = seq.to(device), static.to(device), y.to(device)
                pred = model(seq, static)
                loss = criterion(pred, y)
                val_loss += loss.item()
                all_preds.append(pred.cpu())
                all_tgts.append(y.cpu())

        val_loss /= len(val_loader)
        scheduler.step()

        history['train'].append(train_loss)
        history['val'].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

        if ep % 20 == 0 or ep == epochs - 1:
            print(f"  Ep {ep+1:3d}/{epochs} | train={train_loss:.2f} | val={val_loss:.2f} | best={best_val:.2f}")

    dt = time.time() - t0

    # 鈹€鈹€ 璇勪及 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    all_preds, all_tgts = [], []
    with torch.no_grad():
        for seq, static, y in val_loader:
            seq, static = seq.to(device), static.to(device)
            all_preds.append(model(seq, static).cpu())
            all_tgts.append(y)

    all_preds = torch.cat(all_preds).numpy()
    all_tgts  = torch.cat(all_tgts).numpy()

    rmse = np.sqrt(np.mean((all_preds - all_tgts) ** 2))
    nrmse = rmse / np.mean(all_tgts)
    mape = np.mean(np.abs((all_preds - all_tgts) / (all_tgts + 1e-3))) * 100
    ss_res = np.sum((all_preds - all_tgts) ** 2)
    ss_tot = np.sum((all_tgts - np.mean(all_tgts)) ** 2)
    r2 = 1 - ss_res / max(ss_tot, 1e-10)
    bias = np.mean(all_preds - all_tgts)

    results = {
        'model': model_name,
        'n_params': n_params,
        'rmse_bu': float(rmse),
        'nrmse_pct': float(nrmse * 100),
        'mape_pct': float(mape),
        'r2': float(r2),
        'bias_bu': float(bias),
        'best_val_loss': float(best_val),
        'train_time_sec': float(dt),
        'epochs': epochs,
        'pred_mean': float(np.mean(all_preds)),
        'tgt_mean': float(np.mean(all_tgts)),
    }

    print(f"\n  Results: RMSE={rmse:.1f} bu | NRMSE={nrmse*100:.1f}% | MAPE={mape:.1f}% | R虏={r2:.3f} | bias={bias:.1f}")
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True,
                        choices=list(MODEL_MAP.keys()))
    parser.add_argument('--data', type=str,
                        default=MERGED_DATA_PATH)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    res = train_and_eval(
        model_name=args.model,
        data_path=args.data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=args.device,
    )

    os.makedirs(BENCHMARK_RESULTS_DIR, exist_ok=True)
    path = os.path.join(BENCHMARK_RESULTS_DIR, f"{args.model}.json")
    with open(path, 'w') as f:
        json.dump(res, f, indent=2)
    print(f"Saved to {path}")

