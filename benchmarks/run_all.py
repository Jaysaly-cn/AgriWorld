"""
Benchmarks 鈥?Run All Baselines + Comparison Report
====================================================
渚濇杩愯鎵€鏈夊熀绾挎ā鍨嬪苟鐢熸垚瀵规瘮琛ㄣ€?
鐢ㄦ硶 (鍦?AgriWorld 鏍圭洰褰曚笅鎵ц):
    python benchmarks/run_all.py
    python benchmarks/run_all.py --data ./AgriWorld_Master/_debug_2022.pkl
"""

import os, sys, json, time, argparse
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.train import train_and_eval
from agriworld.paths import BENCHMARK_RESULTS_DIR, MERGED_DATA_PATH

MODELS = ['mlp', 'lstm', 'transformer']


def run_all(data_path: str, epochs: int = 100, device: str = 'cuda'):
    print(f"{'='*70}")
    print(f"  AgriWorld Benchmarks 鈥?All Baselines")
    print(f"  Data: {data_path}  |  Epochs: {epochs}  |  Device: {device}")
    print(f"{'='*70}\n")

    results = []

    for model_name in MODELS:
        print(f"\n{'鈹€'*50}")
        print(f"  [{model_name.upper()}]")
        print(f"{'鈹€'*50}")
        res = train_and_eval(
            model_name=model_name,
            data_path=data_path,
            epochs=epochs,
            batch_size=32,
            lr=1e-3,
            device=device,
        )
        results.append(res)
        # 淇濆瓨涓棿缁撴灉
        os.makedirs(BENCHMARK_RESULTS_DIR, exist_ok=True)
        result_path = os.path.join(
            BENCHMARK_RESULTS_DIR, f'{model_name}.json'
        )
        with open(result_path, 'w') as f:
            json.dump(res, f, indent=2)

    # 鈹€鈹€ 瀵规瘮琛?鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    print(f"\n\n{'='*70}")
    print(f"  COMPARISON TABLE")
    print(f"{'='*70}")
    header = f"{'Model':<15} {'Params':>8} {'RMSE':>8} {'NRMSE%':>8} {'MAPE%':>7} {'R虏':>7} {'Bias':>7} {'Time(s)':>8}"
    print(header)
    print('-' * len(header))

    for r in results:
        print(f"{r['model']:<15} {r['n_params']:>8,} {r['rmse_bu']:>8.1f} {r['nrmse_pct']:>8.1f} "
              f"{r['mape_pct']:>7.1f} {r['r2']:>7.3f} {r['bias_bu']:>7.1f} {r['train_time_sec']:>8.0f}")

    # 鈹€鈹€ 淇濆瓨姹囨€?鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    summary = {
        'data': data_path,
        'epochs': epochs,
        'device': device,
        'results': {r['model']: r for r in results},
    }
    summary_path = os.path.join(BENCHMARK_RESULTS_DIR, 'summary.json')
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary saved to {summary_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str,
                        default=MERGED_DATA_PATH)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    run_all(args.data, args.epochs, args.device)

