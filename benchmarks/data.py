"""Shared data loader for benchmark models."""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split

import agriworld.config as C
from agriworld.dataset import AgriTensorDataset as BaseDataset
from agriworld.units import CORN_T_HA_TO_BU_AC
from agriworld.splits import split_indices


class BenchmarkDataset(Dataset):
    """Flatten AgriTensorDataset samples into (X_seq, X_static, y)."""

    def __init__(self, data_path: str):
        base = BaseDataset(data_path)
        self.samples = []
        self.metadata = []

        for i in range(len(base)):
            s = base[i]
            forcing  = s['forcing']          # [365, 7]
            static   = s['static_features']  # [11]
            # 鎻愬彇 cum_GDD (channel 6) 浣滀负棰濆搴忓垪鐗瑰緛
            gdd_cum  = forcing[:, 6:7]        # [365, 1]
            forcing_6ch = forcing[:, :6]       # [365, 6]
            seq = torch.cat([forcing_6ch, gdd_cum], dim=-1)  # [365, 7] (6+1)
            # 绠€鍖? 鐩存帴鐢ㄥ師濮嬬殑 7 閫氶亾
            seq = forcing                      # [365, 7]

            yield_t_ha = s['target_yield'].item()
            yield_bu = yield_t_ha * CORN_T_HA_TO_BU_AC

            self.samples.append({
                'seq':    seq.float(),
                'static': static.float(),
                'y':      torch.tensor(yield_bu, dtype=torch.float32),
            })
            self.metadata.append({
                "year": s["year"],
                "state": s["state"],
                "county": s["county"],
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return s['seq'], s['static'], s['y']


def create_dataloaders(data_path: str, batch_size: int = 32,
                       val_ratio: float = 0.2, seed: int = 42):
    """
    Return train/val loaders, validation indices, and the full dataset.
    Uses the same split policy as AgriWorld.
    """
    ds = BenchmarkDataset(data_path)
    N = len(ds)
    train_idx, val_idx = split_indices(
        ds.metadata, val_ratio, seed, getattr(C, "SPLIT_MODE", "temporal")
    )
    train_ds = torch.utils.data.Subset(ds, train_idx)
    val_ds = torch.utils.data.Subset(ds, val_idx)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, val_ds.indices, ds

