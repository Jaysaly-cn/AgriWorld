"""Leakage-aware dataset splitting utilities."""

import random

from torch.utils.data import Subset


def _split_indices_with_mode(samples, val_ratio=0.2, seed=42, mode="temporal"):
    n = len(samples)
    if n < 2:
        return list(range(n)), [], "single"

    if mode in {"temporal", "auto"}:
        years = sorted({int(sample.get("year", 0)) for sample in samples})
        if len(years) > 1:
            val_year = years[-1]
            val_idx = [
                i for i, sample in enumerate(samples)
                if int(sample.get("year", 0)) == val_year
            ]
            val_set = set(val_idx)
            train_idx = [i for i in range(n) if i not in val_set]
            min_train = max(32, int(round(0.60 * n)))
            max_val = max(1, int(round(0.40 * n)))
            if len(train_idx) >= min_train and len(val_idx) <= max_val:
                return train_idx, val_idx, f"temporal(year={val_year})"

    if mode in {"spatial_state"}:
        groups = {}
        for i, sample in enumerate(samples):
            group = sample.get("state")
            if group:
                groups.setdefault(str(group), []).append(i)
        if len(groups) > 1:
            keys = sorted(groups)
            random.Random(seed).shuffle(keys)
            target = max(1, int(round(val_ratio * n)))
            val_idx = []
            for key in keys:
                if len(val_idx) >= target:
                    break
                val_idx.extend(groups[key])
            val_set = set(val_idx)
            train_idx = [i for i in range(n) if i not in val_set]
            if train_idx and val_idx:
                return train_idx, val_idx, "spatial(state)"

    if mode in {"temporal", "auto", "spatial", "spatial_county"}:
        groups = {}
        for i, sample in enumerate(samples):
            group = sample.get("county") or sample.get("state")
            if group:
                groups.setdefault(str(group), []).append(i)
        if len(groups) > 1:
            keys = sorted(groups)
            random.Random(seed).shuffle(keys)
            target = max(1, int(round(val_ratio * n)))
            val_idx = []
            for key in keys:
                if len(val_idx) >= target:
                    break
                val_idx.extend(groups[key])
            val_set = set(val_idx)
            train_idx = [i for i in range(n) if i not in val_set]
            if train_idx and val_idx:
                return train_idx, val_idx, "spatial(county)"

    indices = list(range(n))
    random.Random(seed).shuffle(indices)
    n_val = max(1, int(val_ratio * n))
    return indices[n_val:], indices[:n_val], "random"


def split_indices(samples, val_ratio=0.2, seed=42, mode="temporal"):
    train_idx, val_idx, _ = _split_indices_with_mode(
        samples, val_ratio, seed, mode
    )
    return train_idx, val_idx


def split_dataset(dataset, val_ratio=0.2, seed=42, mode="temporal"):
    train_idx, val_idx, used_mode = _split_indices_with_mode(
        dataset.samples, val_ratio, seed, mode
    )
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)
    train_subset.split_mode_used = used_mode
    val_subset.split_mode_used = used_mode
    return train_subset, val_subset
