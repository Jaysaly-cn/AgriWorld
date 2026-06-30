"""Audit a merged AgriWorld pickle without requiring PyTorch."""

import argparse
import os
import pickle
from collections import Counter, defaultdict

import numpy as np

from agriworld.data_quality import forcing_quality_issue
from agriworld.paths import MERGED_DATA_PATH


CHANNELS = [
    "Precip", "ETo", "SRAD", "PAR", "Tmax",
    "Tmin", "Tmean", "VPD", "GDD",
]


def audit(path):
    with open(path, "rb") as handle:
        data = pickle.load(handle)

    years = Counter()
    accepted_years = Counter()
    corn_years = Counter()
    lai_years = Counter()
    yield_years = Counter()
    eligible_years = Counter()
    rejected = Counter()
    examples = defaultdict(list)
    channel_nonzero = defaultdict(list)

    for sample_id, sample in data.items():
        year = int(sample.get("year", str(sample_id).split("_")[0]))
        years[year] += 1
        meta = sample.get("meta") or {}
        planting = int(meta.get("planting_doy", 120))
        forcing = np.asarray(sample.get("stress_forcing", []), dtype=np.float32)
        doy = sample.get("DOY")

        issue = forcing_quality_issue(forcing, doy=doy, planting_doy=planting)
        if issue is not None:
            rejected[issue] += 1
            if len(examples[issue]) < 10:
                examples[issue].append(str(sample_id))
            continue

        accepted_years[year] += 1
        static = np.asarray(sample.get("static_features", []), dtype=np.float32)
        is_corn = static.size > 10 and int(round(float(static[10]))) == 1
        if is_corn:
            corn_years[year] += 1

        obs_lai = np.asarray(sample.get("obs_LAI", []), dtype=np.float32)
        valid_lai = int(np.sum((obs_lai > 0.5) & (obs_lai <= 12.0)))
        has_lai = valid_lai >= 10
        if has_lai:
            lai_years[year] += 1

        has_yield = float(sample.get("target_yield", 0.0)) > 1.0
        if has_yield:
            yield_years[year] += 1
        if is_corn and has_lai and has_yield:
            eligible_years[year] += 1

        for idx, name in enumerate(CHANNELS):
            channel_nonzero[name].append(
                float(np.mean(np.abs(forcing[:, idx]) > 1e-6))
            )

    size_mb = os.path.getsize(path) / 1024 / 1024
    print("=" * 72)
    print("AGRIWORLD DATA AUDIT")
    print("=" * 72)
    print(f"File: {path}")
    print(f"Size: {size_mb:.1f} MB")
    print(f"Raw samples: {len(data)}")
    print(f"Accepted weather: {sum(accepted_years.values())}")
    print(f"Rejected weather: {sum(rejected.values())}")

    print("\nTraining eligibility funnel by year:")
    print("  year: raw -> weather -> corn -> LAI>=10 -> yield -> eligible")
    for year in sorted(years):
        print(
            f"  {year}: {years[year]} -> {accepted_years[year]} -> "
            f"{corn_years[year]} -> {lai_years[year]} -> "
            f"{yield_years[year]} -> {eligible_years[year]}"
        )

    print("\nRejection reasons:")
    if not rejected:
        print("  none")
    for reason, count in sorted(rejected.items()):
        print(f"  {reason}: {count} examples={examples[reason]}")

    print("\nMean non-zero-day fraction among accepted samples:")
    for name in CHANNELS:
        values = channel_nonzero[name]
        if values:
            print(f"  {name:8s}: {np.mean(values):.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default=MERGED_DATA_PATH,
    )
    args = parser.parse_args()
    audit(args.data)

