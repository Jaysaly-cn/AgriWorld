"""Audit a merged AgriWorld pickle without requiring PyTorch."""

import argparse
import csv
import json
import os
import pickle
import sys
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agriworld.data_quality import forcing_quality_issue
from agriworld.paths import MERGED_DATA_PATH, RESULTS_DIR


CHANNELS = [
    "Precip", "ETo", "SRAD", "PAR", "Tmax",
    "Tmin", "Tmean", "VPD", "GDD",
]

CROP_NAMES = {
    1: "Corn",
    5: "Soybean",
    21: "Barley",
    22: "Durum Wheat",
    23: "Spring Wheat",
    24: "Winter Wheat",
    25: "Other Small Grains",
    26: "Dbl Crop WinWht/Soybeans",
    27: "Rye",
    28: "Oats",
    29: "Millet",
    36: "Alfalfa",
    37: "Other Hay/Non Alfalfa",
    42: "Dry Beans",
    43: "Potatoes",
    44: "Other Crops",
    61: "Fallow/Idle Cropland",
    176: "Grass/Pasture",
}


def crop_label(code):
    try:
        code = int(round(float(code)))
    except (TypeError, ValueError):
        return "unknown"
    return f"{code}:{CROP_NAMES.get(code, 'Unknown')}"


def _summary(values):
    arr = np.asarray(values, dtype=np.float32)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"n": 0}
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def audit(path, save_dir=None):
    with open(path, "rb") as handle:
        data = pickle.load(handle)

    years = Counter()
    accepted_years = Counter()
    rejected = Counter()
    examples = defaultdict(list)
    channel_nonzero = defaultdict(list)
    by_crop = defaultdict(Counter)
    by_crop_year = defaultdict(Counter)
    by_crop_state = defaultdict(Counter)
    crop_yields = defaultdict(list)
    crop_lai_counts = defaultdict(list)
    weather_rejected_by_crop = defaultdict(Counter)

    for sample_id, sample in data.items():
        year = int(sample.get("year", str(sample_id).split("_")[0]))
        years[year] += 1
        meta = sample.get("meta") or {}
        state = str(meta.get("state", str(sample_id).split("_")[-1].split("-")[0]))
        planting = int(meta.get("planting_doy", 120))
        forcing = np.asarray(sample.get("stress_forcing", []), dtype=np.float32)
        doy = sample.get("DOY")
        static = np.asarray(sample.get("static_features", []), dtype=np.float32)
        crop = crop_label(static[10] if static.size > 10 else None)

        by_crop[crop]["raw"] += 1
        by_crop_year[crop][year] += 1
        by_crop_state[crop][state] += 1

        issue = forcing_quality_issue(forcing, doy=doy, planting_doy=planting)
        if issue is not None:
            rejected[issue] += 1
            weather_rejected_by_crop[crop][issue] += 1
            if len(examples[issue]) < 10:
                examples[issue].append(str(sample_id))
            continue

        accepted_years[year] += 1
        by_crop[crop]["weather"] += 1

        obs_lai = np.asarray(sample.get("obs_LAI", []), dtype=np.float32)
        valid_lai = int(np.sum((obs_lai > 0.5) & (obs_lai <= 12.0)))
        crop_lai_counts[crop].append(valid_lai)
        has_lai = valid_lai >= 10
        if has_lai:
            by_crop[crop]["lai_ge_10"] += 1

        target_yield = float(sample.get("target_yield", 0.0))
        has_yield = target_yield > 1.0
        if has_yield:
            by_crop[crop]["yield"] += 1
            crop_yields[crop].append(target_yield)
        if has_lai and has_yield:
            by_crop[crop]["eligible"] += 1

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

    print("\nWeather availability by year:")
    print("  year: raw -> weather")
    for year in sorted(years):
        print(f"  {year}: {years[year]} -> {accepted_years[year]}")

    print("\nMulti-crop eligibility funnel:")
    print("  crop: raw -> weather -> LAI>=10 -> yield -> eligible")
    rows = []
    for crop, counts in sorted(
        by_crop.items(),
        key=lambda item: (-item[1]["raw"], item[0]),
    ):
        row = {
            "crop": crop,
            "raw": counts["raw"],
            "weather": counts["weather"],
            "lai_ge_10": counts["lai_ge_10"],
            "yield": counts["yield"],
            "eligible": counts["eligible"],
            "yield_mean": _summary(crop_yields[crop]).get("mean"),
            "valid_lai_mean": _summary(crop_lai_counts[crop]).get("mean"),
        }
        rows.append(row)
        print(
            f"  {crop}: {row['raw']} -> {row['weather']} -> "
            f"{row['lai_ge_10']} -> {row['yield']} -> {row['eligible']}"
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

    print("\nTop crop-year distribution:")
    for crop, counter in sorted(
        by_crop_year.items(),
        key=lambda item: (-sum(item[1].values()), item[0]),
    )[:12]:
        pieces = ", ".join(f"{year}:{counter[year]}" for year in sorted(counter))
        print(f"  {crop}: {pieces}")

    print("\nTop crop-state distribution:")
    for crop, counter in sorted(
        by_crop_state.items(),
        key=lambda item: (-sum(item[1].values()), item[0]),
    )[:12]:
        pieces = ", ".join(f"{state}:{count}" for state, count in counter.most_common(8))
        print(f"  {crop}: {pieces}")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        csv_path = os.path.join(save_dir, "crop_data_audit.csv")
        json_path = os.path.join(save_dir, "crop_data_audit.json")
        with open(csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        payload = {
            "path": path,
            "raw_samples": len(data),
            "accepted_weather": int(sum(accepted_years.values())),
            "rejected_weather": int(sum(rejected.values())),
            "crop_funnel": rows,
            "crop_year": {
                crop: dict(sorted(counter.items()))
                for crop, counter in by_crop_year.items()
            },
            "crop_state": {
                crop: dict(counter.most_common())
                for crop, counter in by_crop_state.items()
            },
            "weather_rejected_by_crop": {
                crop: dict(counter)
                for crop, counter in weather_rejected_by_crop.items()
            },
        }
        with open(json_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"\nSaved crop audit CSV:  {csv_path}")
        print(f"Saved crop audit JSON: {json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        default=MERGED_DATA_PATH,
    )
    parser.add_argument(
        "--save-dir",
        default=RESULTS_DIR,
        help="Directory for crop_data_audit.csv/json. Use empty string to skip.",
    )
    args = parser.parse_args()
    audit(args.data, save_dir=args.save_dir or None)

