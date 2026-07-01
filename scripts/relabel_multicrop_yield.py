"""Relabel an existing merged pickle with crop-specific NASS county yields."""

import argparse
import copy
import json
import os
import pickle
import sys
from collections import Counter

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agriworld.nass import (  # noqa: E402
    CROP_YIELD_SPECS,
    fetch_multi_crop_yields,
    make_yield_key,
    parse_crop_codes,
)
from agriworld.paths import DATA_ROOT, MERGED_DATA_PATH  # noqa: E402


DEFAULT_OUTPUT = os.path.join(DATA_ROOT, "national_ode_tensors_v3_multicrop.pkl")


def _crop_code(sample):
    static = np.asarray(sample.get("static_features", []), dtype=np.float32)
    if static.size <= 10:
        return None
    return int(round(float(static[10])))


def _year(sample_id, sample):
    if sample.get("year") is not None:
        return int(sample["year"])
    return int(str(sample_id).split("_")[0])


def relabel(input_path, output_path, crops):
    crops = parse_crop_codes(crops)
    unsupported = [c for c in crops if c not in CROP_YIELD_SPECS]
    if unsupported:
        raise ValueError(f"Unsupported crop codes: {unsupported}")

    with open(input_path, "rb") as handle:
        data = pickle.load(handle)

    years = sorted({_year(sample_id, sample) for sample_id, sample in data.items()})
    states = sorted({
        str((sample.get("meta") or {}).get("state", "")).upper()
        for sample in data.values()
        if (sample.get("meta") or {}).get("state")
    })
    if not states:
        states = ["IA", "IL", "IN", "MN", "NE"]

    yield_maps = fetch_multi_crop_yields(years, states, crops)
    out = {}
    counts = Counter()
    crop_counts = Counter()
    missing_examples = []

    for sample_id, sample in data.items():
        counts["raw"] += 1
        crop = _crop_code(sample)
        crop_counts[f"raw_{crop}"] += 1
        if crop not in crops:
            counts["unsupported_crop"] += 1
            continue

        year = _year(sample_id, sample)
        meta = sample.get("meta") or {}
        key = make_yield_key(meta.get("state", ""), meta.get("county", sample_id))
        target = yield_maps.get((year, crop), {}).get(key)
        if target is None:
            counts["missing_yield"] += 1
            if len(missing_examples) < 20:
                missing_examples.append({
                    "sample_id": str(sample_id),
                    "year": year,
                    "crop": crop,
                    "yield_key": key,
                })
            continue

        new_sample = copy.deepcopy(sample)
        new_sample["target_yield"] = float(target)
        new_sample["year"] = year
        new_meta = dict(new_sample.get("meta") or {})
        spec = CROP_YIELD_SPECS[crop]
        new_meta.update({
            "yield_crop_type": crop,
            "yield_crop_name": spec["name"],
            "yield_source": "USDA NASS QuickStats county yield",
            "yield_unit": "bu/acre",
            "bushel_lb": spec["bushel_lb"],
            "previous_target_yield": float(sample.get("target_yield", np.nan)),
        })
        new_sample["meta"] = new_meta
        out[sample_id] = new_sample
        counts["kept"] += 1
        crop_counts[f"kept_{crop}"] += 1

    if not out:
        raise RuntimeError("No samples kept after crop-specific relabeling.")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    tmp_path = output_path + ".tmp"
    with open(tmp_path, "wb") as handle:
        pickle.dump(out, handle)
    os.replace(tmp_path, output_path)

    audit = {
        "input": input_path,
        "output": output_path,
        "years": years,
        "states": states,
        "crops": {str(c): CROP_YIELD_SPECS[c]["name"] for c in crops},
        "counts": dict(counts),
        "crop_counts": dict(crop_counts),
        "missing_examples": missing_examples,
    }
    audit_path = output_path + ".audit.json"
    with open(audit_path, "w", encoding="utf-8") as handle:
        json.dump(audit, handle, indent=2)

    print("=" * 72)
    print("MULTI-CROP YIELD RELABEL COMPLETE")
    print("=" * 72)
    print(f"Input : {input_path}")
    print(f"Output: {output_path}")
    print(f"Audit : {audit_path}")
    print(f"Raw={counts['raw']} kept={counts['kept']} "
          f"unsupported={counts['unsupported_crop']} missing_yield={counts['missing_yield']}")
    for crop in crops:
        print(
            f"  {crop}:{CROP_YIELD_SPECS[crop]['name']} "
            f"raw={crop_counts[f'raw_{crop}']} kept={crop_counts[f'kept_{crop}']}"
        )
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=MERGED_DATA_PATH)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--crops", default="1,5", help="Comma-separated CDL crop codes.")
    args = parser.parse_args()
    relabel(args.input, args.output, args.crops)

