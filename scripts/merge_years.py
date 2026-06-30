"""
AgriWorld 鈥?澶氬勾浠ｆ暟鎹悎骞?(v2.0)
===================================
灏?run_pipeline.py 杈撳嚭鐨勫悇骞村害 pkl 鍚堝苟涓虹粺涓€璁粌鏁版嵁闆嗐€?
鏍煎紡: {year}_{MTRS} 浣滀负 key  鈫? 纭繚鍚岀綉鏍间笉鍚屽勾浠戒笉鍐茬獊銆?
鐢ㄦ硶:
    python merge_years.py
    python merge_years.py --years 2020,2021,2022
    python merge_years.py --output unified_v2.pkl
"""

import os
import pickle
import argparse
import numpy as np
from agriworld.data_quality import forcing_quality_issue
from agriworld.paths import DATA_ROOT, MERGED_DATA_PATH

DEFAULT_YEARS = [2019, 2020, 2021, 2022, 2023]
INPUT_DIR     = DATA_ROOT
OUTPUT_FILE   = MERGED_DATA_PATH


def merge(years, input_dir, output_file):
    print(f"{'='*60}")
    print("  AgriWorld multi-year merge")
    print(f"  骞翠唤: {years}")
    print(f"  婧愮洰褰? {input_dir}")
    print(f"  杈撳嚭: {output_file}")
    print(f"{'='*60}\n")

    all_data = {}
    stats = {}
    accepted_stats = {}
    total_grids = 0
    total_grids_before_filter = 0
    rejected = {}

    for year in years:
        pkl_path = os.path.join(input_dir, f"national_ode_tensors_v2_{year}.pkl")
        if not os.path.exists(pkl_path):
            print(f"  [{year}] SKIP 鈥?鏂囦欢涓嶅瓨鍦? {pkl_path}")
            stats[year] = 0
            continue

        with open(pkl_path, "rb") as f:
            year_data = pickle.load(f)

        n = len(year_data)
        stats[year] = n
        print(f"  [{year}] {n} grids")
        accepted_before = len(all_data)

        for mtrs, d in year_data.items():
            meta = d.get("meta") or {}
            state = str(meta.get("state", str(mtrs).split("-")[0]))
            planting = int(meta.get("planting_doy", 120))
            issue = forcing_quality_issue(
                d.get("stress_forcing", []),
                doy=d.get("DOY"),
                planting_doy=planting,
            )
            if issue is not None:
                rejected[issue] = rejected.get(issue, 0) + 1
                continue
            key = f"{year}_{mtrs}"
            d['year'] = year
            all_data[key] = d
        accepted_stats[year] = len(all_data) - accepted_before

        total_grids_before_filter += n

    if len(all_data) == 0:
        print("  鏃犳暟鎹彲鍚堝苟銆傝鍏堣繍琛?run_pipeline.py")
        return

    # 鈹€鈹€ 淇濆瓨 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    with open(output_file, "wb") as f:
        pickle.dump(all_data, f)

    size_mb = os.path.getsize(output_file) / 1024 / 1024
    n_years = len(set(d['year'] for d in all_data.values()))

    # 鈹€鈹€ 缁熻 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    yields = [d['target_yield'] for d in all_data.values()]
    print(f"\n{'='*60}")
    print(f"  鍚堝苟瀹屾垚")
    print(f"{'='*60}")
    print(f"  鎬绘牱鏈?(grid脳year): {len(all_data)}")
    print(f"  璐ㄦ帶鍓旈櫎: {sum(rejected.values())}")
    for reason, count in sorted(rejected.items()):
        print(f"    {reason}: {count}")
    print(f"  骞存暟: {n_years}")
    print(f"  鏂囦欢澶у皬: {size_mb:.1f} MB")
    print(f"  浜ч噺鑼冨洿: {np.min(yields):.0f} - {np.max(yields):.0f} bu/acre")
    print(f"  浜ч噺鍧囧€? {np.mean(yields):.1f}")
    print(f"\n  骞村害鍒嗗竷:")
    for y in sorted(stats.keys()):
        print(f"    {y}: {stats[y]} raw -> {accepted_stats.get(y, 0)} accepted")
    print(f"\n  杈撳嚭: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AgriWorld multi-year merge")
    parser.add_argument('--years', type=str,
                        default=",".join(str(y) for y in DEFAULT_YEARS))
    parser.add_argument('--input', type=str, default=INPUT_DIR)
    parser.add_argument('--output', type=str, default=OUTPUT_FILE)
    args = parser.parse_args()

    years = [int(y.strip()) for y in args.years.split(",")]
    merge(years, args.input, args.output)

