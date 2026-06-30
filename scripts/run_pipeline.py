"""
AgriWorld 鈥?澶氬勾浠ｆ暟鎹绾垮叆鍙?(v3.3)
=======================================
骞惰杩愯 Ag-WorldDataPipeline 璺ㄨ秺 5 骞?(2019-2023)銆?
鏋舵瀯:
    榛樿 1 涓勾搴﹁繘绋?脳 4 涓?Thread锛岄伩鍏嶈Е鍙?GEE 閰嶉
    姣忓勾鐙珛 pkl 杈撳嚭 鈫?merge_years.py 鍚堝苟

鐢ㄦ硶:
    python run_pipeline.py                      # 鍏ㄩ儴 5 骞?    python run_pipeline.py --years 2019,2020    # 鎸囧畾骞翠唤
    python run_pipeline.py --processes 2        # 2 杩涚▼骞惰
"""

import os
import sys
import time
import argparse
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from agriworld.paths import (
    CACHE_ROOT,
    DATA_ROOT,
    GEE_CREDENTIALS_PATH,
    PROJECT_ROOT,
    PROXY_URL,
)

# 鈹€鈹€鈹€ 閰嶇疆 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

DEFAULT_YEARS = [2019, 2020, 2021, 2022, 2023]
OUTPUT_DIR    = DATA_ROOT
SCRIPT        = os.path.join(PROJECT_ROOT, "scripts", "data_pipeline.py")

# 鈹€鈹€鈹€ 鍗曞勾鎵ц鍣?鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

def annual_file_quality(path: str) -> tuple[int, int]:
    import pickle
    from agriworld.data_quality import forcing_quality_issue

    with open(path, "rb") as handle:
        data = pickle.load(handle)
    valid = 0
    for sample in data.values():
        meta = sample.get("meta") or {}
        issue = forcing_quality_issue(
            sample.get("stress_forcing", []),
            doy=sample.get("DOY"),
            planting_doy=int(meta.get("planting_doy", 120)),
        )
        valid += int(issue is None)
    return valid, len(data)


def run_single_year(year: int, output_dir: str, cache_dir: str,
                    credentials: str, force: bool = False,
                    timeout_hours: float = 0.0, gee_workers: int = 6,
                    smap_workers: int = 4, skip_smap: bool = False) -> dict:
    """
    鍦ㄥ瓙杩涚▼涓繍琛?python data_pipeline.py, 浼犻€掑勾浠藉弬鏁般€?    杩斿洖 {year, ok, file, duration_sec, error}
    """
    t0 = time.time()
    output_file = os.path.join(output_dir, f"national_ode_tensors_v2_{year}.pkl")

    if os.path.exists(output_file) and not force:
        try:
            valid, total = annual_file_quality(output_file)
            coverage = valid / max(total, 1)
            if total > 0 and coverage >= 0.95:
                size_mb = os.path.getsize(output_file) / 1024 / 1024
                print(
                    f"  [{year}] valid output exists ({valid}/{total}, "
                    f"{size_mb:.1f} MB), skip"
                )
                return {
                    "year": year, "ok": True, "file": output_file,
                    "duration_sec": 0, "cached": True,
                }
            print(
                f"  [{year}] existing output failed QC ({valid}/{total}); rebuilding"
            )
        except Exception as exc:
            print(f"  [{year}] cannot validate existing output ({exc}); rebuilding")

    cmd = [
        sys.executable, "-u", SCRIPT,
        "--start", f"{year}-01-01",
        "--end",   f"{year}-12-31",
        "--output", output_dir,
        "--cache", cache_dir,
        "--credentials", credentials,
    ]
    if skip_smap:
        cmd.append("--skip-smap")

    try:
        log_dir = os.path.join(PROJECT_ROOT, "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"pipeline_{year}.log")
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"
        env["AGRI_GEE_WORKERS"] = str(gee_workers)
        env["AGRI_SMAP_WORKERS"] = str(smap_workers)

        print(
            f"  [{year}] start | GEE workers={gee_workers} "
            f"SMAP workers={smap_workers} | log={log_path}",
            flush=True,
        )
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        timeout_sec = timeout_hours * 3600.0 if timeout_hours > 0 else None
        with open(log_path, "w", encoding="utf-8") as log_file:
            for line in process.stdout:
                log_file.write(line)
                log_file.flush()
                print(f"[{year}] {line}", end="", flush=True)
                if timeout_sec is not None and time.time() - t0 > timeout_sec:
                    process.kill()
                    process.wait()
                    return {
                        "year": year, "ok": False, "file": None,
                        "duration_sec": time.time() - t0,
                        "error": f"Timeout (>{timeout_hours:g}h)",
                        "log": log_path,
                    }
        return_code = process.wait()
        dt = time.time() - t0
        ok = return_code == 0
        return {
            "year": year, "ok": ok, "file": output_file,
            "duration_sec": dt, "cached": False,
            "error": "" if ok else f"Child exited with code {return_code}",
            "log": log_path,
        }
    except Exception as e:
        dt = time.time() - t0
        return {
            "year": year, "ok": False, "file": None,
            "duration_sec": dt, "error": str(e)
        }


# 鈹€鈹€鈹€ 涓诲叆鍙?鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€

def main():
    parser = argparse.ArgumentParser(description="AgriWorld multi-year data pipeline")
    parser.add_argument("--years", type=str,
                        default=",".join(str(y) for y in DEFAULT_YEARS),
                        help="閫楀彿鍒嗛殧鐨勫勾浠藉垪琛? 濡?2019,2020,2021")
    parser.add_argument("--processes", type=int, default=1,
                        help="骞惰骞村害杩涚▼鏁帮紱GEE 閰嶉鏁忔劅锛屽缓璁?1")
    parser.add_argument("--output", type=str, default=OUTPUT_DIR)
    parser.add_argument("--cache", type=str, default=CACHE_ROOT)
    parser.add_argument("--credentials", type=str, default=GEE_CREDENTIALS_PATH)
    parser.add_argument("--force", action="store_true",
                        help="rebuild existing annual files")
    parser.add_argument("--timeout-hours", type=float, default=0.0,
                        help="鍗曞勾搴﹁秴鏃跺皬鏃舵暟锛? 琛ㄧず涓嶈瓒呮椂")
    parser.add_argument("--gee-workers", type=int, default=6,
                        help="澶╂皵銆侀潤鎬佸拰LAI鐨勭嚎绋嬫暟锛屽缓璁?-8")
    parser.add_argument("--smap-workers", type=int, default=4,
                        help="SMAP绾跨▼鏁帮紝寤鸿2-6")
    parser.add_argument("--skip-smap", action="store_true",
                        help="skip SMAP collection to shorten data build time")
    args = parser.parse_args()

    years = [int(y.strip()) for y in args.years.split(",")]
    print(f"{'='*60}")
    print(f"  AgriWorld 澶氬勾浠ｆ暟鎹绾?v3.3")
    print(f"  骞翠唤: {years}")
    print(f"  骞惰杩涚▼: {args.processes}")
    print(f"  杈撳嚭鐩綍: {args.output}")
    print(f"  缂撳瓨鐩綍: {args.cache}")
    print(f"  椤圭洰鐩綍: {PROJECT_ROOT}")
    print(f"  浠ｇ悊鍦板潃: {PROXY_URL}")
    print(f"  GEE鍑嵁: {args.credentials}")
    print(f"  鍗曞勾瓒呮椂: {'disabled' if args.timeout_hours <= 0 else f'{args.timeout_hours:g}h'}")
    print(f"  GEE绾跨▼: {args.gee_workers} | SMAP绾跨▼: {args.smap_workers}")
    print(f"  璺宠繃SMAP: {args.skip_smap}")
    print(f"{'='*60}\n")

    os.makedirs(args.output, exist_ok=True)
    os.makedirs(args.cache, exist_ok=True)

    t_total = time.time()
    results = []

    # 鈹€鈹€ 杩涚▼姹犲苟琛?鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    with ProcessPoolExecutor(max_workers=args.processes) as pool:
        futures = {
            pool.submit(
                run_single_year,
                y,
                args.output,
                args.cache,
                args.credentials,
                args.force,
                args.timeout_hours,
                args.gee_workers,
                args.smap_workers,
                args.skip_smap,
            ): y
            for y in years
        }
        for future in as_completed(futures):
            res = future.result()
            results.append(res)
            y = res['year']
            status = "OK" if res['ok'] else "FAIL"
            tag = "(cached)" if res.get('cached') else f"({res['duration_sec']:.0f}s)"
            print(f"  [{y}] {status} {tag}")
            if not res['ok']:
                err = res.get('error', '')
                print(f"       Error: {err}")
                if res.get("log"):
                    print(f"       Log: {res['log']}")

    results.sort(key=lambda r: r['year'])

    # 鈹€鈹€ 缁熻 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    dt_total = time.time() - t_total
    ok_years  = sum(1 for r in results if r['ok'])
    fail_years = len(years) - ok_years
    total_size = sum(os.path.getsize(r['file']) / 1024 / 1024
                     for r in results if r['ok'] and r['file'] and os.path.exists(r['file']))

    print(f"\n{'='*60}")
    print(f"  瀹屾垚: {ok_years}/{len(years)} 骞? |  澶辫触: {fail_years}")
    print(f"  鎬昏€楁椂: {dt_total/60:.0f} min  |  鎬诲ぇ灏? {total_size:.0f} MB")
    print(f"  杈撳嚭鏂囦欢:")
    for r in results:
        if r['ok'] and r['file']:
            size = os.path.getsize(r['file']) / 1024 / 1024 if os.path.exists(r['file']) else 0
            tag = " (cached)" if r.get('cached') else ""
            print(f"    {r['file']:60s} {size:6.1f} MB{tag}")
    print(f"{'='*60}")

    if fail_years > 0:
        print(f"\n  澶辫触骞翠唤闇€瑕侀噸璺戙€傚崟鐙噸璺戞煇涓€骞?")
        for r in results:
            if not r['ok']:
                print(f"    python run_pipeline.py --years {r['year']}")

    if ok_years == len(years):
        print(f"\n  鉁?鍏ㄩ儴 {ok_years} 骞存暟鎹氨缁€備笅涓€姝?")
        print(f"    python merge_years.py")
        print(f"    (鍚堝苟 5 涓勾搴?pkl 鈫?缁熶竴 training dataset)")


if __name__ == "__main__":
    main()

