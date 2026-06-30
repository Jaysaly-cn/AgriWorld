"""Run AgriWorld ablation variants and collect evaluation summaries."""

import argparse
import csv
import json
import os
import shutil
import sys
import time
from contextlib import redirect_stdout
from inspect import signature

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from agriworld.log_utils import tee_stdout
from agriworld.paths import MERGED_DATA_PATH, RESULTS_DIR, SAVE_DIR


VARIANTS = {
    "baseline": {
        "desc": "V3.35 mainline: spatial contrast + state adaptation + corn window stress",
        "model_version": "phys_spatial",
        "use_lstm_residual": False,
        "overrides": {},
    },
    "no_window_stress": {
        "desc": "Disable crop growth-window stress factor",
        "model_version": "no_window_stress",
        "use_lstm_residual": False,
        "overrides": {"USE_WINDOW_STRESS": False},
    },
    "reproductive_only_window_stress": {
        "desc": "Use only the reproductive crop-window stress factor",
        "model_version": "reproductive_only_window_stress",
        "use_lstm_residual": False,
        "overrides": {"WINDOW_STRESS_ACTIVE_WINDOWS": "reproductive"},
    },
    "no_spatial_contrast": {
        "desc": "Disable county pairwise yield contrast loss",
        "model_version": "no_spatial_contrast",
        "use_lstm_residual": False,
        "overrides": {"USE_SPATIAL_CONTRAST": False},
    },
    "no_spatial_group_bias": {
        "desc": "Disable state-level regional mean-bias loss",
        "model_version": "no_spatial_group_bias",
        "use_lstm_residual": False,
        "overrides": {"USE_SPATIAL_GROUP_BIAS": False},
    },
    "no_state_embedding": {
        "desc": "Disable state-level adaptation in static crop parameter head",
        "model_version": "no_state_embedding",
        "use_lstm_residual": False,
        "overrides": {"USE_STATE_EMBEDDINGS": False},
    },
    "spatial_embedding_on": {
        "desc": "Enable both state and county embeddings inside static crop parameter head",
        "model_version": "spatial_embedding_on",
        "use_lstm_residual": False,
        "overrides": {"USE_SPATIAL_EMBEDDINGS": True},
    },
    "no_static_crop_params": {
        "desc": "Disable structured county-level crop parameter adjustments",
        "model_version": "no_static_crop_params",
        "use_lstm_residual": False,
        "overrides": {"USE_STATIC_CROP_PARAMS": False},
    },
    "no_static_adapt_reg": {
        "desc": "Disable regularization on structured county-level crop factors",
        "model_version": "no_static_adapt_reg",
        "use_lstm_residual": False,
        "overrides": {"W_STATIC_ADAPT": 0.0},
    },
    "no_reproductive_heat_penalty": {
        "desc": "Disable reproductive-stage heat penalty on harvest index",
        "model_version": "no_reproductive_heat_penalty",
        "use_lstm_residual": False,
        "overrides": {"USE_REPRODUCTIVE_HEAT_PENALTY": False},
    },
    "yield_residual_on": {
        "desc": "Enable static-conditioned yield residual head",
        "model_version": "yield_residual_on",
        "use_lstm_residual": False,
        "overrides": {"USE_YIELD_RESIDUAL": True},
    },
    "no_static_interaction_gates": {
        "desc": "Disable static-conditioned factor interaction gates",
        "model_version": "no_static_interaction_gates",
        "use_lstm_residual": False,
        "overrides": {"USE_STATIC_INTERACTION_GATES": False},
    },
    "no_temperature_stress": {
        "desc": "Disable temperature stress multiplier",
        "model_version": "no_temperature_stress",
        "use_lstm_residual": False,
        "overrides": {"USE_TEMPERATURE_STRESS": False},
    },
    "heat_stress_025": {
        "desc": "Stronger extreme-heat stress penalty",
        "model_version": "heat_stress_025",
        "use_lstm_residual": False,
        "overrides": {
            "TEMPERATURE_STRESS_MODE": "heat",
            "HEAT_STRESS_MAX_REDUCTION": 0.25,
        },
    },
}


DEFAULT_OVERRIDES = {
    "USE_YIELD_YEAR_TREND": True,
    "USE_COUPLING_ANOMALY": True,
    "USE_VPD_STRESS": True,
    "USE_NITROGEN_STRESS": True,
    "USE_TEMPERATURE_STRESS": True,
    "USE_YIELD_RESIDUAL": False,
    "USE_STATIC_INTERACTION_GATES": True,
    "USE_STATIC_CROP_PARAMS": True,
    "USE_SPATIAL_EMBEDDINGS": False,
    "USE_STATE_EMBEDDINGS": True,
    "USE_COUNTY_EMBEDDINGS": False,
    "USE_SPATIAL_CONTRAST": True,
    "USE_SPATIAL_GROUP_BIAS": True,
    "USE_REPRODUCTIVE_HEAT_PENALTY": True,
    "USE_WINDOW_STRESS": True,
    "WINDOW_STRESS_ACTIVE_WINDOWS": "all",
    "TEMPERATURE_STRESS_MODE": "heat",
    "TEMPERATURE_STRESS_FLOOR": 0.95,
    "TEMPERATURE_STRESS_STRENGTH": 0.20,
    "HEAT_STRESS_THRESHOLD_C": 33.0,
    "HEAT_STRESS_WIDTH_C": 2.5,
    "HEAT_STRESS_MAX_REDUCTION": 0.16,
    "HEAT_STRESS_STAGE_CENTER": 0.45,
    "HEAT_STRESS_STAGE_WIDTH": 0.15,
    "SOIL_WATER_STRESS_FLOOR": 0.60,
    "W_CANOPY": 3.0,
    "YIELD_RESIDUAL_MAX_LOG": 0.20,
    "STATIC_INTERACTION_MAX": 0.10,
    "STATIC_HI_MAX_LOG": 0.12,
    "STATIC_YIELD_MAX_LOG": 0.10,
    "STATIC_HEAT_SENS_MAX": 0.50,
    "W_STATIC_ADAPT": 0.15,
    "W_SPATIAL_GROUP_BIAS": 0.15,
    "SPATIAL_GROUP_BIAS_MIN_COUNT": 4,
    "REPRO_HEAT_THRESHOLD_C": 30.0,
    "REPRO_HEAT_WIDTH_C": 2.0,
    "REPRO_HEAT_MAX_HI_REDUCTION": 0.20,
    "WINDOW_STRESS_MAX_REDUCTION": 0.22,
    "WINDOW_STRESS_HEAT_THRESHOLD_C": 30.0,
    "WINDOW_STRESS_COLD_THRESHOLD_C": 8.0,
    "WINDOW_STRESS_TEMP_WIDTH_C": 2.0,
    "WINDOW_STRESS_PAR_REFERENCE": 20.0,
    "W_WINDOW_STRESS": 0.08,
}


def _evaluate_variant(evaluate, ckpt, data_path, device):
    if "save_legacy" in signature(evaluate).parameters:
        return evaluate(ckpt, data_path, device, save_legacy=False)
    return evaluate(ckpt, data_path, device)


def run_variant(name: str, variant_config: dict, data_path: str, epochs: int):
    print(f"\n{'=' * 60}")
    print(f"  Ablation: {name} - {variant_config['desc']}")
    print(f"{'=' * 60}")

    import agriworld.config as C

    C.MODEL_VERSION = variant_config["model_version"]
    C.USE_LSTM_RESIDUAL = variant_config["use_lstm_residual"]
    C.DATA_PATH = data_path
    C.MAX_EPOCHS = epochs
    for key, value in DEFAULT_OVERRIDES.items():
        setattr(C, key, value)
    for key, value in variant_config.get("overrides", {}).items():
        setattr(C, key, value)

    os.environ["AGRI_MODEL_VERSION"] = variant_config["model_version"]
    os.environ["AGRI_USE_LSTM"] = str(int(variant_config["use_lstm_residual"]))

    from scripts.train import train

    os.makedirs(RESULTS_DIR, exist_ok=True)
    train_log = os.path.join(RESULTS_DIR, f"ablation_{name}_train.log")
    if getattr(C, "VERBOSE", False):
        with tee_stdout(train_log):
            train()
    else:
        with open(train_log, "w", encoding="utf-8") as f, redirect_stdout(f):
            train()

    ver = variant_config["model_version"]
    ckpt = os.path.join(SAVE_DIR, f"agriworld_{ver}_best.pth")
    result = {
        "status": "OK",
        "checkpoint": ckpt,
        "description": variant_config["desc"],
        "schema": getattr(C, "MODEL_SCHEMA", None),
        "model_version": variant_config["model_version"],
        "train_log": train_log,
    }

    if not os.path.exists(ckpt):
        print(f"  Checkpoint NOT found: {ckpt}")
        result["status"] = "NO_CHECKPOINT"
        return result

    size = os.path.getsize(ckpt) / 1024
    print(f"  Checkpoint: {ckpt} ({size:.1f} KB)")

    from scripts.evaluate import evaluate
    import agriworld.config as C2

    eval_log = os.path.join(RESULTS_DIR, f"ablation_{name}_eval.log")
    if getattr(C2, "VERBOSE", False) or getattr(C2, "ABLATION_EVALUATE_VERBOSE", False):
        with tee_stdout(eval_log):
            summary = _evaluate_variant(evaluate, ckpt, data_path, C2.DEVICE)
            print(f"\n  Text log saved to {eval_log}")
    else:
        with open(eval_log, "w", encoding="utf-8") as f, redirect_stdout(f):
            summary = _evaluate_variant(evaluate, ckpt, data_path, C2.DEVICE)

    eval_pt = os.path.join(RESULTS_DIR, f"eval_agriworld_{ver}_best.pt")
    variant_pt = os.path.join(RESULTS_DIR, f"ablation_{name}_eval.pt")
    if os.path.exists(eval_pt):
        shutil.copyfile(eval_pt, variant_pt)

    result.update({
        "checkpoint_size_kb": size,
        "eval_log": eval_log,
        "eval_summary_pt": variant_pt if os.path.exists(variant_pt) else eval_pt,
        "yield_all": summary.get("yield_all", {}),
        "yield_per_crop": summary.get("yield_per_crop", {}),
        "factor_responses": summary.get("factor_responses", {}),
    })
    yield_all = result["yield_all"]
    if yield_all:
        print(
            f"  Eval: RMSE={yield_all.get('rmse_bu_acre', yield_all.get('rmse', float('nan'))):.2f} "
            f"bu/ac NRMSE={yield_all.get('nrmse_pct', float('nan')):.2f}% "
            f"R2={yield_all.get('r2', float('nan')):.3f}"
        )
    print(f"  Logs: train={train_log} | eval={eval_log}")
    return result


def run_all(data_path: str, epochs: int = 100):
    import agriworld.config as C

    print(f"  Schema: {getattr(C, 'MODEL_SCHEMA', 'unknown')}")
    print(f"  Active ablation variants: {', '.join(VARIANTS.keys())}")
    results = {}
    for name, variant_config in VARIANTS.items():
        t0 = time.time()
        try:
            variant_result = run_variant(name, variant_config, data_path, epochs)
            variant_result["time_sec"] = time.time() - t0
            results[name] = variant_result
        except Exception as exc:
            results[name] = {
                "status": "FAIL",
                "error": str(exc),
                "time_sec": time.time() - t0,
            }
            print(f"  {name} FAILED: {exc}")

    print(f"\n{'=' * 60}")
    print("  Ablation summary")
    print(f"{'=' * 60}")
    for name, result in results.items():
        print(f"  {name}: {result['status']} ({result['time_sec']:.0f}s)")
        yield_all = result.get("yield_all") or {}
        if yield_all:
            print(
                "    "
                f"RMSE={yield_all.get('rmse_bu_acre', yield_all.get('rmse', float('nan'))):.3f} "
                f"R2={yield_all.get('r2', float('nan')):.3f}"
            )

    os.makedirs(RESULTS_DIR, exist_ok=True)
    result_path = os.path.join(RESULTS_DIR, "ablation_results.json")
    with open(result_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"  Results JSON: {result_path}")
    csv_path = os.path.join(RESULTS_DIR, "ablation_results.csv")
    write_results_csv(results, csv_path)
    print(f"  Results CSV:  {csv_path}")
    return results


def write_results_csv(results, csv_path):
    fields = [
        "variant",
        "status",
        "schema",
        "model_version",
        "description",
        "rmse_bu_acre",
        "nrmse_pct",
        "r2",
        "corn_rmse",
        "corn_mape_pct",
        "precipitation_pct",
        "radiation_pct",
        "vpd_pct",
        "nitrogen_pct",
        "temperature_pct",
        "heat_extreme_pct",
        "window_heat_pct",
        "window_vpd_pct",
        "window_radiation_pct",
        "window_water_pct",
        "time_sec",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for variant, result in results.items():
            yield_all = result.get("yield_all") or {}
            corn = (result.get("yield_per_crop") or {}).get("Corn", {})
            factors = result.get("factor_responses") or {}
            row = {
                "variant": variant,
                "status": result.get("status"),
                "schema": result.get("schema"),
                "model_version": result.get("model_version"),
                "description": result.get("description"),
                "rmse_bu_acre": yield_all.get("rmse_bu_acre", yield_all.get("rmse")),
                "nrmse_pct": yield_all.get("nrmse_pct"),
                "r2": yield_all.get("r2"),
                "corn_rmse": corn.get("rmse"),
                "corn_mape_pct": corn.get("mape_pct"),
                "time_sec": result.get("time_sec"),
            }
            for name in [
                "precipitation",
                "radiation",
                "vpd",
                "nitrogen",
                "temperature",
                "heat_extreme",
                "window_heat",
                "window_vpd",
                "window_radiation",
                "window_water",
            ]:
                row[f"{name}_pct"] = (
                    factors.get(name, {}).get("mean_response_pct")
                )
            writer.writerow(row)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", type=str, default=None, choices=list(VARIANTS.keys()))
    parser.add_argument("--data", type=str, default=MERGED_DATA_PATH)
    parser.add_argument("--epochs", type=int, default=100)
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)
    if args.variant:
        log_path = os.path.join(RESULTS_DIR, f"ablation_{args.variant}.log")
        with tee_stdout(log_path):
            result = run_variant(args.variant, VARIANTS[args.variant], args.data, args.epochs)
            result["time_sec"] = None
            result_path = os.path.join(RESULTS_DIR, f"ablation_{args.variant}_results.json")
            with open(result_path, "w", encoding="utf-8") as f:
                json.dump({args.variant: result}, f, indent=2)
            write_results_csv(
                {args.variant: result},
                os.path.join(RESULTS_DIR, f"ablation_{args.variant}_results.csv"),
            )
            print(f"  Results JSON: {result_path}")
            print(f"  Text log saved to {log_path}")
    else:
        log_path = os.path.join(RESULTS_DIR, "ablation.log")
        with tee_stdout(log_path):
            run_all(args.data, args.epochs)
            print(f"  Text log saved to {log_path}")

