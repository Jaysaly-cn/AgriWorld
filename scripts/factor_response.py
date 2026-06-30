"""Counterfactual response audit for the main agricultural drivers."""

import argparse
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

import agriworld.config as C
from agriworld.dataset import AgriTensorDataset
from agriworld.simulator import AgriWorldSimulator
from agriworld.splits import split_dataset
from agriworld.paths import RESULTS_DIR, SAVE_DIR


def _predict(model, batch, device, forcing=None, n_init=None):
    forcing = batch["forcing"].to(device) if forcing is None else forcing
    n_init = batch["n_init"].to(device) if n_init is None else n_init
    static = batch["static_features"].to(device)
    year = batch.get("year", None)
    if year is not None:
        year = year.to(device)
    state_id = batch.get("state_id", None)
    county_id = batch.get("county_id", None)
    if state_id is not None:
        state_id = state_id.to(device)
    if county_id is not None:
        county_id = county_id.to(device)
    model.set_static_features(static, state_id=state_id, county_id=county_id)
    _, pred = model(forcing, n_init, year=year)
    return pred


def _crop_progress(model, forcing):
    gdd_cum = forcing[..., 6:7]
    gdd_em = torch.abs(model.ode_func.pheno_expert.gdd_emergence)
    gdd_ma = torch.abs(model.ode_func.pheno_expert.gdd_maturity)
    return torch.clamp((gdd_cum - gdd_em) / (gdd_ma - gdd_em + 1e-6), 0.0, 1.0)


@torch.no_grad()
def audit_factor_responses(model, dataloader, device, max_batches=3, print_results=True):
    """Report mean yield response to controlled one-factor perturbations."""
    model.eval()
    accum = {
        "precipitation": [],
        "radiation": [],
        "vpd": [],
        "nitrogen": [],
        "temperature": [],
        "heat_extreme": [],
        "window_heat": [],
        "window_vpd": [],
        "window_radiation": [],
        "window_water": [],
    }

    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= max_batches:
            break
        forcing = batch["forcing"].to(device)
        n_init = batch["n_init"].to(device)
        baseline = _predict(model, batch, device, forcing, n_init).clamp(min=1e-4)

        perturbations = {}
        for name, channel in [("precipitation", 0), ("radiation", 2), ("vpd", 4)]:
            low = forcing.clone()
            high = forcing.clone()
            low[..., channel] *= 0.8
            high[..., channel] *= 1.2
            perturbations[name] = (
                _predict(model, batch, device, low, n_init),
                _predict(model, batch, device, high, n_init),
            )

        # Use absolute low/high N scenarios. Most dataset samples are already
        # N-sufficient and simulator.forward clamps n_init to <=400 kg/ha, so
        # proportional 0.8/1.2 perturbations can collapse to the same state.
        n_low = torch.full_like(n_init, 60.0)
        n_high = torch.full_like(n_init, 240.0)
        perturbations["nitrogen"] = (
            _predict(model, batch, device, forcing, n_low),
            _predict(model, batch, device, forcing, n_high),
        )

        temp_low = forcing.clone()
        temp_high = forcing.clone()
        temp_low[..., 3] -= 2.0
        temp_high[..., 3] += 2.0
        perturbations["temperature"] = (
            _predict(model, batch, device, temp_low, n_init),
            _predict(model, batch, device, temp_high, n_init),
        )

        heat_low = forcing.clone()
        heat_high = forcing.clone()
        hot_day_c = float(getattr(C, "HEAT_AUDIT_HOT_DAY_C", 28.0))
        heat_delta_c = float(getattr(C, "HEAT_AUDIT_DELTA_C", 6.0))
        hot_day_mask = heat_high[..., 3] >= hot_day_c
        heat_high[..., 3] = torch.where(
            hot_day_mask,
            heat_high[..., 3] + heat_delta_c,
            heat_high[..., 3],
        )
        perturbations["heat_extreme"] = (
            _predict(model, batch, device, heat_low, n_init),
            _predict(model, batch, device, heat_high, n_init),
        )

        progress = _crop_progress(model, forcing)
        repro_mask = ((progress >= 0.50) & (progress <= 0.70)).squeeze(-1)
        for name, channel, low_mult, high_mult in [
            ("window_vpd", 4, 1.0, 1.25),
            ("window_radiation", 2, 0.80, 1.20),
        ]:
            low = forcing.clone()
            high = forcing.clone()
            low[..., channel] = torch.where(
                repro_mask,
                low[..., channel] * low_mult,
                low[..., channel],
            )
            high[..., channel] = torch.where(
                repro_mask,
                high[..., channel] * high_mult,
                high[..., channel],
            )
            perturbations[name] = (
                _predict(model, batch, device, low, n_init),
                _predict(model, batch, device, high, n_init),
            )

        water_low = forcing.clone()
        water_high = forcing.clone()
        water_low[..., 0] = torch.where(
            repro_mask,
            water_low[..., 0] * 0.10,
            water_low[..., 0],
        )
        water_low[..., 1] = torch.where(
            repro_mask,
            water_low[..., 1] * 1.25,
            water_low[..., 1],
        )
        water_high[..., 0] = torch.where(
            repro_mask,
            water_high[..., 0] + 5.0,
            water_high[..., 0],
        )
        water_high[..., 1] = torch.where(
            repro_mask,
            water_high[..., 1] * 0.90,
            water_high[..., 1],
        )
        perturbations["window_water"] = (
            _predict(model, batch, device, water_low, n_init),
            _predict(model, batch, device, water_high, n_init),
        )

        heat_window_low = forcing.clone()
        heat_window_high = forcing.clone()
        heat_window_high[..., 3] = torch.where(
            repro_mask,
            heat_window_high[..., 3] + heat_delta_c,
            heat_window_high[..., 3],
        )
        perturbations["window_heat"] = (
            _predict(model, batch, device, heat_window_low, n_init),
            _predict(model, batch, device, heat_window_high, n_init),
        )

        for name, (low_pred, high_pred) in perturbations.items():
            response_pct = 100.0 * (high_pred - low_pred) / baseline
            accum[name].append(response_pct.cpu())

    results = {}
    expected = {
        "precipitation": "context",
        "radiation": "positive",
        "vpd": "negative",
        "nitrogen": "positive",
        "temperature": "context",
        "heat_extreme": "negative",
        "window_heat": "negative",
        "window_vpd": "negative",
        "window_radiation": "positive",
        "window_water": "positive",
    }
    eps = float(getattr(C, "FACTOR_RESPONSE_EPS", 0.5))
    context_warn_abs = {
        "precipitation": 50.0,
        "temperature": 20.0,
    }
    if print_results:
        print("\n" + "=" * 80)
        print("AGRICULTURAL FACTOR RESPONSE AUDIT")
        print("=" * 80)
        print("  Note: nitrogen uses absolute low/high N scenarios (60 vs 240 kg/ha).")
    heat_delta_c = float(getattr(C, "HEAT_AUDIT_DELTA_C", 6.0))
    hot_day_c = float(getattr(C, "HEAT_AUDIT_HOT_DAY_C", 28.0))
    if print_results:
        print(
            f"  Note: heat_extreme adds +{heat_delta_c:g}C only on days "
            f"with Tmean >= {hot_day_c:g}C."
        )
        print("  Note: window_* perturbations act only during post-emergence progress 0.50-0.70.")
    for name, chunks in accum.items():
        if not chunks:
            continue
        values = torch.cat(chunks)
        mean = values.mean().item()
        median = values.median().item()
        q25 = torch.quantile(values, 0.25).item()
        q75 = torch.quantile(values, 0.75).item()
        expected_direction = expected[name]
        if expected_direction == "positive":
            active_fraction = (values >= eps).float().mean().item() * 100.0
            median_pass = median >= eps
            status = "PASS" if mean >= eps and median_pass else (
                "PARTIAL" if mean >= eps else "WARN"
            )
        elif expected_direction == "negative":
            active_fraction = (values <= -eps).float().mean().item() * 100.0
            median_pass = median <= -eps
            status = "PASS" if mean <= -eps and median_pass else (
                "PARTIAL" if mean <= -eps else "WARN"
            )
        else:
            warn_abs = context_warn_abs.get(name)
            status = (
                "WARN"
                if warn_abs is not None and abs(mean) > warn_abs
                else "INFO"
            )
            active_fraction = (values.abs() >= eps).float().mean().item() * 100.0
        if print_results:
            print(
                f"  {status:4s} | {name:14s} high-minus-low yield response: "
                f"mean={mean:+7.2f}% median={median:+7.2f}% "
                f"active={active_fraction:5.1f}%"
            )
        results[name] = {
            "mean_response_pct": mean,
            "median_response_pct": median,
            "q25_response_pct": q25,
            "q75_response_pct": q75,
            "active_fraction_pct": active_fraction,
            "expected": expected_direction,
            "status": status,
            "epsilon_pct": eps,
            "heat_audit_hot_day_c": hot_day_c if name == "heat_extreme" else None,
            "heat_audit_delta_c": heat_delta_c if name in {"heat_extreme", "window_heat"} else None,
        }
    return results


def save_factor_results(results, prefix="factor_response"):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    json_path = os.path.join(RESULTS_DIR, f"{prefix}.json")
    csv_path = os.path.join(RESULTS_DIR, f"{prefix}.csv")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "factor",
                "status",
                "expected",
                "mean_response_pct",
                "median_response_pct",
                "q25_response_pct",
                "q75_response_pct",
                "active_fraction_pct",
                "epsilon_pct",
                "heat_audit_hot_day_c",
                "heat_audit_delta_c",
            ],
        )
        writer.writeheader()
        for factor, values in results.items():
            writer.writerow({"factor": factor, **values})
    print(f"  Factor response JSON saved to {json_path}")
    print(f"  Factor response CSV saved to {csv_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpt",
        default=os.path.join(
            SAVE_DIR, f"agriworld_{C.MODEL_VERSION}_best.pth"
        ),
    )
    parser.add_argument("--data", default=C.DATA_PATH)
    parser.add_argument("--batches", type=int, default=3)
    parser.add_argument("--out-prefix", default="factor_response")
    args = parser.parse_args()

    model = AgriWorldSimulator().to(C.DEVICE)
    state = torch.load(args.ckpt, map_location=C.DEVICE, weights_only=True)
    model.load_state_dict(state, strict=False)
    ds = AgriTensorDataset(args.data)
    _, vds = split_dataset(
        ds, C.VAL_RATIO, C.SEED, getattr(C, "SPLIT_MODE", "temporal")
    )
    loader = DataLoader(vds, batch_size=C.BATCH_VAL, shuffle=False)
    results = audit_factor_responses(model, loader, C.DEVICE, args.batches)
    save_factor_results(results, args.out_prefix)


if __name__ == "__main__":
    main()

