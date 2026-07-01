"""Generate paper-oriented figures from saved AgriWorld results.

The script reads only files under results/ and writes PNGs to
paper_experiment_records/figures/. It does not import torch or run the model.

Usage:
    python scripts/make_paper_figures.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "paper_experiment_records" / "figures"


def _float(value, default=0.0):
    try:
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def _json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _style():
    plt.rcParams.update({
        "figure.dpi": 150,
        "savefig.dpi": 240,
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    })


def fig_data_composition():
    labels = ["Source", "Accepted", "Dropped"]
    values = [1327, 1198, 129]
    colors = ["#64748B", "#2563EB", "#F59E0B"]
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.4))

    bars = axes[0].bar(labels, values, color=colors, width=0.6)
    axes[0].set_title("Dataset QC")
    axes[0].set_ylabel("samples")
    axes[0].set_ylim(0, max(values) * 1.18)
    for bar, value in zip(bars, values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, value + 25, str(value),
                     ha="center", fontweight="bold")

    crop_labels = ["Corn", "Soybean"]
    crop_values = [794, 404]
    axes[1].pie(crop_values, labels=crop_labels, autopct="%1.1f%%",
                colors=["#22C55E", "#A855F7"], startangle=90)
    axes[1].set_title("Crop composition")

    fig.tight_layout()
    fig.savefig(OUT / "01_data_composition.png")
    plt.close(fig)


def fig_training_curves():
    rows = _rows(RESULTS / "training_history_phys_spatial.csv")
    if not rows:
        return
    epoch = [_float(r["epoch"]) for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.2))

    axes[0].plot(epoch, [_float(r["train_loss"]) for r in rows], label="train", color="#2563EB")
    axes[0].plot(epoch, [_float(r["val_loss"]) for r in rows], label="val", color="#DC2626")
    axes[0].set_title("Total loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend(frameon=False)

    axes[1].plot(epoch, [_float(r["yield_val"]) for r in rows], color="#16A34A")
    axes[1].set_title("Validation yield loss")
    axes[1].set_xlabel("epoch")

    axes[2].plot(epoch, [_float(r["spatial_contrast_loss"]) for r in rows],
                 label="contrast", color="#7C3AED")
    axes[2].plot(epoch, [_float(r["spatial_group_bias_loss"]) for r in rows],
                 label="group bias", color="#EA580C")
    axes[2].set_title("Spatial losses")
    axes[2].set_xlabel("epoch")
    axes[2].legend(frameon=False)

    fig.tight_layout()
    fig.savefig(OUT / "02_training_curves.png")
    plt.close(fig)


def fig_v38_v39_improvement():
    labels = ["Overall", "Corn", "Soybean"]
    v38 = [33.59, 37.68, 24.42]
    v39 = [16.31, 19.51, 7.56]
    x = range(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.bar([i - width / 2 for i in x], v38, width, label="V3.38 shared yield", color="#94A3B8")
    ax.bar([i + width / 2 for i in x], v39, width, label="V3.39 crop-conditioned", color="#2563EB")
    ax.set_xticks(list(x), labels)
    ax.set_ylabel("RMSE (bu/ac)")
    ax.set_title("Crop-conditioned yield formation improves multicrop prediction")
    ax.legend(frameon=False)
    for i, value in enumerate(v39):
        ax.text(i + width / 2, value + 0.8, f"{value:.1f}", ha="center", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "03_v38_v39_rmse.png")
    plt.close(fig)


def fig_crop_parameters(summary: dict):
    params = summary.get("crop_yield_parameters", {})
    if not params:
        return
    metrics = ["HI", "yield_scale", "yield_year_trend"]
    titles = ["Harvest index", "Yield scale", "Year trend"]
    crops = ["Corn", "Soybean"]
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.2))
    for ax, metric, title in zip(axes, metrics, titles):
        vals = [_float(params.get(c, {}).get(metric)) for c in crops]
        bars = ax.bar(crops, vals, color=["#22C55E", "#A855F7"], width=0.58)
        ax.set_title(title)
        for bar, value in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, value + max(vals) * 0.04,
                    f"{value:.3f}", ha="center", fontweight="bold")
        ax.set_ylim(0, max(vals) * 1.25 if max(vals) > 0 else 1)
    fig.tight_layout()
    fig.savefig(OUT / "04_crop_yield_parameters.png")
    plt.close(fig)


def fig_state_residuals():
    rows = _rows(RESULTS / "eval_agriworld_phys_spatial_best_state_residuals.csv")
    if not rows:
        return
    states = [r["state"] for r in rows]
    bias = [_float(r["bias_bu_ac"]) for r in rows]
    rmse = [_float(r["rmse_bu_ac"]) for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.3))
    axes[0].axhline(0, color="#111827", lw=0.8)
    axes[0].bar(states, bias, color=["#DC2626" if v < 0 else "#2563EB" for v in bias])
    axes[0].set_title("State bias")
    axes[0].set_ylabel("bu/ac")
    axes[1].bar(states, rmse, color="#64748B")
    axes[1].set_title("State RMSE")
    axes[1].set_ylabel("bu/ac")
    fig.tight_layout()
    fig.savefig(OUT / "05_state_residuals.png")
    plt.close(fig)


def fig_factor_response(summary: dict):
    factors = summary.get("factor_responses", {})
    order = ["radiation", "vpd", "nitrogen", "window_heat", "window_vpd", "window_radiation"]
    labels = ["Radiation", "VPD", "Nitrogen", "Window heat", "Window VPD", "Window rad."]
    values = [_float(factors.get(k, {}).get("mean_response_pct")) for k in order]
    fig, ax = plt.subplots(figsize=(7.8, 3.6))
    ax.axhline(0, color="#111827", lw=0.8)
    bars = ax.bar(labels, values, color=["#2563EB" if v >= 0 else "#DC2626" for v in values])
    ax.set_title("Counterfactual factor-response audit")
    ax.set_ylabel("yield response (%)")
    ax.tick_params(axis="x", rotation=18)
    for bar, value in zip(bars, values):
        offset = 1.0 if value >= 0 else -1.0
        va = "bottom" if value >= 0 else "top"
        ax.text(bar.get_x() + bar.get_width() / 2, value + offset,
                f"{value:+.1f}%", ha="center", va=va, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "06_factor_response.png")
    plt.close(fig)


def fig_factor_response_by_crop(summary: dict):
    by_crop = summary.get("factor_responses_by_crop", {})
    if not by_crop:
        return
    order = ["radiation", "vpd", "nitrogen", "window_heat", "window_vpd", "window_radiation"]
    labels = ["Radiation", "VPD", "Nitrogen", "Win heat", "Win VPD", "Win rad."]
    corn = [_float(by_crop.get("Corn", {}).get(k, {}).get("mean_response_pct")) for k in order]
    soy = [_float(by_crop.get("Soybean", {}).get(k, {}).get("mean_response_pct")) for k in order]
    x = range(len(labels))
    width = 0.36
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    ax.axhline(0, color="#111827", lw=0.8)
    ax.bar([i - width / 2 for i in x], corn, width, label="Corn", color="#22C55E")
    ax.bar([i + width / 2 for i in x], soy, width, label="Soybean", color="#A855F7")
    ax.set_xticks(list(x), labels)
    ax.tick_params(axis="x", rotation=18)
    ax.set_ylabel("yield response (%)")
    ax.set_title("Per-crop factor-response audit")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "08_factor_response_by_crop.png")
    plt.close(fig)


def fig_ablation_screening():
    labels, deltas = [], []
    compact = _rows(RESULTS / "ablation_results.csv")
    if compact:
        for row in compact:
            labels.append(row["variant"].replace("_", "\n"))
            deltas.append(_float(row.get("delta_rmse_bu_acre")))
    else:
        files = [
            RESULTS / "ablation_no_spatial_contrast_results.csv",
            RESULTS / "ablation_no_spatial_group_bias_results.csv",
            RESULTS / "ablation_no_window_stress_results.csv",
            RESULTS / "ablation_reproductive_only_window_stress_results.csv",
        ]
        for path in files:
            rows = _rows(path)
            if not rows:
                continue
            labels.append(rows[0]["variant"].replace("_", "\n"))
            deltas.append(_float(rows[0].get("delta_rmse_bu_acre")))
    if not labels:
        return
    fig, ax = plt.subplots(figsize=(7.6, 3.6))
    bars = ax.bar(labels, deltas, color="#F97316", width=0.6)
    ax.set_title("V3.39 compact ablation")
    ax.set_ylabel("Delta RMSE (bu/ac)")
    for bar, value in zip(bars, deltas):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.15,
                f"+{value:.1f}", ha="center", fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "07_ablation_screening.png")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    _style()
    summary = _json(RESULTS / "eval_agriworld_phys_spatial_best.json")
    fig_data_composition()
    fig_training_curves()
    fig_v38_v39_improvement()
    fig_crop_parameters(summary)
    fig_state_residuals()
    fig_factor_response(summary)
    fig_ablation_screening()
    fig_factor_response_by_crop(summary)
    print(f"Paper figures written to {OUT}")


if __name__ == "__main__":
    main()
