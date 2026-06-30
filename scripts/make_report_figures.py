"""Generate lightweight figures for the AgriWorld progress report.

This script only depends on the files already produced under ``results/``.
It does not import torch or run the model, so it can be used on a laptop
after downloading the result folder.

Usage:
    python scripts/make_report_figures.py
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "docs" / "report_assets"


def _safe_float(value, default=0.0):
    try:
        if value in (None, ""):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _load_current_ablation_rows() -> list[dict]:
    """Prefer individually re-run v3.23 results, then append full matrix rows."""
    rows_by_variant: dict[str, dict] = {}
    for name in [
        "ablation_baseline_results.csv",
        "ablation_no_temperature_stress_results.csv",
        "ablation_heat_stress_025_results.csv",
    ]:
        for row in _load_csv_rows(RESULTS / name):
            rows_by_variant[row["variant"]] = row

    for row in _load_csv_rows(RESULTS / "ablation_results.csv"):
        rows_by_variant.setdefault(row["variant"], row)
    return list(rows_by_variant.values())


def _style():
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 220,
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def fig_data_funnel():
    labels = ["Raw samples", "Accepted", "Insufficient LAI", "Non-corn"]
    values = [1857, 867, 147, 843]
    colors = ["#6B7280", "#2F80ED", "#F2C94C", "#EB5757"]

    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    bars = ax.bar(labels, values, color=colors, width=0.58)
    ax.set_title("Dataset QC funnel (2019-2023 merged data)")
    ax.set_ylabel("County-year samples")
    ax.set_ylim(0, max(values) * 1.18)
    ax.tick_params(axis="x", rotation=12)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 35,
            f"{value}",
            ha="center",
            va="bottom",
            fontweight="bold",
        )
    fig.tight_layout()
    fig.savefig(OUT / "01_data_funnel.png")
    plt.close(fig)


def fig_model_workflow():
    fig, ax = plt.subplots(figsize=(9.5, 3.8))
    ax.axis("off")
    boxes = [
        ("Weather\nDaymet", 0.03, 0.62, "#D6EAF8"),
        ("Soil + terrain\nSoilGrids/3DEP", 0.03, 0.28, "#E8F6EF"),
        ("Management + crop\nN rate / CDL", 0.03, -0.06, "#FEF5E7"),
        ("Expert modules\nwater, N, radiation,\nstomatal, phenology", 0.31, 0.28, "#EBF5FB"),
        ("Differentiable ODE\nLAI, biomass,\nN pool, soil water", 0.58, 0.28, "#FDEDEC"),
        ("Outputs\nLAI trajectory\nyield + factor response", 0.82, 0.28, "#F4ECF7"),
    ]
    for text, x, y, color in boxes:
        ax.add_patch(
            plt.Rectangle(
                (x, y),
                0.18,
                0.24,
                facecolor=color,
                edgecolor="#374151",
                linewidth=1.1,
                transform=ax.transAxes,
            )
        )
        ax.text(x + 0.09, y + 0.12, text, ha="center", va="center", transform=ax.transAxes)

    arrows = [
        ((0.21, 0.74), (0.31, 0.45)),
        ((0.21, 0.40), (0.31, 0.40)),
        ((0.21, 0.06), (0.31, 0.35)),
        ((0.49, 0.40), (0.58, 0.40)),
        ((0.76, 0.40), (0.82, 0.40)),
    ]
    for start, end in arrows:
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            xycoords=ax.transAxes,
            arrowprops=dict(arrowstyle="->", lw=1.4, color="#374151"),
        )
    ax.text(
        0.5,
        0.94,
        "AgriWorld v3.23: physics-guided differentiable agricultural world model",
        ha="center",
        va="center",
        fontsize=13,
        fontweight="bold",
        transform=ax.transAxes,
    )
    fig.tight_layout()
    fig.savefig(OUT / "02_model_workflow.png")
    plt.close(fig)


def fig_metric_cards(summary: dict):
    yield_all = summary.get("yield_all", {})
    corn = (summary.get("yield_per_crop", {}) or {}).get("Corn", {})
    metrics = [
        ("Validation samples", summary.get("n_val", 187), ""),
        ("RMSE", yield_all.get("rmse_bu_acre", 21.53), "bu/ac"),
        ("NRMSE", yield_all.get("nrmse_pct", 11.14), "%"),
        ("Corn MAPE", corn.get("mape_pct", 8.71), "%"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(9.5, 2.2))
    for ax, (title, value, unit) in zip(axes, metrics):
        ax.axis("off")
        ax.add_patch(
            plt.Rectangle((0.02, 0.08), 0.96, 0.84, facecolor="#F8FAFC", edgecolor="#CBD5E1")
        )
        if isinstance(value, (int, float)):
            text = f"{value:.2f}" if title != "Validation samples" else f"{int(value)}"
        else:
            text = str(value)
        ax.text(0.5, 0.62, text, ha="center", va="center", fontsize=19, fontweight="bold")
        ax.text(0.5, 0.42, unit, ha="center", va="center", color="#64748B")
        ax.text(0.5, 0.22, title, ha="center", va="center", fontsize=9)
    fig.suptitle("Current baseline performance", y=1.05, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "03_metric_cards.png", bbox_inches="tight")
    plt.close(fig)


def fig_factor_response(summary: dict):
    factors = summary.get("factor_responses", {})
    order = ["precipitation", "radiation", "vpd", "nitrogen", "temperature", "heat_extreme"]
    values = [_safe_float(factors.get(name, {}).get("mean_response_pct")) for name in order]
    labels = ["Precip", "Radiation", "VPD", "Nitrogen", "Temp", "Extreme heat"]
    colors = ["#2F80ED" if v >= 0 else "#EB5757" for v in values]

    fig, ax = plt.subplots(figsize=(8.5, 4.0))
    ax.axhline(0, color="#111827", linewidth=0.8)
    bars = ax.bar(labels, values, color=colors, width=0.62)
    ax.set_title("Counterfactual factor-response audit")
    ax.set_ylabel("High-minus-low yield response (%)")
    for bar, value in zip(bars, values):
        va = "bottom" if value >= 0 else "top"
        offset = 1.0 if value >= 0 else -1.0
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + offset,
            f"{value:+.1f}%",
            ha="center",
            va=va,
            fontweight="bold",
        )
    ax.set_ylim(min(values + [0]) - 8, max(values + [0]) + 10)
    fig.tight_layout()
    fig.savefig(OUT / "04_factor_response.png")
    plt.close(fig)


def fig_ablation(rows: list[dict]):
    preferred = [
        "baseline",
        "no_temperature_stress",
        "heat_stress_025",
        "no_year_trend",
        "no_vpd_stress",
        "no_nitrogen_stress",
        "hard_temperature_stress",
    ]
    row_by_variant = {row.get("variant"): row for row in rows}
    variants = [v for v in preferred if v in row_by_variant]
    values = [_safe_float(row_by_variant[v].get("rmse_bu_acre")) for v in variants]
    labels = [
        "baseline",
        "no temp",
        "heat 0.25",
        "no year",
        "no VPD",
        "no N",
        "hard temp",
    ][: len(variants)]

    fig, ax = plt.subplots(figsize=(8.8, 4.0))
    colors = ["#2F80ED" if v == "baseline" else "#9CA3AF" for v in variants]
    bars = ax.bar(labels, values, color=colors, width=0.62)
    ax.set_title("Ablation comparison")
    ax.set_ylabel("Yield RMSE (bu/ac)")
    ax.tick_params(axis="x", rotation=18)
    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(values) * 0.015,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    ax.set_ylim(0, max(values) * 1.2 if values else 1)
    fig.tight_layout()
    fig.savefig(OUT / "05_ablation_rmse.png")
    plt.close(fig)


def fig_next_steps():
    steps = [
        ("1", "Tighten\nfactor audit", "Correct PASS rules\nfor negative response"),
        ("2", "Spatial\nheterogeneity", "Static-conditioned\nresidual / interactions"),
        ("3", "Water-state\nprobe", "Align ODE soil water\nwith SMAP scale/depth"),
        ("4", "Paper-ready\nexperiments", "Fixed protocol for\nAAAI-style evaluation"),
    ]
    fig, ax = plt.subplots(figsize=(9.5, 3.2))
    ax.axis("off")
    for i, (num, title, detail) in enumerate(steps):
        x = 0.05 + i * 0.235
        ax.add_patch(
            plt.Circle((x, 0.68), 0.045, color="#2F80ED", transform=ax.transAxes)
        )
        ax.text(x, 0.68, num, ha="center", va="center", color="white", fontweight="bold", transform=ax.transAxes)
        ax.text(x, 0.47, title, ha="center", va="center", fontsize=11, fontweight="bold", transform=ax.transAxes)
        ax.text(x, 0.22, detail, ha="center", va="center", fontsize=9, color="#4B5563", transform=ax.transAxes)
        if i < len(steps) - 1:
            ax.annotate(
                "",
                xy=(x + 0.17, 0.68),
                xytext=(x + 0.06, 0.68),
                xycoords=ax.transAxes,
                arrowprops=dict(arrowstyle="->", lw=1.5, color="#6B7280"),
            )
    ax.text(0.5, 0.94, "Recommended next work plan", ha="center", va="center", fontsize=13, fontweight="bold", transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(OUT / "06_next_steps.png")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    _style()
    summary = _load_json(RESULTS / "eval_agriworld_baseline_best.json")
    rows = _load_current_ablation_rows()

    fig_data_funnel()
    fig_model_workflow()
    fig_metric_cards(summary)
    fig_factor_response(summary)
    fig_ablation(rows)
    fig_next_steps()

    print(f"Report figures saved to: {OUT}")


if __name__ == "__main__":
    main()
