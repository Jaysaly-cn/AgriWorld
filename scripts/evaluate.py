"""
AgriWorld 鈥?妯″瀷鍩虹嚎璇勪及
========================
鍔犺浇宸茶缁冪殑 best checkpoint锛屽叏闈㈣瘎浼?
  1. 浜ч噺绮惧害 (鎸変綔鐗╃被鍨嬪垎缁?
  2. LAI 鏃堕棿搴忓垪鎷熷悎
  3. SMAP 鍦熷￥姘村垎鎺㈤拡 (R虏 绾挎€у洖褰?
  4. 鐗╃悊涓€鑷存€?(姘村垎骞宠　, 鐘舵€佽竟鐣?
  5. 鍙傛暟鏀舵暃璇婃柇 (鍝簺鍙傛暟瀛︿範浜? 鍝簺鍋滄粸浜?
  6. 璇勪及缁撴灉搴忓垪鍖栦负 eval_baseline.pt

鐢ㄦ硶:
    python evaluate.py                            # 榛樿璺緞
    python evaluate.py --ckpt /path/to/model.pth  # 鎸囧畾 checkpoint
"""

import os, sys, argparse, csv, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
from torch.utils.data import DataLoader

import agriworld.config as C
from agriworld.paths import RESULTS_DIR, SAVE_DIR
from agriworld.log_utils import tee_stdout
from agriworld.dataset import AgriTensorDataset
from agriworld.simulator import AgriWorldSimulator
from agriworld.validate import validate_physics, validate_smap, validate_yield_all
from agriworld.units import t_ha_to_bu_ac_factor
from agriworld.splits import split_dataset
from agriworld.window_stress import WINDOW_NAMES, FACTOR_NAMES
from scripts.factor_response import audit_factor_responses


def _load_model_state(model, state):
    if "model_state_dict" in state:
        payload = state["model_state_dict"]
        epoch_info = state.get("epoch", "?")
        best_val = state.get("best_val_loss", "?")
    else:
        payload = state
        epoch_info, best_val = "? (raw state_dict)", "?"
    current = model.state_dict()
    skipped = []
    for key in list(payload.keys()):
        if key in current and payload[key].shape != current[key].shape:
            skipped.append(key)
            payload.pop(key)
    if skipped:
        print(f"  Shape-mismatched checkpoint keys skipped: {len(skipped)}")
    incompatible = model.load_state_dict(payload, strict=False)
    if incompatible.missing_keys:
        print(f"  Missing checkpoint keys initialized from defaults: {len(incompatible.missing_keys)}")
    if incompatible.unexpected_keys:
        print(f"  Unexpected checkpoint keys ignored: {len(incompatible.unexpected_keys)}")
    return epoch_info, best_val


def _crop_yield_parameter_summary(model):
    crop_ids = torch.tensor([1, 5], device=next(model.parameters()).device)
    return {
        "Corn": {
            "HI": float(model.crop_harvest_index(crop_ids[:1]).item()),
            "yield_scale": float(model.crop_yield_scale(crop_ids[:1]).item()),
            "yield_year_trend": float(model.crop_yield_year_slope(crop_ids[:1]).item()),
        },
        "Soybean": {
            "HI": float(model.crop_harvest_index(crop_ids[1:]).item()),
            "yield_scale": float(model.crop_yield_scale(crop_ids[1:]).item()),
            "yield_year_trend": float(model.crop_yield_year_slope(crop_ids[1:]).item()),
        },
    }


def _stress_summary(model, traj, forcing):
    ode = model.ode_func
    f_temp = ode.temp_expert(forcing[..., 3:4])
    dev_index, _, _, _ = ode.pheno_expert.from_cumulative(forcing[..., 6:7])
    f_temp = ode._effective_temperature_stress(
        f_temp,
        tmean=forcing[..., 3:4],
        dev_index=dev_index,
    )
    f_water = ode.water_expert(traj[..., 3:4])
    f_n, _ = ode.n_expert(traj[..., 2:3], traj[..., 1:2])
    f_vpd = ode.stom_expert(forcing[..., 4:5])
    return {
        "mean_f_water": float(f_water.mean().item()),
        "mean_f_vpd": float(f_vpd.mean().item()),
        "mean_f_n": float(f_n.mean().item()),
        "mean_f_temp": float(f_temp.mean().item()),
    }


def _county_adaptation_summary(model, forcing):
    values = {
        "county_hi_factor": 1.0,
        "county_yield_factor": 1.0,
        "county_heat_sensitivity": 1.0,
        "repro_heat_factor": 1.0,
    }
    adjustments = model.static_crop_adjustments()
    if adjustments is not None:
        hi_factor, yield_factor, heat_sens = adjustments
        values["county_hi_factor"] = float(hi_factor.mean().item())
        values["county_yield_factor"] = float(yield_factor.mean().item())
        values["county_heat_sensitivity"] = float(heat_sens.mean().item())
    if getattr(C, "USE_REPRODUCTIVE_HEAT_PENALTY", True):
        heat_factor = model._reproductive_heat_factor(
            forcing,
            forcing[:, :, 6],
            torch.tensor(
                [[values["county_heat_sensitivity"]]],
                dtype=forcing.dtype,
                device=forcing.device,
            ),
        )
        values["repro_heat_factor"] = float(heat_factor.mean().item())
    return values


def _window_stress_row(model):
    values = {
        "window_stress_factor": 1.0,
        "dominant_stress_window": "",
        "dominant_stress_factor": "",
    }
    for name in WINDOW_NAMES:
        values[f"stress_{name}"] = 0.0
    summary = getattr(model, "last_window_stress", None)
    if not summary:
        return values
    factor = summary["factor"].detach().reshape(-1)[0].item()
    window_stress = summary["window_stress"].detach()[0]
    contributions = summary["contributions"].detach()[0]
    flat_index = int(torch.argmax(contributions).item())
    window_idx = flat_index // len(FACTOR_NAMES)
    factor_idx = flat_index % len(FACTOR_NAMES)
    values["window_stress_factor"] = float(factor)
    values["dominant_stress_window"] = WINDOW_NAMES[window_idx]
    values["dominant_stress_factor"] = FACTOR_NAMES[factor_idx]
    for idx, name in enumerate(WINDOW_NAMES):
        values[f"stress_{name}"] = float(window_stress[idx].item())
    return values


def _group_residuals(sample_rows, group_key):
    groups = {}
    for row in sample_rows:
        key = str(row.get(group_key, ""))
        groups.setdefault(key, []).append(row)
    summaries = []
    for key, rows in groups.items():
        pred = np.array([float(row["pred_bu_ac"]) for row in rows])
        target = np.array([float(row["target_bu_ac"]) for row in rows])
        residual = pred - target
        rmse = float(np.sqrt(np.mean(residual ** 2)))
        mae = float(np.mean(np.abs(residual)))
        target_mean = float(np.mean(target))
        summaries.append({
            group_key: key,
            "n": len(rows),
            "target_mean_bu_ac": target_mean,
            "pred_mean_bu_ac": float(np.mean(pred)),
            "bias_bu_ac": float(np.mean(residual)),
            "mae_bu_ac": mae,
            "rmse_bu_ac": rmse,
            "nrmse_pct": float(rmse / max(target_mean, 1e-6) * 100.0),
            "abs_bias_bu_ac": float(abs(np.mean(residual))),
        })
    return sorted(
        summaries,
        key=lambda row: (row["rmse_bu_ac"], row["abs_bias_bu_ac"]),
        reverse=True,
    )


def _spatial_residual_summary(sample_rows):
    min_count = 3
    by_state = _group_residuals(sample_rows, "state")
    by_county = _group_residuals(sample_rows, "county")
    by_county_n3 = [row for row in by_county if row["n"] >= min_count]
    state_rmse = np.array([row["rmse_bu_ac"] for row in by_state], dtype=float)
    state_bias = np.array([row["bias_bu_ac"] for row in by_state], dtype=float)
    county_rmse = np.array([row["rmse_bu_ac"] for row in by_county], dtype=float)
    county_bias = np.array([row["bias_bu_ac"] for row in by_county], dtype=float)
    county_n3_rmse = np.array([row["rmse_bu_ac"] for row in by_county_n3], dtype=float)
    county_n3_bias = np.array([row["bias_bu_ac"] for row in by_county_n3], dtype=float)
    return {
        "by_state": by_state,
        "by_county": by_county,
        "county_reliable_min_count": min_count,
        "county_reliable_group_count": len(by_county_n3),
        "macro_state_rmse_bu_ac": float(np.mean(state_rmse)) if state_rmse.size else float("nan"),
        "macro_county_rmse_bu_ac": float(np.mean(county_rmse)) if county_rmse.size else float("nan"),
        "macro_county_rmse_n3_bu_ac": float(np.mean(county_n3_rmse)) if county_n3_rmse.size else float("nan"),
        "state_bias_std_bu_ac": float(np.std(state_bias)) if state_bias.size else float("nan"),
        "county_bias_std_bu_ac": float(np.std(county_bias)) if county_bias.size else float("nan"),
        "county_bias_std_n3_bu_ac": float(np.std(county_n3_bias)) if county_n3_bias.size else float("nan"),
        "state_bias_range_bu_ac": float(np.max(state_bias) - np.min(state_bias)) if state_bias.size else float("nan"),
        "worst_states_by_rmse": by_state[:5],
        "worst_counties_by_abs_bias": sorted(
            by_county,
            key=lambda row: row["abs_bias_bu_ac"],
            reverse=True,
        )[:20],
        "worst_counties_n3_by_abs_bias": sorted(
            by_county_n3,
            key=lambda row: row["abs_bias_bu_ac"],
            reverse=True,
        )[:20],
    }


def _write_rows_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?#  鏍稿績璇勪及鍑芥暟
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?
@torch.no_grad()
def evaluate(ckpt_path, data_path, device, save_legacy=True):
    verbose = bool(getattr(C, "VERBOSE", False))
    print(f"Evaluate | schema={getattr(C, 'MODEL_SCHEMA', 'unknown')} | device={device}")
    if verbose:
        print(f"  checkpoint: {ckpt_path}")
        print(f"  data:       {data_path}")

    # 鈹€鈹€ 1. 鍔犺浇妯″瀷 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    model = AgriWorldSimulator().to(device)
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    epoch_info, best_val = _load_model_state(model, state)
    model.eval()
    if verbose:
        print(f"  Model loaded | epoch={epoch_info} | best_val={best_val}")

    # 鈹€鈹€ 2. 鍔犺浇鏁版嵁 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    ds = AgriTensorDataset(data_path)
    N = len(ds)
    tds, vds = split_dataset(
        ds, C.VAL_RATIO, C.SEED, getattr(C, "SPLIT_MODE", "temporal")
    )
    n_train, n_val = len(tds), len(vds)
    vdl = DataLoader(vds, batch_size=C.BATCH_VAL, shuffle=False)

    # 鈹€鈹€ 3. 浣滅墿绫诲瀷鍒嗗竷 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    crop_types = [int(ds[i]['static_features'][10].item()) for i in range(N)]
    corn = sum(1 for c in crop_types if c == 1)
    soy  = sum(1 for c in crop_types if c == 5)
    print(f"  Dataset: {N} samples ({n_train} train / {n_val} val) | corn={corn} soy={soy} other={N-corn-soy}")

    groups = {}  # {crop_label: {preds: [], tgts: []}}

    n_val = len(vds)
    sample_rows = []
    trajectory_records = []
    max_trajectories = int(getattr(C, "EVAL_TRAJECTORY_SAMPLES", 24))
    progress_every = int(getattr(C, "EVAL_PROGRESS_EVERY", 0))
    for i in range(n_val):
        if progress_every > 0 and (i % progress_every == 0 or i == n_val - 1):
            print(f"  Inference: {i+1}/{n_val}", flush=True)

        idx = vds.indices[i]
        sample = ds[idx]
        ct = int(sample['static_features'][10].item())
        label = {1: 'Corn', 5: 'Soybean'}.get(ct, f'Other({ct})')

        if label not in groups:
            groups[label] = {'preds': [], 'tgts': []}

        static_f = sample['static_features'].unsqueeze(0).to(device)
        forcing  = sample['forcing'].unsqueeze(0).to(device)
        n_init   = sample['n_init'].unsqueeze(0).to(device)
        year = torch.tensor([sample['year']], device=device)
        state_id = sample.get("state_id", None)
        county_id = sample.get("county_id", None)
        if state_id is not None:
            state_id = state_id.to(device)
        if county_id is not None:
            county_id = county_id.to(device)

        model.set_static_features(static_f, state_id=state_id, county_id=county_id)
        traj, pred = model(forcing, n_init, year=year)

        bu_factor = t_ha_to_bu_ac_factor(ct)
        tgt_raw = sample['target_yield'].item() * bu_factor
        pred_raw = pred.item() * bu_factor
        pred_t_ha = pred.item()
        target_t_ha = sample["target_yield"].item()
        peak_lai_pred = traj[..., 0].amax().item()
        mask_lai = sample["mask_lai"].bool()
        obs_vals = sample["obs_lai"][mask_lai]
        peak_lai_obs = float(obs_vals.max().item()) if obs_vals.numel() else float("nan")
        stress = _stress_summary(model, traj, forcing)
        county_adapt = _county_adaptation_summary(model, forcing)
        window_stress = _window_stress_row(model)

        groups[label]['preds'].append(pred_raw)
        groups[label]['tgts'].append(tgt_raw)
        sample_rows.append({
            "sample_id": sample.get("sample_id", idx),
            "year": sample.get("year", ""),
            "state": sample.get("state", ""),
            "county": sample.get("county", ""),
            "state_id": int(sample.get("state_id", torch.tensor([-1]))[0].item()),
            "county_id": int(sample.get("county_id", torch.tensor([-1]))[0].item()),
            "crop": label,
            "target_t_ha": target_t_ha,
            "pred_t_ha": pred_t_ha,
            "target_bu_ac": tgt_raw,
            "pred_bu_ac": pred_raw,
            "residual_t_ha": pred_t_ha - target_t_ha,
            "residual_bu_ac": pred_raw - tgt_raw,
            "peak_lai_pred": peak_lai_pred,
            "peak_lai_obs": peak_lai_obs,
            "final_biomass": traj[:, -1, 1].item(),
            **stress,
            **county_adapt,
            **window_stress,
        })
        if (
            getattr(C, "SAVE_EVAL_TRAJECTORIES", True) and
            len(trajectory_records) < max_trajectories
        ):
            trajectory_records.append({
                "sample_id": sample.get("sample_id", idx),
                "year": sample.get("year", ""),
                "state": sample.get("state", ""),
                "county": sample.get("county", ""),
                "traj": traj.squeeze(0).detach().cpu(),
                "forcing": forcing.squeeze(0).detach().cpu(),
                "obs_lai": sample["obs_lai"].detach().cpu(),
                "mask_lai": sample["mask_lai"].detach().cpu(),
                "pred_t_ha": pred_t_ha,
                "target_t_ha": target_t_ha,
                **county_adapt,
                **window_stress,
            })

    # 缁熻
    all_preds, all_tgts = [], []
    for label, g in groups.items():
        p = np.array(g['preds']); t = np.array(g['tgts'])
        all_preds.extend(p); all_tgts.extend(t)
        mape = np.mean(np.abs((p - t) / (t + 1e-3))) * 100
        rmse = np.sqrt(np.mean((p - t) ** 2))
        r2 = 1 - np.sum((p - t)**2) / max(np.sum((t - t.mean())**2), 1e-10) if len(t) > 1 else 0
        if verbose:
            print(f"  {label:10s}  n={len(p):3d}  "
                  f"pred={np.mean(p):6.1f}  tgt={np.mean(t):6.1f}  "
                  f"RMSE={rmse:5.1f}  MAPE={mape:5.1f}%  R2={r2:.3f}")

    # 鎬讳綋
    ap = np.array(all_preds); at = np.array(all_tgts)
    r2_all = 1 - np.sum((ap-at)**2) / max(np.sum((at-at.mean())**2), 1e-10)
    rmse_all = np.sqrt(np.mean((ap-at)**2))
    adaptation_summary = {}
    for key in [
        "county_hi_factor",
        "county_yield_factor",
        "county_heat_sensitivity",
        "repro_heat_factor",
    ]:
        vals = np.array([row[key] for row in sample_rows], dtype=float)
        adaptation_summary[key] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }
    window_summary = {}
    if sample_rows:
        vals = np.array([row["window_stress_factor"] for row in sample_rows], dtype=float)
        window_summary["factor"] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "min": float(np.min(vals)),
            "max": float(np.max(vals)),
        }
        for name in WINDOW_NAMES:
            vals = np.array([row[f"stress_{name}"] for row in sample_rows], dtype=float)
            window_summary[name] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
            }
    spatial_residuals = _spatial_residual_summary(sample_rows)
    print(
        f"  Yield all: n={len(ap)} pred={np.mean(ap):.1f} tgt={np.mean(at):.1f} "
        f"RMSE={rmse_all:.1f} bu/ac NRMSE={rmse_all/at.mean()*100:.1f}% R2={r2_all:.3f}"
    )
    for label, g in groups.items():
        p, t = np.array(g['preds']), np.array(g['tgts'])
        bias = np.mean(p - t)
        rmse = np.sqrt(np.mean((p - t) ** 2))
        print(f"  {label}: n={len(p)} RMSE={rmse:.1f} bu/ac bias={bias:+.1f} bu/ac")
    crop_params = _crop_yield_parameter_summary(model)
    print(
        "  Crop yield params: "
        f"Corn HI={crop_params['Corn']['HI']:.3f} "
        f"YS={crop_params['Corn']['yield_scale']:.3f} "
        f"YT={crop_params['Corn']['yield_year_trend']:.3f} | "
        f"Soybean HI={crop_params['Soybean']['HI']:.3f} "
        f"YS={crop_params['Soybean']['yield_scale']:.3f} "
        f"YT={crop_params['Soybean']['yield_year_trend']:.3f}"
    )
    if spatial_residuals["worst_states_by_rmse"]:
        worst = spatial_residuals["worst_states_by_rmse"][0]
        print(
            "  Spatial residual: "
            f"worst_state={worst['state']} "
            f"RMSE={worst['rmse_bu_ac']:.1f} "
            f"bias={worst['bias_bu_ac']:+.1f} bu/ac | "
            f"macro_state_RMSE={spatial_residuals['macro_state_rmse_bu_ac']:.1f} "
            f"macro_county_RMSE_n3={spatial_residuals['macro_county_rmse_n3_bu_ac']:.1f} "
            f"state_bias_std={spatial_residuals['state_bias_std_bu_ac']:.1f}"
        )

    # 鈹€鈹€ 5. 鐗╃悊涓€鑷存€?鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    if getattr(C, "EVAL_RUN_PHYSICS", False) or verbose:
        print("\n  Physical consistency validation")
        validate_yield_all(model, vdl, device)
        validate_physics(model, vdl, device, n_show=10)

    # 鈹€鈹€ 6. SMAP 鎺㈤拡 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    if getattr(C, "EVAL_RUN_SMAP", False) or verbose:
        print("\n  SMAP probe")
        validate_smap(model, vdl, device)

    # 鈹€鈹€ 7. 鍥犲瓙鍝嶅簲瀹¤ 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    factor_results = audit_factor_responses(
        model,
        vdl,
        device,
        max_batches=3,
        print_results=bool(getattr(C, "EVAL_PRINT_FACTOR_TABLE", False) or verbose),
        by_crop=True,
    )
    factor_results_by_crop = factor_results.pop("_by_crop", {})
    factor_line = " ".join(
        f"{name}={vals.get('mean_response_pct', float('nan')):+.1f}%"
        for name, vals in factor_results.items()
    )
    print(f"  Factor response: {factor_line}")

    # 鈹€鈹€ 8. 鎵€鏈夊彲瀛︿範鍙傛暟 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    if getattr(C, "EVAL_PRINT_PARAMS", False) or verbose:
        print("\n  Learnable parameters")
        for name, param in model.named_parameters():
            if param.numel() == 1:
                print(f"  {name:50s} = {param.item():.4f}")
            elif param.numel() < 30:
                vals = param.detach().flatten().tolist()
                print(f"  {name:50s} = [{', '.join(f'{v:.4f}' for v in vals)}]")
            else:
                print(f"  {name:50s}  shape={str(list(param.shape)):16s}  "
                      f"mean={param.mean().item():.4f}  std={param.std().item():.4f}")

    # 鈹€鈹€ 9. 淇濆瓨姹囨€?鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    summary = {
        'checkpoint': ckpt_path,
        'n_grids': N, 'n_val': len(vds),
        'crop_counts': {'corn': corn, 'soy': soy, 'other': N-corn-soy},
        'yield_all': {
            'rmse_bu_acre': float(rmse_all),
            'r2': float(r2_all),
            'nrmse_pct': float(rmse_all / at.mean() * 100),
        },
        'yield_per_crop': {
            label: {
                'n': len(g['preds']),
                'rmse': float(np.sqrt(np.mean((np.array(g['preds'])-np.array(g['tgts']))**2))),
                'mape_pct': float(np.mean(np.abs((np.array(g['preds'])-np.array(g['tgts'])) / (np.array(g['tgts'])+1e-3))) * 100),
            }
            for label, g in groups.items()
        },
        'factor_responses': factor_results,
        'factor_responses_by_crop': factor_results_by_crop,
        'county_adaptation': adaptation_summary,
        'window_stress': window_summary,
        'crop_yield_parameters': _crop_yield_parameter_summary(model),
        'spatial_residuals': spatial_residuals,
        'params': {
            name: float(param.item())
            for name, param in model.named_parameters()
            if param.numel() == 1
        },
    }
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ckpt_stem = os.path.splitext(os.path.basename(ckpt_path))[0]
    named_save_path = os.path.join(RESULTS_DIR, f"eval_{ckpt_stem}.pt")
    named_json_path = os.path.join(RESULTS_DIR, f"eval_{ckpt_stem}.json")
    named_samples_path = os.path.join(RESULTS_DIR, f"eval_{ckpt_stem}_samples.csv")
    named_state_path = os.path.join(RESULTS_DIR, f"eval_{ckpt_stem}_state_residuals.csv")
    named_county_path = os.path.join(RESULTS_DIR, f"eval_{ckpt_stem}_county_residuals.csv")
    named_traj_path = os.path.join(RESULTS_DIR, f"eval_{ckpt_stem}_trajectories.pt")
    torch.save(summary, named_save_path)
    with open(named_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    if getattr(C, "SAVE_EVAL_TABLES", True) and sample_rows:
        with open(named_samples_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(sample_rows[0].keys()))
            writer.writeheader()
            writer.writerows(sample_rows)
        if verbose:
            print(f"  Per-sample CSV saved to {named_samples_path}")
        _write_rows_csv(named_state_path, spatial_residuals["by_state"])
        _write_rows_csv(named_county_path, spatial_residuals["by_county"])
        if verbose:
            print(f"  State residual CSV saved to {named_state_path}")
            print(f"  County residual CSV saved to {named_county_path}")
    if getattr(C, "SAVE_EVAL_TRAJECTORIES", True) and trajectory_records:
        torch.save(trajectory_records, named_traj_path)
        if verbose:
            print(f"  Trajectory sample saved to {named_traj_path}")
    print(f"  Saved: {named_save_path}")
    print(f"  Saved: {named_json_path}")
    if save_legacy:
        save_path = os.path.join(RESULTS_DIR, "eval_baseline.pt")
        torch.save(summary, save_path)
        json_path = os.path.join(RESULTS_DIR, "eval_baseline.json")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        print(f"  Saved: {save_path}")
        print(f"  Saved: {json_path}")

    # D0 璇婃柇
    for name, param in model.named_parameters():
        if 'D0' in name and param.numel() == 1:
            status = "unchanged" if abs(param.item() - 2.0) < 0.01 else "updated"
            print(f"  D0 (VPD half-saturation) = {param.item():.3f}  {status}")

    return summary


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?#  CLI
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='AgriWorld 妯″瀷鍩虹嚎璇勪及')
    parser.add_argument('--ckpt', type=str,
                        default=os.path.join(
                            SAVE_DIR,
                            f'agriworld_{C.MODEL_VERSION}_best.pth',
                        ))
    parser.add_argument('--data', type=str,
                        default=C.DATA_PATH)
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument(
        '--log',
        type=str,
        default=os.path.join(RESULTS_DIR, "evaluate.log"),
        help='Path to a text log file. Use empty string to disable.',
    )
    args = parser.parse_args()

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.log:
        with tee_stdout(args.log):
            summary = evaluate(args.ckpt, args.data, device)
            print(f"\n  Text log saved to {args.log}")
    else:
        evaluate(args.ckpt, args.data, device)

