"""
AgriWorld Validate v2 鈥?鐗╃悊涓€鑷存€ч獙璇?+ SMAP 鎺㈤拡
===================================================
鏂板: static_features 浼犻€掋€丼MAP 鍦熷￥姘村垎绾挎€ф帰閽堥獙璇併€?
"""

import numpy as np
import torch
import agriworld.config as config
from agriworld.units import CORN_T_HA_TO_BU_AC


@torch.no_grad()
def validate_physics(model, dataloader, device, n_show=5):
    model.eval()
    results = []

    for i, batch in enumerate(dataloader):
        if i >= n_show:
            break
        forcing  = batch["forcing"].to(device)
        n_init   = batch["n_init"].to(device)
        static_f = batch["static_features"].to(device)
        tgt_yield = batch["target_yield"].to(device).squeeze(-1)
        year_b = batch.get("year", None)
        if year_b is not None:
            year_b = year_b.to(device)

        model.set_static_features(static_f)
        traj, pred_yield = model(forcing, n_init, year=year_b)

        lai    = traj[0, :, 0].cpu().numpy()
        bio    = traj[0, :, 1].cpu().numpy()
        n_pool = traj[0, :, 2].cpu().numpy()
        sw     = traj[0, :, 3].cpu().numpy()
        pcp    = forcing[0, :, 0].cpu().numpy()   # Precip
        eto    = forcing[0, :, 1].cpu().numpy()   # ETo
        tmean  = forcing[0, :, 3].cpu().numpy()   # Tmean
        vpd    = forcing[0, :, 4].cpu().numpy()   # VPD

        fc, wp, dr = model.ode_func.water_expert.get_soil_params()
        fc_v = fc[0].item() if fc.dim() > 0 else fc.item()
        wp_v = wp[0].item() if wp.dim() > 0 else wp.item()
        dr_v = dr[0].item() if dr.dim() > 0 else dr.item()

        k_ext = torch.abs(model.ode_func.rad_expert.k_ext).item()
        rue   = torch.abs(model.ode_func.rad_expert.rue).item()
        D0    = torch.abs(model.ode_func.stom_expert.D0).item()

        checks = {}

        # 1. N 姹犳湁鐭垮寲 鈥?涓嶈姹傚崟璋冿紝浠呮鏌ョ墿鐞嗚寖鍥?
        checks['n_pool_range_valid'] = bool(np.all(n_pool >= 0) and np.all(n_pool <= 400.0))

        # 2. 姘村垎骞宠　娈嬪樊
        water_balance = []
        for t in range(1, len(sw)):
            gdd_t = forcing[0, t, 6].item()
            gdd_ma = max(torch.abs(model.ode_func.pheno_expert.gdd_maturity).item(), 1.0)
            dev_t = min(max(gdd_t / gdd_ma, 0.0), 1.0)
            root_depth = 300.0 + 900.0 / (1.0 + np.exp(-8.0 * (dev_t - 0.25)))
            pcp_day = pcp[t]
            eto_day  = eto[t]
            fapar_t  = 1.0 - np.exp(-k_ext * max(lai[t], 0))
            f_vpd_t  = 1.0 / (1.0 + vpd[t] / max(D0, 0.1))
            water_avail = np.clip((sw[t-1] - wp_v) / max(fc_v - wp_v, 1e-3), 0, 1) ** 0.5
            runoff_fraction = 0.30 / (1.0 + np.exp(-(sw[t-1] - fc_v) / 0.03))
            infiltration = pcp_day * (1.0 - runoff_fraction)
            et_plant = eto_day * fapar_t * water_avail * f_vpd_t
            et_soil = eto_day * (1.0 - fapar_t) * (0.15 + 0.35 * water_avail)
            drain_approx = max(sw[t-1] - fc_v, 0) * dr_v * root_depth
            expected_delta = (infiltration - et_plant - et_soil - drain_approx) / root_depth
            actual_delta   = sw[t] - sw[t-1]
            water_balance.append(abs(expected_delta - actual_delta))
        checks['water_balance_err'] = float(np.mean(water_balance))

        # 3. 鐢熺墿閲忓闀?
        bio_diff = np.diff(bio)
        checks['bio_negative_growth_days'] = int(np.sum(bio_diff < -0.01))
        checks['bio_max'] = float(np.max(bio))

        # 4. LAI 宄板€?
        checks['lai_peak_doy']   = int(np.argmax(lai)) + 1
        checks['lai_peak_valid'] = bool(120 <= checks['lai_peak_doy'] <= 260)
        checks['lai_max']        = float(np.max(lai))

        # 5. 鐘舵€佽竟鐣?(涓?ODE 鍐?clip 涓€鑷? SW 鈭?[0, 0.60])
        checks['sw_in_bounds']      = bool(np.all(sw >= -0.01) and np.all(sw <= 0.61))
        checks['n_pool_positive']   = bool(np.all(n_pool >= 0))
        checks['bio_positive']      = bool(np.all(bio >= 0))
        checks['lai_positive']      = bool(np.all(lai >= 0))

        # 6. N 娴佸け
        n0_effective = torch.clamp(n_init[0, 0], max=400.0).item()
        checks['n_pool_net_depletion'] = float(n0_effective - n_pool[-1])

        # 7. 娓╁害鍝嶅簲妫€鏌?(浼犲叆鏍锋湰鐨勯潤鎬佺壒寰佸榻愮淮搴?
        tmean_t = torch.tensor(tmean, device=device).view(1, -1)
        static_one = static_f[:1]  # 鍙栫涓€涓牱鏈殑 static 瀵归綈 batch=1
        model.ode_func.temp_expert.set_static_features(static_one)
        f_temp_min = np.min(model.ode_func.temp_expert(tmean_t).cpu().numpy())
        f_temp_max = np.max(model.ode_func.temp_expert(tmean_t).cpu().numpy())
        checks['f_temp_range_valid'] = bool(0.0 <= f_temp_min <= f_temp_max <= 1.0)

        # 8. VPD 鑳佽揩鑼冨洿
        f_vpd_min = np.min(1.0 / (1.0 + vpd / max(D0, 0.1)))
        f_vpd_max = np.max(1.0 / (1.0 + vpd / max(D0, 0.1)))
        checks['f_vpd_range_valid'] = bool(0.0 < f_vpd_min <= f_vpd_max <= 1.0)

        results.append({
            'pred_yield': pred_yield[0].item(),
            'tgt_yield':  tgt_yield[0].item(),
            'checks':     checks,
        })

    print("\n" + "=" * 80)
    print("PHYSICAL CONSISTENCY VALIDATION")
    print("=" * 80)

    all_keys = list(results[0]['checks'].keys())
    for key in all_keys:
        vals = [r['checks'][key] for r in results]
        if isinstance(vals[0], bool):
            rate = sum(vals) / len(vals) * 100
            status = "PASS" if rate >= 80 else "WARN" if rate >= 50 else "FAIL"
            print(f"  {status} | {key}: {sum(vals)}/{len(vals)} ({rate:.0f}%)")
        elif isinstance(vals[0], (int, float)):
            print(f"       | {key}: mean={np.mean(vals):.4f} range=[{min(vals):.4f}, {max(vals):.4f}]")

    print(f"\n  Yield:")
    for r in results:
        print(
            f"    pred={r['pred_yield']:.2f} t/ha "
            f"({r['pred_yield'] * CORN_T_HA_TO_BU_AC:.1f} bu/ac) | "
            f"target={r['tgt_yield']:.2f} t/ha "
            f"({r['tgt_yield'] * CORN_T_HA_TO_BU_AC:.1f} bu/ac)"
        )

    print(f"\n  Learned Parameters:")
    print(f"    RUE: {rue:.3f}  |  k_ext: {k_ext:.3f}  |  D0: {D0:.3f} kPa")
    print(f"    FC: {fc_v:.3f}  |  WP: {wp_v:.3f}  |  drain: {dr_v:.4f}")

    return results


@torch.no_grad()
def validate_yield_all(model, dataloader, device):
    """Print aggregate yield metrics over the entire dataloader."""
    model.eval()
    preds = []
    tgts = []
    years = []

    for batch in dataloader:
        forcing = batch["forcing"].to(device)
        n_init = batch["n_init"].to(device)
        static_f = batch["static_features"].to(device)
        target = batch["target_yield"].to(device).squeeze(-1)
        year_b = batch.get("year", None)
        if year_b is not None:
            year_b = year_b.to(device)

        model.set_static_features(static_f)
        _, pred = model(forcing, n_init, year=year_b)
        preds.append(pred.detach().cpu())
        tgts.append(target.detach().cpu())
        if year_b is not None:
            years.append(year_b.detach().cpu())

    if not preds:
        print("No validation samples available for yield metrics.")
        return {}

    pred = torch.cat(preds).numpy()
    tgt = torch.cat(tgts).numpy()
    err = pred - tgt
    bias = float(np.mean(err))
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((tgt - np.mean(tgt)) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    scale = float(np.median(tgt / np.maximum(pred, 1e-6)))

    print("\n" + "=" * 80)
    print("FULL VALIDATION YIELD METRICS")
    print("=" * 80)
    print(
        f"  n={len(pred)} | bias={bias:+.3f} t/ha | "
        f"MAE={mae:.3f} | RMSE={rmse:.3f} | R2={r2:.3f} | "
        f"median target/pred={scale:.3f}"
    )

    if years:
        year_arr = torch.cat(years).numpy()
        for y in sorted(set(int(v) for v in year_arr)):
            m = year_arr == y
            ye = err[m]
            yp = pred[m]
            yt = tgt[m]
            y_scale = float(np.median(yt / np.maximum(yp, 1e-6)))
            print(
                f"  {y}: n={int(m.sum())} | "
                f"bias={np.mean(ye):+.3f} | "
                f"MAE={np.mean(np.abs(ye)):.3f} | "
                f"median target/pred={y_scale:.3f}"
            )

    return {
        "n": int(len(pred)),
        "bias": bias,
        "mae": mae,
        "rmse": rmse,
        "r2": float(r2),
        "median_target_pred": scale,
    }


@torch.no_grad()
def validate_physics(model, dataloader, device, n_show=5):
    """Validate the first n_show samples, not just the first sample per batch."""
    model.eval()
    results = []
    last_params = None

    for batch in dataloader:
        if len(results) >= n_show:
            break

        forcing = batch["forcing"].to(device)
        n_init = batch["n_init"].to(device)
        static_f = batch["static_features"].to(device)
        tgt_yield = batch["target_yield"].to(device).squeeze(-1)
        obs_lai_batch = batch.get("obs_lai", None)
        if obs_lai_batch is not None:
            obs_lai_batch = obs_lai_batch.to(device)
        sample_ids = batch.get("sample_id", None)
        year_b = batch.get("year", None)
        if year_b is not None:
            year_b = year_b.to(device)

        model.set_static_features(static_f)
        traj, pred_yield = model(forcing, n_init, year=year_b)

        fc, wp, dr = model.ode_func.water_expert.get_soil_params()
        k_ext = torch.abs(model.ode_func.rad_expert.k_ext).item()
        rue = torch.abs(model.ode_func.rad_expert.rue).item()
        d0 = torch.abs(model.ode_func.stom_expert.D0).item()

        for j in range(forcing.size(0)):
            if len(results) >= n_show:
                break

            lai = traj[j, :, 0].cpu().numpy()
            obs_lai = (
                obs_lai_batch[j].detach().cpu().numpy()
                if obs_lai_batch is not None else None
            )
            bio = traj[j, :, 1].cpu().numpy()
            n_pool = traj[j, :, 2].cpu().numpy()
            sw = traj[j, :, 3].cpu().numpy()
            pcp = forcing[j, :, 0].cpu().numpy()
            eto = forcing[j, :, 1].cpu().numpy()
            tmean = forcing[j, :, 3].cpu().numpy()
            vpd = forcing[j, :, 4].cpu().numpy()

            fc_v = fc[j].item() if fc.dim() > 0 else fc.item()
            wp_v = wp[j].item() if wp.dim() > 0 else wp.item()
            dr_v = dr[j].item() if dr.dim() > 0 else dr.item()
            last_params = (rue, k_ext, d0, fc_v, wp_v, dr_v)

            checks = {}
            checks["n_pool_range_valid"] = bool(
                np.all(n_pool >= 0) and np.all(n_pool <= 400.0)
            )

            water_balance = []
            gdd_ma = max(
                torch.abs(model.ode_func.pheno_expert.gdd_maturity).item(),
                1.0,
            )
            for t in range(1, len(sw)):
                dev_t = min(max(forcing[j, t, 6].item() / gdd_ma, 0.0), 1.0)
                root_depth = 300.0 + 900.0 / (
                    1.0 + np.exp(-8.0 * (dev_t - 0.25))
                )
                fapar_t = 1.0 - np.exp(-k_ext * max(lai[t], 0.0))
                f_vpd_t = 1.0 / (1.0 + vpd[t] / max(d0, 0.1))
                water_avail = np.clip(
                    (sw[t - 1] - wp_v) / max(fc_v - wp_v, 1e-3),
                    0.0,
                    1.0,
                ) ** 0.5
                runoff_fraction = 0.30 / (
                    1.0 + np.exp(-(sw[t - 1] - fc_v) / 0.03)
                )
                infiltration = pcp[t] * (1.0 - runoff_fraction)
                et_plant = eto[t] * fapar_t * water_avail * f_vpd_t
                et_soil = eto[t] * (1.0 - fapar_t) * (
                    0.15 + 0.35 * water_avail
                )
                drainage = max(sw[t - 1] - fc_v, 0.0) * dr_v * root_depth
                expected_delta = (
                    infiltration - et_plant - et_soil - drainage
                ) / root_depth
                water_balance.append(abs(expected_delta - (sw[t] - sw[t - 1])))

            checks["water_balance_err"] = float(np.mean(water_balance))
            checks["bio_negative_growth_days"] = int(
                np.sum(np.diff(bio) < -0.01)
            )
            checks["bio_max"] = float(np.max(bio))
            checks["lai_peak_doy"] = int(np.argmax(lai)) + 1
            checks["lai_peak_valid"] = bool(
                120 <= checks["lai_peak_doy"] <= 260
            )
            checks["lai_max"] = float(np.max(lai))
            checks["obs_lai_max"] = (
                float(np.nanmax(obs_lai)) if obs_lai is not None else float("nan")
            )
            checks["sw_in_bounds"] = bool(
                np.all(sw >= -0.01) and np.all(sw <= 0.66)
            )
            checks["n_pool_positive"] = bool(np.all(n_pool >= 0))
            checks["bio_positive"] = bool(np.all(bio >= 0))
            checks["lai_positive"] = bool(np.all(lai >= 0))
            n0_effective = torch.clamp(n_init[j, 0], max=400.0).item()
            checks["n_pool_net_depletion"] = float(n0_effective - n_pool[-1])
            water_floor = float(getattr(config, "SOIL_WATER_STRESS_FLOOR", 0.25))
            f_water_np = water_floor + (1.0 - water_floor) * np.clip(
                (sw - wp_v) / max(fc_v - wp_v, 1e-3),
                0.0,
                1.0,
            ) ** 0.5
            f_n_np = np.clip(n_pool / np.maximum(15.0 + 8.0 * bio, 1e-3), 0.0, 1.0)
            checks["f_water_mean"] = float(np.mean(f_water_np))
            checks["f_n_mean"] = float(np.mean(f_n_np))

            model.ode_func.temp_expert.set_static_features(static_f[j:j + 1])
            f_temp = model.ode_func.temp_expert(
                torch.tensor(tmean, device=device).view(1, -1)
            ).cpu().numpy()
            checks["f_temp_range_valid"] = bool(
                0.0 <= np.min(f_temp) <= np.max(f_temp) <= 1.0
            )
            f_vpd = 1.0 / (1.0 + vpd / max(d0, 0.1))
            checks["f_vpd_mean"] = float(np.mean(f_vpd))
            checks["f_vpd_range_valid"] = bool(
                0.0 < np.min(f_vpd) <= np.max(f_vpd) <= 1.0
            )

            results.append({
                "sample_id": (
                    sample_ids[j] if sample_ids is not None else str(len(results))
                ),
                "pred_yield": pred_yield[j].item(),
                "tgt_yield": tgt_yield[j].item(),
                "checks": checks,
            })

    if not results:
        print("No validation samples available for physical diagnostics.")
        return []

    print("\n" + "=" * 80)
    print("PHYSICAL CONSISTENCY VALIDATION")
    print("=" * 80)

    for key in results[0]["checks"]:
        vals = [r["checks"][key] for r in results]
        if isinstance(vals[0], bool):
            rate = sum(vals) / len(vals) * 100
            status = "PASS" if rate >= 80 else "WARN" if rate >= 50 else "FAIL"
            print(f"  {status} | {key}: {sum(vals)}/{len(vals)} ({rate:.0f}%)")
        else:
            print(
                f"       | {key}: mean={np.mean(vals):.4f} "
                f"range=[{min(vals):.4f}, {max(vals):.4f}]"
            )

    print("\n  Yield:")
    yield_errors = []
    for r in results:
        c = r["checks"]
        yield_errors.append(r["pred_yield"] - r["tgt_yield"])
        print(
            f"    {r['sample_id']} | pred={r['pred_yield']:.2f} t/ha "
            f"({r['pred_yield'] * CORN_T_HA_TO_BU_AC:.1f} bu/ac) | "
            f"target={r['tgt_yield']:.2f} t/ha "
            f"({r['tgt_yield'] * CORN_T_HA_TO_BU_AC:.1f} bu/ac) | "
            f"LAI={c['lai_max']:.2f}/{c['obs_lai_max']:.2f} | "
            f"Bio={c['bio_max']:.2f} | "
            f"Fw={c['f_water_mean']:.2f} Fn={c['f_n_mean']:.2f}"
        )
    yield_errors = np.asarray(yield_errors, dtype=np.float32)
    preds = np.asarray([r["pred_yield"] for r in results], dtype=np.float32)
    tgts = np.asarray([r["tgt_yield"] for r in results], dtype=np.float32)
    scale_hint = float(np.median(tgts / np.maximum(preds, 1e-6)))
    print(
        f"    summary | bias={yield_errors.mean():+.2f} t/ha | "
        f"MAE={np.abs(yield_errors).mean():.2f} t/ha | "
        f"median target/pred={scale_hint:.3f}"
    )

    if last_params is not None:
        rue, k_ext, d0, fc_v, wp_v, dr_v = last_params
        print("\n  Learned Parameters:")
        trend = (
            model.yield_year_slope.item()
            if hasattr(model, "yield_year_slope") else 0.0
        )
        print(f"    RUE: {rue:.3f}  |  k_ext: {k_ext:.3f}  |  D0: {d0:.3f} kPa")
        print(
            f"    HI: {model.harvest_index.item():.3f}  |  "
            f"YS: {model.yield_scale.item():.3f}  |  "
            f"YT: {trend:.3f} log/year"
        )
        print(f"    FC: {fc_v:.3f}  |  WP: {wp_v:.3f}  |  drain: {dr_v:.4f}")

    return results


@torch.no_grad()
def validate_smap(model, dataloader, device):
    """
    SMAP 鍦熷￥姘村垎绾挎€ф帰閽堥獙璇併€?

    鏂囩尞渚濇嵁 (data_instrument.md 搂5):
        鐢ㄧ嚎鎬у洖褰?SMAP_obs = a 脳 SW_ode + b锛?
        R虏 > 0.5 鈫?ODE 缂栫爜浜嗙湡瀹炲湡澹ゆ按鍒嗕俊鍙?
        R虏 < 0.2 鈫?SW 鍙槸涓轰紭鍖栦骇閲忎骇鐢熺殑鏃犳剰涔夋暟鍊?
    """
    model.eval()
    sw_preds_surface = []
    sw_preds_rootzone = []
    smap_surface_obs = []
    smap_rootzone_obs = []

    for batch in dataloader:
        forcing  = batch["forcing"].to(device)
        n_init   = batch["n_init"].to(device)
        static_f = batch["static_features"].to(device)
        year_b = batch.get("year", None)
        if year_b is not None:
            year_b = year_b.to(device)

        model.set_static_features(static_f)
        traj, _ = model(forcing, n_init, year=year_b)

        sw = traj[:, :, 3]  # [B, T]

        # SMAP 鏁版嵁
        smap_s = batch.get("val_smap_surface", None)
        smap_r = batch.get("val_smap_rootzone", None)

        if smap_s is not None:
            smap_s = smap_s.to(device)
            # 鍙彇鏈夋晥 (闈為浂) 鐨?SMAP 瑙傛祴
            valid_s = smap_s > 0.001
            if valid_s.any():
                sw_preds_surface.append(sw[valid_s].cpu())
                smap_surface_obs.append(smap_s[valid_s].cpu())

        if smap_r is not None:
            smap_r = smap_r.to(device)
            valid_r = smap_r > 0.001
            if valid_r.any():
                sw_preds_rootzone.append(sw[valid_r].cpu())
                smap_rootzone_obs.append(smap_r[valid_r].cpu())

    print("\n" + "=" * 80)
    print("SMAP SOIL MOISTURE VALIDATION (Linear Probe)")
    print("=" * 80)

    results = {}

    for name, preds_list, obs_list in [
        ("Surface (0-5cm)", sw_preds_surface, smap_surface_obs),
        ("Rootzone (0-100cm)", sw_preds_rootzone, smap_rootzone_obs),
    ]:
        if len(preds_list) == 0:
            print(f"  {name}: No valid SMAP data")
            continue

        preds = torch.cat(preds_list).numpy()
        obs   = torch.cat(obs_list).numpy()

        # 绾挎€у洖褰? obs = a 脳 pred + b
        A = np.vstack([preds, np.ones_like(preds)]).T
        a, b = np.linalg.lstsq(A, obs, rcond=None)[0]
        obs_pred = a * preds + b

        ss_res = np.sum((obs - obs_pred) ** 2)
        ss_tot = np.sum((obs - np.mean(obs)) ** 2)
        r2 = 1 - ss_res / max(ss_tot, 1e-12)

        status = "PASS" if r2 > 0.5 else "WARN" if r2 > 0.2 else "FAIL"
        print(f"  {status} | {name}: R虏={r2:.4f}  a={a:.4f}  b={b:.4f}  n={len(preds)}")
        results[name] = {"R虏": r2, "a": a, "b": b, "n": len(preds)}

    return results

