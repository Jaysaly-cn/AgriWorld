import numpy as np
import torch
import torch.nn.functional as F
from agriworld.config import W_LAI, W_YIELD, W_L1, W_SMOOTH


def compute_lai_loss(traj, obs_lai, mask, tgt_yield=None):
    """Robust LAI loss.

    Satellite LAI can be biased low for otherwise high-yield corn pixels.
    A plain relative MSE over-penalizes model LAI above small observations
    and can collapse biomass, so low-LAI over-prediction receives reduced
    weight only when the yield label indicates a productive crop.
    """
    pred_lai = traj[:, :, 0]
    lai_err = (pred_lai - obs_lai) / obs_lai.clamp(min=2.0)
    point_loss = F.smooth_l1_loss(
        lai_err,
        torch.zeros_like(lai_err),
        beta=0.5,
        reduction="none",
    )

    reliability = torch.ones_like(point_loss)
    if tgt_yield is not None:
        high_yield = (tgt_yield.view(-1, 1) >= 8.0).float()
        low_obs = (obs_lai < 1.5).float()
        over_pred = (pred_lai > obs_lai).float()
        reliability = reliability - 0.75 * high_yield * low_obs * over_pred

    weighted_mask = mask * reliability.clamp(min=0.25)
    return torch.sum(point_loss * weighted_mask) / (
        weighted_mask.sum() + 1e-8
    )


def compute_yield_loss(pred_yield, tgt_yield):
    """Differentiable robust loss in log-yield space."""
    lp = torch.log(pred_yield.clamp(min=1e-6))
    lt = torch.log(tgt_yield.clamp(min=1e-6))
    diff = lp - lt
    robust = F.smooth_l1_loss(lp, lt, beta=0.5)
    bias_penalty = diff.mean().square()
    loss = robust + 2.5 * bias_penalty
    return loss, lp, lt


def compute_spatial_contrast_loss(
    pred_yield,
    tgt_yield,
    county_id=None,
    crop_id=None,
    min_gap=0.15,
):
    """Pairwise county contrast in log-yield space.

    In multi-crop training, pairwise ranking is only meaningful within the
    same crop because corn and soybean occupy different yield scales.
    """
    if pred_yield.numel() < 2:
        return pred_yield.new_tensor(0.0)
    lp = torch.log(pred_yield.clamp(min=1e-6)).view(-1)
    lt = torch.log(tgt_yield.clamp(min=1e-6)).view(-1)
    pred_diff = lp[:, None] - lp[None, :]
    target_diff = lt[:, None] - lt[None, :]
    mask = torch.triu(torch.ones_like(pred_diff, dtype=torch.bool), diagonal=1)
    if county_id is not None:
        cid = county_id.view(-1)
        mask = mask & (cid[:, None] != cid[None, :])
    if crop_id is not None:
        crop = crop_id.view(-1)
        mask = mask & (crop[:, None] == crop[None, :])
    if min_gap > 0:
        mask = mask & ((tgt_yield.view(-1, 1) - tgt_yield.view(1, -1)).abs() >= min_gap)
    if not mask.any():
        return pred_yield.new_tensor(0.0)
    return F.smooth_l1_loss(
        pred_diff[mask],
        target_diff[mask],
        beta=0.25,
    )


def compute_group_bias_loss(
    pred_yield,
    tgt_yield,
    group_id=None,
    crop_id=None,
    min_count=4,
):
    """Penalize systematic regional mean bias in log-yield space.

    When crop_id is provided, groups become region-crop cells. This prevents
    the regional bias term from mixing corn and soybean baselines.
    """
    if group_id is None or pred_yield.numel() < min_count:
        return pred_yield.new_tensor(0.0)
    residual = (
        torch.log(pred_yield.clamp(min=1e-6)).view(-1) -
        torch.log(tgt_yield.clamp(min=1e-6)).view(-1)
    )
    group = group_id.view(-1)
    if crop_id is not None:
        group = group.long() * 512 + crop_id.view(-1).long()
    terms = []
    for gid in torch.unique(group):
        mask = group == gid
        if int(mask.sum().item()) >= int(min_count):
            terms.append(residual[mask].mean().square())
    if not terms:
        return pred_yield.new_tensor(0.0)
    return torch.stack(terms).mean()


def compute_canopy_yield_consistency_loss(
    traj, tgt_yield, obs_lai=None, mask_lai=None
):
    """High grain yield requires feasible canopy and biomass support."""
    peak_lai = traj[:, :, 0].amax(dim=1)
    final_bio = traj[:, -1, 1]

    required_peak_lai = torch.clamp(0.38 * tgt_yield, min=2.5, max=5.5)
    if obs_lai is not None:
        if mask_lai is not None:
            obs_for_peak = torch.where(
                mask_lai > 0,
                obs_lai,
                torch.zeros_like(obs_lai),
            )
        else:
            obs_for_peak = obs_lai
        obs_peak = obs_for_peak.amax(dim=1)
        obs_supported = torch.clamp(0.65 * obs_peak, min=0.0, max=7.0)
        high_yield = tgt_yield >= 8.0
        required_peak_lai = torch.where(
            high_yield,
            torch.maximum(required_peak_lai, obs_supported),
            required_peak_lai,
        )
    lai_gap = torch.relu(required_peak_lai - peak_lai) / required_peak_lai

    # Use a stricter support threshold than the absolute HI/YS upper bounds
    # so yield cannot be matched mainly by the conversion head.
    required_bio = tgt_yield / (0.62 * 1.70)
    bio_gap = torch.relu(required_bio - final_bio) / required_bio.clamp(min=1.0)

    return (lai_gap.square() + bio_gap.square()).mean()


def compute_anomaly_reg(coupling, f_temp, f_water_raw, f_n_raw,
                        lai_d, n_d, sw_d, B, T):
    _, _, a = coupling(f_temp, f_water_raw, f_n_raw, lai_d, n_d, sw_d)
    a = a.reshape(B, T, 1)
    l1 = torch.mean(torch.abs(a))
    sm = torch.mean((a[:, 1:] - a[:, :-1]) ** 2) if a.size(1) > 1 else torch.tensor(0.0, device=a.device)
    return W_L1 * l1 + W_SMOOTH * sm


def compute_vpd_aux_loss(model, forcing):
    """
    VPD 杈呭姪 Loss: 榧撳姳 D0 浣?f_vpd 鍒嗗竷鍦ㄥ悎鐞嗚寖鍥淬€?
    鐩爣: 鐢熼暱瀛ｅ钩鍧?f_vpd 搴斿湪 0.4-0.8 涔嬮棿 (鐜夌背涓タ閮ㄥ吀鍨嬪€?銆?
    濡傛灉 f_vpd_mean 寮傚父楂?(>0.9) 鎴栧紓甯镐綆 (<0.2), 鏂藉姞鎯╃綒銆?
    """
    vpd = forcing[..., 4:5]  # VPD channel
    f_vpd = model.ode_func.stom_expert(vpd)  # [B, T, 1]
    f_vpd_mean = f_vpd.mean(dim=1)  # [B, 1]  姣忎釜缃戞牸鐨勬椂闂村钩鍧囧€?

    # 鎯╃綒杩囬珮鎴栬繃浣庣殑 f_vpd
    penalty = (
        torch.relu(f_vpd_mean - 0.85).mean() +
        torch.relu(0.25 - f_vpd_mean).mean()
    )
    return 0.05 * penalty  # 灏忔潈閲? 浠呯敤浜庢縺娲?D0 姊害


def compute_state_constraint_loss(model, traj):
    """Penalize impossible states without hiding them inside the ODE solver."""
    lai = traj[..., 0]
    bio = traj[..., 1]
    n_pool = traj[..., 2]
    sw = traj[..., 3]

    fc, wp, _ = model.ode_func.water_expert.get_soil_params()
    while fc.dim() < sw.dim():
        fc = fc.unsqueeze(-1)
        wp = wp.unsqueeze(-1)
    saturation = torch.clamp(fc + 0.12, max=0.60)

    bound_penalty = (
        torch.relu(-lai).square().mean() +
        torch.relu(lai - 12.0).square().mean() +
        torch.relu(-bio).square().mean() +
        torch.relu(bio - 35.0).square().mean() +
        torch.relu(-n_pool).square().mean() / 100.0 +
        torch.relu(n_pool - 400.0).square().mean() / 100.0 +
        torch.relu(0.02 - sw).square().mean() * 100.0 +
        torch.relu(sw - saturation).square().mean() * 100.0
    )
    negative_growth = torch.relu(-(bio[:, 1:] - bio[:, :-1])).square().mean()
    return bound_penalty + negative_growth


def compute_parameter_prior_loss(model):
    """Weak literature priors reduce compensation between physical parameters."""
    ode = model.ode_func
    terms = [
        ((torch.abs(ode.rad_expert.rue) - 4.8) / 1.50).square(),
        ((torch.abs(ode.rad_expert.k_ext) - 0.85) / 0.30).square(),
        ((torch.abs(ode.stom_expert.D0) - 2.0) / 0.75).square(),
        ((torch.abs(ode.sla) - 0.015) / 0.005).square(),
        ((torch.abs(ode.k_sen) - 0.008) / 0.004).square(),
        ((torch.abs(ode.pheno_expert.gdd_flowering) - 850.0) / 150.0).square(),
        ((torch.abs(ode.pheno_expert.gdd_maturity) - 1700.0) / 250.0).square(),
        ((model.harvest_index - 0.56) / 0.08).square(),
        ((model.yield_scale - 1.15) / 0.18).square(),
        ((model.yield_year_slope - 0.035) / 0.025).square(),
    ]
    fc, wp, drain = ode.water_expert.get_soil_params()
    terms.extend([
        ((fc.mean() - 0.30) / 0.08).square(),
        ((wp.mean() - 0.13) / 0.05).square(),
        ((drain.mean() - 0.10) / 0.06).square(),
        torch.relu(wp - fc + 0.04).square().mean(),
    ])
    year_weight = ode.year_embed.embed.weight
    year_penalty = year_weight.square().mean() + year_weight.mean(dim=0).square().mean()
    return torch.stack([term.reshape(()) for term in terms]).mean() + year_penalty


def compute_static_adaptation_regularization(model):
    """Keep county-level structured adapters interpretable and non-memorizing."""
    if not hasattr(model, "static_crop_adjustments"):
        return torch.tensor(0.0, device=model.harvest_index.device)
    adjustments = model.static_crop_adjustments()
    if adjustments is None:
        return torch.tensor(0.0, device=model.harvest_index.device)
    hi_factor, yield_factor, heat_sensitivity = adjustments
    return (
        torch.log(hi_factor.clamp_min(1e-6)).square().mean() +
        torch.log(yield_factor.clamp_min(1e-6)).square().mean() +
        (heat_sensitivity - 1.0).square().mean()
    )


def compute_window_stress_regularization(model):
    """Keep crop-window stress visible but not a hidden yield offset."""
    summary = getattr(model, "last_window_stress", None)
    if not summary:
        return torch.tensor(0.0, device=model.harvest_index.device)
    factor = summary["factor"]
    sensitivity = summary["sensitivity"]
    return (1.0 - factor).square().mean() + 0.02 * sensitivity.square().mean()


@torch.no_grad()
def calibrate_scale(model, dataloader, device):
    model.eval()
    ratios = []
    for batch in dataloader:
        forcing = batch["forcing"].to(device)
        n_init = batch["n_init"].to(device)
        static_f = batch["static_features"].to(device)
        tgt = batch["target_yield"].to(device).squeeze(-1)

        state_id = batch.get("state_id", None)
        county_id = batch.get("county_id", None)
        if state_id is not None:
            state_id = state_id.to(device)
        if county_id is not None:
            county_id = county_id.to(device)
        model.set_static_features(static_f, state_id=state_id, county_id=county_id)
        year_b = batch.get("year", None)
        if year_b is not None: year_b = year_b.to(device)
        traj, _ = model(forcing, n_init, year=year_b)

        bio = traj[:, -1, 1].clamp(min=1e-8)
        gdd_final = forcing[:, -1, 6]
        hi_dynamic = model.harvest_index * torch.sigmoid(
            (gdd_final - torch.abs(model.gdd_flowering)) /
            torch.abs(model.hi_width).clamp(min=1.0)
        )
        unscaled_yield = bio * hi_dynamic
        valid = (
            torch.isfinite(unscaled_yield) &
            torch.isfinite(tgt) &
            (unscaled_yield > 1e-8) &
            (tgt > 0)
        )
        if valid.any():
            ratios.append((tgt[valid] / unscaled_yield[valid]).cpu())

    if not ratios:
        print("  [鏍″噯] 鎵€鏈?batch 鍧囨棤鏁堬紝浣跨敤榛樿 scale=1.0")
        model.set_yield_scale(1.0)
        return 1.0

    # Median ratio is robust to a few samples with near-zero final biomass.
    scl = float(torch.median(torch.cat(ratios)).item())
    raw_scl = scl
    scl = np.clip(scl, 0.5, 2.0)
    if raw_scl < 0.5 or raw_scl > 2.0:
        print(
            f"  [calibration warning] raw yield scale={raw_scl:.3f} "
            f"is outside [0.5, 2.0]; using {scl:.3f}."
        )
    if not np.isfinite(scl):
        print(f"  [calibration] invalid scale={scl}; using default 1.0")
        scl = 1.0
    model.set_yield_scale(scl)
    model.log_yield_scale.requires_grad_(True)
    return scl

