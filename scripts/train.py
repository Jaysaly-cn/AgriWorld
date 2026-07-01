"""
AgriWorld Trainer v2 鈥?鍏ㄥ彉閲忔秷璐圭殑璁粌寰幆
=============================================
鏂板: 姣?batch 浼犲叆 static_features 鈫?鍒濆鍖栧悇 Expert 鐨勭綉鏍肩壒寮傛€у弬鏁般€?
"""

import os
import csv
import time
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import agriworld.config as C
from agriworld.dataset import AgriTensorDataset
from agriworld.simulator import AgriWorldSimulator
from agriworld.losses import (compute_lai_loss, compute_yield_loss,
                     compute_anomaly_reg, compute_vpd_aux_loss,
                     compute_state_constraint_loss,
                     compute_parameter_prior_loss,
                     compute_canopy_yield_consistency_loss,
                     compute_static_adaptation_regularization,
                     compute_spatial_contrast_loss,
                     compute_group_bias_loss,
                     compute_window_stress_regularization)
from agriworld.validate import validate_physics, validate_yield_all
from agriworld.pretrain import run_all_pretrain
from agriworld.splits import split_dataset
from agriworld.paths import DATA_ROOT, PROJECT_ROOT, RESULTS_DIR, SAVE_DIR


def _write_training_history(history, version):
    if not history or not getattr(C, "SAVE_TRAIN_HISTORY", True):
        return
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, f"training_history_{version}.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)
    print(f"Training history saved to {path}")


def _print_split_summary(name, subset):
    samples = [subset.dataset.samples[i] for i in subset.indices]
    targets = torch.tensor([sample["target_yield"].item() for sample in samples])
    final_gdd = torch.tensor([sample["gdd_final"] for sample in samples])
    years = sorted({int(sample["year"]) for sample in samples})
    crops = {}
    for sample in samples:
        crop = sample.get("crop", "unknown")
        crops[crop] = crops.get(crop, 0) + 1
    low_gdd = int((final_gdd < 500.0).sum().item())
    print(
        f"{name}: n={len(samples)} | years={years} | "
        f"crops={crops} | "
        f"yield={targets.mean():.2f} [{targets.min():.2f}, {targets.max():.2f}] t/ha | "
        f"GDD={final_gdd.mean():.0f} [{final_gdd.min():.0f}, {final_gdd.max():.0f}] | "
        f"GDD<500: {low_gdd}"
    )
    if low_gdd:
        bad = [
            sample for sample in samples
            if sample["gdd_final"] < 500.0
        ][:10]
        for sample in bad:
            tmean = sample["forcing"][:, 3]
            print(
                f"  bad GDD | id={sample['sample_id']} "
                f"year={sample['year']} state={sample['state']} "
                f"planting={sample['planting_doy']} "
                f"Tmean=[{tmean.min().item():.1f}, {tmean.max().item():.1f}] "
                f"GDD={sample['gdd_final']:.1f}"
            )
        raise ValueError(
            f"{name} contains {low_gdd} samples with final GDD below 500. "
            "Check daily GDD and planting metadata before training."
        )


def train():
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        allow_tf32 = bool(getattr(C, "ALLOW_TF32", True))
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32

    torch.manual_seed(C.SEED)
    import numpy as np
    np.random.seed(C.SEED)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data root:    {DATA_ROOT}")
    print(f"Data file:    {C.DATA_PATH}")
    print(f"Model output: {SAVE_DIR}")
    if not os.path.exists(C.DATA_PATH):
        raise FileNotFoundError(
            f"Training data not found: {C.DATA_PATH}. "
            "Check AGRI_DATA_ROOT or AGRI_DATA_PATH."
        )
    os.makedirs(SAVE_DIR, exist_ok=True)

    ds = AgriTensorDataset(C.DATA_PATH)
    tds, vds = split_dataset(
        ds, C.VAL_RATIO, C.SEED, getattr(C, "SPLIT_MODE", "temporal")
    )
    n_train, n_val = len(tds), len(vds)
    pin_memory = bool(
        getattr(C, "PIN_MEMORY", True) and C.DEVICE.type == "cuda"
    )
    loader_workers = int(getattr(C, "DATA_LOADER_WORKERS", 0))
    loader_kwargs = {
        "num_workers": loader_workers,
        "pin_memory": pin_memory,
    }
    tdl = DataLoader(
        tds, batch_size=C.BATCH_TRAIN, shuffle=True, **loader_kwargs
    )
    vdl = DataLoader(
        vds, batch_size=C.BATCH_VAL, shuffle=False, **loader_kwargs
    )
    non_blocking = pin_memory

    print(
        f"Schema: {getattr(C, 'MODEL_SCHEMA', 'unknown')} | "
        f"Device: {C.DEVICE} | Dataset: {len(ds)} "
        f"({n_train} train / {n_val} val) | "
        f"Split: {getattr(tds, 'split_mode_used', C.SPLIT_MODE)}"
    )
    print(
        f"Runtime: batch={C.BATCH_TRAIN}/{C.BATCH_VAL} | "
        f"ODE={getattr(C, 'ODE_METHOD', 'euler')} "
        f"step={getattr(C, 'ODE_STEP_SIZE', 1.0):g} day | "
        f"train_stride={getattr(C, 'TRAIN_STEP_DAYS', 1)} day | "
        f"val_every={getattr(C, 'VAL_EVERY', 1)} | "
        f"pin_memory={pin_memory} | workers={loader_workers} | "
        f"TF32={getattr(C, 'ALLOW_TF32', True)} | "
        f"LSTM={getattr(C, 'USE_LSTM_RESIDUAL', False)} | "
        f"year_emb={getattr(C, 'USE_YEAR_EMBEDDING', False)}"
    )
    _print_split_summary("Train", tds)
    _print_split_summary("Val", vds)

    # 鈹€鈹€ 棰勮缁冩潈閲?鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    paths = run_all_pretrain(device=C.DEVICE, force=False)

    model = AgriWorldSimulator().to(C.DEVICE)

    # 鍔犺浇鍙敤鐨勯璁粌鏉冮噸
    if os.path.exists(paths['water']):
        model.ode_func.water_expert.load_state_dict(
            torch.load(paths['water'], weights_only=True), strict=False
        )
    if getattr(C, "FREEZE_SOIL_HYDRAULICS", True):
        for param in model.ode_func.water_expert.parameters():
            param.requires_grad_(False)
        print("Soil hydraulic pedotransfer: frozen at v3 pretrained values")
    if os.path.exists(paths['nitrogen']):
        model.ode_func.n_expert.load_state_dict(
            torch.load(paths['nitrogen'], weights_only=True), strict=False
        )
    if os.path.exists(paths['radiation']):
        model.ode_func.rad_expert.load_state_dict(
            torch.load(paths['radiation'], weights_only=True), strict=False
        )
    if os.path.exists(paths['stomatal']):
        model.ode_func.stom_expert.load_state_dict(
            torch.load(paths['stomatal'], weights_only=True), strict=False
        )
    if os.path.exists(paths['phenology']):
        model.ode_func.pheno_expert.load_state_dict(
            torch.load(paths['phenology'], weights_only=True), strict=False
        )
    if os.path.exists(paths['coupling']):
        model.ode_func.coupling.load_state_dict(
            torch.load(paths['coupling'], weights_only=True), strict=False
        )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params}")

    # 鈹€鈹€ 鍒濆鐗╃悊杞ㄨ抗鍋ュ悍妫€鏌?鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    model.eval()
    with torch.no_grad():
        sb = next(iter(tdl))
        sf = sb["static_features"].to(
            C.DEVICE, non_blocking=non_blocking
        )
        yr_sb = sb.get("year", None)
        if yr_sb is not None:
            yr_sb = yr_sb.to(C.DEVICE, non_blocking=non_blocking)
        state_sb = sb.get("state_id", None)
        county_sb = sb.get("county_id", None)
        if state_sb is not None:
            state_sb = state_sb.to(C.DEVICE, non_blocking=non_blocking)
        if county_sb is not None:
            county_sb = county_sb.to(C.DEVICE, non_blocking=non_blocking)
        model.set_static_features(sf, state_id=state_sb, county_id=county_sb)
        traj0, py = model(
            sb["forcing"].to(C.DEVICE, non_blocking=non_blocking),
            sb["n_init"].to(C.DEVICE, non_blocking=non_blocking),
            year=yr_sb,
        )
        ty = sb["target_yield"].to(
            C.DEVICE, non_blocking=non_blocking
        ).squeeze(-1)
        peak_lai = traj0[..., 0].amax(dim=1).median().item()
        final_bio = traj0[:, -1, 1].median().item()
        pred_median = py.median().item()
        target_median = ty.median().item()
        print(
            f"Initial physics | peak LAI median={peak_lai:.3f} | "
            f"final biomass median={final_bio:.3f} t/ha | "
            f"yield median={pred_median:.3f}/{target_median:.3f} t/ha | "
            f"YS={model.yield_scale.item():.3f}"
        )
        if peak_lai < 0.20 or final_bio < 0.50:
            raise RuntimeError(
                "Initial physical trajectory failed canopy establishment "
                f"(peak LAI={peak_lai:.3f}, biomass={final_bio:.3f})."
            )

    # 鈹€鈹€ 浼樺寲鍣ㄥ垎缁?鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    opt_expert = optim.Adam(model.expert_params(), lr=C.LR_EXPERT, weight_decay=C.WEIGHT_DECAY)
    opt_photo = optim.Adam(model.photo_params(), lr=C.LR_PHOTO, weight_decay=C.WEIGHT_DECAY)
    opt_coupling = optim.Adam(model.coupling_params(), lr=C.LR_COUPLING, weight_decay=C.WEIGHT_DECAY)
    opt_yield = optim.Adam(model.yield_params(), lr=C.LR_YIELD, weight_decay=0.0)
    opt_stomatal = optim.Adam(model.stomatal_params(), lr=C.LR_STOMATAL, weight_decay=0.0)
    opt_phenology = optim.Adam(model.phenology_params(), lr=C.LR_PHENOLOGY, weight_decay=0.0)
    opt_lstm = optim.Adam(model.lstm_params(), lr=getattr(C, 'LR_LSTM', 1e-3), weight_decay=0.0)

    best_val = float("inf")
    wait = 0
    avg_val = float("nan")
    avg_val_lai = float("nan")
    avg_val_yield = float("nan")
    ver = getattr(C, 'MODEL_VERSION', 'baseline')
    best_path = os.path.join(C.SAVE_DIR, f"agriworld_{ver}_best.pth")
    history = []

    for ep in range(C.MAX_EPOCHS):
        epoch_started = time.perf_counter()
        if C.DEVICE.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        model.train()

        # 闃舵鎺у埗
        if ep < C.PHASE_LAI_ONLY:
            train_yield = False
            w_a = 0.0
            w_y = 0.0
        elif ep < C.PHASE_LAI_ONLY + C.PHASE_ANOM_RAMP:
            train_yield = True
            ramp = (ep - C.PHASE_LAI_ONLY + 1) / C.PHASE_ANOM_RAMP
            w_a = min(1.0, ramp)
            w_y = min(1.0, ramp)
        else:
            train_yield = True
            w_a = 1.0
            w_y = 1.0
            # Phase 3 鍏ュ彛: 閲嶇疆楠岃瘉鍩虹嚎, 婵€娲?LSTM 娈嬪樊
            if ep == C.PHASE_LAI_ONLY + C.PHASE_ANOM_RAMP:
                best_val = float('inf')
                wait = 0
                model.ode_func.lstm_active_flag.fill_(True)
                print("  [Phase 3] reset validation baseline; LSTM residual active")

        s_loss, s_lai, s_yield, s_anom, s_state, nb = 0.0, 0.0, 0.0, 0.0, 0.0, 0
        s_canopy = 0.0
        s_static_adapt = 0.0
        s_spatial_contrast = 0.0
        s_spatial_bias = 0.0
        s_window_stress = 0.0

        for batch in tdl:
            forcing = batch["forcing"].to(
                C.DEVICE, non_blocking=non_blocking
            )
            n_init = batch["n_init"].to(
                C.DEVICE, non_blocking=non_blocking
            )
            obs_lai = batch["obs_lai"].to(
                C.DEVICE, non_blocking=non_blocking
            )
            mask = batch["mask_lai"].to(
                C.DEVICE, non_blocking=non_blocking
            )
            tgt_yield = batch["target_yield"].to(
                C.DEVICE, non_blocking=non_blocking
            ).squeeze(-1)
            static_f = batch["static_features"].to(
                C.DEVICE, non_blocking=non_blocking
            )
            crop_b = static_f[:, 10].round().long()
            year_b   = batch.get("year", None)
            if year_b is not None:
                year_b = year_b.to(
                    C.DEVICE, non_blocking=non_blocking
                )
            state_b = batch.get("state_id", None)
            county_b = batch.get("county_id", None)
            if state_b is not None:
                state_b = state_b.to(C.DEVICE, non_blocking=non_blocking)
            if county_b is not None:
                county_b = county_b.to(C.DEVICE, non_blocking=non_blocking)

            # 姣?batch 娉ㄥ叆缃戞牸鐗瑰紓鎬у弬鏁?
            model.set_static_features(static_f, state_id=state_b, county_id=county_b)

            traj, pred_yield = model(forcing, n_init, year=year_b)
            lai_loss = compute_lai_loss(
                traj, obs_lai, mask, tgt_yield=tgt_yield
            )
            yield_loss, _, _ = compute_yield_loss(pred_yield, tgt_yield)
            spatial_contrast = compute_spatial_contrast_loss(
                pred_yield,
                tgt_yield,
                county_id=county_b,
                crop_id=crop_b if getattr(C, "USE_CROP_AWARE_SPATIAL_LOSS", True) else None,
                min_gap=getattr(C, "SPATIAL_CONTRAST_MIN_GAP", 0.15),
            ) if getattr(C, "USE_SPATIAL_CONTRAST", True) else pred_yield.new_tensor(0.0)
            spatial_bias = compute_group_bias_loss(
                pred_yield,
                tgt_yield,
                group_id=state_b,
                crop_id=crop_b if getattr(C, "USE_CROP_AWARE_SPATIAL_LOSS", True) else None,
                min_count=getattr(C, "SPATIAL_GROUP_BIAS_MIN_COUNT", 4),
            ) if getattr(C, "USE_SPATIAL_GROUP_BIAS", True) else pred_yield.new_tensor(0.0)
            state_reg = compute_state_constraint_loss(model, traj)
            prior_reg = compute_parameter_prior_loss(model)
            static_adapt_reg = compute_static_adaptation_regularization(model)
            window_stress_reg = compute_window_stress_regularization(model)
            canopy_reg = compute_canopy_yield_consistency_loss(
                traj, tgt_yield, obs_lai=obs_lai, mask_lai=mask
            )

            # Anomaly 姝ｅ垯鍖?
            anom_reg = torch.tensor(0.0, device=C.DEVICE)
            if w_a > 1e-8:
                B, T, _ = forcing.shape
                # 淇濇寔 [B, T, 1] 缁村害璋冪敤 Expert (flat 浼氱牬鍧忛潤鎬佺壒寰佺殑 batch 瀵归綈)
                f_temp = model.ode_func.temp_expert(forcing[..., 3:4])
                dev_index, _, _, _ = model.ode_func.pheno_expert.from_cumulative(
                    forcing[..., 6:7]
                )
                f_temp = model.ode_func._effective_temperature_stress(
                    f_temp,
                    tmean=forcing[..., 3:4],
                    dev_index=dev_index,
                )
                f_water_raw = model.ode_func.water_expert(
                    traj[..., 3:4].detach()
                )
                f_n_raw, _ = model.ode_func.n_expert(
                    traj[..., 2:3].detach(),
                    traj[..., 1:2].detach(),
                )
                anom_reg = compute_anomaly_reg(
                    model.ode_func.coupling,
                    f_temp.reshape(-1, 1),
                    f_water_raw.reshape(-1, 1),
                    f_n_raw.reshape(-1, 1),
                    traj[..., 0:1].detach().reshape(-1, 1),
                    traj[..., 2:3].detach().reshape(-1, 1),
                    traj[..., 3:4].detach().reshape(-1, 1),
                    B, T
                )

            loss_ode = (
                C.W_LAI * lai_loss +
                w_y * C.W_YIELD * yield_loss +
                w_a * anom_reg +
                C.W_STATE * state_reg +
                C.W_PRIOR * prior_reg +
                C.W_CANOPY * canopy_reg +
                getattr(C, "W_STATIC_ADAPT", 0.0) * static_adapt_reg +
                getattr(C, "W_WINDOW_STRESS", 0.0) * window_stress_reg +
                w_y * getattr(C, "W_SPATIAL_CONTRAST", 0.0) * spatial_contrast +
                w_y * getattr(C, "W_SPATIAL_GROUP_BIAS", 0.0) * spatial_bias
            )

            # VPD 杈呭姪 Loss 鈥?浠?Phase 2+ 鍚敤, 缁?D0 鐩存帴姊害
            if (
                getattr(C, "USE_VPD_STRESS", True) and
                w_a > 0.2 and ep >= C.PHASE_LAI_ONLY
            ):
                loss_ode = loss_ode + compute_vpd_aux_loss(model, forcing)

            if not torch.isfinite(loss_ode):
                continue

            opt_expert.zero_grad(set_to_none=True)
            opt_photo.zero_grad(set_to_none=True)
            opt_coupling.zero_grad(set_to_none=True)
            opt_stomatal.zero_grad(set_to_none=True)
            opt_phenology.zero_grad(set_to_none=True)
            opt_lstm.zero_grad(set_to_none=True)
            opt_yield.zero_grad(set_to_none=True)
            loss_ode.backward()
            torch.nn.utils.clip_grad_norm_(model.expert_params(), C.GRAD_CLIP)
            torch.nn.utils.clip_grad_norm_(model.photo_params(), C.GRAD_CLIP)
            torch.nn.utils.clip_grad_norm_(model.coupling_params(), C.GRAD_CLIP)
            torch.nn.utils.clip_grad_norm_(model.stomatal_params(), C.GRAD_CLIP)
            torch.nn.utils.clip_grad_norm_(model.phenology_params(), C.GRAD_CLIP)
            torch.nn.utils.clip_grad_norm_(model.lstm_params(), C.GRAD_CLIP)
            torch.nn.utils.clip_grad_norm_(model.yield_params(), C.GRAD_CLIP)
            opt_expert.step()
            opt_photo.step()
            opt_coupling.step()
            opt_stomatal.step()
            opt_phenology.step()
            opt_lstm.step()
            if train_yield:
                opt_yield.step()

            model.clamp_physical_parameters()

            s_loss += loss_ode.item()
            s_lai  += lai_loss.item()
            s_yield += yield_loss.item()
            s_anom += anom_reg.item() if torch.is_tensor(anom_reg) else anom_reg
            s_state += state_reg.item()
            s_canopy += canopy_reg.item()
            s_static_adapt += static_adapt_reg.item()
            s_spatial_contrast += spatial_contrast.item()
            s_spatial_bias += spatial_bias.item()
            s_window_stress += window_stress_reg.item()
            nb += 1

        if nb == 0:
            continue

        # 鈹€鈹€ 楠岃瘉 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        phase3_entry = ep == C.PHASE_LAI_ONLY + C.PHASE_ANOM_RAMP
        val_every = max(1, int(getattr(C, "VAL_EVERY", 1)))
        should_validate = (
            ep == 0 or
            (ep + 1) % val_every == 0 or
            phase3_entry or
            ep == C.MAX_EPOCHS - 1
        )
        if should_validate:
            model.eval()
            val_sum, val_lai_sum, val_yield_sum, vn = 0.0, 0.0, 0.0, 0
            with torch.no_grad():
                for batch in vdl:
                    forcing = batch["forcing"].to(
                        C.DEVICE, non_blocking=non_blocking
                    )
                    n_init = batch["n_init"].to(
                        C.DEVICE, non_blocking=non_blocking
                    )
                    obs_lai = batch["obs_lai"].to(
                        C.DEVICE, non_blocking=non_blocking
                    )
                    mask = batch["mask_lai"].to(
                        C.DEVICE, non_blocking=non_blocking
                    )
                    tgt_yield = batch["target_yield"].to(
                        C.DEVICE, non_blocking=non_blocking
                    ).squeeze(-1)
                    static_f = batch["static_features"].to(
                        C.DEVICE, non_blocking=non_blocking
                    )
                    crop_vb = static_f[:, 10].round().long()
                    year_vb = batch.get("year", None)
                    if year_vb is not None:
                        year_vb = year_vb.to(
                            C.DEVICE, non_blocking=non_blocking
                        )
                    state_vb = batch.get("state_id", None)
                    county_vb = batch.get("county_id", None)
                    if state_vb is not None:
                        state_vb = state_vb.to(
                            C.DEVICE, non_blocking=non_blocking
                        )
                    if county_vb is not None:
                        county_vb = county_vb.to(
                            C.DEVICE, non_blocking=non_blocking
                        )

                    model.set_static_features(
                        static_f,
                        state_id=state_vb,
                        county_id=county_vb,
                    )
                    traj, pred_yield = model(
                        forcing, n_init, year=year_vb
                    )

                    lai_l = compute_lai_loss(
                        traj, obs_lai, mask, tgt_yield=tgt_yield
                    )
                    yl, _, _ = compute_yield_loss(
                        pred_yield, tgt_yield
                    )
                    spatial_l = compute_spatial_contrast_loss(
                        pred_yield,
                        tgt_yield,
                        county_id=county_vb,
                        crop_id=crop_vb if getattr(C, "USE_CROP_AWARE_SPATIAL_LOSS", True) else None,
                        min_gap=getattr(C, "SPATIAL_CONTRAST_MIN_GAP", 0.15),
                    ) if getattr(C, "USE_SPATIAL_CONTRAST", True) else pred_yield.new_tensor(0.0)
                    spatial_bias_l = compute_group_bias_loss(
                        pred_yield,
                        tgt_yield,
                        group_id=state_vb,
                        crop_id=crop_vb if getattr(C, "USE_CROP_AWARE_SPATIAL_LOSS", True) else None,
                        min_count=getattr(C, "SPATIAL_GROUP_BIAS_MIN_COUNT", 4),
                    ) if getattr(C, "USE_SPATIAL_GROUP_BIAS", True) else pred_yield.new_tensor(0.0)
                    state_l = compute_state_constraint_loss(model, traj)
                    prior_l = compute_parameter_prior_loss(model)
                    static_l = compute_static_adaptation_regularization(model)
                    window_l = compute_window_stress_regularization(model)
                    canopy_l = compute_canopy_yield_consistency_loss(
                        traj, tgt_yield, obs_lai=obs_lai, mask_lai=mask
                    )
                    val_sum += (
                        C.W_LAI * lai_l +
                        w_y * C.W_YIELD * yl +
                        C.W_STATE * state_l +
                        C.W_PRIOR * prior_l +
                        C.W_CANOPY * canopy_l +
                        getattr(C, "W_STATIC_ADAPT", 0.0) * static_l +
                        getattr(C, "W_WINDOW_STRESS", 0.0) * window_l +
                        w_y * getattr(C, "W_SPATIAL_CONTRAST", 0.0) * spatial_l +
                        w_y * getattr(C, "W_SPATIAL_GROUP_BIAS", 0.0) * spatial_bias_l
                    ).item()
                    val_lai_sum += lai_l.item()
                    val_yield_sum += yl.item()
                    vn += 1

            avg_val = val_sum / max(vn, 1)
            avg_val_lai = val_lai_sum / max(vn, 1)
            avg_val_yield = val_yield_sum / max(vn, 1)

            if avg_val < best_val:
                best_val = avg_val
                wait = 0
                os.makedirs(C.SAVE_DIR, exist_ok=True)
                torch.save(model.state_dict(), best_path)
            else:
                wait += 1

        epoch_seconds = time.perf_counter() - epoch_started
        peak_memory_gb = (
            torch.cuda.max_memory_allocated() / 1024 ** 3
            if C.DEVICE.type == "cuda" else 0.0
        )
        hi = model.harvest_index.item()
        ys = model.yield_scale.item()
        yt = model.yield_year_slope.item()
        crop_log_ids = torch.tensor([1, 5], device=C.DEVICE)
        crop_hi = model.crop_harvest_index(crop_log_ids)
        crop_ys = model.crop_yield_scale(crop_log_ids)
        crop_yt = model.crop_yield_year_slope(crop_log_ids)
        rue = torch.abs(model.ode_func.rad_expert.rue).item()
        k_ext = torch.abs(model.ode_func.rad_expert.k_ext).item()
        D0 = torch.abs(model.ode_func.stom_expert.D0).item()
        gdd_fl = torch.abs(model.ode_func.pheno_expert.gdd_flowering).item()
        gdd_ma = torch.abs(model.ode_func.pheno_expert.gdd_maturity).item()
        fc, wp, _ = model.ode_func.water_expert.get_soil_params()
        fc_mean = fc.mean().item() if fc.dim() > 0 else fc.item()
        wp_mean = wp.mean().item() if wp.dim() > 0 else wp.item()
        history.append({
            "epoch": ep + 1,
            "train_loss": s_loss / nb,
            "val_loss": avg_val,
            "lai_train": s_lai / nb,
            "lai_val": avg_val_lai,
            "yield_train": s_yield / nb,
            "yield_val": avg_val_yield,
            "state_loss": s_state / nb,
            "canopy_loss": s_canopy / nb,
            "static_adapt_loss": s_static_adapt / nb,
            "spatial_contrast_loss": s_spatial_contrast / nb,
            "spatial_group_bias_loss": s_spatial_bias / nb,
            "window_stress_loss": s_window_stress / nb,
            "anom_loss": s_anom / nb,
            "HI": hi,
            "HI_corn": crop_hi[0].item(),
            "HI_soybean": crop_hi[1].item(),
            "yield_scale": ys,
            "yield_scale_corn": crop_ys[0].item(),
            "yield_scale_soybean": crop_ys[1].item(),
            "yield_year_trend": yt,
            "yield_year_trend_corn": crop_yt[0].item(),
            "yield_year_trend_soybean": crop_yt[1].item(),
            "RUE": rue,
            "k_ext": k_ext,
            "D0": D0,
            "GDD_flowering": gdd_fl,
            "GDD_maturity": gdd_ma,
            "FC": fc_mean,
            "WP": wp_mean,
            "epoch_seconds": epoch_seconds,
            "gpu_peak_gb": peak_memory_gb,
        })

        # 鈹€鈹€ 鏃ュ織 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        log_every = max(1, int(getattr(C, "TRAIN_LOG_EVERY", 25)))
        if (ep + 1) % log_every == 0 or ep == 0 or ep == C.MAX_EPOCHS - 1:
            print(
                f"Ep [{ep+1:03d}/{C.MAX_EPOCHS}] | T: {s_loss/nb:.4f} | V: {avg_val:.4f} "
                f"| LAI: {s_lai/nb:.4f}/{avg_val_lai:.4f} "
                f"| Yld: {s_yield/nb:.4f}/{avg_val_yield:.4f} "
                f"| State: {s_state/nb:.4f} | Can: {s_canopy/nb:.4f} "
                f"| Stat: {s_static_adapt/nb:.4f} "
                f"| Sp: {s_spatial_contrast/nb:.4f} "
                f"| SB: {s_spatial_bias/nb:.4f} "
                f"| Win: {s_window_stress/nb:.4f} "
                f"| Anom: {s_anom/nb:.6f} | HI: {hi:.3f} "
                f"({crop_hi[0].item():.3f}/{crop_hi[1].item():.3f}) "
                f"| YS: {ys:.3f} "
                f"({crop_ys[0].item():.3f}/{crop_ys[1].item():.3f}) "
                f"| YT: {yt:.3f} "
                f"({crop_yt[0].item():.3f}/{crop_yt[1].item():.3f}) "
                f"| RUE: {rue:.2f} | k: {k_ext:.3f} | D0: {D0:.2f} "
                f"| GDDfl: {gdd_fl:.0f} | GDDma: {gdd_ma:.0f} "
                f"| FC: {fc_mean:.3f} | WP: {wp_mean:.3f} "
                f"| wy: {w_y:.2f} | wa: {w_a:.2f} "
                f"| sec: {epoch_seconds:.1f} | GPU: {peak_memory_gb:.2f}G"
            )

        if should_validate and wait >= C.PATIENCE:
            print(f"Early stopping at epoch {ep + 1}")
            break

    final_path = os.path.join(C.SAVE_DIR, f"agriworld_{ver}_final.pth")
    torch.save(model.state_dict(), final_path)
    _write_training_history(history, ver)
    print(f"Training done. Best val: {best_val:.4f}")

    model.load_state_dict(torch.load(
        best_path, weights_only=True
    ))
    if getattr(C, "TRAIN_FINAL_VALIDATE", False):
        validate_yield_all(model, vdl, C.DEVICE)
        validate_physics(model, vdl, C.DEVICE, n_show=10)


if __name__ == "__main__":
    train()

