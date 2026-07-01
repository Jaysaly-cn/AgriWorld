"""
AgriWorld 鈥?鍩虹嚎妯″瀷鍙鍖?===========================
鐢ㄦ硶: python visualize.py

浠庡綋鍓?MODEL_VERSION 瀵瑰簲鐨?best checkpoint 鍔犺浇妯″瀷, 鐢熸垚 3 寮犲浘:
  baseline_trajectory.png   鈥?LAI 鏃跺簭 + 鍥涚淮鐘舵€佽建杩?  baseline_smap.png         鈥?SMAP 鍦熷￥姘村垎鎺㈤拡鏁ｇ偣
  baseline_params.png       鈥?瀛︿範鍙傛暟 vs 鐗╃悊鍏堥獙瀵规瘮
"""

import numpy as np
import os
import torch
import torch
import matplotlib
matplotlib.use('Agg')  # 鏃?GUI 鐜
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import agriworld.config as C
from agriworld.dataset import AgriTensorDataset
from agriworld.simulator import AgriWorldSimulator
from agriworld.units import t_ha_to_bu_ac_factor
from agriworld.paths import RESULTS_DIR, SAVE_DIR


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # 鈹€鈹€ 鍔犺浇妯″瀷 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    model = AgriWorldSimulator().to(device)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    ckpt_path = os.path.join(
        SAVE_DIR, f"agriworld_{C.MODEL_VERSION}_best.pth"
    )
    ckpt = torch.load(ckpt_path,
                       map_location=device, weights_only=False)
    if 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.eval()
    print("Model loaded.")

    # 鈹€鈹€ 鍔犺浇鏁版嵁 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    ds = AgriTensorDataset(C.DATA_PATH)
    sample = ds[0]
    crop_code = int(sample['static_features'][10].item())
    bu_factor = t_ha_to_bu_ac_factor(crop_code)
    static_f = sample['static_features'].unsqueeze(0).to(device)
    forcing  = sample['forcing'].unsqueeze(0).to(device)
    n_init   = sample['n_init'].unsqueeze(0).to(device)
    tgt = sample['target_yield'].item() * bu_factor
    year = torch.tensor([sample['year']], device=device)
    state_id = sample.get("state_id", None)
    county_id = sample.get("county_id", None)
    if state_id is not None:
        state_id = state_id.to(device)
    if county_id is not None:
        county_id = county_id.to(device)

    # 鈹€鈹€ 鎺ㄧ悊 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    with torch.no_grad():
        model.set_static_features(static_f, state_id=state_id, county_id=county_id)
        traj, pred = model(forcing, n_init, year=year)
    pred_yield = pred.item() * bu_factor
    traj = traj.cpu()

    # 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
    #  鍥?1: LAI 鏃堕棿搴忓垪 + 鍥涚淮鐘舵€佽建杩?    # 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    lai_pred = traj[0, :, 0].numpy()
    obs_lai  = sample['obs_lai'].numpy()
    mask     = sample['mask_lai'].numpy() > 0
    peak_doy = int(np.argmax(lai_pred)) + 1

    ax1.fill_between(range(1, 366), 0, lai_pred, alpha=0.25, color='#2196F3')
    ax1.plot(range(1, 366), lai_pred, color='#1976D2', linewidth=2, label='AgriWorld')
    ax1.scatter(np.where(mask)[0] + 1, obs_lai[mask],
                s=12, color='#E53935', alpha=0.6, label='Sentinel-2')
    ax1.axvline(peak_doy, color='gray', linestyle='--', alpha=0.5,
                label=f'Peak DOY {peak_doy}')
    ax1.set_xlabel('Day of Year')
    ax1.set_ylabel('LAI (m虏/m虏)')
    ax1.set_title(f'LAI: Pred={pred_yield:.0f} bu/ac  |  Actual={tgt:.0f} bu/ac')
    ax1.legend(fontsize=9, loc='upper right')
    ax1.grid(True, alpha=0.2)

    state_names   = ['LAI', 'Biomass', 'N Pool', 'Soil Water']
    state_colors  = ['#4CAF50', '#FF9800', '#9C27B0', '#795548']
    state_offsets = [0, 0, 0, 0.05]  # slight offset to separate curves
    for i, (name, color, off) in enumerate(zip(state_names, state_colors, state_offsets)):
        vals = traj[0, :, i].numpy() + off
        ax2.plot(range(1, 366), vals, color=color, alpha=0.8, linewidth=1.5, label=name)
    ax2.set_xlabel('Day of Year')
    ax2.set_title('Full 4-Dimensional State Trajectory')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.2)

    plt.tight_layout()
    trajectory_path = os.path.join(RESULTS_DIR, 'baseline_trajectory.png')
    plt.savefig(trajectory_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'[1/3] trajectory saved.  LAI peak DOY={peak_doy}  Yield={pred_yield:.0f}')

    # 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
    #  鍥?2: SMAP 鍦熷￥姘村垎鎺㈤拡鏁ｇ偣
    # 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
    n_val = max(1, int(C.VAL_RATIO * len(ds)))
    _, vds = torch.utils.data.random_split(
        ds, [len(ds) - n_val, n_val],
        generator=torch.Generator().manual_seed(C.SEED)
    )
    vdl = DataLoader(vds, batch_size=C.BATCH_VAL, shuffle=False)

    sw_all, smap_all = [], []
    with torch.no_grad():
        for batch in vdl:
            forcing_b  = batch['forcing'].to(device)
            n_init_b   = batch['n_init'].to(device)
            static_b   = batch['static_features'].to(device)
            year_b     = batch['year'].to(device)
            state_b    = batch.get('state_id', None)
            county_b   = batch.get('county_id', None)
            if state_b is not None:
                state_b = state_b.to(device)
            if county_b is not None:
                county_b = county_b.to(device)
            smap_b     = batch.get('val_smap_surface', None)

            model.set_static_features(static_b, state_id=state_b, county_id=county_b)
            traj_b, _ = model(forcing_b, n_init_b, year=year_b)

            sw = traj_b[:, :, 3].flatten().cpu().numpy()
            if smap_b is not None:
                sm = smap_b.numpy().flatten()
                valid = sm > 0.001
                if valid.sum() > 10:
                    sw_all.append(sw[valid])
                    smap_all.append(sm[valid])

    if len(sw_all) > 0:
        sw_all  = np.concatenate(sw_all)
        smap_all = np.concatenate(smap_all)
        # 闅忔満閲囨牱 5000 涓偣闃叉杩囧瘑
        idx = np.random.choice(len(sw_all), min(5000, len(sw_all)), replace=False)
        sw_sample  = sw_all[idx]
        smap_sample = smap_all[idx]

        # R虏
        ss_res = np.sum((smap_all - sw_all) ** 2)
        ss_tot = np.sum((smap_all - np.mean(smap_all)) ** 2)
        r2 = 1 - ss_res / max(ss_tot, 1e-12)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(sw_sample, smap_sample, s=2, alpha=0.3, color='#1565C0')
        ax.plot([0, 0.6], [0, 0.6], '--', color='gray', linewidth=1)
        ax.set_xlabel('AgriWorld SW (volume fraction)')
        ax.set_ylabel('SMAP Surface Soil Moisture')
        ax.set_title(f'SMAP Linear Probe 鈥?R虏={r2:.3f}  (n={len(sw_all)})')
        ax.set_xlim(0, 0.5)
        ax.set_ylim(0, 0.6)
        ax.grid(True, alpha=0.2)
        plt.tight_layout()
        smap_path = os.path.join(RESULTS_DIR, 'baseline_smap.png')
        plt.savefig(smap_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'[2/3] SMAP probe saved.  R虏={r2:.3f}  n_points={len(sw_all)}')
    else:
        print('[2/3] SKIP SMAP 鈥?no valid SMAP data')

    # 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
    #  鍥?3: 瀛︿範鍙傛暟 vs 鐗╃悊鍏堥獙
    # 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲
    param_names  = ['RUE', 'k_ext', 'D0', 'FC', 'WP', 'gdd_flowering', 'gdd_maturity']
    prior_vals   = [3.000, 0.650, 2.000, None, None, 850.0, 1700.0]
    learned_vals = [
        abs(model.ode_func.rad_expert.rue).item(),
        abs(model.ode_func.rad_expert.k_ext).item(),
        abs(model.ode_func.stom_expert.D0).item(),
    ]
    fc, wp, _ = model.ode_func.water_expert.get_soil_params()
    learned_vals += [
        fc.mean().item() if fc.dim() > 0 else fc.item(),
        wp.mean().item() if wp.dim() > 0 else wp.item(),
        abs(model.ode_func.pheno_expert.gdd_flowering).item(),
        abs(model.ode_func.pheno_expert.gdd_maturity).item(),
    ]
    prior_vals[3:5] = [0.30, 0.15]  # FC/WP prior from default water params

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(param_names))
    w = 0.35
    bars1 = ax.bar(x - w/2, prior_vals, w, label='Prior (Pretraining)', color='#BBDEFB', edgecolor='#1565C0')
    bars2 = ax.bar(x + w/2, learned_vals, w, label='Learned (Data)', color='#FFE0B2', edgecolor='#EF6C00')
    ax.set_xticks(x)
    ax.set_xticklabels(param_names)
    ax.set_ylabel('Parameter Value')
    ax.set_title('Physics Prior vs. Data-Driven Learning')
    ax.legend(fontsize=10)

    # Annotate parameter shifts.
    for i, (pr, lr) in enumerate(zip(prior_vals, learned_vals)):
        if abs(lr - pr) > 0.01:
            ax.annotate(f'螖={lr-pr:+.2f}', (x[i] + w/2, lr),
                        textcoords="offset points", xytext=(0, 8),
                        ha='center', fontsize=8, color='#EF6C00')

    ax.grid(True, alpha=0.2, axis='y')
    plt.tight_layout()
    params_path = os.path.join(RESULTS_DIR, 'baseline_params.png')
    plt.savefig(params_path, dpi=150, bbox_inches='tight')
    plt.close()
    print('[3/3] params saved.')

    print(f'\nDone. Output directory: {RESULTS_DIR}')


if __name__ == '__main__':
    main()

