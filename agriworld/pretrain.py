"""
AgriWorld 鈥?Expert 棰勮缁?
==========================
鐢ㄥ悎鎴愮墿鐞嗘暟鎹垵濮嬪寲鍚?Expert 鐨勬潈閲嶃€?

棰勮缁冧换鍔?
    WaterExpert:   pedotransfer 缃戠粶 鈫?鍥炲綊 Saxton-Rawls 鍦熷￥姘村姏鍙傛暟
    Nitrogen/Radiation/Stomatal/Phenology:
                   浠庢壈鍔ㄥ垵鍊兼仮澶嶆枃鐚弬鏁帮紝骞跺湪鐙珛鍚堟垚楠岃瘉闆嗘鏌?    CouplingHead:   楠岃瘉闆朵腑蹇冩亽绛夊垵濮嬪寲锛屼笉娉ㄥ叆浜轰负鍚堟垚鍋忓樊

杈撳嚭: ./pretrained/*.pth 鈥?渚?train.py 鍔犺浇
"""

import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from agriworld.config import DEVICE, PRETRAINED_DIR
from agriworld.experts import WaterExpert, NitrogenExpert, RadiationExpert, StomatalExpert, PhenologyExpert
from agriworld.coupling import CouplingHead


def _make_train_val_loaders(*tensors, batch_size, seed, val_ratio=0.2):
    """Create a deterministic synthetic holdout instead of reporting train loss only."""
    n = tensors[0].shape[0]
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(n, generator=generator)
    n_val = max(1, int(round(n * val_ratio)))
    val_idx = order[:n_val].to(tensors[0].device)
    train_idx = order[n_val:].to(tensors[0].device)

    train_ds = TensorDataset(*(tensor[train_idx] for tensor in tensors))
    val_ds = TensorDataset(*(tensor[val_idx] for tensor in tensors))
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False),
    )


@torch.no_grad()
def _loader_loss(model, dataloader, loss_fn):
    total = 0.0
    for batch in dataloader:
        total += loss_fn(model, batch).item()
    return total / max(len(dataloader), 1)


def _fit_synthetic(model, optimizer, train_loader, val_loader, loss_fn,
                   epochs, label):
    """Fit a synthetic task and restore the checkpoint with best holdout loss."""
    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())

    for ep in range(epochs):
        model.train()
        total = 0.0
        for batch in train_loader:
            loss = loss_fn(model, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()

        model.eval()
        val_loss = _loader_loss(model, val_loader, loss_fn)
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())

        if (ep + 1) % 50 == 0 or ep == 0:
            print(
                f"  {label} [{ep+1}/{epochs}] "
                f"train={total/max(len(train_loader), 1):.8f} "
                f"val={val_loss:.8f}"
            )

    model.load_state_dict(best_state)
    return best_val


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?# WaterExpert 鈥?璁粌 pedotransfer 缃戠粶鍥炲綊鍦熷￥姘村姏鍙傛暟
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?

def pretrain_water(n=8000, epochs=100, device=DEVICE):
    """
    璁粌 WaterExpert 鐨?pedotransfer 缃戠粶:
    杈撳叆 (clay, sand, BD, OM) 鈫?杈撳嚭 (fc, wp, log_drain)
    鐩爣: 绠€鍖?Saxton-Rawls 鍨嬫柟绋嬬敓鎴愬悎鎴愭爣绛俱€?
    """
    print("Pre-training WaterExpert (pedotransfer: clay/sand/BD/OM 鈫?fc/wp/drain)...")

    rng = np.random.default_rng(42)

    B = n
    clay = rng.uniform(5, 55, (B, 1)).astype(np.float32)          # % 绮樼矑
    sand = rng.uniform(10, 75, (B, 1)).astype(np.float32)          # % 鐮傜矑
    bd   = rng.uniform(1.0, 1.65, (B, 1)).astype(np.float32)       # g/cm鲁 瀹归噸
    soc  = rng.uniform(0.3, 4.0, (B, 1)).astype(np.float32)        # g/kg
    om   = soc / 10.0 * 1.72                                        # 鏈夋満璐?%

    # 鈹€鈹€ 鐩爣鍊? 绠€鍖?Saxton-Rawls 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
    fc_target = 0.15 + 0.30 * (clay / 55.0) + 0.05 * (om / 5.0) - 0.08 * (sand / 75.0)
    fc_target = np.clip(fc_target, 0.15, 0.50).astype(np.float32)

    wp_target = 0.05 + 0.20 * (clay / 55.0) + 0.02 * (om / 5.0)
    wp_target = np.clip(wp_target, 0.05, 0.30).astype(np.float32)

    drain_target = 0.05 + 0.15 * (sand / 75.0)
    drain_target = np.clip(drain_target, 0.05, 0.25).astype(np.float32)

    # 鈹€鈹€ 缁勮 static_features (11 缁? 浠呭～鍏?4 涓綅缃? 鈹€鈹€鈹€鈹€鈹€鈹€
    static = np.zeros((B, 11), dtype=np.float32)
    static[:, 3] = bd[:, 0]       # Bulk_Density    (index 3)
    static[:, 4] = soc[:, 0]      # SOC             (index 4)
    static[:, 5] = clay[:, 0]     # Clay_Fraction   (index 5)
    static[:, 6] = sand[:, 0]     # Sand_Fraction   (index 6)

    X   = torch.tensor(static, device=device)
    y_fc = torch.tensor(fc_target, device=device).view(-1)
    y_wp = torch.tensor(wp_target, device=device).view(-1)
    y_dr = torch.tensor(drain_target, device=device).view(-1)

    train_dl, val_dl = _make_train_val_loaders(
        X, y_fc, y_wp, y_dr, batch_size=256, seed=42
    )

    expert = WaterExpert().to(device)
    opt = optim.Adam(expert.pedotransfer.parameters(), lr=2e-3)

    def loss_fn(model, batch):
        x, t_fc, t_wp, t_dr = batch
        model.set_static_features(x)
        fc_pred, wp_pred, dr_pred = model.get_soil_params()
        return (
            F.mse_loss(fc_pred.view(-1), t_fc) +
            F.mse_loss(wp_pred.view(-1), t_wp) +
            F.mse_loss(dr_pred.view(-1), t_dr)
        )

    best_val = _fit_synthetic(
        expert, opt, train_dl, val_dl, loss_fn, epochs, "Water"
    )
    print(f"  Water best holdout loss: {best_val:.8f}")

    return expert.state_dict()


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?
# NitrogenExpert 鈥?璁粌 M-M 鍚告敹閫熺巼鍙傛暟
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?
#
# 娉ㄦ剰: NitrogenExpert.forward() 杩斿洖鐨?f_n = smooth_clamp01(n_pool/demand)
#       涓嶇粡杩?Vmax/Km (璁＄畻鍥炬柇寮€), 鍥犳涓嶈兘璁粌 f_n銆?
#       鏀逛负璁粌 uptake = Vmax * n_pool / (Km + n_pool) 鈥斺€?杩欐墠鏄?
#       鐪熸浣跨敤 log_vmax 鍜?log_km 鍙傛暟鐨勮緭鍑恒€?
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?

def pretrain_nitrogen(n=8000, epochs=100, device=DEVICE):
    """
    璁粌 NitrogenExpert 鐨?M-M 鍚告敹鍔ㄥ姏瀛﹀弬鏁?(Vmax, Km)銆?
    鐩爣: uptake = Vmax 脳 n_pool / (Km + n_pool)
    """
    print("Pre-training NitrogenExpert (uptake = V路n/(K+n) Michaelis-Menten)...")

    rng = np.random.default_rng(43)

    B = n
    # Generate physical ranges for mineral N and crop biomass.
    n_pool = rng.uniform(1.0, 250.0, (B, 1)).astype(np.float32)
    bio = rng.uniform(0.01, 35.0, (B, 1)).astype(np.float32)

    # 鐢?鐪熷疄" Vmax/Km 鐢熸垚鍚堟垚鐩爣
    VMAX_TRUE = 1.5
    KM_TRUE   = 40.0
    uptake_target = (
        VMAX_TRUE * n_pool / (KM_TRUE + n_pool) *
        bio / (bio + 1.0)
    )
    uptake_target = uptake_target.astype(np.float32)

    X = torch.tensor(np.hstack([n_pool, bio]), device=device)
    Y = torch.tensor(uptake_target, device=device)

    train_dl, val_dl = _make_train_val_loaders(
        X, Y, batch_size=512, seed=43
    )

    expert = NitrogenExpert().to(device)
    with torch.no_grad():
        expert.log_vmax.fill_(np.log(0.6))
        expert.log_km.fill_(np.log(100.0))
    opt = optim.Adam([expert.log_vmax, expert.log_km], lr=2e-3)

    def loss_fn(model, batch):
        x, y = batch
        _, uptake = model(x[:, 0:1], x[:, 1:2])
        return F.mse_loss(uptake.view(-1), y.view(-1))

    best_val = _fit_synthetic(
        expert, opt, train_dl, val_dl, loss_fn, epochs, "Nitro"
    )
    print(
        f"  Nitro recovered: Vmax={torch.exp(expert.log_vmax).item():.4f} "
        f"(target 1.5000), Km={torch.exp(expert.log_km).item():.3f} "
        f"(target 40.000), val={best_val:.8f}"
    )

    return expert.state_dict()


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?
# RadiationExpert 鈥?璁粌 Monteith RUE + 娑堝厜绯绘暟
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?

def pretrain_radiation(n=8000, epochs=100, device=DEVICE):
    """
    璁粌 RadiationExpert 鐨?RUE (杈愬皠鍒╃敤鏁堢巼) 鍜?k_ext (娑堝厜绯绘暟)銆?

    dB = RUE 脳 PAR 脳 (1 - exp(-k_ext 脳 LAI))
    鐩爣: 鐢ㄥ凡鐭?RUE=3.0, k_ext=0.65 鐢熸垚鍚堟垚鏍囩銆?
    """
    print("Pre-training RadiationExpert (RUE 脳 PAR 脳 fAPAR)...")

    rng = np.random.default_rng(45)
    B = n

    LAI = rng.uniform(0.01, 8.0, (B, 1)).astype(np.float32)
    PAR = rng.uniform(2.0, 25.0, (B, 1)).astype(np.float32)

    RUE_TRUE = 3.0
    K_TRUE   = 0.65
    fAPAR_target = 1.0 - np.exp(-K_TRUE * LAI)                    # 鏃犻噺绾?
    dB_target = (RUE_TRUE * PAR * fAPAR_target * 0.01).astype(np.float32)

    lai_t = torch.tensor(LAI, device=device)
    par_t = torch.tensor(PAR, device=device)
    y_db  = torch.tensor(dB_target, device=device).view(-1)

    train_dl, val_dl = _make_train_val_loaders(
        lai_t, par_t, y_db, batch_size=512, seed=45
    )

    expert = RadiationExpert().to(device)
    with torch.no_grad():
        expert.rue.fill_(1.5)
        expert.k_ext.fill_(0.35)
    opt = optim.Adam([expert.rue, expert.k_ext], lr=2e-3)

    def loss_fn(model, batch):
        lai_b, par_b, y_b = batch
        dB_pred, _ = model(par_b, lai_b)
        return F.mse_loss(dB_pred.view(-1), y_b)

    best_val = _fit_synthetic(
        expert, opt, train_dl, val_dl, loss_fn, epochs, "Radiation"
    )
    print(
        f"  Radiation recovered: RUE={expert.rue.item():.4f} "
        f"(target 3.0000), k_ext={expert.k_ext.item():.4f} "
        f"(target 0.6500), val={best_val:.8f}"
    )

    return expert.state_dict()


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?
# StomatalExpert 鈥?璁粌 VPD 鍗婅“鍑忓弬鏁?D0
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?

def pretrain_stomatal(n=8000, epochs=100, device=DEVICE):
    """
    璁粌 StomatalExpert 鐨?D0 (鍗婅“鍑?VPD 鍊?銆?

    f_vpd = 1 / (1 + VPD / D0)
    鐩爣: 鐢ㄥ凡鐭?D0=2.0 kPa 鐢熸垚鍚堟垚鏍囩銆?    """
    print("Pre-training StomatalExpert (f_vpd = 1/(1+VPD/D0))...")

    rng = np.random.default_rng(46)
    B = n

    VPD = rng.uniform(0.1, 5.0, (B, 1)).astype(np.float32)

    D0_TRUE = 2.0
    f_vpd_target = (1.0 / (1.0 + VPD / D0_TRUE)).astype(np.float32)

    X = torch.tensor(VPD, device=device)
    Y = torch.tensor(f_vpd_target, device=device).view(-1)

    train_dl, val_dl = _make_train_val_loaders(
        X, Y, batch_size=512, seed=46
    )

    expert = StomatalExpert().to(device)
    with torch.no_grad():
        expert.D0.fill_(0.8)
    opt = optim.Adam([expert.D0], lr=2e-3)

    def loss_fn(model, batch):
        x, y = batch
        return F.mse_loss(model(x).view(-1), y)

    best_val = _fit_synthetic(
        expert, opt, train_dl, val_dl, loss_fn, epochs, "Stomatal"
    )
    print(
        f"  Stomatal recovered: D0={expert.D0.item():.4f} "
        f"(target 2.0000), val={best_val:.8f}"
    )

    return expert.state_dict()


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?
# PhenologyExpert 鈥?璁粌 GDD鈫掑彂鑲查樁娈佃妭鐐?
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?

def pretrain_phenology(n=8000, epochs=100, device=DEVICE):
    """
    璁粌 PhenologyExpert 鐨勭Н娓╄妭鐐?(gdd_flowering, gdd_maturity)銆?

    dev_index = min(GDD_cum / gdd_maturity, 1)
    鐩爣: 鐢ㄥ凡鐭?GDD_fl=850, GDD_mat=1700 鐢熸垚鍚堟垚鏍囩銆?
    """
    print("Pre-training PhenologyExpert (GDD 鈫?dev_index)...")

    rng = np.random.default_rng(47)
    B = n

    # 浠庢挱绉嶅埌鎴愮啛鐨勫叏鑼冨洿
    GDD_cum = rng.uniform(0.0, 2200.0, (B, 1)).astype(np.float32)

    GDD_FL_TRUE = 850.0
    GDD_MA_TRUE = 1700.0

    dev_target = (np.clip(GDD_cum / GDD_MA_TRUE, 0, 1)).astype(np.float32)
    fl_target = (
        1.0 / (1.0 + np.exp(-(GDD_cum - GDD_FL_TRUE) / 40.0))
    ).astype(np.float32)

    X = torch.tensor(GDD_cum, device=device)
    Y = torch.tensor(dev_target, device=device).view(-1)
    Y_fl = torch.tensor(fl_target, device=device).view(-1)

    train_dl, val_dl = _make_train_val_loaders(
        X, Y, Y_fl, batch_size=512, seed=47
    )

    expert = PhenologyExpert().to(device)
    with torch.no_grad():
        expert.gdd_flowering.fill_(650.0)
        expert.gdd_maturity.fill_(1300.0)
    opt = optim.Adam(
        [expert.gdd_flowering, expert.gdd_maturity], lr=0.5
    )

    def loss_fn(model, batch):
        x, y, y_fl = batch
        dev_pred, _, _, _ = model.from_cumulative(x)
        flowering_prob = torch.sigmoid(
            (x.view(-1) - torch.abs(model.gdd_flowering)) / 40.0
        )
        return (
            F.mse_loss(dev_pred.view(-1), y) +
            0.25 * F.mse_loss(flowering_prob, y_fl)
        )

    best_val = _fit_synthetic(
        expert, opt, train_dl, val_dl, loss_fn, epochs, "Phenology"
    )
    print(
        f"  Phenology recovered: flowering="
        f"{expert.gdd_flowering.item():.2f} (target 850), maturity="
        f"{expert.gdd_maturity.item():.2f} (target 1700), "
        f"val={best_val:.8f}"
    )

    return expert.state_dict()


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?
# CouplingHead 鈥?璁粌鑳佽揩铻嶅悎閫昏緫
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?

def pretrain_coupling(n=10000, epochs=100, device=DEVICE):
    """
    楠岃瘉 CouplingHead 鐨勬亽绛夊垵濮嬪寲銆?    涓嬫父璁粌鍐嶄粠鐪熷疄鏁版嵁瀛︿範灏忓箙淇锛岄伩鍏嶇敤浠绘剰鍚堟垚鍏崇郴姹℃煋鍝嶅簲銆?    """
    print("Initializing CouplingHead (zero-centered identity correction)...")

    rng = np.random.default_rng(44)

    B = n
    inputs = {
        'f_temp':   rng.uniform(0.0, 1.0, (B, 1)).astype(np.float32),
        'f_water':  rng.uniform(0.0, 1.0, (B, 1)).astype(np.float32),
        'f_nitro':  rng.uniform(0.0, 1.0, (B, 1)).astype(np.float32),
        'lai':      rng.uniform(0.01, 10.0, (B, 1)).astype(np.float32),
        'n_input':  rng.uniform(1.0, 300.0, (B, 1)).astype(np.float32),
        'sw_input': rng.uniform(0.10, 0.50, (B, 1)).astype(np.float32),
    }

    f_w_target = inputs['f_water']
    f_n_target = inputs['f_nitro']

    X_all = torch.tensor(np.hstack([inputs[k] for k in
        ['f_temp', 'f_water', 'f_nitro', 'lai', 'n_input', 'sw_input']]),
        device=device,
    )
    Y_w = torch.tensor(f_w_target, device=device)
    Y_n = torch.tensor(f_n_target, device=device)

    model = CouplingHead().to(device)
    with torch.no_grad():
        pw, pn, anomaly = model(
            X_all[:, 0:1], X_all[:, 1:2], X_all[:, 2:3],
            X_all[:, 3:4], X_all[:, 4:5], X_all[:, 5:6],
        )
        identity_error = (
            F.mse_loss(pw, Y_w) +
            F.mse_loss(pn, Y_n) +
            anomaly.square().mean()
        ).item()
    print(
        "  Coupling identity initialization: "
        f"error={identity_error:.8f} (expected 0; no synthetic fitting needed)"
    )

    return model.state_dict()


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?
# 缁熶竴鍏ュ彛
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?

def run_all_pretrain(device=DEVICE, force=False):
    os.makedirs(PRETRAINED_DIR, exist_ok=True)
    pth = {
        'water':     os.path.join(PRETRAINED_DIR, "water_expert_v3.pth"),
        'nitrogen':  os.path.join(PRETRAINED_DIR, "nitrogen_expert_v3.pth"),
        'radiation': os.path.join(PRETRAINED_DIR, "radiation_expert_v3.pth"),
        'stomatal':  os.path.join(PRETRAINED_DIR, "stomatal_expert_v3.pth"),
        'phenology': os.path.join(PRETRAINED_DIR, "phenology_expert_v3.pth"),
        'coupling':  os.path.join(PRETRAINED_DIR, "coupling_head_v3.pth"),
    }

    for name, fn, path in [
        ('WaterExpert',    pretrain_water,     pth['water']),
        ('NitrogenExpert', pretrain_nitrogen,  pth['nitrogen']),
        ('RadiationExpert', pretrain_radiation, pth['radiation']),
        ('StomatalExpert',  pretrain_stomatal,  pth['stomatal']),
        ('PhenologyExpert', pretrain_phenology, pth['phenology']),
        ('CouplingHead',   pretrain_coupling,  pth['coupling']),
    ]:
        if force or not os.path.exists(path):
            torch.save(fn(device=device), path)
            print(f"  鉁?{name} 棰勮缁冨畬鎴?鈫?{path}")
        else:
            print(f"  - {name} weights exist, skipped. Pass force=True to retrain.")

    return pth


if __name__ == "__main__":
    run_all_pretrain(force=True)

