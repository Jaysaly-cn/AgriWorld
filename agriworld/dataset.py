"""
AgriWorld Dataset v2 鈥?鍏ㄥ彉閲忔彁鍙?
===================================
杈撳嚭 7 閫氶亾 forcing + static_features + 楠岃瘉瀛楁銆?

Forcing 閫氶亾 (鏉ヨ嚜 stress_forcing, 365脳9):
  [0] Precip    mm/day    Daymet prcp
  [1] ETo       mm/day    Hargreaves
  [2] PAR       MJ/m虏/day SRAD 脳 0.48
  [3] Tmean     掳C        (Tmax+Tmin)/2
  [4] VPD       kPa       Magnus - vp
  [5] GDD_daily 掳C路day    max(Tmean-10,0)
  [6] GDD_cum   掳C路day    cumulative GDD from planting

Static features (11 缁?: 瑙?data_index.txt
"""

import os
import pickle
import hashlib
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from agriworld.units import (
    BACKGROUND_MINERAL_N_FRACTION,
    bu_ac_to_t_ha_factor,
    FERTILIZER_AVAILABILITY,
    TOPSOIL_DEPTH_M,
)
from agriworld.data_quality import forcing_quality_issue
import agriworld.config as config


# stress_forcing 鍒楃储寮?鈫?鎻愬彇椤哄簭
SF_IDX = {
    'Precip': 0, 'ETo': 1, 'SRAD': 2, 'PAR': 3,
    'Tmax': 4, 'Tmin': 5, 'Tmean': 6, 'VPD': 7, 'GDD': 8,
}

# 鎻愬彇鐨?forcing 鍒?(7 閫氶亾)
FORCING_EXTRACT = ['Precip', 'ETo', 'PAR', 'Tmean', 'VPD', 'GDD']
#            鈫?绱㈠紩: 0:Pcp, 1:ETo, 2:PAR, 3:Tmean, 4:VPD, 5:GDD_daily

DEFAULT_PLANTING_DOY = {
    "IL": 110,
    "IN": 115,
    "IA": 121,
    "NE": 125,
    "MN": 130,
}

STATE_EMBED_BUCKETS = 32
COUNTY_EMBED_BUCKETS = 4096


def _allowed_crops():
    spec = str(getattr(config, "ALLOWED_CROPS", "1")).strip()
    if spec.lower() in {"all", "*"}:
        return None
    return {int(item.strip()) for item in spec.split(",") if item.strip()}


def _stable_bucket(value, buckets):
    digest = hashlib.blake2b(str(value).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % int(buckets)


class AgriTensorDataset(Dataset):
    def __init__(self, pkl_path):
        if not os.path.exists(pkl_path):
            raise FileNotFoundError(f"Missing file: {pkl_path}")
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        self.samples = []
        self.qc_total = len(data)
        self.qc_dropped = Counter()
        self.qc_examples = defaultdict(list)

        for sample_id, d in data.items():
            doy = np.asarray(d["DOY"], dtype=np.int32)
            raw_forcing = np.asarray(d["stress_forcing"], dtype=np.float32)  # [365, 9]
            static_f   = np.asarray(d["static_features"], dtype=np.float32)  # [11]

            # 鎻愬彇 6 涓?forcing 閫氶亾
            col_indices = [SF_IDX[c] for c in FORCING_EXTRACT]
            forcing_6ch = raw_forcing[:, col_indices]  # [365, 6]

            meta = d.get("meta") or {}
            state = str(meta.get("state", str(sample_id).split("-")[0]))
            planting_doy = int(
                meta.get("planting_doy", DEFAULT_PLANTING_DOY.get(state, 120))
            )
            weather_issue = forcing_quality_issue(
                raw_forcing, doy=doy, planting_doy=planting_doy
            )
            if weather_issue is not None:
                self.qc_dropped[weather_issue] += 1
                if len(self.qc_examples[weather_issue]) < 5:
                    self.qc_examples[weather_issue].append(str(sample_id))
                continue

            # GDD is rebuilt after interpolation from Tmean. Some historical
            # pickle files contain an all-zero or incomplete GDD channel.
            gdd_cum = np.zeros_like(doy, dtype=np.float32)

            # 缁勫悎: [Precip, ETo, PAR, Tmean, VPD, GDD_daily, GDD_cum]
            forcing_full = np.column_stack([forcing_6ch, gdd_cum])  # [365, 7]

            obs_raw = np.asarray(d["obs_LAI"], dtype=np.float32)

            # Initial mineral N in kg N/ha. Total soil N is converted using
            # bulk density and a conservative plant-available fraction.
            n_rate = max(float(static_f[9]), 0.0)
            tot_n = max(float(static_f[7]), 0.0)  # g N / kg soil
            bulk_density = np.clip(float(static_f[3]), 0.8, 1.8)  # Mg/m3
            soil_mass_kg_ha = bulk_density * 1000.0 * TOPSOIL_DEPTH_M * 10000.0
            background_n = (
                tot_n / 1000.0 * soil_mass_kg_ha *
                BACKGROUND_MINERAL_N_FRACTION
            )
            n_init_val = FERTILIZER_AVAILABILITY * n_rate + background_n

            df = pd.DataFrame({
                'DOY': doy,
                'Pcp':  forcing_full[:, 0],
                'ETo':  forcing_full[:, 1],
                'PAR':  forcing_full[:, 2],
                'Tmean': forcing_full[:, 3],
                'VPD':  forcing_full[:, 4],
                'GDDd': forcing_full[:, 5],
                'GDDc': forcing_full[:, 6],
                'obs':  obs_raw,
            })

            # 鎸?DOY 鑱氬悎 (澶氭簮鍙兘鏈夐噸澶?DOY)
            df_agg = df.groupby('DOY').mean()
            df_agg = df_agg.reindex(range(1, 366)).interpolate(
                method='linear', limit_direction='both'
            ).fillna(0.0)

            # Authoritative GDD source: interpolated Daymet Tmean.
            doy_full = df_agg.index.to_numpy(dtype=np.int32)
            daily_gdd = np.maximum(
                df_agg["Tmean"].to_numpy(dtype=np.float32) - 10.0,
                0.0,
            )
            daily_gdd[doy_full < planting_doy] = 0.0
            df_agg["GDDd"] = daily_gdd
            df_agg["GDDc"] = np.cumsum(daily_gdd, dtype=np.float32)

            forcing = df_agg[[
                'Pcp', 'ETo', 'PAR', 'Tmean', 'VPD', 'GDDd', 'GDDc'
            ]].values.astype(np.float32)
            # 缁熶竴鎴柇鍒?365 澶?(闂板勾 366鈫?65, 涓㈠け 12/31 涓嶅奖鍝嶆敹鑾峰悗)
            if forcing.shape[0] > 365:
                forcing = forcing[:365, :]

            obs_lai  = df_agg['obs'].values.astype(np.float32)
            if len(obs_lai) > 365:
                obs_lai = obs_lai[:365]
            mask_lai = np.where((obs_lai > 0.5) & (obs_lai <= 12.0), 1.0, 0.0)

            crop_code = int(round(float(static_f[10]))) if static_f.size > 10 else 1
            allowed = _allowed_crops()
            if allowed is not None and crop_code not in allowed:
                self.qc_dropped["disallowed_crop"] += 1
                continue
            if np.sum(mask_lai) < 10 or d["target_yield"] <= 1.0:
                reason = "insufficient_lai" if np.sum(mask_lai) < 10 else "invalid_yield"
                self.qc_dropped[reason] += 1
                continue

            # LAI 骞虫粦
            s = pd.Series(obs_lai)
            s[mask_lai == 0] = np.nan
            s = s.interpolate(method='linear', limit_direction='both')
            s = s.rolling(window=15, min_periods=1, center=True).mean().fillna(0.0)

            # SMAP 楠岃瘉瀛楁 (鎴柇闂板勾鍒?365)
            smap_surface  = np.asarray(d.get("val_smap_surface",  np.zeros(365)), dtype=np.float32)
            smap_rootzone = np.asarray(d.get("val_smap_rootzone", np.zeros(365)), dtype=np.float32)
            if smap_surface.shape[0] > 365:
                smap_surface  = smap_surface[:365]
                smap_rootzone = smap_rootzone[:365]

            self.samples.append({
                "forcing":         torch.tensor(forcing),
                "static_features": torch.tensor(static_f, dtype=torch.float32),
                "n_init":          torch.tensor([n_init_val], dtype=torch.float32),
                "obs_lai":         torch.tensor(s.values, dtype=torch.float32),
                "mask_lai":        torch.tensor(mask_lai, dtype=torch.float32),
                # Internal yield unit is metric tonnes of grain per hectare.
                "target_yield":    torch.tensor(
                    [d["target_yield"] * bu_ac_to_t_ha_factor(crop_code)],
                    dtype=torch.float32,
                ),
                "val_smap_surface":  torch.tensor(smap_surface, dtype=torch.float32),
                "val_smap_rootzone": torch.tensor(smap_rootzone, dtype=torch.float32),
                "year":            d.get("year", 2022),
                "planting_doy":    planting_doy,
                "gdd_final":       float(forcing[-1, 6]),
                "sample_id":       str(sample_id),
                "state":           state,
                "crop_code":       torch.tensor([crop_code], dtype=torch.long),
                "crop":            {1: "Corn", 5: "Soybean"}.get(
                    crop_code, f"Crop({crop_code})"
                ),
                "county":          (
                    f"{meta.get('state', '')}:{meta.get('county', sample_id)}"
                ),
                "state_id":        torch.tensor(
                    [_stable_bucket(state, STATE_EMBED_BUCKETS)],
                    dtype=torch.long,
                ),
                "county_id":       torch.tensor(
                    [_stable_bucket(
                        f"{meta.get('state', '')}:{meta.get('county', sample_id)}",
                        COUNTY_EMBED_BUCKETS,
                    )],
                    dtype=torch.long,
                ),
            })

        n_dropped = sum(self.qc_dropped.values())
        print(
            f"[Dataset QC] source={self.qc_total} accepted={len(self.samples)} "
            f"dropped={n_dropped}"
        )
        for reason, count in sorted(self.qc_dropped.items()):
            examples = self.qc_examples.get(reason, [])
            suffix = f" examples={examples}" if examples else ""
            print(f"  - {reason}: {count}{suffix}")

        if not self.samples:
            raise ValueError("No valid samples remain after dataset quality control.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

