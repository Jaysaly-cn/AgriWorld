"""Composite ODE for AgriWorld.

State: [LAI, Biomass, N_Pool, Soil_Water].
Forcing: [Precip, ETo, PAR, Tmean, VPD, GDD_daily, GDD_cum].
"""

import torch
import torch.nn as nn
import agriworld.config as config
from agriworld.smooth import smooth_min, smooth_clamp01
from agriworld.experts import (
    TemperatureExpert, WaterExpert, NitrogenExpert,
    RadiationExpert, StomatalExpert, PhenologyExpert,
    LSTMResidual, YearEmbedding,
)
from agriworld.coupling import CouplingHead


# Forcing 閫氶亾绱㈠紩
IDX_PCP     = 0   # Precip     mm/day
IDX_ETO     = 1   # ETo        mm/day
IDX_PAR     = 2   # PAR        MJ/m虏/day
IDX_TMEAN   = 3   # Tmean      掳C
IDX_VPD     = 4   # VPD        kPa
IDX_GDD_D   = 5   # GDD_daily  掳C路day/day
IDX_GDD_C   = 6   # GDD_cum    掳C路day


class CompositeODE(nn.Module):
    def __init__(self):
        super().__init__()

        # 鈹€鈹€ 6 涓笓瀹?鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        self.temp_expert    = TemperatureExpert()
        self.water_expert   = WaterExpert()
        self.n_expert       = NitrogenExpert()
        self.rad_expert     = RadiationExpert()
        self.stom_expert    = StomatalExpert()
        self.pheno_expert   = PhenologyExpert()

        # 鈹€鈹€ 鑰﹀悎澶?鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        self.coupling = CouplingHead()

        # 鈹€鈹€ LSTM 鏃跺簭娈嬪樊 (娑堣瀺瀹為獙涓彲寮€鍏? 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        self.lstm_residual = LSTMResidual(forcing_dim=7, hidden=32)
        self.use_lstm_residual = getattr(config, 'USE_LSTM_RESIDUAL', False)
        self.register_buffer("lstm_active_flag", torch.tensor(False))

        # 鈹€鈹€ 骞翠唤宓屽叆 (澶氬勾鏁版嵁蹇呭) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        self.year_embed = YearEmbedding(dim=4)
        self.current_year = None  # 鐢?simulator 璁剧疆

        # 鈹€鈹€ 浣滅墿缁撴瀯鍙傛暟 (鍙涔? 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        self.sla   = nn.Parameter(torch.tensor(0.015))    # 姣斿彾闈㈢Н (v1.0)
        self.k_sen = nn.Parameter(torch.tensor(0.008))    # 鍩虹琛拌€侀€熺巼 (v1.0)

        # 鈹€鈹€ 杩愯鏃跺紩鐢?(鐢?simulator 璁剧疆) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        self.current_forcing = None  # [B, T, 7]
        self.current_static_features = None

    def _effective_temperature_stress(self, f_temp_raw, tmean=None, dev_index=None):
        """Map raw cardinal-temperature response to the active stress regime."""
        if not getattr(config, "USE_TEMPERATURE_STRESS", True):
            return torch.ones_like(f_temp_raw)

        mode = str(getattr(config, "TEMPERATURE_STRESS_MODE", "soft")).lower()
        if mode in {"off", "none", "disabled"}:
            return torch.ones_like(f_temp_raw)
        if mode in {"hard", "raw", "wang_engel"}:
            return torch.clamp(f_temp_raw, 0.0, 1.0)
        if mode in {"heat", "extreme", "heat_extreme", "heat_only"}:
            if tmean is None:
                return torch.ones_like(f_temp_raw)
            threshold = float(getattr(config, "HEAT_STRESS_THRESHOLD_C", 33.0))
            width = max(float(getattr(config, "HEAT_STRESS_WIDTH_C", 2.5)), 0.1)
            max_reduction = float(
                getattr(config, "HEAT_STRESS_MAX_REDUCTION", 0.12)
            )
            max_reduction = min(max(max_reduction, 0.0), 0.80)
            heat_load = torch.sigmoid((tmean - threshold) / width)
            if dev_index is not None:
                stage_center = float(
                    getattr(config, "HEAT_STRESS_STAGE_CENTER", 0.45)
                )
                stage_width = max(
                    float(getattr(config, "HEAT_STRESS_STAGE_WIDTH", 0.15)),
                    0.02,
                )
                reproductive_gate = torch.sigmoid(
                    (dev_index - stage_center) / stage_width
                )
                stage_weight = 0.35 + 0.65 * reproductive_gate
            else:
                stage_weight = 1.0
            return torch.clamp(
                1.0 - max_reduction * heat_load * stage_weight,
                min=1.0 - max_reduction,
                max=1.0,
            )

        floor = float(getattr(config, "TEMPERATURE_STRESS_FLOOR", 0.90))
        strength = float(getattr(config, "TEMPERATURE_STRESS_STRENGTH", 0.35))
        floor = min(max(floor, 0.0), 1.0)
        strength = min(max(strength, 0.0), 1.0)
        softened = 1.0 - strength * (1.0 - torch.clamp(f_temp_raw, 0.0, 1.0))
        return torch.clamp(softened, min=floor, max=1.0)

    def _interp(self, t):
        """Linearly interpolate forcing at time t."""
        n = self.current_forcing.size(1) - 1
        tc = torch.clamp(t, 0.0, float(n))
        t0 = torch.floor(tc).long()
        t1 = torch.clamp(t0 + 1, 0, n)
        w = (tc - t0.float()).view(1, 1)
        f0 = self.current_forcing[:, t0, :]
        f1 = self.current_forcing[:, t1, :]
        return f0 + w * (f1 - f0)

    def forward(self, t, state):
        """
        ODE right-hand side: f(t, state) -> dstate/dt.
        state: [B, 4] = [LAI, Biomass, N_Pool, Soil_Water]
        """
        # States are stored directly in physical units:
        # LAI [-], biomass [t/ha], mineral N [kg N/ha], soil water [m3/m3].
        state = torch.nan_to_num(state, nan=0.01, posinf=100.0, neginf=0.0)
        lai    = torch.clamp(state[..., 0:1], 0.0, 12.0)
        bio    = torch.clamp(state[..., 1:2], 0.0, 50.0)
        n_pool = torch.clamp(state[..., 2:3], 0.0, 400.0)
        sw_state = state[..., 3:4]
        sw = torch.clamp(sw_state, 0.02, 0.65)
        if torch.isnan(state).any():
            print(f"[NaN] state at t={t.item():.1f} contains NaN, reset to 0.01")

        # 鈹€鈹€ 鎻掑€肩幆澧?鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        env = self._interp(t)
        pcp      = env[..., IDX_PCP:IDX_PCP+1]
        eto      = env[..., IDX_ETO:IDX_ETO+1]
        par      = env[..., IDX_PAR:IDX_PAR+1]
        tmean    = env[..., IDX_TMEAN:IDX_TMEAN+1]
        vpd      = env[..., IDX_VPD:IDX_VPD+1]
        gdd_daily = env[..., IDX_GDD_D:IDX_GDD_D+1]
        gdd_cum  = env[..., IDX_GDD_C:IDX_GDD_C+1]

        # 鈹€鈹€ 鍚勪笓瀹跺墠鍚?+ NaN 婧簮 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        f_temp = self.temp_expert(tmean)
        if torch.isnan(f_temp).any():
            print(f"[NaN-SRC] TemperatureExpert t={t.item():.1f} tmean={tmean[0,0].item():.4f}")

        f_water_raw = self.water_expert(sw)
        if torch.isnan(f_water_raw).any():
            fc, wp, _ = self.water_expert.get_soil_params()
            print(f"[NaN-SRC] WaterExpert t={t.item():.1f} sw={sw[0,0].item():.4f} fc={fc[0].item():.4f} wp={wp[0].item():.4f}")

        f_n_raw, uptake = self.n_expert(n_pool, bio)
        if torch.isnan(f_n_raw).any():
            print(f"[NaN-SRC] NitroExpert t={t.item():.1f} n_pool={n_pool[0,0].item():.4f} bio={bio[0,0].item():.4f}")

        dB_pot, fAPAR = self.rad_expert(par, lai)
        if torch.isnan(dB_pot).any() or torch.isnan(fAPAR).any():
            print(f"[NaN-SRC] RadExpert t={t.item():.1f} par={par[0,0].item():.4f} lai={lai[0,0].item():.4f}")

        f_vpd = self.stom_expert(vpd)
        if torch.isnan(f_vpd).any():
            print(f"[NaN-SRC] StomatalExpert t={t.item():.1f} vpd={vpd[0,0].item():.4f}")

        # 鈹€鈹€ 鏁板€煎畨鍏?鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        f_temp      = torch.nan_to_num(f_temp, nan=0.5)
        f_water_raw = torch.nan_to_num(f_water_raw, nan=0.5)
        f_n_raw     = torch.nan_to_num(f_n_raw, nan=0.5)
        uptake      = torch.nan_to_num(uptake, nan=0.0)
        dB_pot      = torch.nan_to_num(dB_pot, nan=0.0)
        fAPAR       = torch.nan_to_num(fAPAR, nan=0.0)
        f_vpd       = torch.nan_to_num(f_vpd, nan=0.5)

        if not getattr(config, "USE_NITROGEN_STRESS", True):
            f_n_raw = torch.ones_like(f_n_raw)
        if not getattr(config, "USE_VPD_STRESS", True):
            f_vpd = torch.ones_like(f_vpd)

        dev_index, _, _, _ = self.pheno_expert.from_cumulative(gdd_cum)
        f_temp = self._effective_temperature_stress(
            f_temp,
            tmean=tmean,
            dev_index=dev_index,
        )
        static_gates = None
        if (
            getattr(config, "USE_STATIC_INTERACTION_GATES", True) and
            self.current_static_features is not None
        ):
            static_gates = self.coupling.static_interaction_gates(
                self.current_static_features,
                max_adjust=getattr(config, "STATIC_INTERACTION_MAX", 0.10),
            )
            if static_gates is not None:
                _, _, gate_vpd, gate_heat = static_gates
                f_temp = torch.clamp(f_temp * gate_heat, 0.0, 1.0)
                f_vpd = torch.clamp(f_vpd * gate_vpd, 0.0, 1.0)

        # 鈹€鈹€ 鑰﹀悎鑳佽揩 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        year_emb = None
        if (
            getattr(config, "USE_YEAR_EMBEDDING", False) and
            self.current_year is not None
        ):
            year_emb = self.year_embed(self.current_year)

        f_w, f_n, anomaly = self.coupling(
            f_temp, f_water_raw, f_n_raw, lai, n_pool, sw,
            year_emb=year_emb,
        )
        if static_gates is not None:
            gate_w, gate_n, _, _ = static_gates
            f_w = torch.clamp(f_w * gate_w, 0.0, 1.0)
            f_n = torch.clamp(f_n * gate_n, 0.0, 1.0)
        if not getattr(config, "USE_COUPLING_ANOMALY", True):
            anomaly = torch.zeros_like(anomaly)

        # 鈹€鈹€ LSTM 鏃跺簭娈嬪樊淇 (Phase 3 鍓嶅喕缁? 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        if (self.use_lstm_residual and self.current_forcing is not None
            and bool(self.lstm_active_flag.item())):
            t_idx = int(t.item()) if t.dim() == 0 else t.long()
            lstm_delta = self.lstm_residual(self.current_forcing, t_idx)
            f_w     = f_w * (1.0 + lstm_delta[:, 0:1])
            f_n     = f_n * (1.0 + lstm_delta[:, 1:2])
            anomaly = anomaly + lstm_delta[:, 2:3]
            f_w = torch.clamp(f_w, 0.0, 1.0)
            f_n = torch.clamp(f_n, 0.0, 1.0)

        # 鈹€鈹€ Liebig 脳 multiplicative 鑳佽揩 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        f_stress = smooth_min(f_w, f_n, tau=0.1)
        f_stress = f_stress * (1.0 + anomaly)
        f_stress = f_stress * f_temp * f_vpd
        f_stress = torch.clamp(smooth_clamp01(f_stress * 1.1), 0.0, 1.0)

        # 鈹€鈹€ 鏂圭▼ 1: 鐢熺墿閲?d(Biomass)/dt 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        # Growth begins after emergence; dB_pot is in t/ha/day.
        gdd_em = torch.abs(self.pheno_expert.gdd_emergence)
        emergence_width = 20.0
        emerged_gate = torch.sigmoid(
            (gdd_cum - gdd_em) / emergence_width
        )
        # Time derivative of the emergence sigmoid. Its integral is one, so
        # these establishment pools are added once around emergence.
        emergence_rate = (
            emerged_gate * (1.0 - emerged_gate) *
            torch.clamp(gdd_daily, min=0.0) / emergence_width
        )
        dB_photo = dB_pot * f_stress * emerged_gate
        dB_establishment = (
            float(getattr(config, "ESTABLISHMENT_BIOMASS_T_HA", 0.30)) *
            emergence_rate
        )
        dB = dB_photo + dB_establishment

        # 鈹€鈹€ 鏂圭▼ 2: 鍙堕潰绉?d(LAI)/dt 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        # 琛拌€佸姞閫? v1.0 鍩虹嚎 鈥?鐏屾祮鍚庢湡 (dev>0.7) 瑙﹀彂
        sen_accel = torch.sigmoid(10.0 * (dev_index - 0.7))
        k_sen_eff = torch.abs(self.k_sen) * (1.0 + 3.0 * sen_accel)
        # Only a phenology-dependent share of new biomass is allocated to
        # leaves. Treating all biomass as leaf mass forces total biomass far
        # too low when fitting LAI and makes yield scale compensate.
        leaf_allocation = (
            0.05 + 0.35 * torch.sigmoid(12.0 * (0.45 - dev_index))
        )
        # 1 t/ha = 100 g/m2, while SLA is m2 leaf per g dry matter.
        dL_establishment = (
            float(getattr(config, "ESTABLISHMENT_LAI", 0.45)) *
            emergence_rate
        )
        dL = (
            torch.abs(self.sla) * 100.0 * leaf_allocation * dB_photo
            + dL_establishment
            - k_sen_eff * lai
        )

        # 鈹€鈹€ 鏂圭▼ 3: 鍦熷￥姘村垎 dW/dt 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        fc, wp, drain_rate = self.water_expert.get_soil_params()
        while fc.dim() < sw.dim():
            fc = fc.unsqueeze(-1)
            wp = wp.unsqueeze(-1)
            drain_rate = drain_rate.unsqueeze(-1)

        # Root-zone bucket: all fluxes are mm/day and are converted to
        # volumetric-water change using the phenology-dependent root depth.
        root_depth_mm = 300.0 + 900.0 * torch.sigmoid(8.0 * (dev_index - 0.25))
        saturation = torch.clamp(fc + 0.12, max=0.60)
        runoff_fraction = 0.30 * torch.sigmoid((sw - fc) / 0.03)
        infiltration = torch.clamp(pcp, min=0.0) * (1.0 - runoff_fraction)
        et_soil = eto * (1.0 - fAPAR) * (0.15 + 0.35 * f_water_raw)
        et_plant = eto * fAPAR * f_w * f_vpd
        drainage = torch.relu(sw - fc) * drain_rate * root_depth_mm
        lower_restore = torch.relu(0.02 - sw_state) * 1.0
        upper_restore = torch.relu(sw_state - saturation) * 1.0
        dW = (
            (infiltration - et_soil - et_plant - drainage) /
            root_depth_mm
            + lower_restore
            - upper_restore
        )

        # 鈹€鈹€ 鏂圭▼ 4: 姘睜 dN/dt 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        # 鐭垮寲杈撳叆 + 鍙嶇鍖?+ M-M 鍚告敹 + 鍩虹娴佸け
        min_rate = self.n_expert.mineralization_rate(
            tmean.squeeze(-1), sw.squeeze(-1)
        )
        denit_rate = self.n_expert.denitrification_rate(
            tmean.squeeze(-1), sw.squeeze(-1)
        )
        # 骞挎挱鍒?n_pool 褰㈢姸
        while min_rate.dim() < n_pool.dim():
            min_rate = min_rate.unsqueeze(-1)
            denit_rate = denit_rate.unsqueeze(-1)

        organic_n = self.n_expert._organic_n
        if organic_n is not None:
            while organic_n.dim() < n_pool.dim():
                organic_n = organic_n.unsqueeze(-1)
            dN_mineralization = min_rate * organic_n
        else:
            dN_mineralization = 0.0

        dN = (
            dN_mineralization
            - uptake
            - denit_rate * n_pool
            - 0.0005 * n_pool
        )

        dstate = torch.cat([dL, dB, dN, dW], -1)

        # 鈹€鈹€ NaN 杩借釜 (璋冭瘯鐢? 璁粌涓Е鍙戝垯鎵撳嵃) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        if torch.isnan(dstate).any() or torch.isinf(dstate).any():
            bad = torch.isnan(dstate) | torch.isinf(dstate)
            idx = bad.nonzero()
            print(f"[NaN] t={t.item():.1f} | pos={idx[0].tolist() if len(idx) else '?'}",
                  f"| L={lai[0,0].item():.4f} B={bio[0,0].item():.4f}",
                  f"N={n_pool[0,0].item():.4f} W={sw[0,0].item():.4f}",
                  f"| f_temp={f_temp[0,0].item():.4f} f_w={f_w[0,0].item():.4f} f_n={f_n[0,0].item():.4f}",
                  f"| dL={dL[0,0].item():.2f} dB={dB[0,0].item():.2f} dN={dN[0,0].item():.2f} dW={dW[0,0].item():.2f}")
            dstate = torch.nan_to_num(dstate, nan=0.0, posinf=10.0, neginf=-10.0)

        return dstate

