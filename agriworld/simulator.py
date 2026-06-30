"""
AgriWorld Simulator v2 鈥?ODE 姹傝В鍣ㄥ寘瑁?
=========================================
浣跨敤 torchdiffeq.odeint 姹傝В CompositeODE銆?

鏂板: 浼犲叆 static_features 鈫?鍒濆鍖栧悇 Expert 鐨勭綉鏍肩壒寮傛€у弬鏁般€?
"""

import torch
import torch.nn as nn
from torchdiffeq import odeint
from agriworld.ode import CompositeODE
from agriworld.window_stress import CornWindowStressExpert
import agriworld.config as config


def _bounded_inverse(value, low, high):
    p = (float(value) - low) / (high - low)
    p = min(max(p, 1e-4), 1.0 - 1e-4)
    return torch.logit(torch.tensor(p))


def _atanh_inverse(value, scale):
    x = float(value) / float(scale)
    x = min(max(x, -0.999), 0.999)
    return torch.atanh(torch.tensor(x))


class YieldResidualHead(nn.Module):
    """Small bounded residual head for county-level yield heterogeneity."""

    def __init__(self, static_dim=11, dynamic_dim=4, hidden=32):
        super().__init__()
        self.static_norm = nn.LayerNorm(static_dim)
        self.net = nn.Sequential(
            nn.Linear(static_dim + dynamic_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, 1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, static_features, dynamic_features, max_log=0.15):
        static_features = torch.nan_to_num(static_features.float(), nan=0.0)
        dynamic_features = torch.nan_to_num(dynamic_features.float(), nan=0.0)
        x = torch.cat([self.static_norm(static_features), dynamic_features], dim=-1)
        return torch.exp(float(max_log) * torch.tanh(self.net(x)))


class StaticCropParameterHead(nn.Module):
    """Structured county-level crop parameter adjustments from static features."""

    def __init__(
        self,
        static_dim=11,
        hidden=24,
        state_embed_dim=4,
        county_embed_dim=8,
    ):
        super().__init__()
        self.input_dim = static_dim - 1  # drop crop_type; current training data is all corn
        self.state_embed = nn.Embedding(
            int(getattr(config, "STATE_EMBED_BUCKETS", 32)),
            state_embed_dim,
        )
        self.county_embed = nn.Embedding(
            int(getattr(config, "COUNTY_EMBED_BUCKETS", 4096)),
            county_embed_dim,
        )
        total_dim = self.input_dim + state_embed_dim + county_embed_dim
        self.static_norm = nn.LayerNorm(self.input_dim)
        self.context_norm = nn.LayerNorm(total_dim)
        self.net = nn.Sequential(
            nn.Linear(total_dim, hidden, bias=False),
            nn.SiLU(),
            nn.Linear(hidden, 3, bias=False),
        )
        nn.init.normal_(self.state_embed.weight, mean=0.0, std=0.08)
        nn.init.normal_(self.county_embed.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.net[-1].weight, mean=0.0, std=0.02)

    def forward(
        self,
        static_features,
        state_id=None,
        county_id=None,
        hi_max_log=0.10,
        yield_max_log=0.08,
        heat_sens_max=0.50,
    ):
        x = torch.nan_to_num(static_features.float(), nan=0.0)
        x = x[..., :self.input_dim]
        batch = x.shape[0]
        dev = x.device
        if state_id is None:
            state_id = torch.zeros(batch, 1, dtype=torch.long, device=dev)
        if county_id is None:
            county_id = torch.zeros(batch, 1, dtype=torch.long, device=dev)
        state_id = state_id.reshape(batch).to(device=dev, dtype=torch.long)
        county_id = county_id.reshape(batch).to(device=dev, dtype=torch.long)
        use_all_spatial = getattr(config, "USE_SPATIAL_EMBEDDINGS", False)
        use_state = use_all_spatial or getattr(config, "USE_STATE_EMBEDDINGS", True)
        use_county = use_all_spatial or getattr(config, "USE_COUNTY_EMBEDDINGS", False)
        if use_state:
            state_context = self.state_embed(state_id)
        else:
            state_context = torch.zeros(
                batch, self.state_embed.embedding_dim, device=dev
            )
        if use_county:
            county_context = self.county_embed(county_id)
        else:
            county_context = torch.zeros(
                batch, self.county_embed.embedding_dim, device=dev
            )
        context = torch.cat(
            [self.static_norm(x), state_context, county_context],
            dim=-1,
        )
        raw = torch.tanh(self.net(self.context_norm(context)))
        hi_factor = torch.exp(float(hi_max_log) * raw[..., 0:1])
        yield_factor = torch.exp(float(yield_max_log) * raw[..., 1:2])
        heat_sensitivity = 1.0 + float(heat_sens_max) * raw[..., 2:3]
        return hi_factor, yield_factor, torch.clamp(heat_sensitivity, 0.25, 2.0)


class AgriWorldSimulator(nn.Module):
    def __init__(self):
        super().__init__()
        self.ode_func = CompositeODE()
        self.hi_raw = nn.Parameter(_bounded_inverse(
            getattr(config, "INITIAL_HARVEST_INDEX", 0.52),
            0.3,
            0.65,
        ))
        # Raw value maps to a bounded grain conversion correction in [0.5, 2].
        self.log_yield_scale = nn.Parameter(_bounded_inverse(
            getattr(config, "INITIAL_YIELD_SCALE", 1.20),
            0.5,
            2.0,
        ))
        self.yield_year_slope_raw = nn.Parameter(_atanh_inverse(
            getattr(config, "INITIAL_YIELD_YEAR_TREND_LOG", 0.0),
            0.12,
        ))

        self.hi_width = nn.Parameter(torch.tensor(200.0))
        self.yield_residual = YieldResidualHead()
        self.static_crop_params = StaticCropParameterHead()
        self.window_stress = CornWindowStressExpert()
        self.last_window_stress = None
        self.current_static_features = None
        self.current_state_id = None
        self.current_county_id = None

    @property
    def harvest_index(self):
        return 0.3 + 0.35 * torch.sigmoid(self.hi_raw)

    @property
    def yield_scale(self):
        return 0.5 + 1.5 * torch.sigmoid(self.log_yield_scale)

    @property
    def yield_year_slope(self):
        return 0.12 * torch.tanh(self.yield_year_slope_raw)

    @property
    def gdd_flowering(self):
        return self.ode_func.pheno_expert.gdd_flowering

    def set_static_features(
        self,
        static_features,
        latitude_deg=None,
        state_id=None,
        county_id=None,
    ):
        """
        灏嗛潤鎬佺壒寰佸垎鍙戠粰鎵€鏈?Expert锛屽垵濮嬪寲缃戞牸鐗瑰紓鎬у弬鏁般€?

        Args:
            static_features: [B, 11]
            latitude_deg:    [B] 鍙€夛紝鐢ㄤ簬鍧￠潰杈愬皠淇
        """
        self.ode_func.temp_expert.set_static_features(static_features)
        self.ode_func.water_expert.set_static_features(static_features)
        self.ode_func.n_expert.set_static_features(static_features)
        self.ode_func.rad_expert.set_static_features(static_features, latitude_deg)
        self.ode_func.current_static_features = static_features
        self.current_static_features = static_features
        self.current_state_id = state_id
        self.current_county_id = county_id

    def expert_params(self):
        params = (
            list(self.ode_func.water_expert.parameters()) +
            list(self.ode_func.n_expert.parameters())
        )
        return [param for param in params if param.requires_grad]

    def lstm_params(self):
        return list(self.ode_func.lstm_residual.parameters())

    def stomatal_params(self):
        return [self.ode_func.stom_expert.D0]

    def phenology_params(self):
        return [self.ode_func.pheno_expert.gdd_emergence,
                self.ode_func.pheno_expert.gdd_flowering,
                self.ode_func.pheno_expert.gdd_maturity]

    def photo_params(self):
        return [
            self.ode_func.sla, self.ode_func.k_sen,
            self.ode_func.rad_expert.rue, self.ode_func.rad_expert.k_ext,
        ]

    def coupling_params(self):
        params = list(self.ode_func.coupling.parameters())
        if getattr(config, "USE_YEAR_EMBEDDING", False):
            params += list(self.ode_func.year_embed.parameters())
        return params

    def yield_params(self):
        # Flowering GDD is owned by the phenology optimizer.
        params = [
            self.hi_raw,
            self.hi_width,
            self.log_yield_scale,
            self.yield_year_slope_raw,
        ]
        if getattr(config, "USE_YIELD_RESIDUAL", True):
            params += list(self.yield_residual.parameters())
        if getattr(config, "USE_STATIC_CROP_PARAMS", True):
            params += list(self.static_crop_params.parameters())
        if getattr(config, "USE_WINDOW_STRESS", True):
            params += list(self.window_stress.parameters())
        return params

    @torch.no_grad()
    def clamp_physical_parameters(self):
        self.ode_func.sla.clamp_(0.008, 0.030)
        self.ode_func.k_sen.clamp_(0.003, 0.020)
        self.ode_func.rad_expert.rue.clamp_(2.0, 7.0)
        self.ode_func.rad_expert.k_ext.clamp_(0.40, 1.15)
        self.ode_func.stom_expert.D0.clamp_(0.75, 4.0)
        self.ode_func.pheno_expert.gdd_emergence.clamp_(60.0, 150.0)
        self.ode_func.pheno_expert.gdd_flowering.clamp_(650.0, 1050.0)
        self.ode_func.pheno_expert.gdd_maturity.clamp_(1300.0, 2200.0)
        self.hi_width.clamp_(75.0, 400.0)

    @torch.no_grad()
    def set_yield_scale(self, scale):
        """Set the bounded yield scale through its inverse-logit parameter."""
        scale = float(min(max(scale, 0.5001), 1.9999))
        p = (scale - 0.5) / 1.5
        self.log_yield_scale.fill_(torch.logit(torch.tensor(p)).item())

    def static_crop_adjustments(
        self,
        static_features=None,
        state_id=None,
        county_id=None,
    ):
        """Return interpretable county-level crop factors for logging/losses."""
        if static_features is None:
            static_features = self.current_static_features
        if state_id is None:
            state_id = self.current_state_id
        if county_id is None:
            county_id = self.current_county_id
        if (
            static_features is None or
            not getattr(config, "USE_STATIC_CROP_PARAMS", True)
        ):
            return None
        return self.static_crop_params(
            static_features,
            state_id=state_id,
            county_id=county_id,
            hi_max_log=getattr(config, "STATIC_HI_MAX_LOG", 0.10),
            yield_max_log=getattr(config, "STATIC_YIELD_MAX_LOG", 0.08),
            heat_sens_max=getattr(config, "STATIC_HEAT_SENS_MAX", 0.50),
        )

    def forward(self, forcing, n_init, year=None):
        """
        Args:
            forcing: [B, T, 7]  7 閫氶亾 forcing
            n_init:  [B, 1]     鍒濆鏃犳満姘?
            year:    [B] ints or None  骞翠唤鏍囩 (澶氬勾鏁版嵁鏃朵紶鍏?

        Returns:
            traj:       [B, T, 4]
            pred_yield: [B]
        """
        B, T, _ = forcing.shape
        dev = forcing.device

        # 璁剧疆骞翠唤
        self.ode_func.current_year = year

        # GDD_cum 浠?forcing 鎻愬彇 (channel 6)
        gdd_cum_full = forcing[:, :, 6]  # [B, T]
        gdd_flowering = torch.abs(self.gdd_flowering)  # scalar

        h0 = torch.cat([
            torch.full(
                (B, 1),
                float(getattr(config, "INITIAL_LAI", 0.05)),
                device=dev,
            ),
            torch.full(
                (B, 1),
                float(getattr(config, "INITIAL_BIOMASS_T_HA", 0.05)),
                device=dev,
            ),
            torch.clamp(n_init, min=1.0, max=400.0),  # kg N/ha
            self._initial_soil_water(B, dev),          # m3/m3
        ], -1)

        # LSTM 娈嬪樊: 杞ㄨ抗寮€濮嬪墠棰勮绠椾竴娆? 缂撳瓨 hidden states
        lstm_active = (
            self.ode_func.use_lstm_residual and
            bool(self.ode_func.lstm_active_flag.item())
        )
        if lstm_active:
            self.ode_func.lstm_residual.precompute(forcing)

        self.ode_func.current_forcing = forcing
        t = torch.arange(T, dtype=torch.float32, device=dev)

        method = getattr(config, "ODE_METHOD", "euler").lower()
        if method == "euler":
            traj = self._euler_rollout(h0, t)
        else:
            traj = odeint(
                self.ode_func, h0, t, method=method,
                options={
                    "step_size": float(
                        getattr(config, "ODE_STEP_SIZE", 1.0)
                    )
                },
            ).transpose(0, 1)

        # 鈹€鈹€ NaN 闃叉姢 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        if torch.isnan(traj).any() or torch.isinf(traj).any():
            print(f"[NaN] 杞ㄨ抗妫€娴嬪埌 NaN/Inf, batch={B}, 鐢ㄩ浂鏇挎崲")
            traj = torch.nan_to_num(traj, nan=0.01, posinf=10.0, neginf=0.0)

        self.ode_func.current_forcing = None
        if lstm_active:
            self.ode_func.lstm_residual._lstm_cache = None

        final_bio = traj[:, -1, 1:2]  # [B, 1]

        # 鈹€鈹€ HI 鍔ㄦ€佸寲: Sigmoid(GDD - GDD_flowering) 鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€鈹€
        gdd_final = gdd_cum_full[:, -1:]  # [B, 1]
        hi_dynamic = self.harvest_index * torch.sigmoid(
            (gdd_final - gdd_flowering) / torch.abs(self.hi_width)
        )

        county_hi_factor = 1.0
        county_yield_factor = 1.0
        county_heat_sensitivity = 1.0
        if (
            getattr(config, "USE_STATIC_CROP_PARAMS", True) and
            self.current_static_features is not None
        ):
            adjustments = self.static_crop_adjustments(self.current_static_features)
            county_hi_factor, county_yield_factor, county_heat_sensitivity = adjustments
            hi_dynamic = hi_dynamic * county_hi_factor

        if getattr(config, "USE_REPRODUCTIVE_HEAT_PENALTY", True):
            hi_dynamic = hi_dynamic * self._reproductive_heat_factor(
                forcing,
                gdd_cum_full,
                county_heat_sensitivity,
            )

        if year is not None and getattr(config, "USE_YIELD_YEAR_TREND", True):
            year_delta = (
                year.float().reshape(B, 1) -
                float(getattr(config, "YIELD_TREND_CENTER_YEAR", 2021.0))
            )
            year_effect = torch.exp(torch.clamp(
                self.yield_year_slope * year_delta,
                min=-0.35,
                max=0.35,
            ))
        else:
            year_effect = 1.0

        physical_yield = (
            self.yield_scale *
            year_effect *
            county_yield_factor *
            final_bio *
            hi_dynamic
        )
        residual_factor = 1.0
        if (
            getattr(config, "USE_YIELD_RESIDUAL", True) and
            self.current_static_features is not None
        ):
            peak_lai = traj[..., 0].amax(dim=1, keepdim=True)
            year_delta = (
                year.float().reshape(B, 1) -
                float(getattr(config, "YIELD_TREND_CENTER_YEAR", 2021.0))
            ) if year is not None else torch.zeros(B, 1, device=dev)
            dynamic = torch.cat(
                [
                    torch.clamp(final_bio / 25.0, 0.0, 3.0),
                    torch.clamp(peak_lai / 8.0, 0.0, 2.0),
                    torch.clamp(gdd_final / 2000.0, 0.0, 2.0),
                    torch.clamp(year_delta / 5.0, -2.0, 2.0),
                ],
                dim=-1,
            )
            residual_factor = self.yield_residual(
                self.current_static_features,
                dynamic,
                max_log=getattr(config, "YIELD_RESIDUAL_MAX_LOG", 0.15),
            )
        window_factor = 1.0
        self.last_window_stress = None
        if getattr(config, "USE_WINDOW_STRESS", True):
            window_factor, self.last_window_stress = self.window_stress(
                forcing,
                traj,
                self.ode_func,
            )
        pred_yield = physical_yield * residual_factor * window_factor
        return traj, pred_yield.squeeze(-1)

    def _reproductive_heat_factor(
        self,
        forcing,
        gdd_cum_full,
        heat_sensitivity=1.0,
    ):
        tmean = forcing[:, :, 3]
        threshold = float(getattr(config, "REPRO_HEAT_THRESHOLD_C", 30.0))
        width = max(float(getattr(config, "REPRO_HEAT_WIDTH_C", 2.0)), 0.1)
        max_reduction = float(
            getattr(config, "REPRO_HEAT_MAX_HI_REDUCTION", 0.14)
        )
        max_reduction = min(max(max_reduction, 0.0), 0.60)

        gdd_flowering = torch.abs(self.ode_func.pheno_expert.gdd_flowering)
        gdd_maturity = torch.abs(self.ode_func.pheno_expert.gdd_maturity)
        enter_repro = torch.sigmoid((gdd_cum_full - gdd_flowering) / 80.0)
        before_maturity = torch.sigmoid((gdd_maturity - gdd_cum_full) / 120.0)
        repro_gate = enter_repro * before_maturity

        heat_load = torch.sigmoid((tmean - threshold) / width)
        exposure = (
            (heat_load * repro_gate).sum(dim=1, keepdim=True) /
            repro_gate.sum(dim=1, keepdim=True).clamp_min(1e-4)
        )
        sensitivity = torch.as_tensor(
            heat_sensitivity,
            dtype=forcing.dtype,
            device=forcing.device,
        )
        return torch.clamp(
            1.0 - max_reduction * exposure * sensitivity,
            min=max(0.20, 1.0 - max_reduction * 2.0),
            max=1.0,
        )

    def _euler_rollout(self, initial_state, times):
        """Low-overhead daily rollout for training on daily forcing."""
        state = initial_state
        states = [state]
        base_step = float(getattr(config, "ODE_STEP_SIZE", 1.0))
        stride = max(1, int(getattr(config, "TRAIN_STEP_DAYS", 1)))
        for index in range(0, times.numel() - 1, stride):
            days = min(stride, times.numel() - 1 - index)
            previous = state
            derivative = self.ode_func(times[index], state)
            state = state + base_step * days * derivative
            state = self._project_state(state)
            state = torch.nan_to_num(
                state, nan=0.01, posinf=100.0, neginf=0.0
            )
            for offset in range(1, days + 1):
                frac = float(offset) / float(days)
                states.append(previous + frac * (state - previous))
        return torch.stack(states, dim=1)

    def _project_state(self, state):
        parts = [
            torch.clamp(state[..., 0:1], 0.0, 12.0),
            torch.clamp(state[..., 1:2], 0.0, 50.0),
            torch.clamp(state[..., 2:3], 0.0, 400.0),
            torch.clamp(state[..., 3:4], 0.02, 0.65),
        ]
        return torch.cat(parts, dim=-1)

    def _initial_soil_water(self, batch_size, device):
        """Initialize root-zone water between wilting point and field capacity."""
        fc, wp, _ = self.ode_func.water_expert.get_soil_params()
        if fc.dim() == 0:
            fc = fc.expand(batch_size)
            wp = wp.expand(batch_size)
        initial = wp + 0.70 * (fc - wp)
        return initial.to(device).reshape(batch_size, 1)

