"""Interpretable corn growth-window stress factors."""

import torch
import torch.nn as nn

import agriworld.config as config


WINDOW_NAMES = ("establishment", "vegetative", "reproductive", "grain_fill")
FACTOR_NAMES = ("water", "vpd", "heat", "cold", "radiation", "nitrogen")


def _logit(value):
    value = min(max(float(value), 1e-4), 1.0 - 1e-4)
    return torch.logit(torch.tensor(value))


class CornWindowStressExpert(nn.Module):
    """Crop-window stress multiplier with visible agronomic components."""

    def __init__(self):
        super().__init__()
        priors = torch.tensor([
            [0.25, 0.10, 0.15, 0.45, 0.15, 0.20],  # establishment
            [0.30, 0.20, 0.20, 0.05, 0.25, 0.35],  # vegetative
            [0.55, 0.45, 0.55, 0.02, 0.45, 0.20],  # reproductive
            [0.35, 0.25, 0.25, 0.05, 0.50, 0.25],  # grain fill
        ])
        self.raw_sensitivity = nn.Parameter(torch.stack([
            _logit(v) for v in priors.flatten()
        ]).reshape_as(priors))

    def forward(self, forcing, traj, ode_func):
        gdd_cum = forcing[..., 6:7]
        progress = self._crop_progress(gdd_cum, ode_func)
        gates = self._window_gates(progress, gdd_cum, ode_func)
        active = self._active_mask(forcing.device, forcing.dtype)
        gates = gates * active.view(1, 1, -1)

        exposure = self._stress_exposure(forcing, traj, ode_func)
        weighted_exposure = (
            exposure.unsqueeze(2) *
            gates.unsqueeze(-1)
        )
        window_exposure = weighted_exposure.sum(dim=1) / (
            gates.sum(dim=1).unsqueeze(-1).clamp_min(1e-4)
        )
        sensitivity = torch.sigmoid(self.raw_sensitivity)
        contributions = window_exposure * sensitivity.unsqueeze(0)
        window_stress = contributions.sum(dim=-1)
        stress_score = (
            window_stress.sum(dim=-1, keepdim=True) /
            active.sum().clamp_min(1.0)
        )
        max_reduction = float(getattr(config, "WINDOW_STRESS_MAX_REDUCTION", 0.22))
        max_reduction = min(max(max_reduction, 0.0), 0.60)
        factor = 1.0 - max_reduction * torch.tanh(stress_score)
        factor = torch.clamp(factor, min=1.0 - max_reduction, max=1.0)
        return factor, {
            "factor": factor,
            "window_stress": window_stress,
            "contributions": contributions,
            "sensitivity": sensitivity,
            "active_windows": active,
        }

    def _stress_exposure(self, forcing, traj, ode_func):
        tmean = forcing[..., 3:4]
        par = forcing[..., 2:3]
        vpd = forcing[..., 4:5]
        sw = traj[..., 3:4]
        n_pool = traj[..., 2:3]
        bio = traj[..., 1:2]

        f_water = ode_func.water_expert(sw)
        f_n, _ = ode_func.n_expert(n_pool, bio)
        f_vpd = ode_func.stom_expert(vpd)

        width = max(float(getattr(config, "WINDOW_STRESS_TEMP_WIDTH_C", 2.0)), 0.1)
        heat = torch.sigmoid((
            tmean - float(getattr(config, "WINDOW_STRESS_HEAT_THRESHOLD_C", 30.0))
        ) / width)
        cold = torch.sigmoid((
            float(getattr(config, "WINDOW_STRESS_COLD_THRESHOLD_C", 8.0)) - tmean
        ) / width)
        par_ref = max(float(getattr(config, "WINDOW_STRESS_PAR_REFERENCE", 20.0)), 1.0)
        radiation = torch.clamp((par_ref - par) / par_ref, 0.0, 1.0)

        return torch.cat([
            1.0 - f_water,
            1.0 - f_vpd,
            heat,
            cold,
            radiation,
            1.0 - f_n,
        ], dim=-1)

    def _crop_progress(self, gdd_cum, ode_func):
        gdd_em = torch.abs(ode_func.pheno_expert.gdd_emergence)
        gdd_ma = torch.abs(ode_func.pheno_expert.gdd_maturity)
        return torch.clamp((gdd_cum - gdd_em) / (gdd_ma - gdd_em + 1e-6), 0.0, 1.0)

    def _window_gates(self, progress, gdd_cum, ode_func):
        d = progress.squeeze(-1)
        ranges = ((0.00, 0.20), (0.20, 0.50), (0.50, 0.70), (0.70, 1.01))
        width = 0.035
        gates = []
        for low, high in ranges:
            enter = torch.sigmoid((d - low) / width)
            leave = torch.sigmoid((high - d) / width)
            gates.append(enter * leave)
        gdd_em = torch.abs(ode_func.pheno_expert.gdd_emergence)
        emerged = torch.sigmoid((gdd_cum.squeeze(-1) - gdd_em) / 25.0)
        return torch.stack(gates, dim=-1) * emerged.unsqueeze(-1)

    def _active_mask(self, device, dtype):
        spec = str(getattr(config, "WINDOW_STRESS_ACTIVE_WINDOWS", "all")).lower()
        if spec in {"all", "*", ""}:
            return torch.ones(len(WINDOW_NAMES), device=device, dtype=dtype)
        names = {item.strip() for item in spec.split(",")}
        return torch.tensor(
            [1.0 if name in names else 0.0 for name in WINDOW_NAMES],
            device=device,
            dtype=dtype,
        )
