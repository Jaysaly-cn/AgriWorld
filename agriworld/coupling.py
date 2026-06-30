"""
CouplingHead v2 — 带年份感知的胁迫融合
=========================================
在原有 6→16→3 MLP 基础上增加年份嵌入投影 (4→3),
让模型区分不同年份的气候背景。

init: 年份投影零初始化 → 初始行为等同 v1, 预训练权重兼容。
"""

import torch
import torch.nn as nn


class CouplingHead(nn.Module):
    def __init__(self, year_dim: int = 4, static_dim: int = 11):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(6, 16), nn.SiLU(),
            nn.Linear(16, 3),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

        # 年份投影: 4-dim embedding → 3-dim [Δf_w, Δf_n, Δanomaly]
        self.year_proj = nn.Linear(year_dim, 3)
        nn.init.zeros_(self.year_proj.weight)
        nn.init.zeros_(self.year_proj.bias)

        self.static_norm = nn.LayerNorm(static_dim)
        self.static_gate = nn.Sequential(
            nn.Linear(static_dim, 16),
            nn.SiLU(),
            nn.Linear(16, 4),
        )
        nn.init.zeros_(self.static_gate[-1].weight)
        nn.init.zeros_(self.static_gate[-1].bias)

    def static_interaction_gates(self, static_features, max_adjust=0.10):
        """Return bounded static gates for water, nitrogen, VPD, and heat."""
        if static_features is None:
            return None
        x = torch.nan_to_num(static_features.float(), nan=0.0)
        raw = self.static_gate(self.static_norm(x))
        gates = 1.0 + float(max_adjust) * torch.tanh(raw)
        return (
            gates[..., 0:1],
            gates[..., 1:2],
            gates[..., 2:3],
            gates[..., 3:4],
        )

    def forward(self, f_temp, f_water, f_nitrogen, lai, n_pool, sw, year_emb=None):
        """
        year_emb: [B, 4] or None (单年时不传入)
        """
        # Normalize state inputs so their scales cannot dominate expert scores.
        x = torch.cat([
            f_temp,
            f_water,
            f_nitrogen,
            torch.clamp(lai / 6.0, 0.0, 2.0),
            torch.clamp(n_pool / 150.0, 0.0, 3.0),
            torch.clamp(sw / 0.5, 0.0, 1.5),
        ], -1)
        o = self.net(x)  # [B, 3]

        if year_emb is not None:
            yr = self.year_proj(year_emb)  # [B, 3]
            o = o + yr

        water_correction = 1.0 + 0.15 * torch.tanh(o[..., 0:1])
        nitrogen_correction = 1.0 + 0.15 * torch.tanh(o[..., 1:2])
        return (
            torch.clamp(f_water * water_correction, 0.0, 1.0),
            torch.clamp(f_nitrogen * nitrogen_correction, 0.0, 1.0),
            0.10 * torch.tanh(o[..., 2:3]),
        )
