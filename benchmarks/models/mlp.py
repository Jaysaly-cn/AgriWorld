"""
Benchmarks — MLP 产量预测基线
================================
输入: 365×7 序列展平为统计量 + 11 维静态特征
输出: 产量 (bu/acre)
"""

import torch
import torch.nn as nn


class MLPBaseline(nn.Module):
    def __init__(self, seq_dim=7, static_dim=11, hidden=128, dropout=0.3):
        super().__init__()
        # 将 365 天序列聚合为 4 个统计量 per channel: mean, std, min, max → 7×4=28
        self.seq_agg_dim = seq_dim * 4
        input_dim = self.seq_agg_dim + static_dim  # 28 + 11 = 39
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden // 2, 1),
        )

    def aggregate_sequence(self, x_seq):
        """将 [B, 365, 7] 聚合成 [B, 28] (mean/std/min/max per channel)."""
        return torch.cat([
            x_seq.mean(dim=1),
            x_seq.std(dim=1),
            x_seq.min(dim=1).values,
            x_seq.max(dim=1).values,
        ], dim=-1)

    def forward(self, x_seq, x_static):
        seq_agg = self.aggregate_sequence(x_seq)
        x = torch.cat([seq_agg, x_static], dim=-1)
        return self.net(x).squeeze(-1)
