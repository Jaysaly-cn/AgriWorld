"""
Benchmarks — LSTM 产量预测基线
=================================
输入: 365 天 × 7ch 序列 + 11 维静态特征
输出: 产量 (bu/acre)
"""

import torch
import torch.nn as nn


class LSTMBaseline(nn.Module):
    def __init__(self, seq_dim=7, static_dim=11, hidden=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(seq_dim, hidden, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0.0)
        self.static_proj = nn.Linear(static_dim, hidden)
        self.head = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x_seq, x_static):
        """
        x_seq:    [B, 365, 7]
        x_static: [B, 11]
        """
        _, (h_n, _) = self.lstm(x_seq)          # h_n: [L, B, hidden]
        seq_feat = h_n[-1]                       # [B, hidden]  最后一层最后时间步
        static_feat = self.static_proj(x_static) # [B, hidden]
        feat = torch.cat([seq_feat, static_feat], dim=-1)  # [B, hidden*2]
        return self.head(feat).squeeze(-1)       # [B]
