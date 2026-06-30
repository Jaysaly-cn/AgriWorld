"""
Benchmarks — Transformer 产量预测基线
=======================================
输入: 365 天 × 7ch 序列 + 11 维静态特征
输出: 产量 (bu/acre)

使用轻量 Transformer Encoder，便于与 AgriWorld 对比。
"""

import torch
import torch.nn as nn


class TransformerBaseline(nn.Module):
    def __init__(self, seq_dim=7, static_dim=11, d_model=64, nhead=4,
                 num_layers=2, dropout=0.2):
        super().__init__()
        self.input_proj = nn.Linear(seq_dim, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, 365, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation='gelu',
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.static_proj = nn.Linear(static_dim, d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x_seq, x_static):
        """
        x_seq:    [B, 365, 7]
        x_static: [B, 11]
        """
        B = x_seq.shape[0]
        x = self.input_proj(x_seq) + self.pos_embed[:, :x_seq.size(1), :]  # [B, 365, d_model]
        x = self.encoder(x)                        # [B, 365, d_model]
        seq_feat = x.mean(dim=1)                   # [B, d_model]  池化
        static_feat = self.static_proj(x_static)   # [B, d_model]
        feat = torch.cat([seq_feat, static_feat], dim=-1)
        return self.head(feat).squeeze(-1)
