"""
LSTMResidual — 时序残差修正模块
=================================
用轻量 LSTM 处理全年 forcing 序列, 在每个时间步输出 ODE 物理方程的残差修正。

设计:
    7ch forcing × 365 天 → 单层 LSTM(7→32) → 每步 hidden → 
    Linear(32→3) → [Δf_w, Δf_n, anomaly] (scale by 0.1)

与原有 CouplingHead 的 anomaly 输出加和, 形成混合修正:
    f_w_final = f_w_mlp + Δf_w_lstm
    f_n_final = f_n_mlp + Δf_n_lstm  
    anom_final = anomaly_mlp + anomaly_lstm

参考文献:
    "Neural ODE with LSTM residual" — 结合时序 RNN 残差的可微物理模型
"""

import torch
import torch.nn as nn


class LSTMResidual(nn.Module):
    """
    单层 LSTM, 输入全年 forcing, 输出逐时间步的残差修正。

    Args:
        forcing_dim: forcing 通道数 (7)
        hidden:      LSTM 隐状态维度
    """

    def __init__(self, forcing_dim: int = 7, hidden: int = 32):
        super().__init__()
        self.lstm = nn.LSTM(forcing_dim, hidden, num_layers=1,
                            batch_first=True)
        self.head = nn.Linear(hidden, 3)  # [Δf_w, Δf_n, Δanomaly]
        self.scale = 0.05
        self.register_buffer(
            "forcing_mean",
            torch.tensor([3.0, 4.0, 9.0, 15.0, 1.2, 5.0, 800.0]),
        )
        self.register_buffer(
            "forcing_scale",
            torch.tensor([6.0, 3.0, 6.0, 12.0, 1.0, 5.0, 700.0]),
        )

        # 初始化: 零输出
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def precompute(self, forcing_seq):
        """
        轨迹开始前调用一次: 跑完整 LSTM 并缓存输出。

        Args:
            forcing_seq: [B, T, 7]  完整 forcing 序列
        """
        normalized = (forcing_seq - self.forcing_mean) / self.forcing_scale
        normalized = torch.clamp(normalized, -5.0, 5.0)
        self._lstm_cache, _ = self.lstm(normalized)  # [B, T, hidden]

    def forward(self, forcing_seq, t_idx):
        """
        每个时间步调用: O(1) 索引已缓存的 hidden state。

        Args:
            forcing_seq: [B, T, 7]  (不再使用, 保留签名兼容)
            t_idx:       int 或 [B]  当前时间步索引 (0..T-1)

        Returns:
            residual: [B, 3]  [Δf_w, Δf_n, Δanomaly]
        """
        lstm_out = self._lstm_cache  # [B, T, hidden]
        B = lstm_out.shape[0]

        if isinstance(t_idx, int):
            t_idx = torch.full((B,), t_idx, dtype=torch.long, device=lstm_out.device)
        batch_idx = torch.arange(B, device=lstm_out.device)
        hidden_t = lstm_out[batch_idx, t_idx, :]  # [B, hidden]

        return torch.tanh(self.head(hidden_t)) * self.scale
