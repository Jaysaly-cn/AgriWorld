"""
PhenologyExpert — 积温驱动的玉米发育阶段模型
=============================================
消费变量: GDD (forcing index 8, max(Tmean-10, 0) 日增量)

参考文献:
    Ritchie, J.T. & Hanway, J.J. (1982). How a corn plant develops.
    Iowa State Univ. Special Report No. 48.

    Yang, H.S. et al. (2004). Hybrid-Maize — a maize simulation model
    that combines two crop modeling approaches. Agronomy Journal, 96(5).

关键 GDD 节点 (玉米, 基温 10°C):
    emergence  ~90 °C·day
    flowering  ~850 °C·day  (Ritchie 阶段 VT/R1)
    maturity   ~1700 °C·day (Ritchie 阶段 R6, 黑层形成)

输出 dev_index ∈ [0, 1] 用于下游:
    - 收获指数 HI = HI_max × sigmoid((dev - dev_flower) / width)
    - 衰老加速: k_sen_eff = k_sen × (1 + 3 × sigmoid(10(dev - 0.7)))
    - 阶段特异性胁迫敏感性
"""

import torch
import torch.nn as nn


class PhenologyExpert(nn.Module):
    def __init__(self):
        super().__init__()
        # 关键 GDD 节点 (可学习，初始化为玉米文献值)
        self.gdd_emergence = nn.Parameter(torch.tensor(90.0))
        self.gdd_flowering  = nn.Parameter(torch.tensor(850.0))
        self.gdd_maturity   = nn.Parameter(torch.tensor(1700.0))

        # 内部状态: 从播种日开始的累积 GDD
        self.cum_gdd = None  # 需要在 forward 前初始化

    def reset(self, batch_size, device):
        """每个新序列开始时重置累积 GDD。"""
        self.cum_gdd = torch.zeros(batch_size, device=device)

    def forward(self, gdd_daily):
        """
        Args:
            gdd_daily: 日 GDD 增量 [B]  (max(Tmean-10, 0))

        Returns:
            dev_index:  发育指数 [B], 范围 [0, 1]
            is_emerged: bool [B]
            is_flowering: bool [B]
            is_mature:  bool [B]
        """
        if self.cum_gdd is None or self.cum_gdd.shape[0] != gdd_daily.shape[0]:
            self.reset(gdd_daily.shape[0], gdd_daily.device)

        self.cum_gdd = self.cum_gdd + gdd_daily

        gdd_em = torch.abs(self.gdd_emergence)
        gdd_fl = torch.abs(self.gdd_flowering)
        gdd_ma = torch.abs(self.gdd_maturity)

        dev_index = torch.clamp(self.cum_gdd / gdd_ma, 0.0, 1.0)

        is_emerged   = self.cum_gdd >= gdd_em
        is_flowering = self.cum_gdd >= gdd_fl
        is_mature    = self.cum_gdd >= gdd_ma

        return dev_index, is_emerged, is_flowering, is_mature

    def from_cumulative(self, cum_gdd):
        """
        无状态接口 — 直接从累积 GDD 计算发育指数 (用于 ODE 内部)。

        Args:
            cum_gdd: [B] 或 [B, T]  累积积温 (°C·day)

        Returns:
            (dev_index, is_emerged, is_flowering, is_mature)
        """
        gdd_em = torch.abs(self.gdd_emergence)
        gdd_fl = torch.abs(self.gdd_flowering)
        gdd_ma = torch.abs(self.gdd_maturity) + 1e-6

        dev_index = torch.clamp(cum_gdd / gdd_ma, 0.0, 1.0)

        is_emerged   = cum_gdd >= gdd_em
        is_flowering = cum_gdd >= gdd_fl
        is_mature    = cum_gdd >= gdd_ma

        return dev_index, is_emerged, is_flowering, is_mature
