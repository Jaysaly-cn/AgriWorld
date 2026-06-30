"""
StomatalExpert — Leuning 气孔导度 VPD 响应模型
===============================================
消费变量: VPD (forcing index 7, Magnus 公式推导)

参考文献:
    Leuning, R. (1995). A critical appraisal of a combined stomatal-
    photosynthesis model for C3 plants. Plant, Cell & Environment,
    18(4): 339-355. (引用 >3000)

核心方程:
    f_vpd = 1 / (1 + VPD / D0)
    D0 = 半衰减 VPD 值 (玉米 ~1.5-2.5 kPa)

下游效应:
    - 蒸腾速率: ET_plant × f_vpd (高 VPD → 气孔关闭 → 蒸腾降低)
    - 光合速率: 同样乘 f_vpd (气孔关闭 → CO2 进入减少)
"""

import torch
import torch.nn as nn


class StomatalExpert(nn.Module):
    def __init__(self):
        super().__init__()
        # 半衰减 VPD 值 (kPa) — 玉米 ~2.0
        self.D0 = nn.Parameter(torch.tensor(2.0))

    def forward(self, VPD):
        """
        Args:
            VPD: 饱和水汽压差 [B] 或 [B, T]  (kPa)

        Returns:
            f_vpd: VPD 胁迫因子 [same shape], 范围 (0, 1]
                   VPD=0 → 1.0, VPD=D0 → 0.5, VPD→∞ → 0
        """
        D0 = torch.clamp(torch.abs(self.D0), 0.5, 5.0)
        return 1.0 / (1.0 + torch.clamp(VPD, min=0.0) / D0)
