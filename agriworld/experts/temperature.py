"""
TemperatureExpert 鈥?娓╁害鍝嶅簲涓撳 (澧炲己鐗?
=========================================
娑堣垂鍙橀噺: Elevation (static 0, USGS 3DEP)

Daymet 宸插寘鍚湴褰㈠奖鍝嶃€傚綋鍓嶆暟鎹病鏈?Daymet 缃戞牸鍙傝€冮珮绋嬶紝鍥犳涓嶅啀浣跨敤
batch 鍧囧€煎仛浜屾楂樼▼鏍℃锛岄伩鍏嶅悓涓€鏍锋湰鍥?batch 缁勬垚涓嶅悓鑰屽緱鍒颁笉鍚屾俯搴︺€?
娓╁害鍝嶅簲:
    Wang-Engel cardinal-temperature curve, 鍦ㄥ熀娓╁拰涓婇檺娓╁害涓?0锛?    鍦ㄦ渶閫傛俯搴︿负 1銆?"""

import torch
import torch.nn as nn
from agriworld.config import T_BASE, T_OPT, T_CEIL


class TemperatureExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('T_base', torch.tensor(T_BASE))
        self.register_buffer('T_opt', torch.tensor(T_OPT))
        self.register_buffer('T_ceil', torch.tensor(T_CEIL))

        # 娓╁害閫掑噺鐜?(掳C/m), 鏍囧噯澶ф皵 ~0.0065
        self.register_buffer('lapse_rate', torch.tensor(0.0065))

        # Daymet temperature already represents local topography. A second
        # correction requires Daymet grid elevation, which is not in the data.
        self._elev_offset = None

    def set_static_features(self, static_features):
        """
        浠庨珮绋嬭绠楁俯搴︿慨姝ｅ亸绉汇€?

        Args:
            static_features: [B, 11]
                idx 0: Elevation (m, USGS 3DEP)

        Daymet 缃戞牸鍒嗚鲸鐜?1 km, 鍏舵俯搴︿唬琛ㄨ缃戞牸骞冲潎楂樼▼銆?
        鐢ㄥ疄闄?DEM 楂樼▼淇鍚庢洿绮剧‘銆?
        """
        # Keep a zero offset to avoid batch-composition-dependent temperatures.
        # Elevation still affects the model through the observed Daymet forcing.
        self._elev_offset = torch.zeros(
            static_features.shape[0],
            device=static_features.device,
            dtype=static_features.dtype,
        )

    def forward(self, T):
        """
        Args:
            T: 娓╁害 [B] 鎴?[B, T] (掳C, Daymet Tmean)

        Returns:
            f_temp: 娓╁害鍝嶅簲鍥犲瓙 [same shape], range [0, 1]
        """
        # 楂樼▼淇
        if self._elev_offset is not None:
            # 骞挎挱鍒?T 鐨勫舰鐘?
            offset = self._elev_offset
            while offset.dim() < T.dim():
                offset = offset.unsqueeze(-1)
            T = T + offset

        valid = (T > self.T_base) & (T < self.T_ceil)
        tc = torch.clamp(T, self.T_base, self.T_ceil)

        # Wang-Engel cardinal-temperature response: zero at base/ceiling and
        # one at the optimum, with a steeper high-temperature decline.
        alpha = torch.log(torch.tensor(2.0, device=T.device, dtype=T.dtype)) / torch.log(
            (self.T_ceil - self.T_base) / (self.T_opt - self.T_base)
        )
        x = torch.clamp(
            (tc - self.T_base) / (self.T_opt - self.T_base),
            min=0.0,
        )
        response = 2.0 * torch.pow(x, alpha) - torch.pow(x, 2.0 * alpha)
        return torch.where(valid, torch.clamp(response, 0.0, 1.0), torch.zeros_like(T))

