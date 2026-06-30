"""
RadiationExpert 鈥?杈愬皠鍒╃敤鏁堢巼 + 鍧￠潰淇 (澧炲己鐗?
====================================================
娑堣垂鍙橀噺: PAR (forcing index 3), Slope (static 1), Aspect (static 2)

鏍稿績绠楁硶:
    - Monteith RUE 妯″瀷 (Monteith, 1977)
    - 鍧￠潰澶槼杈愬皠鍑犱綍淇 (Allen et al., 2006, FAO-56 闄勫綍)

鍙傝€冩枃鐚?
    Monteith, J.L. (1977). Climate and the efficiency of crop production
    in Britain. Phil. Trans. R. Soc. B, 281: 277-294.

    Allen, R.G. et al. (2006). FAO-56: Dual crop coefficient method.
    Appendix: Slope/aspect radiation adjustment.

鏍稿績鏂圭▼:
    PAR_slope 鈮?PAR_flat 脳 cos(胃_i) / cos(胃_z)  (绠€鍖栨棩灏哄害淇)
    绠€鍖? PAR_eff = PAR 脳 (1 + 尾 脳 tan(slope) 脳 cos(aspect - 纬))
    鍏朵腑 尾 涓虹含搴︾浉鍏冲洜瀛? 纬 涓哄崡鍚戝亸绉?(鍖楀崐鐞?纬=180掳)
"""

import torch
import torch.nn as nn
import numpy as np
from agriworld.units import G_M2_TO_T_HA


class RadiationExpert(nn.Module):
    def __init__(self):
        super().__init__()
        # 鍙涔犵殑鍏夊悎鍙傛暟
        self.rue = nn.Parameter(torch.tensor(3.0))      # g/MJ, RUE
        self.k_ext = nn.Parameter(torch.tensor(0.65))    # 娑堝厜绯绘暟

        # 鍧￠潰杈愬皠淇鍙傛暟
        self._slope_factor = None    # [B] 鍧￠潰杈愬皠淇鍥犲瓙
        self._cos_zenith = None      # [B, T] 澶╅《瑙掍綑寮?(鐢?ODE 娉ㄥ叆)

    def set_static_features(self, static_features, latitude_deg=None):
        """
        璁＄畻鍧″害/鍧″悜瀵?PAR 鐨勪慨姝ｅ洜瀛愩€?
        Args:
            static_features: [B, 11]
                idx 1: Slope  (degree)
                idx 2: Aspect (degree, 姝ｅ寳=0 椤烘椂閽?
            latitude_deg: [B] 绾害 (搴?, 鐢ㄤ簬璁＄畻澶槼浣嶇疆
        """
        slope_deg  = static_features[:, 1]   # [B]
        aspect_deg = static_features[:, 2]   # [B]

        slope_rad  = slope_deg * (np.pi / 180.0)
        aspect_rad = aspect_deg * (np.pi / 180.0)

        # 绠€鍖栨棩灏哄害淇: 鍗楀悜鍧¤幏寰楁洿澶氳緪灏?        # PAR_eff/PAR 鈮?1 + tan(slope) 脳 cos(aspect - south) 脳 lat_factor
        south_rad = torch.tensor(np.pi)  # 鍗?= 180掳 = 蟺 rad
        aspect_diff = torch.cos(aspect_rad - south_rad)

        if latitude_deg is not None:
            lat_rad = latitude_deg * (np.pi / 180.0)
            # 绾害瓒婇珮, 鍧￠潰鏁堝簲瓒婃樉钁?            lat_factor = torch.sin(lat_rad)
        else:
            lat_factor = torch.tensor(0.5)  # 涓含搴﹂粯璁?
        self._slope_factor = 1.0 + torch.tan(slope_rad) * aspect_diff * lat_factor

        # 瑁佸壀鍒板悎鐞嗚寖鍥?[0.7, 1.5]
        self._slope_factor = torch.clamp(self._slope_factor, 0.7, 1.5)

    def forward(self, PAR, LAI):
        """
        Args:
            PAR: 鍏夊悎鏈夋晥杈愬皠 [B] 鎴?[B, T]  (MJ/m虏/day)
            LAI: 鍙堕潰绉寚鏁?   [B] 鎴?[B, T]

        Returns:
            dB_potential: 娼滃湪鐢熺墿閲忓閲?[same shape]
            fAPAR:        鍏夋埅鑾锋瘮渚?    [same shape]
        """
        k = torch.abs(self.k_ext)
        fAPAR = 1.0 - torch.exp(-k * LAI)
        rue = torch.abs(self.rue)

        # 鍧￠潰杈愬皠淇
        PAR_eff = PAR
        if self._slope_factor is not None:
            sf = self._slope_factor
            while sf.dim() < PAR.dim():
                sf = sf.unsqueeze(-1)
            PAR_eff = PAR * sf

        # RUE [g/MJ] 脳 PAR [MJ/m2/day] -> g/m2/day -> t/ha/day.
        dB_potential = rue * PAR_eff * fAPAR * G_M2_TO_T_HA
        return dB_potential, fAPAR

