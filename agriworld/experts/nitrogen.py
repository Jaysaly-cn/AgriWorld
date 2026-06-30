"""
NitrogenExpert 鈥?鍦熷￥姘姩鍔涘涓撳 (澧炲己鐗?
==========================================
娑堣垂鍙橀噺: SOC (static 4), Total_Nitrogen (static 7), pH (static 8)

鏍稿績绠楁硶:
    - Stanford & Smith (1972) 涓€闃剁熆鍖栧姩鍔涘
    - DAYCENT pH 淇 (Parton et al., 1998)
    - Michaelis-Menten 浣滅墿鍚告敹鍔ㄥ姏瀛?

鍙傝€冩枃鐚?
    Stanford, G. & Smith, S.J. (1972). Nitrogen mineralization potentials
    of soils. Soil Sci. Soc. Am. J., 36(3): 465-472. (寮曠敤 >3000)

    Parton, W.J. et al. (1998). DAYCENT and its land surface submodel.
    Global and Planetary Change, 19(1-4): 35-48.

鏍稿績鏂圭▼:
    鐭垮寲:  dN/dt += k_opt 脳 Q10(T) 脳 f_W(W) 脳 Organic_N 脳 f_pH
    鍚告敹:  uptake = Vmax 脳 N_pool / (Km + N_pool)    [M-M 鍔ㄥ姏瀛
    鍙嶇鍖? dN/dt -= k_denit 脳 f_T(T) 脳 f_WFPS(W) 脳 N_pool

鐢ㄦ硶:
    expert.set_static_features(static)  # 鍒濆鍖?SOC, Total_N, pH
    f_n, uptake = expert(n_pool, bio, T, W)  # 鍓嶅悜
"""

import torch
import torch.nn as nn
from agriworld.smooth import smooth_clamp01, smooth_relu


class NitrogenExpert(nn.Module):
    def __init__(self):
        super().__init__()
        # M-M uptake parameters in kg N ha-1 day-1 and kg N ha-1.
        self.log_vmax = nn.Parameter(torch.tensor(0.4055))  # log(1.5)
        self.log_km   = nn.Parameter(torch.tensor(3.6889))  # log(40)

        # 鐭垮寲鍙傛暟 (鍙涔?
        self.log_k_min = nn.Parameter(torch.tensor(-9.0))   # k_opt 鍩哄噯鐭垮寲閫熺巼
        self.log_k_denit = nn.Parameter(torch.tensor(-7.0)) # k_denit_max 鍩哄噯鍙嶇鍖栭€熺巼

        # 缃戞牸鐗瑰紓鎬ф爱鍙傛暟
        self._organic_n = None   # [B] 鏈夋満姘睜 (g N / kg soil)
        self._pH = None          # [B] 鍦熷￥ pH

    def set_static_features(self, static_features):
        """
        浠庨潤鎬佺壒寰佹帹瀵兼瘡涓綉鏍肩殑姘弬鏁般€?

        Args:
            static_features: [B, 11]
                idx 4: SOC             (g/kg) 鈫?Organic_N 鈮?SOC 脳 0.05 (C/N鈮?0)
                idx 7: Total_Nitrogen  (g/kg) 鈫?鍒濆鏈夋満姘睜
                idx 8: pH              (pH units)
        """
        tot_n  = static_features[:, 7]            # [B] g/kg
        ph     = static_features[:, 8]            # [B] pH
        bd     = torch.clamp(static_features[:, 3], 0.8, 1.8)  # Mg/m3

        # Top 0.3 m organic N stock, kg N/ha.
        soil_mass_kg_ha = bd * 1000.0 * 0.30 * 10000.0
        self._organic_n = torch.clamp(
            tot_n / 1000.0 * soil_mass_kg_ha,
            min=500.0,
            max=15000.0,
        )
        self._pH = ph

    def _pH_modifier(self):
        """
        pH 瀵圭熆鍖栫殑淇鍥犲瓙 (DAYCENT 鍨嬫褰㈠嚱鏁?銆?

        pH 6-7: 鏈€澶?= 1.0
        pH 4-5: 绾挎€ц“鍑忚嚦 0.2
        pH 8-9: 绾挎€ц“鍑忚嚦 0.2
        """
        if self._pH is None:
            return 1.0
        ph = self._pH
        # 浣跨敤 sigmoid 鏋勯€犲钩婊戞褰?
        left  = torch.sigmoid((ph - 4.5) * 3.0)    # pH>4.5 涓婂崌鍒?1
        right = torch.sigmoid((8.5 - ph) * 3.0)    # pH<8.5 淇濇寔 1
        return 0.2 + 0.8 * left * right              # [0.2, 1.0]

    def mineralization_rate(self, T, W=None):
        """
        鐭垮寲閫熺巼 (鏃モ伝鹿)銆?

        Q10 娓╁害鍝嶅簲: k(T) = k_opt 脳 2^((T-25)/10)
        """
        if self._organic_n is None:
            return 0.0

        k_opt = torch.exp(self.log_k_min)
        q10 = 2.0 ** ((T - 25.0) / 10.0)
        f_pH = self._pH_modifier()

        rate = k_opt * q10 * f_pH

        # 姘村垎淇: 濡傛灉鏈?W, 鍦ㄧ敯闂存寔姘撮噺闄勮繎鐭垮寲鏈€蹇?
        if W is not None:
            f_w = torch.clamp(1.0 - 2.0 * torch.abs(W - 0.35), 0.1, 1.0)
            rate = rate * f_w

        return rate

    def denitrification_rate(self, T, W):
        """
        鍙嶇鍖栭€熺巼銆傚綋鍦熷￥鍚按閲忚秴杩?~60% WFPS 鏃舵樉钁楀鍔犮€?
        WFPS 鈮?W / porosity, 瀛旈殭搴?鈮?0.45, 闃堝€?鈮?0.27
        """
        k_denit = torch.exp(self.log_k_denit)
        q10 = 2.0 ** ((T - 25.0) / 10.0)

        # WFPS 鎸囨爣: W 瓒呰繃 0.25 鍚庡弽纭濆寲鍔犻€?
        wfps_ratio = smooth_clamp01((W - 0.20) / 0.20)
        return k_denit * q10 * wfps_ratio

    def forward(self, n_pool, bio, T=None, W=None):
        """
        Args:
            n_pool: 鍦熷￥鏃犳満姘?[B] 鎴?[B, T]
            bio:    鐢熺墿閲?      [B] 鎴?[B, T]
            T:      娓╁害         [B] 鎴?[B, T] (鐢ㄤ簬鐭垮寲/鍙嶇鍖栭€熺巼)
            W:      鍦熷￥鍚按閲?  [B] 鎴?[B, T]

        Returns:
            f_n:    姘厖瓒虫寚鏁?[same shape], range [0, 1]
            uptake: 浣滅墿鍚告敹閫熺巼 [same shape]
        """
        Vmax = torch.exp(self.log_vmax)
        Km   = torch.exp(self.log_km) + 1e-6

        # Remaining mineral N relative to a biomass-dependent reserve.
        demand_reserve = 15.0 + 8.0 * bio
        f_n = torch.clamp(
            smooth_clamp01(n_pool / (demand_reserve + 1e-6)),
            0.0,
            1.0,
        )

        # Canopy-dependent Michaelis-Menten uptake, kg N/ha/day.
        canopy_activity = bio / (bio + 1.0)
        uptake = Vmax * n_pool / (Km + n_pool) * canopy_activity

        return f_n, uptake

