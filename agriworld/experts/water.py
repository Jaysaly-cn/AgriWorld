"""
WaterExpert 鈥?鍦熷￥姘村垎鏈夋晥鎬т笓瀹?(澧炲己鐗?
=========================================
娑堣垂鍙橀噺: Clay_Fraction (static 5), Sand_Fraction (static 6),
          Bulk_Density (static 3), SOC (static 4)

鏍稿績绠楁硶: Saxton-Rawls 鍦熷￥浼犻€掑嚱鏁?(Pedotransfer Functions)

鍙傝€冩枃鐚?
    Saxton, K.E. & Rawls, W.J. (2006). Soil water characteristic estimates
    by texture and organic matter for hydrologic solutions.
    Soil Sci. Soc. Am. J., 70(5): 1569-1578. (寮曠敤 >4000)

鏍稿績鍏崇郴 (绠€鍖栬嚜 SR2006 鏂圭▼):
    胃_FC  鈭?f(clay%, sand%, OM, BD)   鈥?鐢伴棿鎸佹按閲?
    胃_WP  鈭?f(clay%, OM)              鈥?姘镐箙钀庤敨鐐?
    K_sat 鈭?f(sand%, BD)             鈥?楗卞拰瀵兼按鐜?

    OM 鈮?SOC 脳 1.72  (van Bemmelen 鍥犲瓙)

鐢ㄦ硶:
    expert.set_static_features(static)  # 鍒濆鍖栫綉鏍肩壒寮傛€у湡澹ゅ弬鏁?
    f_water = expert(sw)               # 璁＄畻姘村垎鏈夋晥搴?
    fc, wp, drain = expert.get_soil_params()  # 鑾峰彇鍦熷￥姘村姏鍙傛暟
"""

import torch
import torch.nn as nn
import agriworld.config as config
from agriworld.smooth import smooth_clamp01


class WaterExpert(nn.Module):
    def __init__(self):
        super().__init__()
        # 鍦熷￥璐ㄥ湴 鈫?姘村姏鍙傛暟鐨勫皬鍨?pedotransfer 缃戠粶
        # 杈撳叆: [clay%, sand%, BD, OM]  4 缁?
        # 杈撳嚭: [fc_adj, wp_ratio, log_drain]  3 缁?
        self.pedotransfer = nn.Sequential(
            nn.Linear(4, 8), nn.SiLU(),
            nn.Linear(8, 8), nn.SiLU(),
            nn.Linear(8, 3),
        )

        # 榛樿鐨勫叏灞€鍙傛暟 (褰撻潤鎬佺壒寰佷笉鍙敤鏃跺洖閫€)
        self.register_buffer('default_fc', torch.tensor(0.35))
        self.register_buffer('default_wp', torch.tensor(0.15))
        self.register_buffer('default_drain_rate', torch.tensor(0.15))

        # 缃戞牸鐗瑰紓鎬ф按鍔涘弬鏁?(鐢?set_static_features 濉厖)
        self._fc = None    # [B] 鐢伴棿鎸佹按閲?(浣撶Н鍚按閲?
        self._wp = None    # [B] 钀庤敨绯绘暟
        self._drain_raw = None # [B] 鎺掓按閫熺巼鐨勬湭绾︽潫鍙傛暟

    def set_static_features(self, static_features):
        """
        浠庨潤鎬佸湡澹ょ壒寰佹帹瀵兼瘡涓綉鏍肩殑姘村姏鍙傛暟銆?

        Args:
            static_features: [B, 11] 闈欐€佺壒寰佸紶閲?
                idx 5: Clay_Fraction  (%)
                idx 6: Sand_Fraction  (%)
                idx 3: Bulk_Density    (g/cm鲁)
                idx 4: SOC             (g/kg)  鈫?OM = SOC 脳 0.172 (g/g)
        """
        B = static_features.shape[0]
        device = static_features.device

        clay  = static_features[:, 5:6] / 100.0        # [B, 1], fraction
        sand  = static_features[:, 6:7] / 100.0
        bd    = static_features[:, 3:4]                  # g/cm鲁
        soc   = static_features[:, 4:5] / 10.0           # g/kg 鈫?g/100g = %
        om    = soc * 1.72                                # 鏈夋満璐?%

        # pedotransfer 杈撳叆: clay, sand, BD, OM
        x = torch.cat([clay, sand, bd, om], dim=-1)     # [B, 4]
        raw = self.pedotransfer(x)                       # [B, 3]

        # 鏄犲皠鍒扮墿鐞嗚寖鍥?
        fc_adj = torch.sigmoid(raw[:, 0]) * 0.24 + 0.22   # [0.22, 0.46]
        wp_ratio = torch.sigmoid(raw[:, 1]) * 0.35 + 0.35 # [0.35, 0.70]

        self._fc    = fc_adj.squeeze(-1)        # [B]
        self._wp    = (fc_adj * wp_ratio).squeeze(-1)  # [B], wp = fc 脳 ratio
        self._drain_raw = raw[:, 2].squeeze(-1)

    def get_soil_params(self):
        """
        杩斿洖姣忎釜缃戞牸鐨?(fc, wp, drain_rate)銆?
        drain 琚蒋涓婇檺绾︽潫鍦?[0.02, 0.25] 涔嬮棿銆?
        """
        if self._fc is None:
            return self.default_fc, self.default_wp, self.default_drain_rate
        drain_soft = 0.04 + 0.14 * torch.sigmoid(self._drain_raw)
        return self._fc, self._wp, drain_soft

    def forward(self, sw):
        """
        Args:
            sw: 鍦熷￥鍚按閲?[B] 鎴?[B, T]  (浣撶Н鍚按閲? 0-1)

        Returns:
            aw: 妞嶇墿鍙敤姘存湁鏁堝害 [same shape], range [0, 1]
        """
        fc, wp, _ = self.get_soil_params()

        # 骞挎挱鍒?sw 鐨勫舰鐘?
        while fc.dim() < sw.dim():
            fc = fc.unsqueeze(-1)
            wp = wp.unsqueeze(-1)

        aw = smooth_clamp01((sw - wp) / (fc - wp + 1e-8))
        aw = torch.clamp(aw, 0.0, 1.0) ** 0.5
        floor = float(getattr(config, "SOIL_WATER_STRESS_FLOOR", 0.25))
        floor = min(max(floor, 0.0), 0.8)
        return floor + (1.0 - floor) * aw

