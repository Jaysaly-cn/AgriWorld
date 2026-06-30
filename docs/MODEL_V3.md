# AgriWorld v3.23 妯″瀷璇存槑

鏈枃妗ｆ弿杩板綋鍓嶄富绾?`agriworld-v3.23`銆傝鐗堟湰涓嶆槸鏃х殑鍗曞勾 2023 瀹為獙锛岃€屾槸鍩轰簬 2019-2023 浜斿勾鍚堝苟鏁版嵁璁粌涓庤瘎浼帮紱2023 浠呬綔涓烘椂闂村鎺ㄩ獙璇佸勾銆?
## 1. 浠诲姟瀹氫箟

AgriWorld 鐨勭洰鏍囨槸瀛︿範涓€涓彲瑙ｉ噴銆佸彲寰€佸彲鍙嶄簨瀹炴壈鍔ㄧ殑鍐滀笟涓栫晫妯″瀷銆傜粰瀹氬幙鍩熺綉鏍肩殑澶╂皵銆佸湡澹ゃ€佸湴褰€佷綔鐗╁拰绠＄悊淇℃伅锛屾ā鍨嬫ā鎷熸暣涓敓闀垮鐘舵€佽建杩癸紝骞堕娴嬫渶缁堜骇閲忋€?
涓昏棰勬祴瀵硅薄锛?
- LAI 鏃堕棿搴忓垪锛?- 鍦颁笂鐢熺墿閲忥紱
- 鍦熷￥鏃犳満姘簱锛?- 鍦熷￥鍚按鐘舵€侊紱
- 鏈€缁?grain yield銆?
妯″瀷涓嶆妸 Sentinel-2 LAI 鎴?SMAP soil moisture 浣滀负 ODE forcing 杈撳叆銆侺AI 鍙敤浜庣洃鐫ｏ紝SMAP 鍙敤浜庤缁冨悗 probe锛屼粠鑰岄伩鍏嶅洜鏋滄硠婕忋€?
## 2. 鏁版嵁濂戠害

鏃ュ昂搴?forcing 閫氶亾涓猴細

```text
0 Precip      mm/day
1 ETo         mm/day
2 PAR         MJ/m2/day
3 Tmean       C
4 VPD         kPa
5 GDD_daily   C day/day
6 GDD_cum     C day
```

ODE 鐘舵€佷负锛?
```text
0 LAI         m2/m2
1 Biomass     t dry matter/ha
2 N_pool      kg N/ha, scaled internally where needed
3 Soil_water  m3/m3
```

闈欐€佺壒寰佸寘鎷湴褰€佸湡澹ゃ€佺鐞嗗拰浣滅墿缂栫爜銆傚綋鍓嶆湁鏁堣缁冨瓙闆嗕负 corn锛屾湭鏉ユ墿灞曞埌 soybean 鎴栨洿澶氫綔鐗╂椂锛屽簲寮曞叆浣滅墿鐗瑰紓鐨勭敓鐞嗗弬鏁版垨浣滅墿鏉′欢鍖栬€﹀悎鍙傛暟銆?
## 3. 妯″瀷缁撴瀯

AgriWorld 鐢变笓瀹舵ā鍧楀拰鍙井 ODE 缁勬垚锛?
| 妯″潡 | 鍔熻兘 |
|---|---|
| `WaterExpert` | 鐢?clay/sand/BD/OM 鎺ㄦ柇 FC銆乄P銆乨rain锛屽苟璁＄畻姘村垎鏈夋晥鎬?|
| `NitrogenExpert` | Michaelis-Menten 姘惛鏀躲€佺熆鍖栥€佸弽纭濆寲鍜屾爱鑳佽揩 |
| `RadiationExpert` | Monteith RUE 妗嗘灦锛岃绠?PAR銆乫APAR 涓庡厜鍚堢敓鐗╅噺澧為噺 |
| `StomatalExpert` | VPD 瀵规皵瀛?钂歌吘/鐢熼暱鐨勬姂鍒?|
| `PhenologyExpert` | GDD 鍒?dev index 鐨勭墿鍊欐槧灏?|
| `CouplingHead` | 瀵瑰鍥犲瓙鑰﹀悎銆佹畫宸拰浜や簰杩涜灏忓箙鏈夌晫淇 |
| `CompositeODE` | 鑱氬悎涓撳杈撳嚭锛岃绠楃姸鎬佸鏁?|

鏍稿績褰㈠紡鍙鎷负锛?
```text
dB/dt = RUE * PAR * fAPAR(LAI, k_ext) * f_stress
dL/dt = SLA * allocation(dev) * dB/dt - senescence(dev) * LAI
dW/dt = Precip - ET - Drainage
dN/dt = Mineralization - Uptake - Denitrification
Yield = HI(dev_final) * Biomass_final * yield_scale * year_adjustment
```

鍏朵腑 `f_stress` 鐢辨按鍒嗐€佹爱绱犮€乂PD銆佹俯搴﹀拰鑰﹀悎娈嬪樊鍏卞悓鍐冲畾銆?
## 4. v3.23 娓╁害鏈哄埗

鍘嗗彶鐗堟湰鐨勯棶棰樻槸锛氱‖娓╁害鍝嶅簲浼氭妸璁稿姝ｅ父鏆栨棩璇綋浣滃己鑳佽揩锛屽鑷?LAI 鍜屼骇閲忚鍘嬩綆銆傚綋鍓嶄富绾块噰鐢ㄦ瀬绔珮娓╅棬鎺э細

```text
heat_load = sigmoid((Tmean - 33.0) / 2.5)
stage_weight = stage-aware factor centered near dev_index = 0.45
f_temp = 1 - 0.12 * heat_load * stage_weight
```

閰嶇疆椤逛綅浜?`agriworld/config.py`锛?
```python
TEMPERATURE_STRESS_MODE = "heat"
HEAT_STRESS_THRESHOLD_C = 33.0
HEAT_STRESS_WIDTH_C = 2.5
HEAT_STRESS_MAX_REDUCTION = 0.12
HEAT_STRESS_STAGE_CENTER = 0.45
HEAT_STRESS_STAGE_WIDTH = 0.15
```

鍙€夋ā寮忥細

- `heat`: 褰撳墠涓荤嚎锛屾瀬绔珮娓╅棬鎺э紱
- `off` / `none` / `disabled`: 鍏抽棴娓╁害鑳佽揩锛?- `hard` / `raw` / `wang_engel`: 浣跨敤鍘熷娓╁害鍝嶅簲锛?- `soft`: 鏃х殑杞俯搴︽槧灏勩€?
褰撳墠缁撹鏄細`heat` 姣?`off` 鐣ヤ紭锛屽苟涓旀彁渚涙柟鍚戞纭殑鏋佺楂樻俯鍝嶅簲锛沗heat_stress_025` 杩囧己锛岄獙璇佽宸暐宸紝鍥犳涓嶄綔涓轰富绾裤€?
## 5. 璁粌绛栫暐

榛樿 split 涓?`auto`銆傚湪褰撳墠浜斿勾鏁版嵁涓婏紝绋嬪簭閫夋嫨鏃堕棿澶栨帹楠岃瘉锛?
```text
Train: 2019, 2020, 2021, 2022
Val:   2023
```

鎹熷け椤瑰寘鎷細

- LAI 杞ㄨ抗璇樊锛?- yield 璇樊锛?- 鐘舵€佽竟鐣岀害鏉燂紱
- canopy/biomass 涓€鑷存€х害鏉燂紱
- anomaly 姝ｅ垯锛?- 鏂囩尞鍙傛暟鍏堥獙涓庣墿鐞嗚寖鍥寸害鏉熴€?
褰撳墠璁粌閫熷害宸查€氳繃浠ヤ笅鏂瑰紡浼樺寲锛?
- 鏁版嵁闆嗗垵濮嬪寲鏃跺姞杞藉埌鍐呭瓨锛?- 榛樿鏃ュ昂搴?Euler 姹傝В锛屽噺灏?RHS 璋冪敤锛?- batch size 閫傞厤绾?867 涓牱鏈妯★紱
- 鍙湪蹇呰 epoch 鍋氬畬鏁撮獙璇侊紱
- CUDA 涓嬪惎鐢ㄨ交閲忔樉瀛樺崰鐢ㄦā寮忋€?
## 6. 璇勪及鍗忚

褰撳墠涓荤嚎璇勪及鍖呮嫭涓夌被杈撳嚭锛?
1. `scripts/evaluate.py`: 浜ч噺绮惧害銆佷綔鐗╃粺璁°€佸弬鏁拌瘖鏂€佺墿鐞嗕竴鑷存€э紱
2. `scripts/factor_response.py`: 鍙嶄簨瀹炲啘涓氬洜绱犲搷搴旓紱
3. `scripts/ablation.py`: 缁撴瀯鍙樹綋鍜屾満鍒舵秷铻嶃€?
褰撳墠涓荤嚎缁撴灉锛?
| 鎸囨爣 | 鏁板€?|
|---|---:|
| n_grids | 867 |
| n_val | 187 |
| RMSE | 21.53 bu/ac |
| NRMSE | 11.14% |
| Corn MAPE | 8.71% |
| R2 | -0.298 |

鍥犵礌鍝嶅簲锛?
| 鍥犵礌 | high-minus-low yield response |
|---|---:|
| precipitation | +1.97% |
| radiation | +49.13% |
| VPD | -29.34% |
| nitrogen | +30.46% |
| temperature | -0.36% |
| heat_extreme | -0.22% |

瑙ｈ锛氫富绾挎ā鍨嬪凡缁忓叿澶囨纭殑杈愬皠銆乂PD銆佹爱绱犲拰鏋佺楂樻俯鍝嶅簲锛涢檷姘村搷搴斿亸寮憋紝璇存槑姘村垎妯″潡瀵归珮浜х帀绫冲甫鐨勯檺鍒跺己搴︿粛闇€杩涗竴姝ュ尯鍒嗗尯鍩熶笌骞翠唤銆?
## 7. 褰撳墠涓嶈冻

1. 2023 楠岃瘉 R2 浠嶄负璐燂紝璇存槑鍘垮煙闂寸浉瀵规帓搴忚兘鍔涗笉瓒炽€?2. SMAP probe 杈冨急锛屽湡澹ゆ按鍒嗙姸鎬佷笌閬ユ劅 soil moisture 鐨勫昂搴?娣卞害瀵瑰簲杩樻湭鍏呭垎鏍″噯銆?3. 姘礌鍝嶅簲鐜板湪鏄€氳繃缁濆浣?楂?N 鍦烘櫙纭鐨勶紝鐪熷疄鍘垮煙绠＄悊宸紓浠嶈緝绮椼€?4. 褰撳墠浜や簰椤瑰ぇ澶氫负鍏ㄥ眬鍏变韩鍙傛暟锛岀己灏戜綔鐗?鍖哄煙/鍦熷￥鏉′欢鍖栨満鍒躲€?
## 8. 涓嬩竴姝ョ粨鏋勪紭鍖栨柟鍚?
浼樺厛绾т粠楂樺埌浣庯細

1. 浣滅墿涓庡尯鍩熸潯浠跺寲鑰﹀悎鍙傛暟锛氳姘村垎銆佹爱绱犮€侀珮娓╀氦浜掔郴鏁扮敱闈欐€佺壒寰佹垨 crop embedding 璋冨埗銆?2. 绌洪棿寮傝川鎬т骇閲忔畫宸細淇濈暀鐗╃悊涓诲共锛岀敤灏忓箙鏈夌晫 head 淇 county-level bias銆?3. 鍦熷￥姘村垎 probe 鏍″噯锛氬尯鍒嗚〃灞傚拰鏍瑰尯姘村垎锛岃皟鏁?ET/drain/root-depth 鏄犲皠銆?4. 鏇翠弗鏍煎洜绱犲璁★細淇璐熷悜鍝嶅簲 PASS 鍒ゆ嵁锛屽鍔?per-state/per-year 鍝嶅簲鍒嗗竷銆?
