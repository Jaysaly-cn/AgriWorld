# AgriWorld 数据管线与数据契约

本文档描述当前主线使用的数据版本。当前数据不是单年数据，而是 2019-2023 五年合并数据；当前训练实验使用 corn 子集，并采用 2023 作为时间外推验证年。

## 1. 当前数据版本

服务器路径：

```text
数据目录: /data4/Agri/yukaijie/AgriWorld/AgriWorld/AgriWorld_Master
缓存目录: /data4/Agri/yukaijie/AgriWorld/AgriWorld/AgriWorld_Master/cache_v
合并文件: /data4/Agri/yukaijie/AgriWorld/AgriWorld/AgriWorld_Master/national_ode_tensors_v2_merged.pkl
项目目录: /data4/Agri/yukaijie/AgriWorld/AgriWorld/Newest_version
```

当前 QC 统计：

```text
source=1857
accepted=867
dropped=990
  - insufficient_lai: 147
  - non_corn: 843
```

训练/验证划分：

```text
Train: 2019, 2020, 2021, 2022  n=680
Val:   2023                    n=187
```

## 2. 数据源

| 数据源 | 用途 | 是否进入 ODE forcing |
|---|---|---|
| Daymet | 日降水、辐射、温度、水汽压等天气驱动 | 是 |
| USDA NASS QuickStats | county-level yield label | 否，作为监督 |
| USDA CDL | 作物类型过滤和 crop code | 静态特征 |
| SoilGrids | clay、sand、BD、OM、TN、pH 等土壤属性 | 静态特征 |
| USGS 3DEP | elevation、slope、aspect | 静态特征 |
| USDA ARMS / 管理先验 | 州级氮肥用量 | 静态/管理特征 |
| Sentinel-2 | LAI 观测，用于 LAI loss | 否 |
| SMAP | soil moisture probe，用于训练后验证 | 否 |
| TIGER Counties | 县域边界和网格定位 | 元数据 |

原则：

1. Yield 必须来自真实 USDA NASS 观测，不使用合成产量标签。
2. Sentinel-2 LAI 只监督状态轨迹，不作为未来信息输入。
3. SMAP 只用于训练后 probe，不参与梯度更新。
4. 每个输入变量必须有明确的物理消费者；没有被模型使用的变量不进入主张量。

## 3. 天气 forcing

当前模型使用 7 通道 forcing：

```text
0 Precip      mm/day
1 ETo         mm/day
2 PAR         MJ/m2/day
3 Tmean       C
4 VPD         kPa
5 GDD_daily   C day/day
6 GDD_cum     C day
```

派生规则：

- `Tmean = (Tmax + Tmin) / 2`
- `PAR = SRAD * 0.48`
- `GDD_daily = max(Tmean - 10, 0)`，从 planting DOY 起累积；
- `GDD_cum` 由清洗后的 `Tmean` 重新计算，不信任历史 pickle 中可能错误的 GDD 列；
- `ETo` 由 Hargreaves 类公式近似；
- `VPD` 由饱和水汽压和实际水汽压计算。

早期数据曾出现 `Tmean=[0,0]` 和 `GDD=0` 的坏样本，当前管线已通过天气 QC 和重新计算 GDD 解决。

## 4. 静态特征

静态特征供专家模块和耦合模块使用，主要包括：

- elevation、slope、aspect；
- bulk density；
- organic matter / organic carbon；
- clay；
- sand；
- total nitrogen；
- pH；
- N fertilizer rate；
- crop code。

水分专家主要消费 clay/sand/BD/OM；氮素专家消费 TN、OM、pH、N rate；辐射模块可使用坡度坡向；作物编码用于过滤当前 corn 子集，并为未来多作物条件化保留入口。

## 5. LAI 与 yield 监督

Sentinel-2 LAI 通过 EVI 等植被指数转换得到。当前 QC 要求每个样本至少有足够的有效 LAI 观测，否则标记为 `insufficient_lai` 并剔除。

Yield 由 NASS county-level 产量转换到 `t/ha`：

```text
1 bu/ac corn = 0.06277 t/ha 近似
```

评估时同时报告：

- `t/ha`；
- `bu/ac`；
- RMSE；
- NRMSE；
- MAPE；
- bias；
- R2。

## 6. 质量控制

当前 QC 主要检查：

| 检查项 | 目的 |
|---|---|
| weather coverage | 防止全零温度、缺辐射、缺 VPD 等坏天气样本 |
| GDD range | 防止种植日或温度字段错误 |
| crop filter | 当前仅保留 corn |
| LAI coverage | 保证 LAI loss 有足够监督点 |
| yield availability | 保证真实产量标签存在 |
| physical range | 防止静态土壤或管理值离谱 |

最新 funnel：

```text
year: raw -> weather -> corn -> LAI>=10 -> yield -> eligible
2019: 358 -> 358 -> 220 -> 323 -> 358 -> 187
2020: 354 -> 354 -> 185 -> 331 -> 354 -> 162
2021: 357 -> 357 -> 185 -> 329 -> 357 -> 157
2022: 408 -> 408 -> 205 -> 377 -> 408 -> 174
2023: 380 -> 380 -> 219 -> 347 -> 380 -> 187
```

其中 `eligible` 代表同时满足作物、LAI、yield 和天气条件后的训练可用样本。

## 7. 缓存与断点续传

数据管线使用年度隔离缓存：

```text
/data4/Agri/yukaijie/AgriWorld/AgriWorld/AgriWorld_Master/cache_v/<year>/
```

关键策略：

- 每个网格/县域请求独立缓存；
- 已完成样本在重启后自动跳过；
- 日志显示 `ok/fail/cached/rate/elapsed/ETA`；
- 年度任务不再使用硬 2 小时 timeout；
- 可通过并行参数提高下载速度，但应避免触发 GEE quota 限制。

如果 GEE 显示 restricted mode，通常表示当前项目配额受限；更新 quota 后，重新启动任务即可，缓存会保留已完成部分。

## 8. 当前数据解释注意事项

1. 当前 867 个样本是经过严格 corn + LAI + weather + yield QC 后的样本，不代表原始五年文件只有 867 条。
2. 2023 是验证年，不是唯一数据年。
3. 产量标签为 county-level，输入网格与县域标签之间存在尺度不匹配，这是 R2 偏低的重要潜在原因。
4. SMAP 分辨率和深度与 ODE soil water 状态不同，probe 结果应作为结构诊断，而非主训练指标。
5. 当前 precipitation 响应较弱，可能与高产玉米带水分非限制、ETo/drain 参数和土壤水分尺度有关。
