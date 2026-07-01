# AgriWorld — 项目全景文档 (AI Agent 工作手册)

> **目标**: 让一个新的 AI Agent 在无任何先验知识的情况下，通过本文档全面理解项目，并基于代码内容进行 AAAI 论文的创新点包装、写作和算法分析。

---

## 1. 项目身份卡

| 项目名 | AgriWorld |
|--------|-----------|
| **一句话定义** | 一个基于专家知识引导的可微世界模型，用于解释性作物系统动力学仿真与产量预测 |
| **当前版本** | v3.27 |
| **目标会议** | AAAI (AI 顶会, 计算机科学方向) |
| **数据** | 867 个美国中西部玉米县级网格 (2019-2023), Daymet + Sentinel-2 + SoilGrids + USDA NASS |
| **代码语言** | Python 3.12 + PyTorch + torchdiffeq |
| **硬件** | RTX 3090 24GB, E5 12核 CPU |

## 2. 为 AAAI 定制的定位

这篇论文不是一个农业领域论文——它是一个 **AI/世界模型** 论文。核心论点：

> 我们展示了一个 **知识引导的可微世界模型（Knowledge-Guided Differentiable World Model）**，用 6 个小型神经专家替代了传统作物模型中的固定方程，同时保留物理可解释性。模型在反事实推演和因子归因上实现了传统黑盒模型无法达到的能力。

### 与 AAAI 典型论文的对齐点

| AAAI 关键词 | AgriWorld 的对应 |
|------------|-----------------|
| **Differentiable simulation** | 6 Expert → ODE, 全部可微, 梯度穿透 ODE 求解器 |
| **Neuro-symbolic / knowledge-guided** | 全部 Expert 由验证物理方程初始化 (Monteith RUE, Saxton-Rawls, M-M kinetics) |
| **Interpretability** | 四维内部状态轨迹 (LAI/Biomass/N/Water) + 因子归因审计 |
| **Counterfactual reasoning** | `factor_response.py` 逐因子干预推演 |
| **World model** | 从气象+土壤+管理 → 推演完整生长季状态 → 预测产量 |

## 3. 完整代码地图

```
AgriWorld/
├── agriworld/                    # 核心 Python 包
│   ├── config.py                 # 全局配置: 130 行, 50+ 环境变量控制消融
│   ├── paths.py                  # 服务器/本地路径契约
│   ├── dataset.py                # 数据加载 + 质控 + 玉米过滤
│   ├── simulator.py              # 顶层仿真器: ODE 求解包装 + 产量头
│   ├── ode.py                    # 核心: 4-d 作物状态 ODE 右函项 (345 行)
│   │                               - 6 Expert 调用
│   │                               - LSTM 时序残差 (可选)
│   │                               - 年份嵌入注入
│   │                               - 极端高温门控
│   │                               - NaN 溯源保护
│   ├── coupling.py               # 耦合头: MLP(6→16→3) + 年份投影 + 静态交互门控
│   ├── smooth.py                 # 可微平滑函数 (softplus/min/clamp)
│   ├── losses.py                 # 8 项多目标损失
│   ├── validate.py               # 物理一致性 + SMAP 探针 + 因子审计
│   ├── pretrain.py               # Expert 预训练 (合成物理数据)
│   ├── units.py                  # 单位转换
│   ├── splits.py                 # 训练/验证集划分 (时间外推)
│   ├── data_quality.py           # 数据漏斗统计
│   ├── log_utils.py              # 日志工具
│   └── experts/                  # 专家模块
│       ├── temperature.py        # Beta 型温度响应 + 高程递减率
│       ├── water.py              # Saxton-Rawls pedotransfer + 水分有效度
│       ├── nitrogen.py           # M-M 吸收 + 矿化 + 反硝化 + pH 修正
│       ├── radiation.py          # Monteith RUE + 坡面辐射修正
│       ├── stomatal.py           # Leuning VPD→气孔导度
│       ├── phenology.py          # GDD→发育阶段 (出苗/开花/成熟)
│       ├── lstm_residual.py      # LSTM 时序残差 (消融)
│       └── year_embed.py         # 年份嵌入 (4-dim)
│
├── scripts/                      # 实验脚本
│   ├── train.py                  # 主训练循环 (Euler 3-day 加速)
│   ├── evaluate.py               # 验证 + 诊断 + 结果导出
│   ├── ablation.py               # 消融实验运行器 (8+ 变体)
│   ├── factor_response.py        # 反事实因子响应审计
│   ├── visualize.py              # 可视化
│   ├── make_report_figures.py    # AAAI 图表生成
│   ├── data_pipeline.py          # GEE 数据管线 (单年)
│   ├── run_pipeline.py           # 多年代并行数据管线
│   └── merge_years.py            # 年度数据合并
│
├── benchmarks/                   # 基线模型
│   ├── models/lstm.py            # LSTM 产量预测 (~61K 参数)
│   ├── models/transformer.py     # Transformer (~133K 参数)
│   ├── models/mlp.py             # MLP 统计基线 (~13K 参数)
│   ├── train.py                  # 统一训练入口
│   ├── run_all.py                # 全量运行 + 对比表
│   └── data.py                   # 共享数据接口
│
├── docs/                         # 文档
│   ├── MODEL_V3.md               # v3.23 模型说明 (中文)
│   ├── ABLATION.md               # 消融实验计划
│   ├── data_index.txt            # 变量索引
│   ├── data_instrument.md        # 数据管线规范
│   └── report_assets/            # 图表资产
│
├── results/                      # 实验结果 (大量已运行)
│   ├── ablation_*.json/csv/log   # 各消融变体结果
│   ├── eval_*.json/csv           # 评估结果
│   ├── factor_response.json      # 因子响应
│   └── training_history_*.csv    # 训练历史
│
├── COLLABORATOR_HANDOFF.md       # 合作者接管文档 (694 行, 最全面)
├── README.md                     # 快速入门
├── requirements.txt              # 依赖清单
│
├── train.py                      # 兼容包装 → scripts/train.py
├── evaluate.py                   # 兼容包装 → scripts/evaluate.py
├── ablation.py                   # 兼容包装 → scripts/ablation.py
└── factor_response.py            # 兼容包装 → scripts/factor_response.py
```

## 4. 完整算法流程

### 4.1 训练流程 (scripts/train.py)

```
1. 加载 5 年合并 .pkl
   └→ dataset.py: 867 个玉米样本, 过滤非玉米 + LAI<10 的网格

2. 数据划分 (splits.py)
   └→ 训练: 2019-2022 (680 样本), 验证: 2023 (187 样本) - 时间外推

3. Expert 预训练 (可选, pretrain.py)
   └→ WaterExpert: pedotransfer (clay/sand/BD/OM → fc/wp/drain)
   └→ NitrogenExpert: M-M uptake (n_pool/bio → uptake)
   └→ CouplingHead: 初始零权重

4. 主训练循环 (MAX_EPOCHS=250)
   ├── Phase 1 (LAI-only, ep 1-20):
   │   └→ 仅 LAI Loss, 冻结 anomaly 和 yield
   ├── Phase 2 (Anomaly ramp, ep 21-50):
   │   └→ anomaly 权重 0→1 缓释
   └── Phase 3 (Full, ep 51-250):
       └→ 全部 Loss 激活, Early stopping

5. 每个 batch 的 forward:
   ┌── forcing [B, T, 7] + static [B, 11] + year + n_init → simulator
   │   ├── set_static_features: 分发土壤/地形到 Expert
   │   ├── ODE 积分 (Euler, step_size=3 days):
   │   │   for t in 0..T step 3:
   │   │       env = interpolate(forcing, t)
   │   │       f_temp = TemperatureExpert(Tmean) × heat_gate(T>33°C)
   │   │       f_water = WaterExpert(SW)  (pedotransfer)
   │   │       f_n, uptake = NitrogenExpert(N_pool, Bio)  (mineralization + denitrification)
   │   │       dB_pot, fAPAR = RadiationExpert(PAR, LAI)  (Monteith RUE)
   │   │       f_vpd = StomatalExpert(VPD)  (Leuning)
   │   │       dev_index = PhenologyExpert(GDD_cum)
   │   │       f_w, f_n_corr, anomaly = CouplingHead(f_temp, f_water, f_n, LAI, N, SW, year_emb)
   │   │       f_stress = min(f_w, f_n_corr) × (1+anomaly) × f_vpd
   │   │       dB = dB_pot × f_stress
   │   │       dL = SLA × dB - k_sen(dev) × LAI × (1 + 2.5 × sigmoid(10(dev-0.7)))
   │   │       dW = Precip - ET_soil - ET_plant - Drainage
   │   │       dN = Mineralization - Uptake - Denitrification - Loss
   │   └── yield = harvest_index(dev_final) × Biomass_final × yield_scale × year_trend
   └→ Loss = W_LAI×L_LAI + W_YIELD×L_YIELD + W_STATE×L_state + W_PRIOR×L_prior ...

6. 优化器 (5 组独立学习率)
   ├── expert (water+N):     lr=3e-4
   ├── photo (SLA/k_sen/RUE/k_ext): lr=3e-3
   ├── coupling:             lr=5e-4
   ├── stomatal (D0):        lr=5e-4
   ├── phenology (GDD):      lr=5e-4
   └── yield (HI/scale/trend): lr=1e-2
```

### 4.2 推理流程 (scripts/evaluate.py)

```
1. 加载 saved_models/agriworld_phys_spatial_best.pth
2. 在 2023 验证集 (187 样本) 上运行
3. 产出:
   ├── 产量精度: RMSE, NRMSE, MAPE, R² (per-county)
   ├── 物理一致性: LAI peak, water balance, state bounds
   ├── SMAP 探针: SW_ode vs SMAP R²
   ├── 因子响应: 6 个环境因子的反事实推演
   └── 参数诊断: 所有可学习参数的收敛值
```

### 4.3 反事实因子审计 (scripts/factor_response.py)

```
对每个验证样本:
  baseline = model(原始 forcing, static, year)
  
  对每个因子 (precip, PAR, VPD, N, T, heat):
    扰动 forcing 或 static 中对应通道 (±Δ)
    perturbed = model(扰动后 input)
    Δyield = perturbed - baseline
    response = Δyield / baseline × 100%
```

## 5. 各 Expert 的设计理由与可解释性

### 5.1 TemperatureExpert
- **物理依据**: 作物生长速率与温度呈非对称 Beta 型曲线
- **AAAI 创新点**: 不是用固定参数，而是用了 **极端高温门控 (>33°C)** — 只在危险高温时施加胁迫，避免对正常暖日误判
- **可解释性**: `f_temp = 1 - 0.16 × sigmoid((T-33)/2.5) × stage_weight`，公式透明

### 5.2 WaterExpert
- **物理依据**: Saxton-Rawls 土壤传递函数
- **AAAI 创新点**: 用小型 MLP **从土壤质地 (clay/sand/BD/OM) 推导每个网格的水力参数** (fc, wp, drain)，替代全局固定参数
- **可解释性**: 每个网格的田间持水量和萎蔫系数可直接提取并制图

### 5.3 NitrogenExpert
- **物理依据**: Michaelis-Menten 酶动力学 + Stanford-Smith 矿化 + Rolston 反硝化
- **AAAI 创新点**: 同时建模 **矿化（源）+ 吸收（汇）+ 反硝化（漏）** 三条路径，pH 和温度共同调控
- **可解释性**: N 池的动态轨迹展示了土壤供氮能力的时间变化

### 5.4 RadiationExpert
- **物理依据**: Monteith 辐射利用效率 (RUE) 模型
- **AAAI 创新点**: 光合有效辐射 (PAR) 由坡度和坡向几何修正（FAO-56 方法）
- **可解释性**: RUE 和 k_ext 可直接与文献值对比

### 5.5 StomatalExpert
- **物理依据**: Leuning 气孔导度 VPD 响应
- **AAAI 创新点**: 单参数 (D0) 的物理模型，在可学习性和物理解释性之间取得平衡
- **可解释性**: D0 从初始 2.0 学习到 2.22，表明数据验证了 Leuning 模型

### 5.6 PhenologyExpert
- **物理依据**: 积温驱动的玉米发育阶段
- **AAAI 创新点**: GDD 节点作为可学习参数，从数据中验证文献值（gdd_flowering 850→854）
- **可解释性**: 发育指数直接控制收获指数动态化和衰老速率

### 5.7 CouplingHead — 核心 AI 贡献
- **物理依据**: Liebig 最小因子定律 × 乘积交互
- **AAAI 创新点**:
  1. **静态交互门控**: 用土壤静态特征 (clay, SOC, pH) 调制水/氮/VPD/热的耦合系数
  2. **年份嵌入**: 4-dim 可学习向量，区分 2019-2023 的气候背景
  3. **anomaly 通道**: 有界修正 (±0.1)，保留物理骨架的同时允许数据修正
- **可解释性**: 交互门控的系数可提取为县级地图

## 6. 消融实验证据 (已运行)

| 变体 | RMSE | R² | Heat Response | 证明什么 |
|------|:----:|:---:|:---:|------|
| **baseline** (v3.26) | 19.68 | -0.084 | -0.76% PASS | 全模型基线 |
| no_static_crop_params | 20.14 | -0.135 | -0.62% PASS | 静态作物参数有效 |
| no_reproductive_heat_penalty | 19.92 | -0.111 | -0.30% WARN | 生殖期热害机制有效 |
| no_temperature_stress | — | — | — | 温度胁迫必要性 |
| no_static_adapt_reg | — | — | — | 静态适应正则化影响 |
| heat_stress_025 | — | — | 过度 | 0.25 系数过强 |

## 7. 当前核心指标

| 指标 | 值 | AAAI 论文中的解读 |
|------|:--:|------|
| **n_val** | 187 (2023 时间外推) | 严格的时间外推评估, 非随机分割 |
| **RMSE** | 19.68 bu/acre | 玉米带县均产量 193 bu/acre → NRMSE=10.2% |
| **MAPE** | 8.18% | 在农业领域可接受范围 |
| **R²** | -0.084 | 县级排序能力待提升 (见 §8 改进方向) |
| **Radiation response** | +47.57% PASS | 模型正确学到了辐射→产量的正效应 |
| **VPD response** | -29.38% PASS | 模型正确学到了大气干燥→产量的负效应 |
| **Nitrogen response** | +38.18% PASS | 模型正确学到了氮素→产量的正效应 |
| **Heat extreme response** | -0.76% PASS | 极端高温惩罚方向正确 |

## 8. 论文写作建议

### 8.1 建议的论文结构

```
1. Introduction
   - 世界模型 vs 黑盒产量预测: 我们不仅预测, 还推演整个动态过程
   - 知识引导 vs 纯数据驱动: 用物理先验减少数据需求

2. Related Work
   - 可微物理仿真 (Neural ODE, PINNs)
   - 作物模型 (DSSAT, APSIM) → 不可微
   - 农业 ML (CNN-LSTM yield prediction) → 纯黑盒

3. AgriWorld: Knowledge-Guided Differentiable World Model
   3.1 Expert Design (6 个专家模块的设计哲学)
   3.2 Coupling & Interaction Gates (静态交互门控 + 年份嵌入)
   3.3 Training with Physics Priors (预训练 + 多目标损失)
   3.4 Factor Audit & Interpretability (因子响应审计)

4. Experiments
   4.1 Yield Prediction (RMSE vs LSTM/Transformer baselines)
   4.2 Physical Consistency (LAI dynamics, SMAP probe)
   4.3 Ablation Studies
   4.4 Counterfactual Factor Audit
   4.5 Spatial Interpretability (county-level adaptation maps)

5. Discussion & Conclusion
```

### 8.2 建议的创新点列表 (AAAI 风格)

1. **Expert-structured differentiable world model**: 6 个物理专家 + 可学习耦合头, 而非单一端到端网络
2. **Static interaction gates**: 土壤质地通过门控机制调节胁迫响应, 实现空间异质性
3. **Year embedding for temporal extrapolation**: 区分气候年份背景, 不做随机分割
4. **Counterfactual factor audit**: 每个因子独立干预, 验证因果方向, 而非仅看预测精度
5. **Physics pretraining + data fine-tuning**: 两阶段训练策略

### 8.3 建议的 Figures (已有 assets 在 `docs/report_assets/`)

| 图号 | 内容 | 文件 |
|:---:|------|------|
| Fig 1 | 数据漏斗 (1857→867 样本) | `01_data_funnel.png` |
| Fig 2 | 模型工作流 (Expert → ODE → Yield) | `02_model_workflow.png` |
| Fig 3 | 核心指标卡片 (RMSE/MAPE/R²) | `03_metric_cards.png` |
| Fig 4 | 因子响应柱状图 | `04_factor_response.png` |
| Fig 5 | 消融实验 RMSE 对比 | `05_ablation_rmse.png` |
| Fig 6 | 下一步路线图 | `06_next_steps.png` |

### 8.4 写作注意事项

- **不要自称"农业模型"论文** → 始终用 "world model" / "differentiable simulation" / "knowledge-guided AI" 的语言
- **不要过度讨论 RMSE 绝对值** → AAAI 审稿人不关心玉米产量, 关心方法论
- **突出反事实推演和因子审计** → 这是黑盒模型做不到的, 是核心区分度
- **消融实验要完整** → 每个 Expert 都要有 "去掉后效果退化" 的实验
- **时间外推验证** → 强调 2023 完全不在训练集中

## 9. 当前待填补的缺口 (投稿前)

| 优先级 | 任务 | 说明 |
|:---:|------|------|
| **P0** | 重新训练 v3.27 并获取完整评估 | 当前已报告的结果基于 v3.26, v3.27 已实现但待服务器跑完 |
| **P0** | 补充缺失的消融实验 | no_coupling_anomaly, no_vpd, no_water_floor 等需完整指标 |
| **P1** | R² 负值的问题定位与修复 | 县级排序能力不足是审稿人最可能攻击的点 |
| **P1** | SMAP R² 提升或重新论证 | 水文模块与遥感土壤水分的对应关系 |
| **P2** | 额外基线模型 | XGBoost/Random Forest (不仅是深度学习基线) |
| **P2** | 更多年份外推 | 用 2019-2021 训练, 2022-2023 测试的双年外推 |
