# AgriWorld V3.27 -> V3.36 改动说明

本文档记录当前主线代码相对 V3.27 handoff 版本的主要变化。V3.27 是此前打包给协作者的版本，当前代码 schema 为 `agriworld-v3.36`。

## 1. 总体变化

V3.27 的核心是知识引导的可微分农业世界模型，重点包括 ODE 状态模拟、六类生物物理 expert、静态作物参数适应、reproductive heat penalty 和多目标物理正则。

V3.36 在此基础上主要做了四类升级：

1. 从纵向年份验证扩展到横向空间验证，支持 `spatial_county` / `spatial_state`。
2. 加入县域/州域空间信息传递与空间对比学习，提升跨地区外推评估能力。
3. 新增 corn 生育窗口胁迫模块，把胁迫响应从单一 reproductive heat 扩展到多个作物窗口。
4. 精简并强化 evaluate / ablation / factor_response，使输出更适合结果评估和论文包装。

## 2. 模型结构变化

### 2.1 生育窗口胁迫模块

新增文件：

```text
agriworld/window_stress.py
```

新增模块：

```text
CornWindowStressExpert
```

该模块将玉米生长过程划分为四个可解释窗口：

```text
establishment
vegetative
reproductive
grain_fill
```

每个窗口审计六类环境胁迫：

```text
water
vpd
heat
cold
radiation
nitrogen
```

当前实现是可解释的加权胁迫公式，而不是大黑盒 MLP。输出为一个 bounded yield multiplier：

```text
yield = physical_yield * residual_factor * window_stress_factor
```

V3.36 进一步把窗口进度改为“出苗后作物发育进度”，并用 emergence gate 避免把出苗前冷天误计入 establishment stress。

### 2.2 静态空间适应保留，但县级 ID 不强行记忆

`StaticCropParameterHead` 继续输出：

```text
county_hi_factor
county_yield_factor
county_heat_sensitivity
```

但默认策略是：

```text
USE_STATE_EMBEDDINGS = 1
USE_COUNTY_EMBEDDINGS = 0
```

也就是说，县级差异主要通过静态特征和空间验证/对比学习体现，不强迫模型用 county embedding 记忆每个县。

### 2.3 Reproductive heat penalty 保留

原有 reproductive-stage heat penalty 仍保留：

```text
USE_REPRODUCTIVE_HEAT_PENALTY = 1
```

它负责 flowering-to-maturity 高温对 HI 的影响；新增 window stress 则覆盖更宽泛的窗口期综合胁迫。

## 3. 数据与划分变化

### 3.1 Dataset 增加空间标识

`AgriTensorDataset` 现在为样本保留：

```text
sample_id
state
county
state_id
county_id
```

这些字段用于空间划分、空间对比 loss、evaluate CSV 导出和可视化。

### 3.2 Split 支持空间验证

`agriworld/splits.py` 新增/强化：

```text
spatial_county
spatial_state
```

当前推荐主线：

```bash
export AGRI_SPLIT_MODE=spatial_county
```

这使模型不仅能做年份间外推，也能做地区间横向外推。

## 4. Loss 与训练变化

### 4.1 新增空间对比 loss

新增函数：

```text
compute_spatial_contrast_loss
```

目标是在同一 batch 内对不同 county 的产量差异进行 pairwise log-yield contrast，使模型学习地区间相对差异，而不仅拟合均值。

相关配置：

```text
USE_SPATIAL_CONTRAST = 1
W_SPATIAL_CONTRAST = 0.40
SPATIAL_CONTRAST_MIN_GAP = 0.15
```

### 4.2 新增窗口胁迫正则

新增函数：

```text
compute_window_stress_regularization
```

用于限制 window stress 不退化为隐藏的全局 yield offset。

相关配置：

```text
USE_WINDOW_STRESS = 1
W_WINDOW_STRESS = 0.08
WINDOW_STRESS_MAX_REDUCTION = 0.22
```

### 4.3 训练日志增加空间与窗口项

训练历史和日志新增：

```text
spatial_contrast_loss
window_stress_loss
```

控制台日志新增：

```text
Sp
Win
```

## 5. Evaluate / Factor Response / Ablation 变化

### 5.1 Evaluate 输出新增窗口解释字段

`scripts/evaluate.py` 现在在 sample CSV 和 JSON summary 中导出：

```text
window_stress_factor
stress_establishment
stress_vegetative
stress_reproductive
stress_grain_fill
dominant_stress_window
dominant_stress_factor
```

这为后续论文图提供了窗口期解释入口。

### 5.2 Factor response 增加窗口扰动

在原有因子响应基础上新增：

```text
window_heat
window_vpd
window_radiation
window_water
```

这些扰动仅作用于敏感窗口，当前使用出苗后进度的 reproductive window。

### 5.3 Ablation 精简为关键项

当前保留和新增的重点消融包括：

```text
no_window_stress
reproductive_only_window_stress
no_spatial_contrast
no_state_embedding
spatial_embedding_on
no_static_crop_params
no_reproductive_heat_penalty
```

其中最推荐优先跑：

```bash
python ablation.py --variant no_window_stress --epochs 100
python ablation.py --variant reproductive_only_window_stress --epochs 100
python ablation.py --variant no_spatial_contrast --epochs 100
```

## 6. 最新结果摘要

注意：V3.27 handoff 文档中的旧结果主要是 temporal/year split；当前最新评估是 `spatial_county` 横向空间验证，二者不完全可直接横向比较。

当前 V3.35/V3.36 前置结果，主线 checkpoint：

```text
eval_agriworld_phys_spatial_best.json
n_val = 174
RMSE = 20.95 bu/ac
NRMSE = 10.99%
R2 = -0.069
```

关键消融：

```text
mainline                    RMSE = 20.95 bu/ac
no_window_stress            RMSE = 22.93 bu/ac
reproductive_only_window    RMSE = 22.60 bu/ac
no_spatial_contrast         RMSE = 22.55 bu/ac
```

结论：

1. 全窗口 window stress 是正贡献。
2. 只保留 reproductive window 不如全窗口。
3. spatial contrast 是正贡献。
4. state adaptation 仍是正贡献。

窗口因子响应：

```text
window_heat       -3.18% PASS
window_vpd        -2.35% PASS
window_radiation  +6.64% PASS
window_water      +0.65% PASS
```

全季节 `heat_extreme` 仍偏弱或接近阈值，但窗口内高温响应已经成立。这说明后续论文中更适合强调“stage/window-aware stress response”，而不是泛化为全年任意高温响应。

## 7. AAAI 包装角度的新增价值

相对 V3.27，当前版本更适合包装为：

1. 空间外推世界模型  
   从单纯年份验证扩展到 county/state holdout，强调横向地区泛化。

2. 机制可解释的窗口胁迫建模  
   生育窗口不是黑盒 attention，而是以 crop calendar / phenology progress 驱动的可解释 stress expert。

3. Counterfactual faithfulness  
   factor_response 不只看全年因子，也能看敏感窗口内的高温、VPD、辐射和水分响应。

4. Knowledge-guided MoE + structured adaptation  
   expert 分工仍保持物理语义，空间适应通过 bounded physiological factors 和 contrastive objective 实现。

## 8. 相对 V3.27 的变动文件

```text
agriworld/config.py
agriworld/dataset.py
agriworld/losses.py
agriworld/simulator.py
agriworld/splits.py
agriworld/validate.py
agriworld/window_stress.py
scripts/ablation.py
scripts/evaluate.py
scripts/factor_response.py
scripts/train.py
scripts/visualize.py
docs/ABLATION.md
docs/AI_AGENT_BRIEF.md
README.md
docs/V3_27_TO_V3_36_CHANGES.md
```

## 9. 当前建议

下一步建议先在服务器跑 V3.36：

```bash
export AGRI_SPLIT_MODE=spatial_county
python train.py
python evaluate.py
python factor_response.py
python ablation.py --variant no_window_stress --epochs 100
python ablation.py --variant reproductive_only_window_stress --epochs 100
python ablation.py --variant no_spatial_contrast --epochs 100
```

重点观察：

1. V3.36 的 window stress dominant window 是否不再被出苗前 cold 过度主导。
2. `window_heat/window_vpd/window_radiation/window_water` 是否仍保持 PASS。
3. `no_window_stress` 和 `no_spatial_contrast` 是否继续弱于主线。
4. 州别残差，尤其 IN 的正 bias 是否需要后续单独处理。

