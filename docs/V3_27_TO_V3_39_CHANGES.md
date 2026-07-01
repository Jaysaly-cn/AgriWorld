# AgriWorld V3.27 -> V3.39 更新对比说明

更新日期：2026-07-01

本文档记录当前主线版本 `agriworld-v3.39` 相对 V3.27 handoff 版本的主要变化。V3.27 是此前打包交接给协作者的版本；V3.39 是当前准备进入论文撰写阶段的多作物空间外推版本。

## 1. 总体变化

V3.27 的核心是知识引导的可微分农业世界模型，重点包括：

- ODE 状态模拟；
- water / nitrogen / radiation / stomatal / phenology / coupling 等专家模块；
- 静态作物参数适应；
- 生殖期高温惩罚；
- 以玉米为主的 yield / LAI / physical consistency 训练与评估。

V3.39 在此基础上完成了五类关键升级：

1. 从玉米单作物主线扩展到玉米 + 大豆多作物建模。
2. 从共享产量形成参数改为作物条件化产量形成。
3. 从普通空间损失改为作物感知的空间对比与州组偏差损失。
4. 从单一高温生殖期惩罚扩展到作物窗口期综合胁迫。
5. 从普通训练日志扩展到面向论文的 evaluate / ablation / factor response / visualization 证据链。

## 2. 数据层变化

### 2.1 多作物标签

V3.27 主要使用玉米样本。V3.39 使用当前重建后的多作物 merged tensor：

```text
Source samples: 1327
Accepted samples: 1198
Dropped samples: 129 insufficient LAI
Corn samples: 794
Soybean samples: 404
Train / Val: 958 / 240
Validation Corn / Soybean: 155 / 85
```

新增脚本：

```text
scripts/relabel_multicrop_yield.py
agriworld/nass.py
```

用途：

- 从 NASS QuickStats 获取作物特异县级产量标签；
- 支持 Corn 与 Soybean 的不同 bu/ac 到 t/ha 转换；
- 解决此前非玉米样本不能直接用玉米 yield 标签的问题。

### 2.2 单位转换

`agriworld/units.py` 从固定玉米转换扩展为按作物转换：

```text
Corn:    56 lb/bu
Soybean: 60 lb/bu
```

影响：

- dataset target 构造；
- evaluate 输出；
- visualize 输出；
- per-crop yield metrics。

## 3. 模型结构变化

### 3.1 作物条件化产量形成

V3.27/V3.38 中，`HI / yield_scale / yield_year_trend` 基本被所有作物共享。V3.39 改为按 crop code 索引：

```text
crop-conditioned HI
crop-conditioned yield_scale
crop-conditioned yield_year_trend
```

当前学到的参数：

| Crop | HI | Yield Scale | Year Trend |
|---|---:|---:|---:|
| Corn | 0.538 | 1.058 | 0.032 |
| Soybean | 0.345 | 0.584 | 0.014 |

这是当前版本最重要的算法升级。它修复了 V3.38 中玉米系统性低估、大豆系统性高估的问题。

### 3.2 作物窗口期胁迫模块

V3.27 保留生殖期高温惩罚。V3.39 新增并稳定使用：

```text
agriworld/window_stress.py
CropWindowStressExpert
```

窗口包括：

```text
establishment
vegetative
reproductive
grain_fill
```

因子包括：

```text
water
vpd
heat
cold
radiation
nitrogen
```

V3.39 已经将窗口机制扩展到 Corn 与 Soybean 的不同先验敏感性。

### 3.3 静态空间适应

V3.27 已有静态适应思想。V3.39 保留并强化：

- state embedding 默认开启；
- county embedding 默认关闭，避免小样本县级记忆；
- 静态作物参数头保留 crop embedding；
- 输出 bounded 的 `county_hi_factor / county_yield_factor / county_heat_sensitivity`。

## 4. Loss 与训练变化

### 4.1 作物感知空间损失

V3.39 中：

```text
compute_spatial_contrast_loss(..., crop_id=...)
compute_group_bias_loss(..., crop_id=...)
```

空间对比和州组偏差只在同一作物内部比较，避免把玉米与大豆的自然产量尺度差异误当成县域差异。

新增配置：

```text
USE_CROP_AWARE_SPATIAL_LOSS
```

### 4.2 作物条件化产量消融开关

新增配置：

```text
USE_CROP_CONDITIONED_YIELD
```

用于关键消融：

```text
no_crop_conditioned_yield
```

该消融证明作物条件化产量形成是当前版本最强贡献。

### 4.3 训练日志增强

训练历史现在记录：

```text
HI
HI_corn
HI_soybean
yield_scale
yield_scale_corn
yield_scale_soybean
yield_year_trend
yield_year_trend_corn
yield_year_trend_soybean
spatial_contrast_loss
spatial_group_bias_loss
window_stress_loss
```

控制台日志也显示：

```text
HI: mean (Corn/Soybean)
YS: mean (Corn/Soybean)
YT: mean (Corn/Soybean)
```

## 5. Evaluate / Factor Response / Ablation 变化

### 5.1 Evaluate

`scripts/evaluate.py` 新增：

- per-crop yield metrics；
- crop-specific yield parameters；
- factor responses by crop；
- state residual table；
- county residual table；
- sample-level stress window fields。

JSON 中新增关键字段：

```text
crop_yield_parameters
factor_responses_by_crop
spatial_residuals
window_stress
county_adaptation
```

### 5.2 Factor Response

`scripts/factor_response.py` 从整体响应扩展为整体 + 分作物响应。

新增输出：

```text
factor_response_v3_39_by_crop.json
factor_response_v3_39_by_crop.csv
```

当前分作物响应：

| Crop | Radiation | VPD | Nitrogen | Window Heat | Window VPD | Window Radiation | Window Water |
|---|---:|---:|---:|---:|---:|---:|---:|
| Corn | +45.58% | -24.36% | +34.67% | -2.88% | -2.40% | +6.54% | +1.37% |
| Soybean | +46.05% | -25.23% | +35.07% | -2.62% | -2.35% | +6.48% | +0.68% |

### 5.3 Ablation

`scripts/ablation.py` 默认不再运行完整旧矩阵，而是运行论文关键 compact ablation：

```text
no_crop_conditioned_yield
no_crop_aware_spatial_loss
no_window_stress
no_spatial_group_bias
```

完整旧矩阵仍可通过：

```bash
python ablation.py --all --epochs 100
```

## 6. 当前核心结果

### 6.1 主结果

| Metric | Value |
|---|---:|
| Overall RMSE | 16.31 bu/ac |
| Overall NRMSE | 11.33% |
| Overall R2 | 0.939 |
| Corn RMSE | 19.51 bu/ac |
| Corn MAPE | 8.48% |
| Corn bias | -3.30 bu/ac |
| Soybean RMSE | 7.56 bu/ac |
| Soybean MAPE | 10.39% |
| Soybean bias | -0.20 bu/ac |

### 6.2 V3.38 -> V3.39 对比

| Metric | V3.38 shared yield | V3.39 crop-conditioned yield |
|---|---:|---:|
| Overall RMSE | 33.59 | 16.31 |
| Overall R2 | 0.743 | 0.939 |
| Corn RMSE | 37.68 | 19.51 |
| Corn bias | -32.13 | -3.30 |
| Soybean RMSE | 24.42 | 7.56 |
| Soybean bias | +22.85 | -0.20 |

### 6.3 Compact Ablation

| Variant | RMSE | R2 | Delta RMSE | Delta reliable county RMSE | Conclusion |
|---|---:|---:|---:|---:|---|
| Mainline V3.39 | 16.31 | 0.939 | 0.00 | 0.00 | keep |
| No crop-conditioned yield | 35.54 | 0.713 | +19.22 | +19.27 | essential |
| No crop-aware spatial loss | 17.92 | 0.927 | +1.61 | +1.82 | useful |
| No window stress | 19.92 | 0.910 | +3.60 | +3.72 | important |
| No spatial group bias | 18.50 | 0.922 | +2.19 | +2.20 | important |

## 7. 论文包装角度的新增价值

相对 V3.27，V3.39 更适合包装为：

1. Crop-conditioned physics-guided world model  
   不同作物共享物理框架，但不共享关键产量形成参数。

2. Crop-aware spatial extrapolation  
   县/州外推目标保留，但空间对比在同作物内部进行。

3. Stage/window-aware stress expert  
   环境响应不只看全年平均，而是进入作物窗口期。

4. Counterfactual response audit  
   不只报告 RMSE，还报告 radiation、VPD、nitrogen、window stress 等响应方向。

5. Interpretable multicrop parameters  
   Corn 与 Soybean 的 HI、yield scale、year trend 可直接解释。

## 8. 相对 V3.27 的变动文件

新增文件：

```text
COLLABORATOR_HANDOFF.md
agriworld/nass.py
agriworld/window_stress.py
docs/AI_AGENT_BRIEF.md
docs/V3_27_TO_V3_36_CHANGES.md
docs/V3_27_TO_V3_39_CHANGES.md
docs/archive/config_v1_backup.py
paper_experiment_records/CURRENT_EXPERIMENT_SUMMARY.md
paper_experiment_records/IMPORTANT_EXPERIMENT_RESULTS.md
paper_experiment_records/SUPPLEMENTARY_EXPERIMENT_RUNBOOK.md
paper_experiment_records/V3_38_MULTICROP_SPATIAL_EVAL.md
paper_experiment_records/V3_39_CROP_CONDITIONED_YIELD.md
paper_experiment_records/V3_39_FINAL_CHANGELOG.md
paper_experiment_records/VISUALIZATION_PLAN.md
scripts/make_paper_figures.py
scripts/relabel_multicrop_yield.py
```

修改文件：

```text
README.md
agriworld/config.py
agriworld/dataset.py
agriworld/losses.py
agriworld/paths.py
agriworld/simulator.py
agriworld/splits.py
agriworld/units.py
agriworld/validate.py
docs/ABLATION.md
scripts/ablation.py
scripts/audit_data.py
scripts/data_pipeline.py
scripts/evaluate.py
scripts/factor_response.py
scripts/train.py
scripts/visualize.py
```

新增论文图资产：

```text
paper_experiment_records/figures/01_data_composition.png
paper_experiment_records/figures/02_training_curves.png
paper_experiment_records/figures/03_v38_v39_rmse.png
paper_experiment_records/figures/04_crop_yield_parameters.png
paper_experiment_records/figures/05_state_residuals.png
paper_experiment_records/figures/06_factor_response.png
paper_experiment_records/figures/07_ablation_screening.png
paper_experiment_records/figures/08_factor_response_by_crop.png
```

## 9. 是否还需要传统 ML / 深度学习对比

建议补，但不应阻塞当前文章草稿。

最低成本推荐：

```text
Random Forest or XGBoost
MLP
optional LSTM / Transformer
```

输入建议：

- crop code；
- state/county ID；
- soil/static features；
- seasonal weather aggregates；
- GDD / radiation / precipitation / VPD summaries。

表格建议：

| Model | RMSE | R2 | Corn RMSE | Soybean RMSE | Factor response | Interpretable states |
|---|---:|---:|---:|---:|---|---|
| RF / XGBoost | TBD | TBD | TBD | TBD | no | no |
| MLP | TBD | TBD | TBD | TBD | weak | no |
| AgriWorld V3.39 | 16.31 | 0.939 | 19.51 | 7.56 | yes | yes |

AAAI 审稿视角下，这个 baseline 表会增强说服力；但当前 V3.39 的内部消融和可解释响应已经足够支撑开始论文写作。

