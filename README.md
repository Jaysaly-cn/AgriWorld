# AgriWorld

AgriWorld is a physics-guided differentiable agricultural world model for county-level crop systems. The current mainline is `agriworld-v3.39`, a multicrop Corn + Soybean version with spatial county validation.

## Current Status

| Item | Value |
|---|---|
| Current schema | `agriworld-v3.39` |
| Main task | multicrop county-level yield and growth simulation |
| Crops | Corn, Soybean |
| Source samples | 1327 |
| Accepted samples | 1198 |
| Dropped samples | 129 insufficient LAI |
| Train / Val | 958 / 240 |
| Split | spatial county split |
| Overall RMSE | 16.31 bu/ac |
| Overall R2 | 0.939 |
| Corn RMSE | 19.51 bu/ac |
| Soybean RMSE | 7.56 bu/ac |

## Main V3.39 Ideas

- Knowledge-guided differentiable ODE simulator for LAI, biomass, nitrogen pool, and soil water.
- Modular experts for water, nitrogen, radiation, stomatal/VPD, phenology, and coupling.
- Crop-conditioned yield formation: Corn and Soybean have separate learned `HI`, `yield_scale`, and `yield_year_trend`.
- Crop-aware spatial contrast and state-group bias losses.
- Crop-window stress expert for establishment, vegetative, reproductive, and grain-fill windows.
- Counterfactual factor-response audit for radiation, VPD, nitrogen, heat, water, and crop-window stresses.

## Directory Layout

```text
AgriWorld/
|-- agriworld/                  # Core package: model, ODE, dataset, losses, validation, config
|-- scripts/                    # Train/evaluate/ablation/data-pipeline implementations
|-- benchmarks/                 # Optional baseline model area
|-- docs/                       # Model, data, ablation, report and visualization docs
|-- paper_experiment_records/   # Paper-facing experiment summaries and figures
|-- results/                    # Evaluation, ablation, factor-response outputs
|-- saved_models/               # Checkpoints
|-- train.py                    # Compatibility wrapper for scripts.train
|-- evaluate.py                 # Compatibility wrapper for scripts.evaluate
|-- ablation.py                 # Compatibility wrapper for scripts.ablation
|-- factor_response.py          # Compatibility wrapper for scripts.factor_response
`-- requirements.txt
```

## Server Paths

```text
Project: /data4/Agri/yukaijie/AgriWorld/AgriWorld/Newest_version
Data:    /data4/Agri/yukaijie/AgriWorld/AgriWorld/AgriWorld_Master
Cache:   /data4/Agri/yukaijie/AgriWorld/AgriWorld/AgriWorld_Master/cache_v
Merged:  /data4/Agri/yukaijie/AgriWorld/AgriWorld/AgriWorld_Master/national_ode_tensors_v3_multicrop.pkl
```

Paths are configured in `agriworld/paths.py` and `agriworld/config.py`, and can be overridden with environment variables.

## Quick Commands

```bash
python train.py
python evaluate.py
python factor_response.py --out-prefix factor_response_v3_39
python ablation.py --epochs 100
python scripts/make_paper_figures.py
```

Default compact ablations:

```text
no_crop_conditioned_yield
no_crop_aware_spatial_loss
no_window_stress
no_spatial_group_bias
```

Run one variant:

```bash
python ablation.py --variant no_crop_conditioned_yield --epochs 100
```

Run all registered variants only if needed:

```bash
python ablation.py --all --epochs 100
```

## Key Results

### V3.38 to V3.39

| Metric | V3.38 shared yield | V3.39 crop-conditioned yield |
|---|---:|---:|
| Overall RMSE | 33.59 | 16.31 |
| Overall R2 | 0.743 | 0.939 |
| Corn RMSE | 37.68 | 19.51 |
| Corn bias | -32.13 | -3.30 |
| Soybean RMSE | 24.42 | 7.56 |
| Soybean bias | +22.85 | -0.20 |

### Compact Ablation

| Variant | RMSE | R2 | Delta RMSE |
|---|---:|---:|---:|
| Mainline V3.39 | 16.31 | 0.939 | 0.00 |
| No crop-conditioned yield | 35.54 | 0.713 | +19.22 |
| No crop-aware spatial loss | 17.92 | 0.927 | +1.61 |
| No window stress | 19.92 | 0.910 | +3.60 |
| No spatial group bias | 18.50 | 0.922 | +2.19 |

## Documentation

| File | Purpose |
|---|---|
| `docs/MODEL_V3.md` | Current model specification |
| `docs/data_instrument.md` | Data pipeline and tensor contract |
| `docs/ABLATION.md` | V3.39 ablation strategy |
| `docs/V3_27_TO_V3_39_CHANGES.md` | Difference from V3.27 handoff version |
| `paper_experiment_records/CURRENT_EXPERIMENT_SUMMARY.md` | Paper-facing experiment summary |
| `paper_experiment_records/V3_39_FINAL_CHANGELOG.md` | Final V3.39 experiment changelog |
| `paper_experiment_records/VISUALIZATION_PLAN.md` | Paper visualization plan |

## Paper Stage

The V3.39 experiment set is strong enough to start paper writing. A lightweight Random Forest/XGBoost or MLP baseline is still recommended before submission to answer reviewer questions about conventional ML comparisons.
