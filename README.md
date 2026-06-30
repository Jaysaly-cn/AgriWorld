# AgriWorld

AgriWorld is a physics-guided differentiable agricultural world model for county-level crop systems. The current development mainline is `agriworld-v3.28`.

V3.28 makes structured county-level crop adaptation bias-free, so learned factors must come from static features rather than a global output bias.

## Current Status

| Item | Status |
|---|---|
| Current schema | `agriworld-v3.28` |
| Stable baseline | `agriworld-v3.27` |
| Dataset | 2019-2023 merged data |
| Active crop subset | 867 corn county-year samples |
| Split | train 2019-2022, validate 2023 |
| V3.23 RMSE | 21.53 bu/ac |
| V3.23 NRMSE | 11.14% |
| V3.23 Corn MAPE | 8.71% |

## Directory Layout

```text
AgriWorld/
├── agriworld/      # Core package: model, ODE, dataset, losses, validation, config
├── scripts/        # Train/evaluate/ablation/data-pipeline implementations
├── benchmarks/     # LSTM/Transformer/MLP baselines
├── tests/          # Unit tests
├── docs/           # Model, data, ablation, report and visualization docs
├── results/        # Evaluation, ablation and figure outputs
├── saved_models/   # Checkpoints
├── train.py        # Compatibility wrapper for scripts.train
├── evaluate.py     # Compatibility wrapper for scripts.evaluate
└── requirements.txt
```

## Server Paths

```text
Project: /data4/Agri/yukaijie/AgriWorld/AgriWorld/Newest_version
Data:    /data4/Agri/yukaijie/AgriWorld/AgriWorld/AgriWorld_Master
Cache:   /data4/Agri/yukaijie/AgriWorld/AgriWorld/AgriWorld_Master/cache_v
Merged:  /data4/Agri/yukaijie/AgriWorld/AgriWorld/AgriWorld_Master/national_ode_tensors_v2_merged.pkl
```

Paths are configured in `agriworld/paths.py` and `agriworld/config.py`, and can be overridden with environment variables.

## Quick Commands

Compatibility wrappers:

```bash
python train.py
python evaluate.py
python factor_response.py
python ablation.py --variant baseline --epochs 250
```

Module entry points:

```bash
python -m scripts.train
python -m scripts.evaluate
python -m scripts.factor_response
python -m scripts.ablation --variant baseline --epochs 250
```

Data pipeline:

```bash
python run_pipeline.py
python merge_years.py
python audit_data.py
```

## V3.28 Additions

- Yield residual head is disabled by default and retained as an ablation option.
- Static-conditioned interaction gates for water, nitrogen, VPD and heat responses.
- Structured county-level crop parameters from static features: HI factor, yield-scale factor and heat sensitivity.
- Static adaptation regularization to keep county-level factors interpretable and reduce memorization.
- Bias-free static crop adapter output to prevent county adaptation from collapsing into global offsets.
- Reproductive-stage heat exposure penalty on harvest index.
- Evaluation exports county-level HI factor, yield factor, heat sensitivity and reproductive heat factor for AAAI-ready interpretability figures.
- Stricter factor-response PASS criteria.
- Training-history CSV export for learning-curve visualization.
- Per-sample evaluation CSV and trajectory export for paper figures.
- Reduced ablation matrix to control training cost.
- Current tuning defaults: `USE_YIELD_RESIDUAL=0`, `USE_STATIC_CROP_PARAMS=1`, `USE_REPRODUCTIVE_HEAT_PENALTY=1`, `W_STATIC_ADAPT=0.15`.

## Documentation

| File | Purpose |
|---|---|
| `docs/MODEL_V3.md` | Model specification |
| `docs/data_instrument.md` | Data pipeline and tensor contract |
| `docs/ABLATION.md` | V3.28 ablation strategy |
| `docs/VISUALIZATION_TODO.md` | AAAI visualization plan and data exports |
| `docs/报告.md` | Progress report for advisor updates |

## Next Work

1. Train V3.28 baseline and compare against V3.27.
2. Run the reduced V3.28 ablation matrix.
3. Inspect R2/ranking improvement, not only RMSE.
4. Use the new exported tables to generate AAAI-ready visualizations.
