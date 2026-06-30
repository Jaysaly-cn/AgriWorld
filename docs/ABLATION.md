# AgriWorld V3.27 Ablation Plan

V3.27 keeps the free yield residual head disabled by default and regularizes structured county-level crop parameters learned from static features.

- `static interaction gates`: small static-conditioned gates for water, nitrogen, VPD and heat responses.
- `YieldResidualHead`: retained as an optional ablation mechanism, disabled in the mainline.
- `StaticCropParameterHead`: static-feature adjustments for harvest index, yield scale and heat sensitivity.
- `static adaptation regularization`: keeps county-level factors near the physical baseline unless data support deviations.
- `reproductive heat penalty`: flowering-to-maturity heat exposure reduces harvest index.

Because ablation training is expensive, V3.27 defaults to a minimal ablation matrix.
The current mainline defaults use `USE_YIELD_RESIDUAL=0` and
`USE_STATIC_CROP_PARAMS=1` with `W_STATIC_ADAPT=0.15`.

## Minimal V3.27 Ablation Matrix

| Variant | Purpose |
|---|---|
| `baseline` | V3.27 mainline: regularized static crop params + static interaction gates |
| `no_static_crop_params` | Tests whether structured county-level crop parameters help |
| `no_static_adapt_reg` | Tests whether regularization prevents memorization |
| `no_reproductive_heat_penalty` | Tests whether reproductive heat damage is necessary |
| `yield_residual_on` | Tests whether the residual head improves county-level ranking |
| `no_static_interaction_gates` | Tests whether static-conditioned factor interactions help |
| `no_temperature_stress` | Checks whether extreme-heat stress remains necessary |
| `heat_stress_025` | Checks whether stronger heat penalty is too strong |

The default `scripts/ablation.py` matrix is intentionally limited to these variants.
New ablation summaries include `schema`, `model_version` and `description` columns.
If `ablation_results.csv` still contains historical variants such as `lstm_res` or
`no_year_trend`, treat it as a stale full-matrix result rather than the current V3.27
minimal ablation.

## Historical V3.23 Findings Kept As Prior Evidence

| Historical variant | Finding |
|---|---|
| `hard_temperature_stress` | Strongly worsened yield accuracy; hard Wang-Engel stress should not be the mainline. |
| `no_water_floor` | Collapsed physically and numerically; water bounds are required. |
| `no_vpd_stress` | Similar RMSE but invalid VPD response; not physically acceptable. |
| `no_nitrogen_stress` | Weakens nitrogen response audit; nitrogen mechanism should stay. |
| `no_year_trend` | Worsened temporal extrapolation; year trend should stay until a better temporal mechanism replaces it. |

These variants should not be rerun every iteration. Re-run them only for a final paper table or if a later structural change directly touches the corresponding mechanism.

## V3.27 Success Criteria

Primary metrics:

- RMSE / NRMSE / MAPE on 2023 holdout.
- R2 or ranking improvement on county-level validation samples.

Mechanism metrics:

- VPD response remains negative.
- Nitrogen response remains positive.
- Radiation response remains positive.
- Extreme-heat response remains negative under the stricter epsilon rule.
- Physical consistency checks remain PASS.

Visualization artifacts:

- `training_history_<version>.csv`
- `eval_<checkpoint>_samples.csv`
- `eval_<checkpoint>_trajectories.pt`
- `factor_response.{json,csv}`

## Recommended Command

```bash
python ablation.py --epochs 250
```

For a cheaper smoke test:

```bash
python ablation.py --variant baseline --epochs 20
python ablation.py --variant no_static_crop_params --epochs 20
```
