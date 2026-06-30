# AAAI Visualization TODO

These visualizations are deferred until the model is finalized, but V3.24 keeps the required data export hooks in training and evaluation code.

## V3.24 Data Exports

- `results/training_history_<version>.csv`
  - epoch-level train/validation loss
  - LAI/yield/state/canopy/anomaly components
  - RUE, k_ext, D0, HI, yield scale, year trend
- `results/eval_<checkpoint>_samples.csv`
  - per-sample prediction, target, residual, year/state/county
  - peak predicted/observed LAI
  - final biomass and mean stress factors
- `results/eval_<checkpoint>_trajectories.pt`
  - selected validation trajectories for LAI, biomass, N, soil water
  - forcing, observed LAI, masks, prediction and target
- `results/factor_response.{json,csv}`
  - factor response mean/median/status/epsilon

## Main Paper Figures To Produce Later

1. Temporal generalization:
   loss curves, prediction-target scatter, residual by year/state.

2. Counterfactual factor response:
   bar plot, county-level distributions, dose-response curves, interaction heatmaps.

3. Interpretable trajectories:
   LAI vs Sentinel-2, biomass accumulation, soil water with precipitation, stress factors.

4. Ablation beyond accuracy:
   RMSE comparison, factor-response validity, physical consistency pass count.

5. Spatial diagnostics:
   target yield map, predicted yield map, residual map, VPD/water sensitivity map.

## Cost Control

Ablation is the most expensive visualization source. V3.24 keeps only the minimum default ablation set:

```text
baseline
no_yield_residual
no_static_interaction_gates
no_temperature_stress
heat_stress_025
```

Full historical ablations should be run only for final paper experiments.
