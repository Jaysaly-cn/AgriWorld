"""Shared quality checks for serialized AgriWorld weather forcing."""

import numpy as np


def forcing_quality_issue(stress_forcing, doy=None, planting_doy=120):
    """Return a rejection reason, or None when weather forcing is usable."""
    forcing = np.asarray(stress_forcing, dtype=np.float32)
    if forcing.ndim != 2 or forcing.shape[0] < 300 or forcing.shape[1] < 9:
        return "bad_forcing_shape"

    required = forcing[:, [0, 1, 3, 6, 7]]
    if np.isfinite(required).mean() < 0.95:
        return "weather_missing"

    tmean = forcing[:, 6]
    finite_t = tmean[np.isfinite(tmean)]
    if finite_t.size < 300:
        return "temperature_missing"
    if np.ptp(finite_t) < 5.0 or np.std(finite_t) < 2.0:
        return "temperature_flat"

    par = forcing[:, 3]
    if np.sum(np.isfinite(par) & (par > 0.1)) < 200:
        return "radiation_missing"

    if doy is None:
        doy = np.arange(1, forcing.shape[0] + 1)
    doy = np.asarray(doy)
    daily_gdd = np.maximum(np.nan_to_num(tmean, nan=0.0) - 10.0, 0.0)
    final_gdd = float(np.sum(np.where(doy >= planting_doy, daily_gdd, 0.0)))
    if final_gdd < 500.0:
        return "gdd_too_low"

    return None

