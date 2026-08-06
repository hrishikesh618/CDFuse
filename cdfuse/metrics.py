"""Cell-wise comparison metrics.

Every metric is computed independently at each latitude/longitude cell over
the time steps the two datasets share. Cells are masked (returned as NaN)
where there are too few valid pairs, or where the metric's denominator is
undefined, rather than being reported as a misleading finite number.

Terminology: ``candidate`` is the dataset being evaluated (a model,
reanalysis, satellite product, or similar) and ``reference`` is the
benchmark it is evaluated against. Metrics are not symmetric, so the order
matters.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import xarray as xr

TIME = "time"


def _paired(
    candidate: xr.DataArray, reference: xr.DataArray
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """Mask both arrays to time steps where *both* have finite values.

    Returns the masked pair plus the per-cell count of valid pairs, so each
    metric can decide how many pairs it needs to be meaningful.
    """
    valid = np.isfinite(candidate) & np.isfinite(reference)
    return candidate.where(valid), reference.where(valid), valid.sum(TIME)


def compute_correlation(candidate: xr.DataArray, reference: xr.DataArray) -> xr.DataArray:
    """Pearson correlation through time. Needs at least two valid pairs."""
    candidate, reference, pairs = _paired(candidate, reference)
    result = xr.corr(candidate, reference, dim=TIME)
    return result.where(pairs >= 2)


def compute_nse(candidate: xr.DataArray, reference: xr.DataArray) -> xr.DataArray:
    """Nash-Sutcliffe Efficiency. Undefined where the reference is constant."""
    candidate, reference, pairs = _paired(candidate, reference)
    numerator = ((candidate - reference) ** 2).sum(TIME, skipna=True)
    denominator = ((reference - reference.mean(TIME, skipna=True)) ** 2).sum(TIME, skipna=True)
    result = 1 - numerator / denominator
    return result.where((pairs >= 2) & (denominator != 0))


def compute_kge(candidate: xr.DataArray, reference: xr.DataArray) -> xr.DataArray:
    """Kling-Gupta Efficiency (Gupta et al., 2009).

    Undefined where the reference has zero standard deviation (alpha term)
    or zero mean (beta term).
    """
    candidate, reference, pairs = _paired(candidate, reference)
    correlation = xr.corr(candidate, reference, dim=TIME)
    reference_std = reference.std(TIME, skipna=True)
    reference_mean = reference.mean(TIME, skipna=True)
    alpha = candidate.std(TIME, skipna=True) / reference_std
    beta = candidate.mean(TIME, skipna=True) / reference_mean
    result = 1 - np.sqrt((correlation - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    return result.where((pairs >= 2) & (reference_std != 0) & (reference_mean != 0))


def compute_pbias(candidate: xr.DataArray, reference: xr.DataArray) -> xr.DataArray:
    """Percentage bias. Undefined where the reference sums to zero."""
    candidate, reference, pairs = _paired(candidate, reference)
    denominator = reference.sum(TIME, skipna=True)
    result = 100 * (candidate - reference).sum(TIME, skipna=True) / denominator
    return result.where((pairs >= 1) & (denominator != 0))


def compute_rmse(candidate: xr.DataArray, reference: xr.DataArray) -> xr.DataArray:
    """Root mean square error, in the units of the compared variable."""
    candidate, reference, pairs = _paired(candidate, reference)
    result = np.sqrt(((candidate - reference) ** 2).mean(TIME, skipna=True))
    return result.where(pairs >= 1)


def compute_mae(candidate: xr.DataArray, reference: xr.DataArray) -> xr.DataArray:
    """Mean absolute error, in the units of the compared variable."""
    candidate, reference, pairs = _paired(candidate, reference)
    result = abs(candidate - reference).mean(TIME, skipna=True)
    return result.where(pairs >= 1)


def compute_bias(candidate: xr.DataArray, reference: xr.DataArray) -> xr.DataArray:
    """Mean signed difference (candidate minus reference)."""
    candidate, reference, pairs = _paired(candidate, reference)
    result = (candidate - reference).mean(TIME, skipna=True)
    return result.where(pairs >= 1)


METRIC_FUNCTIONS: dict[str, Callable[[xr.DataArray, xr.DataArray], xr.DataArray]] = {
    "Correlation": compute_correlation,
    "NSE": compute_nse,
    "KGE": compute_kge,
    "PBIAS": compute_pbias,
    "RMSE": compute_rmse,
    "MAE": compute_mae,
    "Bias": compute_bias,
}


def summarise(result: xr.DataArray, metric_name: str) -> dict[str, float | int | str]:
    """Reduce a metric field to summary statistics, ignoring masked cells."""
    values = np.asarray(result.values, dtype="float64")
    finite = np.isfinite(values)
    total_cells = int(values.size)

    if not finite.any():
        return {
            "Metric": metric_name,
            "Mean": np.nan,
            "Median": np.nan,
            "Minimum": np.nan,
            "Maximum": np.nan,
            "Valid cells": 0,
            "Coverage %": 0.0,
        }

    valid = values[finite]
    return {
        "Metric": metric_name,
        "Mean": float(np.mean(valid)),
        "Median": float(np.median(valid)),
        "Minimum": float(np.min(valid)),
        "Maximum": float(np.max(valid)),
        "Valid cells": int(valid.size),
        "Coverage %": round(100.0 * valid.size / total_cells, 1) if total_cells else 0.0,
    }
