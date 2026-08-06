"""Tests for the cell-wise comparison metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from cdfuse.metrics import (
    METRIC_FUNCTIONS,
    compute_bias,
    compute_correlation,
    compute_kge,
    compute_mae,
    compute_nse,
    compute_pbias,
    compute_rmse,
    summarise,
)


def test_identical_series_are_perfect(simple_grid):
    """A dataset compared against itself should score perfectly everywhere."""
    assert np.allclose(compute_correlation(simple_grid, simple_grid).values, 1.0)
    assert np.allclose(compute_nse(simple_grid, simple_grid).values, 1.0)
    assert np.allclose(compute_kge(simple_grid, simple_grid).values, 1.0)
    assert np.allclose(compute_pbias(simple_grid, simple_grid).values, 0.0, atol=1e-9)
    assert np.allclose(compute_rmse(simple_grid, simple_grid).values, 0.0, atol=1e-9)
    assert np.allclose(compute_mae(simple_grid, simple_grid).values, 0.0, atol=1e-9)
    assert np.allclose(compute_bias(simple_grid, simple_grid).values, 0.0, atol=1e-9)


def test_metrics_reduce_the_time_dimension(simple_grid):
    for name, function in METRIC_FUNCTIONS.items():
        result = function(simple_grid, simple_grid)
        assert "time" not in result.dims, f"{name} kept the time dimension"
        assert result.sizes["lat"] == simple_grid.sizes["lat"]
        assert result.sizes["lon"] == simple_grid.sizes["lon"]


def test_constant_offset_gives_expected_bias_and_pbias(simple_grid):
    """A known +2 offset must show up exactly in Bias, and consistently in PBIAS."""
    candidate = simple_grid + 2.0

    assert np.allclose(compute_bias(candidate, simple_grid).values, 2.0)
    assert np.allclose(compute_mae(candidate, simple_grid).values, 2.0)
    assert np.allclose(compute_rmse(candidate, simple_grid).values, 2.0)

    # PBIAS = 100 * sum(offset) / sum(reference)
    expected = 100 * (2.0 * simple_grid.sizes["time"]) / simple_grid.sum("time").values
    assert np.allclose(compute_pbias(candidate, simple_grid).values, expected)

    # A pure offset does not disturb timing, so correlation stays at 1.
    assert np.allclose(compute_correlation(candidate, simple_grid).values, 1.0)


def test_scaling_is_reflected_in_kge_alpha(simple_grid):
    """Doubling a zero-mean-free series changes variability, so KGE must drop."""
    candidate = simple_grid * 2.0
    kge = compute_kge(candidate, simple_grid)
    assert np.all(kge.values < 1.0)
    # Correlation is scale-invariant and should be untouched.
    assert np.allclose(compute_correlation(candidate, simple_grid).values, 1.0)


def test_nse_is_zero_when_candidate_is_the_reference_mean(simple_grid):
    """NSE compares against the reference mean, so predicting it scores exactly 0."""
    candidate = simple_grid.mean("time").broadcast_like(simple_grid)
    nse = compute_nse(candidate, simple_grid)
    assert np.allclose(nse.values, 0.0, atol=1e-9)


def test_constant_reference_masks_nse_rather_than_dividing_by_zero():
    time = pd.date_range("2020-01-01", periods=5, freq="D")
    reference = xr.DataArray(
        np.full((5, 1, 1), 3.0),
        coords={"time": time, "lat": [0.0], "lon": [0.0]},
        dims=("time", "lat", "lon"),
    )
    candidate = reference + 1.0
    assert np.isnan(compute_nse(candidate, reference).values).all()
    # KGE's alpha term is also undefined for a constant reference.
    assert np.isnan(compute_kge(candidate, reference).values).all()


def test_zero_sum_reference_masks_pbias():
    time = pd.date_range("2020-01-01", periods=4, freq="D")
    reference = xr.DataArray(
        np.array([1.0, -1.0, 2.0, -2.0]).reshape(4, 1, 1),
        coords={"time": time, "lat": [0.0], "lon": [0.0]},
        dims=("time", "lat", "lon"),
    )
    candidate = reference + 0.5
    assert np.isnan(compute_pbias(candidate, reference).values).all()


def test_insufficient_pairs_are_masked():
    """A cell with only one overlapping valid pair cannot support a correlation."""
    time = pd.date_range("2020-01-01", periods=4, freq="D")
    coords = {"time": time, "lat": [0.0], "lon": [0.0]}
    candidate = xr.DataArray(
        np.array([1.0, np.nan, np.nan, np.nan]).reshape(4, 1, 1),
        coords=coords, dims=("time", "lat", "lon"),
    )
    reference = xr.DataArray(
        np.array([1.0, 2.0, 3.0, 4.0]).reshape(4, 1, 1),
        coords=coords, dims=("time", "lat", "lon"),
    )
    assert np.isnan(compute_correlation(candidate, reference).values).all()
    assert np.isnan(compute_nse(candidate, reference).values).all()


def test_nan_in_either_series_is_excluded_pairwise():
    """A NaN on one side must remove that time step from both sides."""
    time = pd.date_range("2020-01-01", periods=4, freq="D")
    coords = {"time": time, "lat": [0.0], "lon": [0.0]}
    candidate = xr.DataArray(
        np.array([1.0, 2.0, np.nan, 4.0]).reshape(4, 1, 1),
        coords=coords, dims=("time", "lat", "lon"),
    )
    reference = xr.DataArray(
        np.array([1.0, 2.0, 999.0, 4.0]).reshape(4, 1, 1),
        coords=coords, dims=("time", "lat", "lon"),
    )
    # The 999 must be ignored rather than dominating the error.
    assert np.allclose(compute_bias(candidate, reference).values, 0.0)
    assert np.allclose(compute_rmse(candidate, reference).values, 0.0)


def test_metrics_are_directional(simple_grid):
    """Swapping candidate and reference must change asymmetric metrics."""
    candidate = simple_grid * 1.5 + 1.0
    forward = compute_pbias(candidate, simple_grid).values
    backward = compute_pbias(simple_grid, candidate).values
    assert not np.allclose(forward, backward)
    # Correlation, by contrast, is symmetric.
    assert np.allclose(
        compute_correlation(candidate, simple_grid).values,
        compute_correlation(simple_grid, candidate).values,
    )


def test_summarise_reports_coverage_and_ignores_masked_cells(simple_grid):
    result = compute_bias(simple_grid + 1.0, simple_grid)
    summary = summarise(result, "Bias")
    assert summary["Metric"] == "Bias"
    assert summary["Valid cells"] == result.size
    assert summary["Coverage %"] == 100.0
    assert summary["Mean"] == pytest.approx(1.0)


def test_summarise_handles_a_fully_masked_field(simple_grid):
    empty = compute_bias(simple_grid, simple_grid).where(False)
    summary = summarise(empty, "Bias")
    assert summary["Valid cells"] == 0
    assert summary["Coverage %"] == 0.0
    assert np.isnan(summary["Mean"])
