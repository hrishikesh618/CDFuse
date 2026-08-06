"""Integration test over a deliberately awkward, but entirely realistic, pair.

Real uploads rarely arrive in canonical form. This exercise combines the
conventions that most often appear together in published gridded products:

* coordinates named ``valid_time`` / ``latitude`` / ``longitude``
* longitudes on the 0..360 convention
* latitude stored north-to-south
* an extra pressure-level dimension
* hourly time steps, offset between the two datasets

No product is named, and nothing here is treated as a special case by the
library — the point is that the generic pipeline copes.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from cdfuse.io import open_netcdf
from cdfuse.pipeline import ComparisonSettings, run_comparison
from cdfuse.preprocess import squeeze_extra_dimensions


@pytest.fixture
def awkward_candidate() -> xr.DataArray:
    """Hourly, 0..360 longitudes, descending latitude, with a level dimension."""
    time = pd.date_range("2021-06-01 00:00", periods=24 * 8, freq="h")
    latitude = np.array([22.0, 21.0, 20.0, 19.0])           # descending
    longitude = np.array([350.0, 355.0, 0.0, 5.0])          # crosses the meridian
    level = np.array([850.0, 500.0])

    rng = np.random.default_rng(7)
    shape = (time.size, level.size, latitude.size, longitude.size)
    values = rng.normal(10.0, 2.0, shape)
    values[:, 1, :, :] += 100.0  # make the 500 hPa slice obviously different

    return xr.DataArray(
        values,
        coords={
            "valid_time": time,
            "level": level,
            "latitude": latitude,
            "longitude": longitude,
        },
        dims=("valid_time", "level", "latitude", "longitude"),
        name="t2m",
    )


@pytest.fixture
def canonical_reference() -> xr.DataArray:
    """Daily, -180..180 longitudes, ascending latitude — already tidy."""
    time = pd.date_range("2021-06-01", periods=8, freq="D")
    lat = np.array([19.0, 20.0, 21.0, 22.0])
    lon = np.array([-10.0, -5.0, 0.0, 5.0])
    rng = np.random.default_rng(11)
    return xr.DataArray(
        rng.normal(10.0, 2.0, (time.size, lat.size, lon.size)),
        coords={"time": time, "lat": lat, "lon": lon},
        dims=("time", "lat", "lon"),
        name="temperature",
    )


def test_awkward_pair_runs_end_to_end(awkward_candidate, canonical_reference):
    candidate = squeeze_extra_dimensions(
        awkward_candidate, {"level": 850.0}, {"valid_time", "latitude", "longitude"}
    )
    assert "level" not in candidate.dims, "the level slice should have been taken"

    settings = ComparisonSettings(
        candidate_time="valid_time",
        candidate_lat="latitude",
        candidate_lon="longitude",
        aggregation_level="Daily",
        aggregation_method="Mean",
        spatial_method="Interpolate candidate to reference grid",
        metrics=["Correlation", "RMSE", "Bias"],
    )
    result = run_comparison(candidate, canonical_reference, settings)

    assert result.time_steps == 8, "hourly data should aggregate to 8 days"
    assert result.lat_cells == 4 and result.lon_cells == 4

    for array in result.arrays.values():
        assert set(array.dims) == {"lat", "lon"}
        assert np.all(array["lat"].values == np.sort(array["lat"].values))
        assert array["lon"].max() <= 180


def test_the_selected_level_is_the_one_compared(awkward_candidate, canonical_reference):
    """Picking 500 hPa (offset by +100) must produce a very different bias."""
    settings = ComparisonSettings(
        candidate_time="valid_time",
        candidate_lat="latitude",
        candidate_lon="longitude",
        aggregation_level="Daily",
        spatial_method="Interpolate candidate to reference grid",
        metrics=["Bias"],
    )

    lower = run_comparison(
        squeeze_extra_dimensions(
            awkward_candidate, {"level": 850.0}, {"valid_time", "latitude", "longitude"}
        ),
        canonical_reference,
        settings,
    )
    upper = run_comparison(
        squeeze_extra_dimensions(
            awkward_candidate, {"level": 500.0}, {"valid_time", "latitude", "longitude"}
        ),
        canonical_reference,
        settings,
    )

    lower_bias = lower.summary.set_index("Metric").loc["Bias", "Mean"]
    upper_bias = upper.summary.set_index("Metric").loc["Bias", "Mean"]
    assert upper_bias - lower_bias == pytest.approx(100.0, abs=1.0)


def test_hour_filtering_on_the_awkward_candidate(awkward_candidate, canonical_reference):
    """Keeping 00:00 and matching by date should still line up with daily data."""
    candidate = squeeze_extra_dimensions(
        awkward_candidate, {"level": 850.0}, {"valid_time", "latitude", "longitude"}
    )
    settings = ComparisonSettings(
        candidate_time="valid_time",
        candidate_lat="latitude",
        candidate_lon="longitude",
        candidate_hour=0,
        match_hours_by_date=True,
        aggregation_level="None (use matched time steps)",
        spatial_method="Interpolate candidate to reference grid",
        metrics=["Correlation"],
    )
    result = run_comparison(candidate, canonical_reference, settings)
    assert result.time_steps == 8
    assert result.provenance["Candidate hour filter"] == "00:00"


def test_netcdf_roundtrip_preserves_the_awkward_conventions(awkward_candidate, tmp_path):
    """Writing and re-reading must not quietly normalise the coordinate names."""
    path = tmp_path / "awkward.nc"
    awkward_candidate.to_dataset().to_netcdf(path)

    reopened = open_netcdf(path.read_bytes(), "awkward.nc")
    assert "t2m" in reopened.data_vars
    assert "valid_time" in reopened["t2m"].dims
    assert float(reopened["longitude"].max()) > 180
