"""Shared fixtures for the CDFuse test suite."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr


@pytest.fixture
def simple_grid() -> xr.DataArray:
    """A small deterministic 3-D array on time/lat/lon."""
    time = pd.date_range("2020-01-01", periods=10, freq="D")
    lat = np.array([10.0, 11.0, 12.0])
    lon = np.array([70.0, 71.0])
    rng = np.random.default_rng(0)
    values = rng.normal(5.0, 1.0, (time.size, lat.size, lon.size))
    return xr.DataArray(
        values,
        coords={"time": time, "lat": lat, "lon": lon},
        dims=("time", "lat", "lon"),
        name="value",
    )


@pytest.fixture
def hourly_grid() -> xr.DataArray:
    """Six days of hourly data, so hour-of-day filtering can be exercised."""
    time = pd.date_range("2020-01-01 00:00", periods=24 * 6, freq="h")
    lat = np.array([10.0, 11.0])
    lon = np.array([70.0, 71.0])
    values = np.arange(time.size * lat.size * lon.size, dtype="float64").reshape(
        time.size, lat.size, lon.size
    )
    return xr.DataArray(
        values,
        coords={"time": time, "lat": lat, "lon": lon},
        dims=("time", "lat", "lon"),
        name="value",
    )
