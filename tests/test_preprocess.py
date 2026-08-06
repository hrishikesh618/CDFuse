"""Tests for standardisation, temporal processing and alignment."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from cdfuse.preprocess import (
    PreprocessError,
    aggregate,
    align,
    describe,
    select_hour_of_day,
    squeeze_extra_dimensions,
    standardise,
)


def test_standardise_renames_coordinates():
    time = pd.date_range("2020-01-01", periods=3, freq="D")
    data = xr.DataArray(
        np.zeros((3, 2, 2)),
        coords={"valid_time": time, "latitude": [1.0, 2.0], "longitude": [10.0, 11.0]},
        dims=("valid_time", "latitude", "longitude"),
    )
    prepared = standardise(data, "valid_time", "latitude", "longitude")
    assert set(prepared.data.dims) == {"time", "lat", "lon"}


def test_standardise_normalises_longitudes_past_180():
    time = pd.date_range("2020-01-01", periods=2, freq="D")
    data = xr.DataArray(
        np.zeros((2, 1, 3)),
        coords={"time": time, "lat": [0.0], "lon": [10.0, 190.0, 350.0]},
        dims=("time", "lat", "lon"),
    )
    prepared = standardise(data, "time", "lat", "lon")
    assert prepared.data["lon"].max() <= 180
    assert prepared.data["lon"].min() >= -180
    # Sorting must follow the conversion.
    assert np.all(np.diff(prepared.data["lon"].values) > 0)
    assert any("longitude" in note.lower() for note in prepared.notes)


def test_standardise_sorts_descending_latitude():
    """Many products store latitude north-to-south; alignment needs ascending."""
    time = pd.date_range("2020-01-01", periods=2, freq="D")
    data = xr.DataArray(
        np.arange(2 * 3 * 1, dtype="float64").reshape(2, 3, 1),
        coords={"time": time, "lat": [30.0, 20.0, 10.0], "lon": [0.0]},
        dims=("time", "lat", "lon"),
    )
    prepared = standardise(data, "time", "lat", "lon")
    assert np.all(np.diff(prepared.data["lat"].values) > 0)


def test_standardise_applies_a_time_shift():
    time = pd.date_range("2020-01-01 06:00", periods=3, freq="D")
    data = xr.DataArray(
        np.zeros((3, 1, 1)),
        coords={"time": time, "lat": [0.0], "lon": [0.0]},
        dims=("time", "lat", "lon"),
    )
    prepared = standardise(data, "time", "lat", "lon", time_shift_hours=-6.0)
    assert pd.Timestamp(prepared.data["time"].values[0]).hour == 0


def test_standardise_rejects_a_missing_coordinate(simple_grid):
    with pytest.raises(PreprocessError, match="not present"):
        standardise(simple_grid, "time", "lat", "does_not_exist")


def test_standardise_rejects_two_dimensional_coordinates():
    time = pd.date_range("2020-01-01", periods=2, freq="D")
    data = xr.DataArray(
        np.zeros((2, 2, 2)),
        coords={
            "time": time,
            "lat2d": (("y", "x"), np.array([[1.0, 1.0], [2.0, 2.0]])),
            "lon2d": (("y", "x"), np.array([[10.0, 11.0], [10.0, 11.0]])),
        },
        dims=("time", "y", "x"),
    )
    with pytest.raises(PreprocessError, match="dimensional"):
        standardise(data, "time", "lat2d", "lon2d")


def test_standardise_averages_duplicate_timestamps():
    time = pd.to_datetime(["2020-01-01", "2020-01-01", "2020-01-02"])
    data = xr.DataArray(
        np.array([1.0, 3.0, 5.0]).reshape(3, 1, 1),
        coords={"time": time, "lat": [0.0], "lon": [0.0]},
        dims=("time", "lat", "lon"),
    )
    prepared = standardise(data, "time", "lat", "lon")
    assert prepared.data.sizes["time"] == 2
    assert prepared.data.values[0, 0, 0] == pytest.approx(2.0)
    assert any("duplicate" in note.lower() for note in prepared.notes)


def test_squeeze_extra_dimensions_keeps_single_cell_grids():
    """A 1x1 grid must survive; a blanket squeeze would destroy it."""
    time = pd.date_range("2020-01-01", periods=3, freq="D")
    data = xr.DataArray(
        np.zeros((3, 1, 1, 2)),
        coords={"time": time, "lat": [0.0], "lon": [0.0], "level": [500.0, 850.0]},
        dims=("time", "lat", "lon", "level"),
    )
    reduced = squeeze_extra_dimensions(data, {"level": 850.0}, {"time", "lat", "lon"})
    assert set(reduced.dims) == {"time", "lat", "lon"}
    assert reduced.sizes["lat"] == 1 and reduced.sizes["lon"] == 1


def test_squeeze_extra_dimensions_selects_the_requested_level():
    time = pd.date_range("2020-01-01", periods=2, freq="D")
    values = np.zeros((2, 2, 2, 2))
    values[..., 1] = 7.0
    data = xr.DataArray(
        values,
        coords={"time": time, "lat": [0.0, 1.0], "lon": [0.0, 1.0], "level": [500.0, 850.0]},
        dims=("time", "lat", "lon", "level"),
    )
    reduced = squeeze_extra_dimensions(data, {"level": 850.0}, {"time", "lat", "lon"})
    assert np.allclose(reduced.values, 7.0)


def test_select_hour_of_day_keeps_one_record_per_day(hourly_grid):
    prepared = select_hour_of_day(hourly_grid, 6, match_by_date=True)
    assert prepared.data.sizes["time"] == 6
    assert set(pd.DatetimeIndex(prepared.data["time"].values).hour) == {0}


def test_select_hour_of_day_can_keep_original_timestamps(hourly_grid):
    prepared = select_hour_of_day(hourly_grid, 6, match_by_date=False)
    assert set(pd.DatetimeIndex(prepared.data["time"].values).hour) == {6}


def test_select_hour_of_day_reports_available_hours(simple_grid):
    """Daily data has only hour 0, so asking for 13:00 must fail helpfully."""
    with pytest.raises(PreprocessError, match="00:00"):
        select_hour_of_day(simple_grid, 13)


def test_different_hours_can_be_matched_by_calendar_date(hourly_grid):
    """The whole point of date matching: pair 23:00 with 00:00 day by day."""
    candidate = select_hour_of_day(hourly_grid, 23, match_by_date=True).data
    reference = select_hour_of_day(hourly_grid, 0, match_by_date=True).data
    aligned_candidate, aligned_reference, _ = align(
        candidate, reference, "Use exact shared coordinates only"
    )
    assert aligned_candidate.sizes["time"] == 6
    assert aligned_reference.sizes["time"] == 6


def test_aggregate_monthly_sum():
    time = pd.date_range("2020-01-01", periods=60, freq="D")
    data = xr.DataArray(
        np.ones((60, 1, 1)),
        coords={"time": time, "lat": [0.0], "lon": [0.0]},
        dims=("time", "lat", "lon"),
    )
    aggregated = aggregate(data, "MS", "Sum")
    assert aggregated.sizes["time"] == 2
    assert aggregated.values[0, 0, 0] == pytest.approx(31.0)
    assert aggregated.values[1, 0, 0] == pytest.approx(29.0)  # 2020 was a leap year


def test_aggregate_none_is_a_passthrough(simple_grid):
    assert aggregate(simple_grid, None, "Mean") is simple_grid


def test_align_rejects_non_overlapping_periods():
    def build(start: str) -> xr.DataArray:
        time = pd.date_range(start, periods=5, freq="D")
        return xr.DataArray(
            np.zeros((5, 1, 1)),
            coords={"time": time, "lat": [0.0], "lon": [0.0]},
            dims=("time", "lat", "lon"),
        )

    with pytest.raises(PreprocessError, match="share no time steps"):
        align(build("2020-01-01"), build("2021-01-01"), "Use exact shared coordinates only")


def test_align_rejects_a_single_shared_timestep():
    time_a = pd.date_range("2020-01-01", periods=3, freq="D")
    time_b = pd.date_range("2020-01-03", periods=3, freq="D")

    def build(time) -> xr.DataArray:
        return xr.DataArray(
            np.zeros((len(time), 1, 1)),
            coords={"time": time, "lat": [0.0], "lon": [0.0]},
            dims=("time", "lat", "lon"),
        )

    with pytest.raises(PreprocessError, match="[Oo]nly one time step"):
        align(build(time_a), build(time_b), "Use exact shared coordinates only")


def test_align_rejects_disjoint_grids_without_interpolation():
    time = pd.date_range("2020-01-01", periods=3, freq="D")

    def build(lats) -> xr.DataArray:
        return xr.DataArray(
            np.zeros((3, len(lats), 1)),
            coords={"time": time, "lat": lats, "lon": [0.0]},
            dims=("time", "lat", "lon"),
        )

    with pytest.raises(PreprocessError, match="share no exact"):
        align(build([1.0, 2.0]), build([50.0, 51.0]), "Use exact shared coordinates only")


def test_align_interpolates_onto_the_reference_grid():
    time = pd.date_range("2020-01-01", periods=3, freq="D")
    candidate = xr.DataArray(
        np.ones((3, 5, 5)),
        coords={"time": time, "lat": np.linspace(0, 4, 5), "lon": np.linspace(0, 4, 5)},
        dims=("time", "lat", "lon"),
    )
    reference = xr.DataArray(
        np.ones((3, 3, 3)),
        coords={"time": time, "lat": np.linspace(1, 3, 3), "lon": np.linspace(1, 3, 3)},
        dims=("time", "lat", "lon"),
    )
    aligned_candidate, aligned_reference, notes = align(
        candidate, reference, "Interpolate candidate to reference grid"
    )
    assert aligned_candidate.sizes["lat"] == 3
    assert aligned_candidate.shape == aligned_reference.shape
    assert any("nterpolat" in note for note in notes)


def test_describe_reports_period_and_extent(simple_grid):
    summary = describe(simple_grid)
    assert "2020-01-01" in summary["Period"]
    assert "10.000" in summary["Extent"]
    assert summary["Grid"] == "3 lat x 2 lon"
