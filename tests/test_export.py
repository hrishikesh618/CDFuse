"""Tests for the download/export helpers and map rendering."""

from __future__ import annotations

import io
import zipfile

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from cdfuse import export, plotting
from cdfuse.metrics import compute_bias, summarise


@pytest.fixture
def metric_field(simple_grid) -> xr.DataArray:
    return compute_bias(simple_grid + 1.5, simple_grid)


def test_safe_name_strips_unsafe_characters():
    assert export.safe_name("PBIAS") == "pbias"
    assert export.safe_name("Corr / ratio") == "corr_ratio"
    assert export.safe_name("../../etc/passwd") == "etc_passwd"
    assert export.safe_name("   ") == "output"


def test_metric_to_netcdf_roundtrips(metric_field, tmp_path):
    payload = export.metric_to_netcdf(metric_field, "Bias", {"Aggregation": "Daily (Mean)"})
    assert isinstance(payload, bytes)
    # Either classic NetCDF ("CDF") or the HDF5-based NetCDF4 container.
    assert payload[:3] == b"CDF" or payload[:4] == b"\x89HDF", "not a NetCDF container"

    path = tmp_path / "bias.nc"
    path.write_bytes(payload)
    with xr.open_dataset(path) as reopened:
        assert "bias" in reopened.data_vars
        assert reopened.attrs["Aggregation"] == "Daily (Mean)"
        assert "CDFuse" in reopened.attrs["source"]
        assert np.allclose(reopened["bias"].values, metric_field.values, equal_nan=True)


def test_metric_to_csv_has_one_row_per_cell(metric_field):
    payload = export.metric_to_csv(metric_field, "Bias")
    frame = pd.read_csv(io.BytesIO(payload))
    assert set(frame.columns) == {"lat", "lon", "bias"}
    assert len(frame) == metric_field.size


def test_summary_to_csv_is_readable(metric_field):
    summary = pd.DataFrame([summarise(metric_field, "Bias")])
    frame = pd.read_csv(io.BytesIO(export.summary_to_csv(summary)))
    assert frame.loc[0, "Metric"] == "Bias"


def test_run_report_records_settings_and_notes():
    report = export.build_run_report(
        {"Aggregation": "Daily (Mean)", "Metrics": "NSE"},
        ["Matched 12 time steps."],
    ).decode("utf-8")
    assert "Aggregation: Daily (Mean)" in report
    assert "- Matched 12 time steps." in report
    assert "CDFuse" in report


def test_archive_contains_every_output(metric_field):
    files = {
        "Bias": {
            "png": b"fake-png",
            "netcdf": export.metric_to_netcdf(metric_field, "Bias"),
            "geotiff": None,  # optional output that legitimately may be absent
            "csv": export.metric_to_csv(metric_field, "Bias"),
        }
    }
    archive_bytes = export.build_archive(files, b"metric,mean\n", b"report")
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        names = set(archive.namelist())
    assert "summary.csv" in names
    assert "run_report.txt" in names
    assert "maps/bias.png" in names
    assert "netcdf/bias.nc" in names
    assert "tables/bias.csv" in names
    assert not any(name.endswith(".tif") for name in names)


def test_format_number_switches_to_scientific_notation():
    assert export.format_number(1.23456) == "1.235"
    assert export.format_number(0.0) == "0.000"
    assert "e" in export.format_number(1.2e-8)
    assert "e" in export.format_number(9.9e12)
    assert export.format_number(float("nan")) == "n/a"


def test_resolve_limits_keeps_fixed_bounds(metric_field):
    assert plotting.resolve_limits(metric_field, "Correlation", -1.0, 1.0) == (-1.0, 1.0)


def test_resolve_limits_is_symmetric_for_diverging_metrics(simple_grid):
    field = compute_bias(simple_grid + 3.0, simple_grid)
    vmin, vmax = plotting.resolve_limits(field, "Bias", None, None)
    assert vmin == pytest.approx(-vmax)


def test_resolve_limits_handles_an_all_nan_field(metric_field):
    empty = metric_field.where(False)
    assert plotting.resolve_limits(empty, "RMSE", None, None) == (None, None)


def test_make_map_produces_a_png(metric_field):
    figure = plotting.make_map(metric_field, "Bias", None, vmin=None, vmax=None, cmap="RdBu_r")
    payload = plotting.figure_to_png(figure, dpi=72)
    assert payload[:8] == b"\x89PNG\r\n\x1a\n"
    import matplotlib.pyplot as plt

    plt.close(figure)


def test_make_map_works_without_cartopy(metric_field, monkeypatch):
    """The fallback path must still render when cartopy is unavailable."""
    monkeypatch.setattr(plotting, "CARTOPY_AVAILABLE", False)
    figure = plotting.make_map(metric_field, "Bias", None, cmap="viridis")
    assert plotting.figure_to_png(figure, dpi=72)[:4] == b"\x89PNG"
    import matplotlib.pyplot as plt

    plt.close(figure)
