"""Tests for the geospatial paths that depend on the rioxarray accessor.

These exist because ``.rio`` is only attached to xarray objects as a side
effect of importing ``rioxarray``. Forgetting that import breaks boundary
clipping and GeoTIFF export at runtime while every other test still passes,
so both paths are exercised explicitly here.
"""

from __future__ import annotations

import io
import zipfile

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from shapely.geometry import box

from cdfuse import export
from cdfuse.io import DataLoadError, clip_to_boundary, load_boundary, open_netcdf


@pytest.fixture
def metric_field() -> xr.DataArray:
    lat = np.linspace(0.0, 9.0, 10)
    lon = np.linspace(0.0, 9.0, 10)
    values = np.arange(100, dtype="float64").reshape(10, 10)
    return xr.DataArray(
        values, coords={"lat": lat, "lon": lon}, dims=("lat", "lon"), name="bias"
    )


@pytest.fixture
def small_boundary() -> gpd.GeoDataFrame:
    """A square covering roughly the lower-left quadrant of the test grid."""
    return gpd.GeoDataFrame(
        {"name": ["box"]}, geometry=[box(0.0, 0.0, 4.0, 4.0)], crs="EPSG:4326"
    )


def test_rio_accessor_is_registered(metric_field):
    """The accessor must exist purely from importing the cdfuse package."""
    assert hasattr(metric_field, "rio"), "rioxarray was not imported; .rio is missing"


def test_clip_to_boundary_masks_outside_cells(metric_field, small_boundary):
    clipped = clip_to_boundary(metric_field, small_boundary)

    assert clipped.shape == metric_field.shape, "clipping must preserve the grid shape"

    finite = np.isfinite(clipped.values)
    assert finite.any(), "cells inside the boundary should survive"
    assert not finite.all(), "cells outside the boundary should be masked"

    # A cell well inside the square is kept; one well outside is dropped.
    assert np.isfinite(clipped.sel(lat=1.0, lon=1.0).values)
    assert np.isnan(clipped.sel(lat=9.0, lon=9.0).values)


def test_clip_reports_a_non_overlapping_boundary(metric_field):
    far_away = gpd.GeoDataFrame(
        {"name": ["elsewhere"]}, geometry=[box(100.0, 60.0, 101.0, 61.0)], crs="EPSG:4326"
    )
    with pytest.raises(DataLoadError):
        clip_to_boundary(metric_field, far_away)


def test_metric_to_geotiff_returns_readable_bytes(metric_field, tmp_path):
    payload = export.metric_to_geotiff(metric_field, "Bias")
    assert payload is not None, "GeoTIFF export returned None"
    assert payload[:4] in (b"II*\x00", b"MM\x00*"), "not a TIFF header"

    import rioxarray

    path = tmp_path / "roundtrip.tif"
    path.write_bytes(payload)
    # Close the handle explicitly; Windows will not release the file otherwise.
    with rioxarray.open_rasterio(path) as reopened:
        assert reopened.rio.crs.to_epsg() == 4326
        assert reopened.squeeze().shape == metric_field.shape


def test_archive_includes_the_geotiff_when_it_is_produced(metric_field):
    files = {
        "Bias": {
            "png": b"fake",
            "netcdf": export.metric_to_netcdf(metric_field, "Bias"),
            "geotiff": export.metric_to_geotiff(metric_field, "Bias"),
            "csv": export.metric_to_csv(metric_field, "Bias"),
        }
    }
    archive_bytes = export.build_archive(files, b"metric\n", b"report")
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        names = set(archive.namelist())
    assert "geotiff/bias.tif" in names


def test_load_boundary_reads_a_zipped_shapefile(tmp_path, small_boundary):
    """The documented happy path: one ZIP holding every shapefile component."""
    shapefile_dir = tmp_path / "shp"
    shapefile_dir.mkdir()
    small_boundary.to_file(shapefile_dir / "boundary.shp")

    zip_path = tmp_path / "boundary.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        for component in shapefile_dir.iterdir():
            archive.write(component, component.name)

    loaded = load_boundary([_Upload("boundary.zip", zip_path.read_bytes())])
    assert loaded is not None
    assert len(loaded) == 1
    assert loaded.crs.to_epsg() == 4326


def test_load_boundary_rejects_a_shapefile_missing_sidecars(tmp_path, small_boundary):
    shapefile_dir = tmp_path / "shp"
    shapefile_dir.mkdir()
    small_boundary.to_file(shapefile_dir / "boundary.shp")

    lone_shp = shapefile_dir / "boundary.shp"
    with pytest.raises(DataLoadError, match="sidecar"):
        load_boundary([_Upload("boundary.shp", lone_shp.read_bytes())])


def test_load_boundary_rejects_a_zip_slip_path(tmp_path):
    """A ZIP entry escaping the extraction folder must be refused."""
    zip_path = tmp_path / "evil.zip"
    with zipfile.ZipFile(zip_path, "w") as archive:
        archive.writestr("../escaped.shp", b"nope")

    with pytest.raises(DataLoadError, match="outside"):
        load_boundary([_Upload("evil.zip", zip_path.read_bytes())])


def test_open_netcdf_roundtrip(tmp_path):
    time = pd.date_range("2020-01-01", periods=3, freq="D")
    dataset = xr.Dataset(
        {"value": (("time", "lat", "lon"), np.ones((3, 2, 2)))},
        coords={"time": time, "lat": [0.0, 1.0], "lon": [0.0, 1.0]},
    )
    path = tmp_path / "sample.nc"
    dataset.to_netcdf(path)

    reopened = open_netcdf(path.read_bytes(), "sample.nc")
    assert "value" in reopened.data_vars
    assert reopened.sizes["time"] == 3


def test_open_netcdf_rejects_junk():
    with pytest.raises(DataLoadError, match="could not be opened"):
        open_netcdf(b"this is definitely not netcdf", "broken.nc")


def test_open_netcdf_rejects_an_empty_upload():
    with pytest.raises(DataLoadError, match="empty"):
        open_netcdf(b"", "nothing.nc")


class _Upload:
    """Minimal stand-in for a Streamlit UploadedFile."""

    def __init__(self, name: str, payload: bytes) -> None:
        self.name = name
        self._payload = payload

    def getvalue(self) -> bytes:
        return self._payload
