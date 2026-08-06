"""Reading uploaded NetCDF files and vector boundaries."""

from __future__ import annotations

import os
import tempfile
import zipfile
from pathlib import Path

import geopandas as gpd
import rioxarray  # noqa: F401 - importing registers the .rio accessor used below
import xarray as xr


class DataLoadError(ValueError):
    """Raised when an upload cannot be read, with a message aimed at the user."""


def open_netcdf(file_bytes: bytes, filename: str) -> xr.Dataset:
    """Load an uploaded NetCDF file fully into memory.

    The file is written to a temporary path because the NetCDF libraries need
    a real file handle. It is removed again as soon as the data is loaded, so
    nothing is left on the server between sessions.
    """
    if not file_bytes:
        raise DataLoadError(f"'{filename}' is empty.")

    suffix = Path(filename).suffix or ".nc"
    tmp_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        try:
            with xr.open_dataset(tmp_path) as dataset:
                loaded = dataset.load()
        except Exception as exc:  # noqa: BLE001 - re-raised with guidance below
            raise DataLoadError(
                f"'{filename}' could not be opened as NetCDF ({exc}). Confirm it is a valid "
                "NetCDF3/NetCDF4/HDF5 file and is not truncated."
            ) from exc
    finally:
        if tmp_path and os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass

    if not loaded.data_vars:
        raise DataLoadError(
            f"'{filename}' contains no data variables, only coordinates or attributes."
        )
    return loaded


def _safe_extract(archive: zipfile.ZipFile, target: Path) -> None:
    """Extract a ZIP, refusing entries that escape the target directory."""
    target_resolved = target.resolve()
    for member in archive.infolist():
        destination = (target / member.filename).resolve()
        if destination != target_resolved and target_resolved not in destination.parents:
            raise DataLoadError(
                "The uploaded ZIP contains a path that points outside the extraction "
                "folder, so it was rejected."
            )
    archive.extractall(target)


def load_boundary(uploaded_files: list) -> gpd.GeoDataFrame | None:
    """Read an uploaded boundary as a GeoDataFrame in EPSG:4326.

    Accepts a zipped shapefile, loose shapefile components uploaded together,
    GeoJSON, or GeoPackage.
    """
    if not uploaded_files:
        return None

    with tempfile.TemporaryDirectory() as directory:
        directory_path = Path(directory)

        for uploaded in uploaded_files:
            destination = directory_path / Path(uploaded.name).name
            destination.write_bytes(uploaded.getvalue())
            if destination.suffix.lower() == ".zip":
                try:
                    with zipfile.ZipFile(destination) as archive:
                        _safe_extract(archive, directory_path)
                except zipfile.BadZipFile as exc:
                    raise DataLoadError(f"'{uploaded.name}' is not a readable ZIP file.") from exc

        candidates: list[Path] = []
        for pattern in ("*.shp", "*.geojson", "*.json", "*.gpkg"):
            candidates.extend(sorted(directory_path.rglob(pattern)))

        if not candidates:
            raise DataLoadError(
                "No boundary layer was found in the upload. Provide a zipped shapefile, all "
                "shapefile components together (.shp, .shx, .dbf, .prj), a GeoJSON, or a GeoPackage."
            )

        source = candidates[0]
        if source.suffix.lower() == ".shp":
            missing = [
                extension
                for extension in (".shx", ".dbf")
                if not source.with_suffix(extension).exists()
            ]
            if missing:
                raise DataLoadError(
                    f"The shapefile '{source.name}' is missing its {', '.join(missing)} "
                    "sidecar file(s). Shapefiles need every component to be uploaded together."
                )

        try:
            boundary = gpd.read_file(source)
        except Exception as exc:  # noqa: BLE001
            raise DataLoadError(f"The boundary could not be read: {exc}") from exc

        if boundary.empty:
            raise DataLoadError("The uploaded boundary contains no features.")

        if boundary.crs is None:
            raise DataLoadError(
                "The boundary has no coordinate reference system. Include a .prj file, or "
                "supply a GeoJSON/GeoPackage that records its CRS."
            )

        try:
            return boundary.to_crs("EPSG:4326")
        except Exception as exc:  # noqa: BLE001
            raise DataLoadError(
                f"The boundary could not be converted to EPSG:4326 (WGS84): {exc}"
            ) from exc


def clip_to_boundary(data: xr.DataArray, boundary: gpd.GeoDataFrame) -> xr.DataArray:
    """Mask a metric field to the boundary polygons, keeping the grid shape."""
    import numpy as np
    from shapely.geometry import mapping

    clipped = data.copy()
    clipped.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
    clipped.rio.write_crs("EPSG:4326", inplace=True)
    try:
        result = clipped.rio.clip(
            boundary.geometry.apply(mapping),
            boundary.crs,
            drop=False,
            all_touched=True,
        )
    except Exception as exc:  # noqa: BLE001
        raise DataLoadError(
            f"Clipping to the boundary failed: {exc}. Check that the boundary overlaps the "
            "data extent."
        ) from exc

    # rioxarray returns an all-missing array rather than raising when the
    # boundary falls outside the grid. Silently handing back a blank map would
    # look like a broken metric, so say what actually happened.
    if np.isfinite(data.values).any() and not np.isfinite(result.values).any():
        data_bounds = (
            float(data["lon"].min()), float(data["lat"].min()),
            float(data["lon"].max()), float(data["lat"].max()),
        )
        boundary_bounds = tuple(round(float(value), 3) for value in boundary.total_bounds)
        raise DataLoadError(
            "Clipping removed every cell, so the boundary does not overlap the data.\n\n"
            f"Data extent (lon/lat): {tuple(round(value, 3) for value in data_bounds)}\n"
            f"Boundary extent (lon/lat): {boundary_bounds}\n\n"
            "Check that the boundary covers the same region, or turn off clipping."
        )

    return result
