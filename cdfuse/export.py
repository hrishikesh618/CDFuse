"""Turning results into downloadable files."""

from __future__ import annotations

import io
import logging
import re
import zipfile
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import rioxarray  # noqa: F401 - importing registers the .rio accessor used below
import xarray as xr

from .config import APP_NAME, VERSION

LOGGER = logging.getLogger(__name__)


def safe_name(value: str) -> str:
    """Return a filesystem-safe lowercase stem."""
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip())
    return cleaned.strip("._").lower() or "output"


def metric_to_netcdf(
    data: xr.DataArray, metric_name: str, provenance: dict[str, str] | None = None
) -> bytes:
    """Serialise one metric field to NetCDF bytes, with provenance attributes."""
    variable = safe_name(metric_name)
    dataset = data.rename(variable).to_dataset()
    dataset[variable].attrs.update(
        {
            "long_name": f"{metric_name} between candidate and reference datasets",
            "metric": metric_name,
        }
    )
    dataset.attrs.update(
        {
            "title": f"{metric_name} computed with {APP_NAME}",
            "source": f"{APP_NAME} v{VERSION}",
            "history": f"Created {datetime.now(timezone.utc):%Y-%m-%dT%H:%M:%SZ} by {APP_NAME}",
            "Conventions": "CF-1.8",
        }
    )
    if provenance:
        dataset.attrs.update({key: str(value) for key, value in provenance.items()})

    for coordinate, attributes in (
        ("lat", {"units": "degrees_north", "standard_name": "latitude"}),
        ("lon", {"units": "degrees_east", "standard_name": "longitude"}),
    ):
        if coordinate in dataset.coords:
            dataset[coordinate].attrs.update(attributes)

    return bytes(dataset.to_netcdf())


def metric_to_geotiff(data: xr.DataArray, metric_name: str) -> bytes | None:
    """Serialise one metric field to GeoTIFF bytes, or None if that fails.

    GeoTIFF export is a convenience for GIS users, so a failure here is not
    fatal — but it is logged rather than swallowed, so the reason is
    recoverable from the server log.
    """
    import tempfile
    from pathlib import Path

    try:
        raster = data.rename(safe_name(metric_name))
        raster.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
        raster.rio.write_crs("EPSG:4326", inplace=True)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "metric.tif"
            raster.rio.to_raster(path)
            return path.read_bytes()
    except Exception:  # noqa: BLE001 - optional output
        LOGGER.warning("GeoTIFF export failed for %s", metric_name, exc_info=True)
        return None


def summary_to_csv(summary: pd.DataFrame) -> bytes:
    return summary.to_csv(index=False).encode("utf-8")


def metric_to_csv(data: xr.DataArray, metric_name: str) -> bytes:
    """Flatten one metric field to a long-format CSV of lat, lon, value."""
    frame = data.to_dataframe(name=safe_name(metric_name)).reset_index()
    keep = [column for column in ("lat", "lon", safe_name(metric_name)) if column in frame.columns]
    return frame[keep].to_csv(index=False).encode("utf-8")


def build_run_report(provenance: dict[str, object], notes: list[str]) -> bytes:
    """A plain-text record of the settings behind a run, for reproducibility."""
    lines = [
        f"{APP_NAME} v{VERSION} — comparison report",
        f"Generated: {datetime.now(timezone.utc):%Y-%m-%d %H:%M:%S} UTC",
        "",
        "SETTINGS",
        "--------",
    ]
    lines.extend(f"{key}: {value}" for key, value in provenance.items())
    if notes:
        lines.extend(["", "PROCESSING NOTES", "----------------"])
        lines.extend(f"- {note}" for note in notes)
    lines.extend(
        [
            "",
            "NOTE",
            "----",
            "Metrics are computed cell by cell over the time steps the two datasets share.",
            "Cells without enough valid pairs, or where a metric's denominator is undefined,",
            "are stored as missing values rather than as numbers.",
            "",
        ]
    )
    return "\n".join(lines).encode("utf-8")


def build_archive(
    files: dict[str, dict[str, bytes | None]],
    summary_csv: bytes,
    report: bytes,
) -> bytes:
    """Bundle every output into a single ZIP for download."""
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("summary.csv", summary_csv)
        archive.writestr("run_report.txt", report)
        for metric_name, outputs in files.items():
            stem = safe_name(metric_name)
            for kind, extension, folder in (
                ("png", "png", "maps"),
                ("netcdf", "nc", "netcdf"),
                ("geotiff", "tif", "geotiff"),
                ("csv", "csv", "tables"),
            ):
                payload = outputs.get(kind)
                if payload:
                    archive.writestr(f"{folder}/{stem}.{extension}", payload)
    return buffer.getvalue()


def format_number(value: float) -> str:
    """Compact display formatting that stays readable across magnitudes."""
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    magnitude = abs(value)
    if magnitude != 0 and (magnitude < 1e-3 or magnitude >= 1e5):
        return f"{value:.3e}"
    return f"{value:.3f}"
