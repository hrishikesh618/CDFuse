"""End-to-end comparison pipeline.

Keeping the pipeline free of Streamlit means the exact code path the web app
uses can also be driven from a script, a notebook, or the test suite.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import geopandas as gpd
import pandas as pd
import xarray as xr

from . import preprocess
from .config import AGGREGATION_FREQUENCIES, METRIC_CONFIG
from .io import clip_to_boundary
from .metrics import METRIC_FUNCTIONS, summarise


@dataclass
class ComparisonSettings:
    """Everything the user chooses for one comparison run."""

    candidate_time: str = "time"
    candidate_lat: str = "lat"
    candidate_lon: str = "lon"
    candidate_shift_hours: float = 0.0
    candidate_hour: int | None = None

    reference_time: str = "time"
    reference_lat: str = "lat"
    reference_lon: str = "lon"
    reference_shift_hours: float = 0.0
    reference_hour: int | None = None

    match_hours_by_date: bool = True
    aggregation_level: str = "Daily"
    aggregation_method: str = "Mean"
    spatial_method: str = "Interpolate candidate to reference grid"
    metrics: Sequence[str] = ("Correlation", "NSE", "KGE", "PBIAS")
    clip_to_boundary: bool = False


@dataclass
class ComparisonResult:
    """Metric fields plus the context needed to interpret and reproduce them."""

    arrays: dict[str, xr.DataArray]
    summary: pd.DataFrame
    notes: list[str] = field(default_factory=list)
    time_steps: int = 0
    lat_cells: int = 0
    lon_cells: int = 0
    provenance: dict[str, object] = field(default_factory=dict)


def run_comparison(
    candidate: xr.DataArray,
    reference: xr.DataArray,
    settings: ComparisonSettings,
    boundary: gpd.GeoDataFrame | None = None,
) -> ComparisonResult:
    """Standardise, align, and compare two data arrays."""
    if not settings.metrics:
        raise preprocess.PreprocessError("Select at least one metric to calculate.")

    unknown = [name for name in settings.metrics if name not in METRIC_FUNCTIONS]
    if unknown:
        raise preprocess.PreprocessError(f"Unknown metric(s): {', '.join(unknown)}.")

    notes: list[str] = []

    prepared_candidate = preprocess.standardise(
        candidate,
        settings.candidate_time,
        settings.candidate_lat,
        settings.candidate_lon,
        settings.candidate_shift_hours,
    )
    prepared_reference = preprocess.standardise(
        reference,
        settings.reference_time,
        settings.reference_lat,
        settings.reference_lon,
        settings.reference_shift_hours,
    )
    notes.extend(f"Candidate: {note}" for note in prepared_candidate.notes)
    notes.extend(f"Reference: {note}" for note in prepared_reference.notes)

    candidate_data = prepared_candidate.data
    reference_data = prepared_reference.data

    if settings.candidate_hour is not None:
        selected = preprocess.select_hour_of_day(
            candidate_data, settings.candidate_hour, settings.match_hours_by_date
        )
        candidate_data = selected.data
        notes.extend(f"Candidate: {note}" for note in selected.notes)

    if settings.reference_hour is not None:
        selected = preprocess.select_hour_of_day(
            reference_data, settings.reference_hour, settings.match_hours_by_date
        )
        reference_data = selected.data
        notes.extend(f"Reference: {note}" for note in selected.notes)

    frequency = AGGREGATION_FREQUENCIES.get(settings.aggregation_level)
    if frequency is not None:
        candidate_data = preprocess.aggregate(candidate_data, frequency, settings.aggregation_method)
        reference_data = preprocess.aggregate(reference_data, frequency, settings.aggregation_method)
        notes.append(
            f"Aggregated both datasets to {settings.aggregation_level.lower()} "
            f"using the {settings.aggregation_method.lower()}."
        )

    candidate_data, reference_data, align_notes = preprocess.align(
        candidate_data, reference_data, settings.spatial_method
    )
    notes.extend(align_notes)

    arrays: dict[str, xr.DataArray] = {}
    summary_rows: list[dict[str, float | int | str]] = []

    for metric_name in settings.metrics:
        result = METRIC_FUNCTIONS[metric_name](candidate_data, reference_data)

        if settings.clip_to_boundary and boundary is not None:
            result = clip_to_boundary(result, boundary)

        result.attrs.update(
            {
                "metric": metric_name,
                "units": str(METRIC_CONFIG.get(metric_name, {}).get("units", "")),
                "aggregation": f"{settings.aggregation_level} / {settings.aggregation_method}",
                "spatial_alignment": settings.spatial_method,
            }
        )
        arrays[metric_name] = result
        summary_rows.append(summarise(result, metric_name))

    provenance = {
        "Aggregation": f"{settings.aggregation_level} ({settings.aggregation_method})",
        "Spatial alignment": settings.spatial_method,
        "Candidate time shift (h)": settings.candidate_shift_hours,
        "Reference time shift (h)": settings.reference_shift_hours,
        "Candidate hour filter": (
            f"{settings.candidate_hour:02d}:00" if settings.candidate_hour is not None else "none"
        ),
        "Reference hour filter": (
            f"{settings.reference_hour:02d}:00" if settings.reference_hour is not None else "none"
        ),
        "Hours matched by calendar date": (
            settings.match_hours_by_date
            if (settings.candidate_hour is not None or settings.reference_hour is not None)
            else "n/a"
        ),
        "Metrics": ", ".join(settings.metrics),
        "Clipped to boundary": bool(settings.clip_to_boundary and boundary is not None),
        "Matched time steps": int(candidate_data.sizes.get("time", 0)),
        "Grid": f"{candidate_data.sizes.get('lat', 0)} lat x {candidate_data.sizes.get('lon', 0)} lon",
    }

    return ComparisonResult(
        arrays=arrays,
        summary=pd.DataFrame(summary_rows),
        notes=notes,
        time_steps=int(candidate_data.sizes.get("time", 0)),
        lat_cells=int(candidate_data.sizes.get("lat", 0)),
        lon_cells=int(candidate_data.sizes.get("lon", 0)),
        provenance=provenance,
    )
