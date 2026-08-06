"""Standardisation, temporal processing and grid alignment.

The functions here turn an arbitrary user-supplied ``DataArray`` into the
canonical shape CDFuse works with: dimensions named ``time``, ``lat`` and
``lon``, sorted ascending, with longitudes on the -180..180 convention.

No product-specific behaviour is applied anywhere in this module. Time
shifts and hour-of-day selection happen only when the user asks for them.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import xarray as xr

CORE_DIMS = ("time", "lat", "lon")


class PreprocessError(ValueError):
    """Raised for problems a user can act on, with an explanatory message."""


@dataclass
class PreparedData:
    """A standardised array plus notes about what was changed along the way."""

    data: xr.DataArray
    notes: list[str] = field(default_factory=list)


def squeeze_extra_dimensions(
    data: xr.DataArray, selections: dict[str, object], core_dims: set[str]
) -> xr.DataArray:
    """Reduce non-core dimensions (level, ensemble member, ...) to one slice.

    Only the extra dimensions are squeezed out. A blanket ``squeeze()`` would
    also drop ``lat`` or ``lon`` on single-row or single-column grids, which
    would break alignment later on.
    """
    extra = [dim for dim in data.dims if dim not in core_dims]
    if not extra:
        return data

    for dim in extra:
        if dim not in selections:
            continue
        selector = selections[dim]
        if isinstance(selector, int) and dim not in data.coords:
            data = data.isel({dim: selector})
        else:
            data = data.sel({dim: selector})

    # Drop any length-1 leftovers, but never the core dimensions.
    droppable = [
        dim for dim in data.dims if dim not in core_dims and data.sizes[dim] == 1
    ]
    if droppable:
        data = data.squeeze(droppable, drop=True)
    return data


def _to_datetime_index(values: np.ndarray) -> pd.DatetimeIndex:
    """Convert a time coordinate to a pandas DatetimeIndex.

    Handles numpy datetime64, cftime objects from non-standard calendars, and
    already-decoded datetimes.
    """
    index = pd.Index(values)
    if isinstance(index, pd.DatetimeIndex):
        return index

    # cftime indexes expose a converter for standard-calendar-compatible dates.
    to_datetimeindex = getattr(index, "to_datetimeindex", None)
    if callable(to_datetimeindex):
        try:
            return to_datetimeindex()
        except Exception:  # noqa: BLE001 - fall through to the generic parser
            pass

    try:
        return pd.DatetimeIndex(pd.to_datetime(values))
    except Exception as exc:  # noqa: BLE001
        raise PreprocessError(
            "The selected time coordinate could not be read as dates. Choose a "
            "different coordinate, or convert the calendar before uploading."
        ) from exc


def standardise(
    data: xr.DataArray,
    time_name: str,
    lat_name: str,
    lon_name: str,
    time_shift_hours: float = 0.0,
) -> PreparedData:
    """Rename core coordinates, decode time, normalise longitudes and sort."""
    notes: list[str] = []

    for label, name in (("time", time_name), ("latitude", lat_name), ("longitude", lon_name)):
        if name not in data.dims and name not in data.coords:
            raise PreprocessError(
                f"'{name}' was chosen as the {label} coordinate but it is not present "
                f"on this variable. Available dimensions: {', '.join(map(str, data.dims))}."
            )

    for name, axis in ((lat_name, "latitude"), (lon_name, "longitude")):
        coord = data.coords.get(name)
        if coord is not None and coord.ndim > 1:
            raise PreprocessError(
                f"The {axis} coordinate '{name}' is {coord.ndim}-dimensional. CDFuse "
                "supports regular 1-D latitude/longitude grids; curvilinear grids need "
                "to be regridded before upload."
            )

    rename_map = {
        original: standard
        for original, standard in ((time_name, "time"), (lat_name, "lat"), (lon_name, "lon"))
        if original != standard
    }
    # A stale coordinate already occupying a target name would collide.
    collisions = [
        target
        for target in rename_map.values()
        if target in data.coords and target not in rename_map
    ]
    if collisions:
        data = data.drop_vars(collisions)
        notes.append(
            "Dropped unused coordinate(s) " + ", ".join(collisions) + " to free the standard names."
        )
    if rename_map:
        data = data.rename(rename_map)

    missing = [dim for dim in CORE_DIMS if dim not in data.dims]
    if missing:
        raise PreprocessError(
            "After mapping the coordinates the variable is still missing: "
            + ", ".join(missing)
            + ". Check that the chosen variable is a gridded time series."
        )

    time_index = _to_datetime_index(data["time"].values)
    if time_shift_hours:
        time_index = time_index + pd.to_timedelta(float(time_shift_hours), unit="h")
        notes.append(f"Applied a {float(time_shift_hours):+g} hour time shift.")
    data = data.assign_coords(time=time_index)

    lon_values = np.asarray(data["lon"].values, dtype="float64")
    if lon_values.size and np.nanmax(lon_values) > 180:
        data = data.assign_coords(lon=((lon_values + 180) % 360) - 180)
        notes.append("Converted longitudes from the 0..360 convention to -180..180.")

    data = data.sortby("time").sortby("lat").sortby("lon")

    for dim in CORE_DIMS:
        if data.get_index(dim).has_duplicates:
            data = data.groupby(dim).mean(skipna=True)
            notes.append(f"Averaged duplicate {dim} values.")

    return PreparedData(data.astype("float64"), notes)


def select_hour_of_day(
    data: xr.DataArray, hour: int, match_by_date: bool = True
) -> PreparedData:
    """Keep a single hourly timestamp per day.

    This is a neutral filter. CDFuse makes no assumption about what any given
    hour represents in the user's product; interpreting the accumulation or
    timestamping convention is the user's responsibility.
    """
    notes: list[str] = []
    hour = int(hour)
    mask = data["time"].dt.hour.values == hour
    if not mask.any():
        available = sorted({int(h) for h in data["time"].dt.hour.values})
        raise PreprocessError(
            f"No timestamps at {hour:02d}:00 were found. Hours present in this dataset: "
            + ", ".join(f"{h:02d}:00" for h in available[:12])
            + ("..." if len(available) > 12 else "")
        )

    selected = data.isel(time=np.flatnonzero(mask))
    notes.append(f"Kept {int(mask.sum())} timestamps at {hour:02d}:00.")

    if match_by_date:
        dates = pd.DatetimeIndex(selected["time"].values).normalize()
        if dates.has_duplicates:
            raise PreprocessError(
                f"More than one record per day was found at {hour:02d}:00, so matching by "
                "calendar date would be ambiguous. Turn off date matching, or reduce the "
                "data to one value per hour before uploading."
            )
        selected = selected.assign_coords(time=dates)
        notes.append("Replaced the selected timestamps with their calendar dates.")

    return PreparedData(selected, notes)


def aggregate(data: xr.DataArray, frequency: str | None, method: str) -> xr.DataArray:
    """Resample in time. ``frequency`` of None leaves the data untouched."""
    if frequency is None:
        return data

    if data.sizes.get("time", 0) < 1:
        raise PreprocessError("There are no time steps left to aggregate.")

    resampler = data.resample(time=frequency)
    operations = {
        "Mean": lambda: resampler.mean(skipna=True),
        "Sum": lambda: resampler.sum(skipna=True),
        "Minimum": lambda: resampler.min(skipna=True),
        "Maximum": lambda: resampler.max(skipna=True),
        "Median": lambda: resampler.median(skipna=True),
    }
    if method not in operations:
        raise PreprocessError(f"Unknown aggregation statistic '{method}'.")
    return operations[method]()


def align(
    candidate: xr.DataArray, reference: xr.DataArray, spatial_method: str
) -> tuple[xr.DataArray, xr.DataArray, list[str]]:
    """Restrict both arrays to shared time steps and a common grid."""
    notes: list[str] = []

    common_time = np.intersect1d(candidate["time"].values, reference["time"].values)
    if common_time.size == 0:
        raise PreprocessError(
            "The two datasets share no time steps. Check the aggregation level, any time "
            "shift, and whether the periods covered actually overlap.\n\n"
            f"Candidate: {_describe_period(candidate)}\n"
            f"Reference: {_describe_period(reference)}"
        )
    if common_time.size < 2:
        raise PreprocessError(
            "Only one time step is shared by the two datasets. Metrics through time need "
            "at least two. Widen the period or use a finer aggregation level."
        )

    candidate = candidate.sel(time=common_time)
    reference = reference.sel(time=common_time)
    notes.append(f"Matched {common_time.size} time steps.")

    if spatial_method == "Interpolate candidate to reference grid":
        candidate = candidate.interp(lat=reference["lat"], lon=reference["lon"])
        notes.append("Interpolated the candidate onto the reference grid (linear).")
    elif spatial_method == "Interpolate reference to candidate grid":
        reference = reference.interp(lat=candidate["lat"], lon=candidate["lon"])
        notes.append("Interpolated the reference onto the candidate grid (linear).")
    else:
        common_lat = np.intersect1d(candidate["lat"].values, reference["lat"].values)
        common_lon = np.intersect1d(candidate["lon"].values, reference["lon"].values)
        if common_lat.size == 0 or common_lon.size == 0:
            raise PreprocessError(
                "The two grids share no exact latitude/longitude values, so they cannot be "
                "compared cell by cell. Choose one of the interpolation options instead."
            )
        candidate = candidate.sel(lat=common_lat, lon=common_lon)
        reference = reference.sel(lat=common_lat, lon=common_lon)
        notes.append(f"Kept {common_lat.size} x {common_lon.size} exactly shared cells.")

    candidate, reference = xr.align(candidate, reference, join="inner")

    if candidate.sizes.get("lat", 0) == 0 or candidate.sizes.get("lon", 0) == 0:
        raise PreprocessError(
            "Spatial alignment produced an empty grid. The two datasets probably cover "
            "different regions.\n\n"
            f"Candidate extent: {_describe_extent(candidate)}\n"
            f"Reference extent: {_describe_extent(reference)}"
        )

    if not np.isfinite(candidate.values).any() or not np.isfinite(reference.values).any():
        raise PreprocessError(
            "One of the datasets has no valid values left after alignment. If the grids "
            "only partially overlap, interpolation can leave the edges empty."
        )

    return candidate, reference, notes


def _describe_period(data: xr.DataArray) -> str:
    times = pd.DatetimeIndex(data["time"].values)
    if times.empty:
        return "no time steps"
    return f"{times.min():%Y-%m-%d %H:%M} to {times.max():%Y-%m-%d %H:%M} ({times.size} steps)"


def _describe_extent(data: xr.DataArray) -> str:
    if "lat" not in data.coords or "lon" not in data.coords:
        return "unknown"
    lat = np.asarray(data["lat"].values, dtype="float64")
    lon = np.asarray(data["lon"].values, dtype="float64")
    if lat.size == 0 or lon.size == 0:
        return "empty"
    return (
        f"lat {np.nanmin(lat):.3f} to {np.nanmax(lat):.3f}, "
        f"lon {np.nanmin(lon):.3f} to {np.nanmax(lon):.3f}"
    )


def describe(data: xr.DataArray) -> dict[str, str]:
    """Human-readable summary of a standardised array, for the UI."""
    return {
        "Period": _describe_period(data),
        "Extent": _describe_extent(data),
        "Grid": f"{data.sizes.get('lat', 0)} lat x {data.sizes.get('lon', 0)} lon",
        "Time steps": f"{data.sizes.get('time', 0):,}",
    }
