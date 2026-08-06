"""Map rendering for metric fields.

Cartopy is used when it is installed, giving coastlines and national borders
as in the original desktop tool. When it is unavailable the same map is drawn
on plain matplotlib axes, so the application still works on hosts where
cartopy cannot be installed. Nothing else in CDFuse depends on cartopy.
"""

from __future__ import annotations

import io

import matplotlib
import numpy as np
import xarray as xr

matplotlib.use("Agg")  # Streamlit renders figures server-side; no GUI backend.

import matplotlib.pyplot as plt  # noqa: E402

try:  # pragma: no cover - depends on the deployment environment
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    CARTOPY_AVAILABLE = True
except Exception:  # noqa: BLE001
    ccrs = None
    cfeature = None
    CARTOPY_AVAILABLE = False

DIVERGING_METRICS = {"Bias", "PBIAS", "Correlation"}


def resolve_limits(
    data: xr.DataArray,
    metric_name: str,
    vmin: float | None,
    vmax: float | None,
) -> tuple[float | None, float | None]:
    """Fill in colour limits for metrics that have no natural fixed range.

    Robust percentiles keep a handful of extreme cells from flattening the
    rest of the map. Diverging metrics are kept symmetric about zero so the
    colour scale stays honest about the sign.
    """
    if vmin is not None and vmax is not None:
        return vmin, vmax

    values = np.asarray(data.values, dtype="float64")
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None, None

    low, high = (float(bound) for bound in np.percentile(finite, [2, 98]))
    if not np.isfinite(low) or not np.isfinite(high) or np.isclose(low, high):
        low, high = float(np.min(finite)), float(np.max(finite))

    diverging = metric_name in DIVERGING_METRICS

    if np.isclose(low, high):
        # A constant field still deserves a sensible scale rather than a
        # matplotlib warning and an arbitrary auto-range.
        level = low
        if diverging:
            bound = abs(level) or 1.0
            return -bound, bound
        padding = abs(level) * 0.05 or 0.5
        return level - padding, level + padding

    if diverging:
        bound = max(abs(low), abs(high))
        return -bound, bound
    return low, high


def make_map(
    data: xr.DataArray,
    metric_name: str,
    boundary=None,
    vmin: float | None = None,
    vmax: float | None = None,
    cmap: str = "viridis",
    units: str = "",
    use_cartopy: bool = True,
    title: str | None = None,
) -> plt.Figure:
    """Render one metric field as a map figure."""
    vmin, vmax = resolve_limits(data, metric_name, vmin, vmax)
    with_cartopy = bool(use_cartopy and CARTOPY_AVAILABLE)

    if with_cartopy:
        fig, ax = plt.subplots(
            figsize=(10.5, 7.0),
            subplot_kw={"projection": ccrs.PlateCarree()},
            constrained_layout=True,
        )
        plot_kwargs = {"transform": ccrs.PlateCarree()}
    else:
        fig, ax = plt.subplots(figsize=(10.5, 7.0), constrained_layout=True)
        plot_kwargs = {}

    mesh = data.plot(
        ax=ax,
        x="lon",
        y="lat",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        add_colorbar=False,
        add_labels=False,
        **plot_kwargs,
    )

    if with_cartopy:
        ax.coastlines(resolution="110m", linewidth=0.7)
        ax.add_feature(cfeature.BORDERS, linestyle=":", linewidth=0.6)
        gridlines = ax.gridlines(
            draw_labels=True, linewidth=0.5, color="gray", alpha=0.4, linestyle="--"
        )
        gridlines.top_labels = False
        gridlines.right_labels = False
    else:
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(True, linewidth=0.5, alpha=0.35, linestyle="--")

    if boundary is not None and not boundary.empty:
        boundary.boundary.plot(ax=ax, linewidth=0.9, edgecolor="black", **plot_kwargs)

    label = metric_name if not units or units == "-" else f"{metric_name} ({units})"
    colorbar = fig.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.06, fraction=0.055)
    colorbar.set_label(label, fontsize=11)

    ax.set_title(title or f"{metric_name} — candidate vs reference", fontsize=14, pad=12)
    return fig


def figure_to_png(fig: plt.Figure, dpi: int = 300) -> bytes:
    """Serialise a figure to PNG bytes for download."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=dpi, bbox_inches="tight")
    buffer.seek(0)
    return buffer.getvalue()
