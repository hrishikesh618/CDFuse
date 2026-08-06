"""Application-wide constants for CDFuse.

Nothing in this module is specific to any single data product. Coordinate
name candidates are common conventions used for auto-detection only; the
user can always override the detected names in the interface.
"""

from __future__ import annotations

APP_NAME = "CDFuse"
APP_TAGLINE = "Compare. Validate. Visualise."
APP_DESCRIPTION = (
    "Compare two gridded NetCDF datasets: align them in time and space, "
    "calculate cell-wise performance metrics, map the results, and export "
    "reproducible outputs."
)
VERSION = "1.0.0"

# Coordinate names tried when guessing which coordinate is which. The first
# match wins. Users can override every guess in the interface.
TIME_CANDIDATES = ("time", "valid_time", "datetime", "date", "t")
LAT_CANDIDATES = ("lat", "latitude", "nav_lat", "y", "rlat")
LON_CANDIDATES = ("lon", "longitude", "nav_lon", "x", "rlon")

NETCDF_SUFFIXES = ("nc", "nc4", "cdf")
VECTOR_SUFFIXES = ("zip", "shp", "shx", "dbf", "prj", "cpg", "geojson", "json", "gpkg")

# Plot defaults per metric. ``vmin``/``vmax`` of None means "scale to the data",
# which suits unbounded error metrics such as RMSE.
METRIC_CONFIG: dict[str, dict[str, object]] = {
    "Correlation": {
        "vmin": -1.0,
        "vmax": 1.0,
        "cmap": "RdBu_r",
        "units": "-",
        "perfect": 1.0,
        "description": "Pearson correlation of the two series through time at each cell.",
        "guidance": "Ranges from -1 to 1. Higher is better; 1 means the timing of variations matches exactly.",
    },
    "NSE": {
        "vmin": -1.0,
        "vmax": 1.0,
        "cmap": "viridis",
        "units": "-",
        "perfect": 1.0,
        "description": "Nash-Sutcliffe Efficiency, comparing the candidate against the reference mean.",
        "guidance": "1 is perfect. 0 means the candidate is no better than the reference mean; below 0 is worse than that mean.",
    },
    "KGE": {
        "vmin": -1.0,
        "vmax": 1.0,
        "cmap": "plasma",
        "units": "-",
        "perfect": 1.0,
        "description": "Kling-Gupta Efficiency, combining correlation, variability and bias.",
        "guidance": "1 is perfect. Values above roughly -0.41 improve on using the reference mean alone.",
    },
    "PBIAS": {
        "vmin": -100.0,
        "vmax": 100.0,
        "cmap": "BrBG",
        "units": "%",
        "perfect": 0.0,
        "description": "Percentage bias of the candidate relative to the reference total.",
        "guidance": "0 is unbiased. Positive means the candidate overestimates; negative means it underestimates.",
    },
    "RMSE": {
        "vmin": None,
        "vmax": None,
        "cmap": "magma_r",
        "units": "same as variable",
        "perfect": 0.0,
        "description": "Root mean square error, in the units of the compared variable.",
        "guidance": "0 is perfect. Penalises large discrepancies more heavily than MAE.",
    },
    "MAE": {
        "vmin": None,
        "vmax": None,
        "cmap": "magma_r",
        "units": "same as variable",
        "perfect": 0.0,
        "description": "Mean absolute error, in the units of the compared variable.",
        "guidance": "0 is perfect. Represents the typical absolute discrepancy.",
    },
    "Bias": {
        "vmin": None,
        "vmax": None,
        "cmap": "RdBu_r",
        "units": "same as variable",
        "perfect": 0.0,
        "description": "Mean difference (candidate minus reference), in the units of the variable.",
        "guidance": "0 is unbiased. Positive means the candidate reads high on average.",
    },
}

DEFAULT_METRICS = ["Correlation", "NSE", "KGE", "PBIAS"]

AGGREGATION_FREQUENCIES = {
    "None (use matched time steps)": None,
    "Daily": "1D",
    "Monthly": "MS",
    "Seasonal (DJF/MAM/JJA/SON)": "QS-DEC",
    "Annual": "YS",
}

AGGREGATION_METHODS = ("Mean", "Sum", "Minimum", "Maximum", "Median")

SPATIAL_METHODS = (
    "Interpolate candidate to reference grid",
    "Interpolate reference to candidate grid",
    "Use exact shared coordinates only",
)

# Guardrails. Uploads are held in memory, so keep an eye on size.
MAX_UPLOAD_MB = 500
LARGE_FILE_WARNING_MB = 150
