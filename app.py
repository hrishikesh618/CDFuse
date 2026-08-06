"""CDFuse — a Streamlit interface for comparing two gridded NetCDF datasets.

This module is the presentation layer only. All data handling lives in the
``cdfuse`` package so it can be tested and reused independently.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import xarray as xr

from cdfuse import export, plotting, preprocess, sample
from cdfuse import io as cdio
from cdfuse.config import (
    AGGREGATION_FREQUENCIES,
    AGGREGATION_METHODS,
    APP_DESCRIPTION,
    APP_NAME,
    APP_TAGLINE,
    DEFAULT_METRICS,
    LARGE_FILE_WARNING_MB,
    LAT_CANDIDATES,
    LON_CANDIDATES,
    METRIC_CONFIG,
    NETCDF_SUFFIXES,
    SPATIAL_METHODS,
    TIME_CANDIDATES,
    VECTOR_SUFFIXES,
    VERSION,
)
from cdfuse.metrics import METRIC_FUNCTIONS
from cdfuse.pipeline import ComparisonSettings, run_comparison

st.set_page_config(
    page_title=f"{APP_NAME} — NetCDF comparison",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={"about": f"{APP_NAME} v{VERSION} — {APP_TAGLINE}"},
)

STYLES = """
<style>
  .block-container {max-width: 1400px; padding-top: 2.2rem; padding-bottom: 3rem;}
  .cdf-kicker {font-size: .8rem; letter-spacing: .14em; text-transform: uppercase; opacity: .6;}
  .cdf-title {font-size: 2.9rem; font-weight: 750; line-height: 1.05; margin: .2rem 0 .3rem;}
  .cdf-tagline {font-size: 1.05rem; opacity: .85; font-weight: 500; margin-bottom: .4rem;}
  .cdf-lead {font-size: 1rem; opacity: .75; max-width: 62rem;}
  .cdf-panel {border: 1px solid rgba(128,128,128,.25); border-radius: 12px;
              padding: 1rem 1.15rem; margin-bottom: .8rem;}
  [data-testid="stMetricValue"] {font-size: 1.45rem;}
  div[data-testid="stDataFrame"] {width: 100%;}

  /* Tabs read as steps, so keep them legible when they wrap on narrow screens. */
  button[data-baseweb="tab"] {font-size: 1rem; font-weight: 600;}
  div[data-baseweb="tab-list"] {flex-wrap: wrap; gap: .15rem;}

  @media (max-width: 900px) {
    .block-container {padding-top: 1.2rem; padding-left: 1rem; padding-right: 1rem;}
    .cdf-title {font-size: 2.1rem;}
    .cdf-tagline {font-size: .98rem;}
    .cdf-lead {font-size: .93rem;}
    button[data-baseweb="tab"] {font-size: .88rem;}
    [data-testid="stMetricValue"] {font-size: 1.2rem;}
  }
  @media (max-width: 640px) {
    .cdf-title {font-size: 1.75rem;}
    .cdf-panel {padding: .8rem .9rem;}
  }
</style>
"""
st.markdown(STYLES, unsafe_allow_html=True)


# --------------------------------------------------------------------------- #
# Session state
# --------------------------------------------------------------------------- #

DEFAULTS: dict[str, object] = {
    "candidate_ds": None,
    "reference_ds": None,
    "candidate_label": "",
    "reference_label": "",
    "boundary": None,
    "boundary_label": "",
    "results": None,
    "result_files": None,
    "demo_loaded": False,
}
for key, value in DEFAULTS.items():
    st.session_state.setdefault(key, value)


@st.cache_data(show_spinner=False, max_entries=4)
def cached_open_netcdf(file_bytes: bytes, filename: str) -> xr.Dataset:
    return cdio.open_netcdf(file_bytes, filename)


@st.cache_data(show_spinner=False)
def cached_demo_pair() -> tuple[xr.Dataset, xr.Dataset]:
    return sample.make_demo_pair()


def detect_index(options: list[str], candidates: tuple[str, ...]) -> int:
    """Index of the first recognised coordinate name, else 0."""
    lowered = [str(option).lower() for option in options]
    for candidate in candidates:
        if candidate in lowered:
            return lowered.index(candidate)
    return 0


def coordinate_options(dataset: xr.Dataset, variable: str) -> list[str]:
    """Names that could plausibly be the time/lat/lon coordinate of a variable."""
    data = dataset[variable]
    names = list(dict.fromkeys(list(data.dims) + list(data.coords)))
    return [str(name) for name in names]


def dataset_overview(dataset: xr.Dataset, variable: str) -> None:
    """Compact description of the selected variable."""
    data = dataset[variable]
    attributes = data.attrs
    units = attributes.get("units", "not recorded")
    long_name = attributes.get("long_name") or attributes.get("standard_name") or variable

    st.caption(f"**{long_name}** · units: `{units}`")
    dimensions = ", ".join(f"{dim}: {size}" for dim, size in data.sizes.items())
    st.caption(f"Dimensions — {dimensions}")

    size_mb = data.nbytes / 1024**2
    if size_mb > LARGE_FILE_WARNING_MB:
        st.warning(
            f"This variable holds about {size_mb:,.0f} MB in memory. Large arrays are slow on "
            "shared hosting; consider cropping the region or period first.",
            icon="⚠️",
        )


def extra_dimension_selection(
    dataset: xr.Dataset, variable: str, core: set[str], prefix: str
) -> dict[str, object]:
    """Ask for one slice per dimension that is not time, latitude or longitude."""
    data = dataset[variable]
    extra = [str(dim) for dim in data.dims if str(dim) not in core]
    if not extra:
        return {}

    selections: dict[str, object] = {}
    st.caption("Additional dimensions — choose the slice to compare.")
    for dim in extra:
        coordinate = data.coords.get(dim)
        if coordinate is not None and coordinate.ndim == 1:
            values = list(coordinate.values)
            labels = [str(value) for value in values]
            chosen = st.selectbox(dim, labels, key=f"{prefix}_dim_{dim}")
            selections[dim] = values[labels.index(chosen)]
        else:
            index = st.number_input(
                f"{dim} (index)",
                min_value=0,
                max_value=max(0, data.sizes[dim] - 1),
                value=0,
                step=1,
                key=f"{prefix}_dim_{dim}",
            )
            selections[dim] = int(index)
    return selections


def reset_results() -> None:
    st.session_state["results"] = None
    st.session_state["result_files"] = None


# --------------------------------------------------------------------------- #
# Header and sidebar
# --------------------------------------------------------------------------- #

st.markdown('<div class="cdf-kicker">Gridded dataset evaluation</div>', unsafe_allow_html=True)
st.markdown(f'<div class="cdf-title">{APP_NAME}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="cdf-tagline">{APP_TAGLINE}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="cdf-lead">{APP_DESCRIPTION}</div>', unsafe_allow_html=True)
st.write("")

with st.sidebar:
    st.subheader("Workflow")
    steps = [
        ("Load data", st.session_state["candidate_ds"] is not None and st.session_state["reference_ds"] is not None),
        ("Configure", st.session_state["candidate_ds"] is not None),
        ("Run comparison", st.session_state["results"] is not None),
        ("Download results", st.session_state["results"] is not None),
    ]
    for index, (label, done) in enumerate(steps, start=1):
        st.markdown(f"{'✅' if done else '⬜'} **{index}. {label}**")

    st.divider()
    st.subheader("Session")
    if st.session_state["candidate_label"]:
        st.caption(f"Candidate: `{st.session_state['candidate_label']}`")
    if st.session_state["reference_label"]:
        st.caption(f"Reference: `{st.session_state['reference_label']}`")
    if st.session_state["boundary_label"]:
        st.caption(f"Boundary: `{st.session_state['boundary_label']}`")
    if not any(
        st.session_state[key] for key in ("candidate_label", "reference_label", "boundary_label")
    ):
        st.caption("Nothing loaded yet.")

    if st.button("Clear session", use_container_width=True):
        for key, value in DEFAULTS.items():
            st.session_state[key] = value
        st.cache_data.clear()
        st.rerun()

    st.divider()
    st.caption(
        "Uploads are held in memory for the length of your session and are not written to "
        "permanent storage. Treat any public deployment as a third-party service and avoid "
        "uploading restricted data."
    )
    st.caption(
        f"{APP_NAME} v{VERSION} · "
        + ("cartopy basemap active" if plotting.CARTOPY_AVAILABLE else "plain matplotlib basemap")
    )


tab_data, tab_configure, tab_results, tab_guide = st.tabs(
    ["1 · Data", "2 · Configure", "3 · Results", "Guide"]
)


# --------------------------------------------------------------------------- #
# Tab 1 — Data
# --------------------------------------------------------------------------- #

with tab_data:
    st.subheader("Load the two datasets")
    st.markdown(
        "The **candidate** is the dataset being evaluated. The **reference** is the benchmark "
        "it is compared against. NSE, KGE and PBIAS all treat the reference as truth, so the "
        "order matters."
    )

    upload_left, upload_right = st.columns(2)
    with upload_left:
        candidate_file = st.file_uploader(
            "Candidate dataset (NetCDF)",
            type=list(NETCDF_SUFFIXES),
            key="candidate_upload",
            help="A model run, reanalysis, satellite product, or any dataset under evaluation.",
        )
    with upload_right:
        reference_file = st.file_uploader(
            "Reference dataset (NetCDF)",
            type=list(NETCDF_SUFFIXES),
            key="reference_upload",
            help="The benchmark or observed dataset.",
        )

    if candidate_file is not None:
        try:
            with st.spinner(f"Reading {candidate_file.name}…"):
                st.session_state["candidate_ds"] = cached_open_netcdf(
                    candidate_file.getvalue(), candidate_file.name
                )
                st.session_state["candidate_label"] = candidate_file.name
                st.session_state["demo_loaded"] = False
                reset_results()
        except cdio.DataLoadError as error:
            st.error(str(error), icon="🚫")
        except Exception as error:  # noqa: BLE001
            st.error(f"Unexpected problem reading the candidate file: {error}", icon="🚫")

    if reference_file is not None:
        try:
            with st.spinner(f"Reading {reference_file.name}…"):
                st.session_state["reference_ds"] = cached_open_netcdf(
                    reference_file.getvalue(), reference_file.name
                )
                st.session_state["reference_label"] = reference_file.name
                st.session_state["demo_loaded"] = False
                reset_results()
        except cdio.DataLoadError as error:
            st.error(str(error), icon="🚫")
        except Exception as error:  # noqa: BLE001
            st.error(f"Unexpected problem reading the reference file: {error}", icon="🚫")

    with st.expander("No data to hand? Load the built-in demonstration pair"):
        st.markdown(
            "Two small synthetic datasets on a 1° grid over 180 days. The candidate is the "
            "reference with a **+12% scaling**, a **+0.15 offset**, independent noise, and a "
            "block of missing values — so correlation comes out high while PBIAS and Bias are "
            "clearly positive. Nothing about them is tied to a real product."
        )
        demo_left, demo_right = st.columns([1, 1])
        with demo_left:
            if st.button("Load demonstration data", use_container_width=True, type="secondary"):
                demo_candidate, demo_reference = cached_demo_pair()
                st.session_state["candidate_ds"] = demo_candidate
                st.session_state["reference_ds"] = demo_reference
                st.session_state["candidate_label"] = "demo_candidate.nc"
                st.session_state["reference_label"] = "demo_reference.nc"
                st.session_state["demo_loaded"] = True
                reset_results()
                st.rerun()
        with demo_right:
            if st.session_state["demo_loaded"]:
                demo_candidate, demo_reference = cached_demo_pair()
                st.download_button(
                    "Download demo candidate (.nc)",
                    data=sample.demo_bytes(demo_candidate),
                    file_name="demo_candidate.nc",
                    mime="application/x-netcdf",
                    use_container_width=True,
                )
                st.download_button(
                    "Download demo reference (.nc)",
                    data=sample.demo_bytes(demo_reference),
                    file_name="demo_reference.nc",
                    mime="application/x-netcdf",
                    use_container_width=True,
                )

    st.divider()
    st.subheader("Optional boundary")
    st.markdown(
        "Overlay a study-area outline on every map, and optionally mask the metric fields to it."
    )
    boundary_files = st.file_uploader(
        "Boundary layer",
        type=list(VECTOR_SUFFIXES),
        accept_multiple_files=True,
        key="boundary_upload",
        help=(
            "Easiest option is a single ZIP containing the shapefile. You can also upload the "
            ".shp/.shx/.dbf/.prj components together, or a GeoJSON or GeoPackage."
        ),
    )

    if boundary_files:
        try:
            with st.spinner("Reading the boundary…"):
                st.session_state["boundary"] = cdio.load_boundary(boundary_files)
                st.session_state["boundary_label"] = ", ".join(
                    uploaded.name for uploaded in boundary_files[:3]
                )
                reset_results()
            boundary = st.session_state["boundary"]
            if boundary is not None:
                bounds = boundary.total_bounds
                st.success(
                    f"Boundary loaded: {len(boundary)} feature(s), extent "
                    f"lon {bounds[0]:.3f} to {bounds[2]:.3f}, lat {bounds[1]:.3f} to {bounds[3]:.3f}.",
                    icon="✅",
                )
        except cdio.DataLoadError as error:
            st.error(str(error), icon="🚫")
            st.session_state["boundary"] = None
        except Exception as error:  # noqa: BLE001
            st.error(f"Unexpected problem reading the boundary: {error}", icon="🚫")
            st.session_state["boundary"] = None

    if st.session_state["candidate_ds"] is not None and st.session_state["reference_ds"] is not None:
        st.success("Both datasets are loaded. Continue to **2 · Configure**.", icon="➡️")
    else:
        st.info("Upload both datasets, or load the demonstration pair, to continue.", icon="ℹ️")


# --------------------------------------------------------------------------- #
# Tab 2 — Configure
# --------------------------------------------------------------------------- #

with tab_configure:
    candidate_ds = st.session_state["candidate_ds"]
    reference_ds = st.session_state["reference_ds"]

    if candidate_ds is None or reference_ds is None:
        st.info("Load both datasets on the **1 · Data** tab first.", icon="ℹ️")
        st.stop()

    st.subheader("Variables and coordinates")
    st.caption(
        "CDFuse guesses the coordinate names from common conventions. Override anything it gets "
        "wrong — no assumption is made about which product the data came from."
    )

    config_left, config_right = st.columns(2)

    with config_left:
        st.markdown("#### Candidate")
        candidate_vars = [str(name) for name in candidate_ds.data_vars]
        candidate_var = st.selectbox("Variable", candidate_vars, key="candidate_var")
        dataset_overview(candidate_ds, candidate_var)

        options = coordinate_options(candidate_ds, candidate_var)
        candidate_time = st.selectbox(
            "Time coordinate", options,
            index=detect_index(options, TIME_CANDIDATES), key="candidate_time",
        )
        candidate_lat = st.selectbox(
            "Latitude coordinate", options,
            index=detect_index(options, LAT_CANDIDATES), key="candidate_lat",
        )
        candidate_lon = st.selectbox(
            "Longitude coordinate", options,
            index=detect_index(options, LON_CANDIDATES), key="candidate_lon",
        )
        candidate_shift = st.number_input(
            "Time shift (hours)", value=0.0, step=1.0, key="candidate_shift",
            help="Shifts every timestamp. Positive moves forward in time, negative moves backward.",
        )
        candidate_extra = extra_dimension_selection(
            candidate_ds, candidate_var,
            {candidate_time, candidate_lat, candidate_lon}, "candidate",
        )

    with config_right:
        st.markdown("#### Reference")
        reference_vars = [str(name) for name in reference_ds.data_vars]
        reference_var = st.selectbox("Variable", reference_vars, key="reference_var")
        dataset_overview(reference_ds, reference_var)

        options = coordinate_options(reference_ds, reference_var)
        reference_time = st.selectbox(
            "Time coordinate", options,
            index=detect_index(options, TIME_CANDIDATES), key="reference_time",
        )
        reference_lat = st.selectbox(
            "Latitude coordinate", options,
            index=detect_index(options, LAT_CANDIDATES), key="reference_lat",
        )
        reference_lon = st.selectbox(
            "Longitude coordinate", options,
            index=detect_index(options, LON_CANDIDATES), key="reference_lon",
        )
        reference_shift = st.number_input(
            "Time shift (hours)", value=0.0, step=1.0, key="reference_shift",
        )
        reference_extra = extra_dimension_selection(
            reference_ds, reference_var,
            {reference_time, reference_lat, reference_lon}, "reference",
        )

    st.divider()
    st.subheader("Time handling")

    hour_mode = st.radio(
        "Sub-daily sampling",
        ["Use every timestamp", "Keep one hour per day"],
        horizontal=True,
        key="hour_mode",
        help=(
            "Use every timestamp for data that is already daily or coarser, or when you want all "
            "sub-daily records included. Keep one hour per day when each dataset stores the value "
            "you care about at a particular hour."
        ),
    )

    candidate_hour = reference_hour = None
    match_by_date = True
    if hour_mode == "Keep one hour per day":
        hour_a, hour_b, hour_c = st.columns(3)
        with hour_a:
            candidate_hour = st.selectbox(
                "Candidate hour (UTC)", list(range(24)),
                index=None, placeholder="Choose an hour",
                format_func=lambda value: f"{value:02d}:00", key="candidate_hour",
            )
        with hour_b:
            reference_hour = st.selectbox(
                "Reference hour (UTC)", list(range(24)),
                index=None, placeholder="Choose an hour",
                format_func=lambda value: f"{value:02d}:00", key="reference_hour",
            )
        with hour_c:
            match_by_date = st.checkbox(
                "Match on calendar date", value=True, key="match_by_date",
                help=(
                    "Replaces the kept timestamps with their dates, so two datasets recorded at "
                    "different hours can still be paired day by day."
                ),
            )
        st.caption(
            "The two hours are chosen independently. CDFuse attaches no special meaning to any "
            "hour — interpreting your product's timestamping and accumulation convention is up "
            "to you."
        )

    st.divider()
    st.subheader("Aggregation and alignment")

    setting_a, setting_b, setting_c = st.columns(3)
    with setting_a:
        aggregation_level = st.selectbox(
            "Temporal aggregation", list(AGGREGATION_FREQUENCIES),
            index=1, key="aggregation_level",
        )
    with setting_b:
        aggregation_method = st.selectbox(
            "Aggregation statistic", list(AGGREGATION_METHODS), key="aggregation_method",
            help="Use Sum for accumulated quantities and Mean for rates or states.",
        )
    with setting_c:
        spatial_method = st.selectbox(
            "Spatial alignment", list(SPATIAL_METHODS), key="spatial_method",
            help=(
                "Interpolation is linear. 'Exact shared coordinates' compares only cells whose "
                "latitude and longitude match precisely in both datasets."
            ),
        )

    st.divider()
    st.subheader("Metrics")

    metric_left, metric_right = st.columns([3, 1])
    with metric_left:
        selected_metrics = st.multiselect(
            "Metrics to calculate", list(METRIC_FUNCTIONS),
            default=DEFAULT_METRICS, key="selected_metrics",
        )
    with metric_right:
        boundary_available = st.session_state["boundary"] is not None
        clip_outputs = st.checkbox(
            "Clip to boundary", value=boundary_available,
            disabled=not boundary_available, key="clip_outputs",
            help="Available once a boundary is uploaded on the Data tab.",
        )

    if selected_metrics:
        with st.expander("What these metrics mean"):
            for name in selected_metrics:
                info = METRIC_CONFIG[name]
                st.markdown(f"**{name}** — {info['description']}  \n{info['guidance']}")

    st.write("")
    run_clicked = st.button(
        "Run comparison", type="primary", use_container_width=True, key="run_button"
    )

    if run_clicked:
        problems: list[str] = []
        if not selected_metrics:
            problems.append("Choose at least one metric.")
        if hour_mode == "Keep one hour per day" and (candidate_hour is None or reference_hour is None):
            problems.append("Choose an hour for both datasets, or switch back to using every timestamp.")
        if len({candidate_time, candidate_lat, candidate_lon}) < 3:
            problems.append("The candidate's time, latitude and longitude must be three different coordinates.")
        if len({reference_time, reference_lat, reference_lon}) < 3:
            problems.append("The reference's time, latitude and longitude must be three different coordinates.")

        if problems:
            for problem in problems:
                st.warning(problem, icon="⚠️")
        else:
            try:
                with st.spinner("Aligning datasets and calculating metrics…"):
                    candidate_da = preprocess.squeeze_extra_dimensions(
                        candidate_ds[candidate_var], candidate_extra,
                        {candidate_time, candidate_lat, candidate_lon},
                    )
                    reference_da = preprocess.squeeze_extra_dimensions(
                        reference_ds[reference_var], reference_extra,
                        {reference_time, reference_lat, reference_lon},
                    )

                    settings = ComparisonSettings(
                        candidate_time=candidate_time,
                        candidate_lat=candidate_lat,
                        candidate_lon=candidate_lon,
                        candidate_shift_hours=float(candidate_shift),
                        candidate_hour=candidate_hour,
                        reference_time=reference_time,
                        reference_lat=reference_lat,
                        reference_lon=reference_lon,
                        reference_shift_hours=float(reference_shift),
                        reference_hour=reference_hour,
                        match_hours_by_date=match_by_date,
                        aggregation_level=aggregation_level,
                        aggregation_method=aggregation_method,
                        spatial_method=spatial_method,
                        metrics=list(selected_metrics),
                        clip_to_boundary=bool(clip_outputs),
                    )
                    result = run_comparison(
                        candidate_da, reference_da, settings, st.session_state["boundary"]
                    )
                    result.provenance["Candidate file"] = st.session_state["candidate_label"]
                    result.provenance["Reference file"] = st.session_state["reference_label"]
                    result.provenance["Candidate variable"] = candidate_var
                    result.provenance["Reference variable"] = reference_var

                with st.spinner("Rendering maps and preparing downloads…"):
                    files: dict[str, dict[str, bytes | None]] = {}
                    for metric_name, array in result.arrays.items():
                        plot_config = METRIC_CONFIG[metric_name]
                        figure = plotting.make_map(
                            array, metric_name, st.session_state["boundary"],
                            vmin=plot_config["vmin"], vmax=plot_config["vmax"],
                            cmap=str(plot_config["cmap"]), units=str(plot_config["units"]),
                        )
                        files[metric_name] = {
                            "png": plotting.figure_to_png(figure),
                            "netcdf": export.metric_to_netcdf(
                                array, metric_name,
                                {key: str(value) for key, value in result.provenance.items()},
                            ),
                            "geotiff": export.metric_to_geotiff(array, metric_name),
                            "csv": export.metric_to_csv(array, metric_name),
                        }
                        plt.close(figure)

                    summary_csv = export.summary_to_csv(result.summary)
                    report = export.build_run_report(result.provenance, result.notes)
                    archive = export.build_archive(files, summary_csv, report)

                st.session_state["results"] = result
                st.session_state["result_files"] = {
                    "per_metric": files,
                    "summary_csv": summary_csv,
                    "report": report,
                    "archive": archive,
                }
                st.success("Comparison finished. Open **3 · Results**.", icon="✅")

            except preprocess.PreprocessError as error:
                reset_results()
                st.error(str(error), icon="🚫")
            except cdio.DataLoadError as error:
                reset_results()
                st.error(str(error), icon="🚫")
            except MemoryError:
                reset_results()
                st.error(
                    "The comparison ran out of memory. Crop the datasets to a smaller region or "
                    "period, or use a coarser aggregation level.",
                    icon="🚫",
                )
            except Exception as error:  # noqa: BLE001
                reset_results()
                st.error(f"The comparison failed: {error}", icon="🚫")
                with st.expander("Technical detail"):
                    st.exception(error)


# --------------------------------------------------------------------------- #
# Tab 3 — Results
# --------------------------------------------------------------------------- #

with tab_results:
    result = st.session_state["results"]
    result_files = st.session_state["result_files"]

    if result is None or result_files is None:
        st.info("Run a comparison on the **2 · Configure** tab to see results here.", icon="ℹ️")
    else:
        st.subheader("Overview")
        overview_a, overview_b, overview_c = st.columns(3)
        overview_a.metric("Matched time steps", f"{result.time_steps:,}")
        overview_b.metric("Latitude cells", f"{result.lat_cells:,}")
        overview_c.metric("Longitude cells", f"{result.lon_cells:,}")

        if result.notes:
            with st.expander("What CDFuse did to your data", expanded=False):
                for note in result.notes:
                    st.markdown(f"- {note}")

        st.subheader("Summary statistics")
        st.caption(
            "Statistics cover only cells with a defined value. Coverage shows what fraction of the "
            "grid that is."
        )
        st.dataframe(
            result.summary.style.format(
                {
                    "Mean": "{:.3f}", "Median": "{:.3f}",
                    "Minimum": "{:.3f}", "Maximum": "{:.3f}",
                    "Coverage %": "{:.1f}",
                },
                na_rep="n/a",
            ),
            use_container_width=True,
            hide_index=True,
        )

        download_a, download_b, download_c = st.columns(3)
        with download_a:
            st.download_button(
                "Summary CSV", data=result_files["summary_csv"],
                file_name="cdfuse_summary.csv", mime="text/csv", use_container_width=True,
            )
        with download_b:
            st.download_button(
                "Run report (.txt)", data=result_files["report"],
                file_name="cdfuse_run_report.txt", mime="text/plain", use_container_width=True,
            )
        with download_c:
            st.download_button(
                "All outputs (.zip)", data=result_files["archive"],
                file_name="cdfuse_outputs.zip", mime="application/zip",
                type="primary", use_container_width=True,
            )

        st.divider()
        st.subheader("Maps")

        for metric_name, array in result.arrays.items():
            plot_config = METRIC_CONFIG[metric_name]
            with st.container():
                st.markdown(f"#### {metric_name}")
                st.caption(str(plot_config["guidance"]))

                map_column, side_column = st.columns([3, 1])
                with map_column:
                    figure = plotting.make_map(
                        array, metric_name, st.session_state["boundary"],
                        vmin=plot_config["vmin"], vmax=plot_config["vmax"],
                        cmap=str(plot_config["cmap"]), units=str(plot_config["units"]),
                    )
                    st.pyplot(figure, use_container_width=True)
                    plt.close(figure)

                with side_column:
                    values = np.asarray(array.values, dtype="float64")
                    finite = values[np.isfinite(values)]
                    if finite.size:
                        st.metric("Mean", export.format_number(float(np.mean(finite))))
                        st.metric("Median", export.format_number(float(np.median(finite))))
                        st.metric("Minimum", export.format_number(float(np.min(finite))))
                        st.metric("Maximum", export.format_number(float(np.max(finite))))
                    else:
                        st.warning("No cells have a defined value for this metric.", icon="⚠️")

                    stem = export.safe_name(metric_name)
                    outputs = result_files["per_metric"][metric_name]
                    st.download_button(
                        "PNG", data=outputs["png"], file_name=f"{stem}_map.png",
                        mime="image/png", key=f"png_{stem}", use_container_width=True,
                    )
                    st.download_button(
                        "NetCDF", data=outputs["netcdf"], file_name=f"{stem}.nc",
                        mime="application/x-netcdf", key=f"nc_{stem}", use_container_width=True,
                    )
                    if outputs.get("geotiff"):
                        st.download_button(
                            "GeoTIFF", data=outputs["geotiff"], file_name=f"{stem}.tif",
                            mime="image/tiff", key=f"tif_{stem}", use_container_width=True,
                        )
                    st.download_button(
                        "CSV", data=outputs["csv"], file_name=f"{stem}.csv",
                        mime="text/csv", key=f"csv_{stem}", use_container_width=True,
                    )
                st.divider()


# --------------------------------------------------------------------------- #
# Tab 4 — Guide
# --------------------------------------------------------------------------- #

with tab_guide:
    st.subheader("How to use CDFuse")
    st.markdown(
        """
**1 · Data** — Upload the candidate and reference NetCDF files, plus an optional boundary.
No data of your own? Load the built-in demonstration pair.

**2 · Configure** — Pick the variable in each file and confirm which coordinates hold time,
latitude and longitude. Set any time shift, decide how sub-daily records are handled, choose
the aggregation and the grid alignment, then select the metrics.

**3 · Results** — Read the summary table, inspect each map, and download the outputs
individually or as a single ZIP.
        """
    )

    st.divider()
    st.subheader("Candidate and reference")
    st.markdown(
        "NSE, KGE and PBIAS are asymmetric: they measure how well the candidate reproduces the "
        "reference, not the other way round. Swapping the two files changes the answer, so assign "
        "them deliberately. Correlation is the exception — it is symmetric."
    )

    st.divider()
    st.subheader("Time handling")
    st.markdown(
        """
CDFuse hard-codes no product conventions — no assumed accumulation windows, no assumed
timestamp offsets, no special treatment of any hour.

- **Use every timestamp** — nothing is filtered before aggregation.
- **Keep one hour per day** — you pick the hour for each dataset separately, from 00:00 to
  23:00 UTC. Optionally the kept timestamps are replaced by their calendar dates so datasets
  recorded at different hours can be paired day by day.

If your product accumulates over a window ending at the timestamp, or is offset from UTC,
apply the correction yourself with the time-shift control. What any given timestamp means in
your data is something only you can know.
        """
    )

    st.divider()
    st.subheader("Metric definitions")
    st.markdown(
        "For candidate values $S_t$ and reference values $O_t$ at one grid cell across $n$ "
        "matched time steps:"
    )
    st.latex(r"\text{Correlation} = \frac{\sum (S_t-\bar S)(O_t-\bar O)}{\sqrt{\sum (S_t-\bar S)^2}\sqrt{\sum (O_t-\bar O)^2}}")
    st.latex(r"\text{NSE} = 1-\frac{\sum (S_t-O_t)^2}{\sum (O_t-\bar O)^2}")
    st.latex(r"\text{KGE} = 1-\sqrt{(r-1)^2+(\alpha-1)^2+(\beta-1)^2},\quad \alpha=\frac{\sigma_S}{\sigma_O},\ \beta=\frac{\mu_S}{\mu_O}")
    st.latex(r"\text{PBIAS} = 100\times\frac{\sum (S_t-O_t)}{\sum O_t}")
    st.latex(r"\text{RMSE} = \sqrt{\tfrac{1}{n}\sum (S_t-O_t)^2},\quad \text{MAE} = \tfrac{1}{n}\sum |S_t-O_t|,\quad \text{Bias} = \tfrac{1}{n}\sum (S_t-O_t)")
    st.markdown(
        "Cells are left undefined where fewer than two valid pairs remain, or where a "
        "denominator would be zero — a constant reference series for NSE, a zero-sum reference "
        "for PBIAS. They are stored as missing values, never as zeros."
    )

    st.divider()
    st.subheader("Boundaries")
    st.markdown(
        """
Upload one ZIP containing the shapefile, or all its components together (`.shp`, `.shx`,
`.dbf`, `.prj`), or a GeoJSON or GeoPackage. The layer must record a CRS; CDFuse reprojects it
to EPSG:4326 before drawing or clipping.
        """
    )

    st.divider()
    st.subheader("Limits and privacy")
    st.markdown(
        f"""
Datasets are loaded fully into memory, so CDFuse suits small to moderate grids. Very large
files should be cropped, coarsened, or processed offline first. Uploads over about
{LARGE_FILE_WARNING_MB} MB trigger a warning.

Files live only in your session and are not written to permanent storage, but a public
deployment is still a third-party service. Do not upload confidential or restricted data
without checking your own institution's rules.
        """
    )

    st.caption(f"{APP_NAME} v{VERSION} · {APP_TAGLINE}")
