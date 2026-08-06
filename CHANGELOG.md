# Changelog

All notable changes to CDFuse are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] — 2026-08-06

First public release, rebuilt from the original desktop NetCDF comparison script.

### Added

- **Step-based interface** — Data → Configure → Results, plus a built-in Guide tab,
  with a sidebar showing workflow progress and session state.
- **Demonstration datasets** — a synthetic candidate/reference pair generated on
  demand, so the whole workflow can be tried without uploading anything. The
  candidate carries a known +12% scaling and +0.15 offset, making the resulting
  metrics predictable.
- **Three further metrics** — RMSE, MAE and Bias, alongside the original
  Correlation, NSE, KGE and PBIAS.
- **More output formats** — GeoTIFF (EPSG:4326) and long-format CSV per metric,
  a plain-text run report recording every setting, and a ZIP bundling everything.
- **Wider input support** — GeoJSON and GeoPackage boundaries as well as
  shapefiles; extra dimensions such as pressure level or ensemble member; and
  non-standard calendars.
- **Automatic coordinate detection** for common naming conventions, always
  overridable in the interface.
- **Responsive layout**, usable from phone to desktop.
- **Test suite** — 70 pytest cases covering metrics, preprocessing, geospatial
  paths, exports and end-to-end integration.
- **Deployment assets** — `Dockerfile`, `packages.txt`, GitHub Actions CI, and
  [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).
- **Importable library** — the `cdfuse` package has no Streamlit dependency, so
  the same pipeline can be driven from scripts and notebooks.

### Changed

- **Removed the ERA5-specific time shift.** The original had a hard-coded
  "Apply −23h time shift (for ERA5)" checkbox. It is replaced by a neutral,
  per-dataset time-shift control and an optional hour-of-day filter where the user
  chooses each dataset's hour independently. CDFuse now assumes nothing about any
  product's timestamping or accumulation convention.
- **Undefined metrics are masked rather than computed.** Cells with fewer than two
  valid pairs, a constant reference (NSE, KGE) or a zero-sum reference (PBIAS) are
  returned as missing values instead of `inf`, `nan` propagated silently, or
  misleading finite numbers.
- **Pairwise validity masking.** A missing value on either side now removes that
  time step from both, so a gap in one dataset cannot bias the score.
- **Longitude and latitude normalisation.** `0…360` longitudes are converted to
  `−180…180`, and both axes are sorted ascending, so grids align regardless of the
  convention each file uses.
- **Split the monolithic script** into a documented package with separate
  responsibilities for I/O, preprocessing, metrics, plotting, export and pipeline.
- **Cartopy is now optional.** Maps gain coastlines and borders when it is
  installed, and fall back to plain matplotlib axes when it is not, so deployment
  no longer depends on a difficult build.

### Fixed

- **`rioxarray` was never imported**, leaving the `.rio` accessor unregistered. This
  broke boundary clipping and GeoTIFF export at runtime with
  `AttributeError: 'DataArray' object has no attribute 'rio'`. Both modules that use
  the accessor now import it explicitly, and `tests/test_geospatial.py` guards
  against the regression.
- **A non-overlapping boundary produced a silently blank map.** `rio.clip` returns
  an all-missing array instead of raising; CDFuse now detects this and reports both
  extents so the mismatch is obvious.
- **Single-row or single-column grids were destroyed** by a blanket `squeeze()` when
  reducing extra dimensions. Only non-core dimensions are squeezed now.
- **Constant metric fields** no longer trigger a matplotlib auto-scaling warning;
  colour limits fall back to a sensible padded range.
- **Duplicate timestamps** are averaged with a note in the run report, rather than
  making alignment ambiguous.
- **Temporary files are always removed** after a NetCDF upload is read, including
  when reading fails.
- **ZIP extraction is path-checked**, refusing archives whose entries would escape
  the extraction directory.
