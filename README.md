<div align="center">

# CDFuse

**Compare. Validate. Visualise.**

A browser-based tool for comparing two gridded NetCDF datasets — align them in time
and space, calculate cell-wise performance metrics, map the results, and export
reproducible outputs.

[![CI](https://github.com/hrishikesh618/CDFuse/actions/workflows/ci.yml/badge.svg)](https://github.com/hrishikesh618/CDFuse/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)

</div>

---

## What it does

Upload a **candidate** dataset (the model run, reanalysis, satellite product, or
whatever is being evaluated) and a **reference** dataset (the benchmark). CDFuse
matches them in time and on a common grid, then scores every grid cell.

| Step | What happens |
| --- | --- |
| **1 · Data** | Upload two NetCDF files, plus an optional study-area boundary. Or load the built-in demonstration pair. |
| **2 · Configure** | Choose the variable and confirm the time/latitude/longitude coordinates, set any time shift, decide how sub-daily records are handled, pick the aggregation, grid alignment and metrics. |
| **3 · Results** | Read the summary table, inspect each map, download outputs individually or as one ZIP. |

### Metrics

| Metric | Range | Perfect | Notes |
| --- | --- | --- | --- |
| Correlation | −1 … 1 | 1 | Pearson, through time. The only symmetric metric here. |
| NSE | −∞ … 1 | 1 | Nash–Sutcliffe Efficiency. 0 = no better than the reference mean. |
| KGE | −∞ … 1 | 1 | Kling–Gupta Efficiency; combines correlation, variability and bias. |
| PBIAS | % | 0 | Percentage bias relative to the reference total. |
| RMSE | ≥ 0 | 0 | Root mean square error, in the variable's units. |
| MAE | ≥ 0 | 0 | Mean absolute error, in the variable's units. |
| Bias | any | 0 | Mean signed difference (candidate − reference). |

Cells are left **undefined** — stored as missing values, never as zeros — where
fewer than two valid pairs remain, or where a denominator would be zero (a
constant reference for NSE, a zero-sum reference for PBIAS).

### Handling of real-world data

CDFuse copes with the conventions gridded products actually arrive in:

- coordinates named `time` / `valid_time` / `latitude` / `lat` / `y`, and so on (auto-detected, always overridable)
- longitudes on either the `0…360` or `−180…180` convention
- latitude stored north-to-south or south-to-north
- extra dimensions such as pressure level or ensemble member (you pick the slice)
- non-standard calendars, decoded where possible
- duplicated timestamps, averaged with a note in the run report

### Outputs

Per metric: a 300-dpi **PNG** map, a CF-attributed **NetCDF**, a **GeoTIFF** (EPSG:4326),
and a long-format **CSV**. Plus a **summary CSV** across all metrics, a plain-text
**run report** recording every setting used, and a **ZIP** containing the lot.

---

## Dataset neutrality

**CDFuse hard-codes no product conventions.** There is no assumed accumulation
window, no assumed timestamp offset, and no special treatment of any particular
hour or dataset.

Where a product-specific adjustment is genuinely needed, it is exposed as an
explicit, user-driven control:

- **Time shift (hours)** — applied only if you set it, independently per dataset.
- **Keep one hour per day** — you choose the hour for each dataset separately, and
  may optionally match them by calendar date so datasets recorded at different
  hours can still be paired day by day.

Selecting, say, 23:00 is therefore *your* decision about *your* data, not a rule
baked into the tool. Interpreting what a timestamp means in your product remains
your responsibility.

---

## Run it locally

### 1. Clone

```bash
git clone https://github.com/hrishikesh618/CDFuse.git
cd CDFuse
```

### 2. Create an environment

Conda is the smoothest route on Windows, because it supplies the GDAL/GEOS/PROJ
libraries that `geopandas` and `rasterio` need:

```bash
conda create -n cdfuse python=3.12 -y
conda activate cdfuse
pip install -r requirements.txt
```

Or with a plain virtual environment:

```bash
python -m venv .venv
```

```powershell
.venv\Scripts\activate      # Windows
```

```bash
source .venv/bin/activate   # macOS / Linux
```

```bash
pip install -r requirements.txt
```

### 3. Start

```bash
streamlit run app.py
```

The app opens at <http://localhost:8501>. Click **Load demonstration data** on the
first tab to try the whole workflow without supplying any files.

---

## Deploy

Full instructions, including Docker and institutional hosting, are in
**[docs/DEPLOYMENT.md](docs/DEPLOYMENT.md)**. The short version:

> **GitHub Pages cannot host CDFuse.** Pages serves static files only and cannot run
> a Python backend. Use Streamlit Community Cloud (free) and link to it from your
> site.

1. Push this repository to GitHub as a **public** repo.
2. Sign in to [share.streamlit.io](https://share.streamlit.io) with GitHub.
3. **New app** → pick the repo and branch → entrypoint `app.py` → **Deploy**.
4. You get a permanent `https://<name>.streamlit.app` URL. Pushes redeploy it.

---

## Development

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest          # 70 tests
ruff check .    # lint
```

### Layout

```text
CDFuse/
├── app.py                  # Streamlit interface (presentation only)
├── cdfuse/                 # importable library — no Streamlit dependency
│   ├── config.py           # constants, metric definitions and plot defaults
│   ├── io.py               # NetCDF and boundary loading, clipping
│   ├── preprocess.py       # standardisation, hour selection, aggregation, alignment
│   ├── metrics.py          # the seven cell-wise metrics
│   ├── pipeline.py         # end-to-end orchestration
│   ├── plotting.py         # map rendering (cartopy optional)
│   ├── export.py           # PNG / NetCDF / GeoTIFF / CSV / ZIP outputs
│   └── sample.py           # synthetic demonstration datasets
├── tests/                  # pytest suite
├── docs/DEPLOYMENT.md
├── Dockerfile
└── requirements.txt
```

The `cdfuse` package deliberately does not import Streamlit, so the same pipeline
the web app runs can be driven from a script or notebook:

```python
import xarray as xr
from cdfuse.pipeline import ComparisonSettings, run_comparison

candidate = xr.open_dataset("candidate.nc")["precip"]
reference = xr.open_dataset("reference.nc")["rainfall"]

result = run_comparison(
    candidate,
    reference,
    ComparisonSettings(aggregation_level="Monthly", metrics=["NSE", "KGE"]),
)
print(result.summary)
```

### Optional: nicer basemaps

Install `cartopy` and maps gain coastlines and national borders automatically:

```bash
pip install cartopy
```

It is deliberately **not** in `requirements.txt`, because it is awkward to build on
some hosts. Without it, CDFuse draws the same maps on plain matplotlib axes.

---

## Limits and privacy

Datasets are loaded fully into memory, so CDFuse suits small to moderate grids.
Crop, coarsen or subset very large files first. The interface warns above roughly
150 MB per variable, and the upload cap is 500 MB (`.streamlit/config.toml`).

Uploads exist only for the life of your browser session and are never written to
permanent storage. A public deployment is nonetheless a third-party service —
check your institution's rules before uploading anything restricted.

---

## Citing

If CDFuse contributes to published work, please cite the release you used. See
[`CITATION.cff`](CITATION.cff). Archiving a GitHub release through
[Zenodo](https://zenodo.org) will mint a DOI.

## Contributing

Bug reports and pull requests are welcome — see [CONTRIBUTING.md](CONTRIBUTING.md).

## License

[MIT](LICENSE) © Hrishikesh Singh
