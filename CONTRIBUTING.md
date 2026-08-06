# Contributing to CDFuse

Thanks for your interest. Bug reports, feature suggestions and pull requests are
all welcome.

## Reporting a bug

Open an issue including:

- what you did, what you expected, and what happened instead
- the **shape and conventions** of your data — dimension names, coordinate names,
  longitude convention, time step — but **not** the data itself
- the full error message shown by the app
- your operating system and Python version

Please do not attach restricted or unpublished datasets. A small synthetic file
that reproduces the problem is far more useful.

## Development setup

```bash
git clone https://github.com/hrishikesh618/CDFuse.git
cd CDFuse

conda create -n cdfuse python=3.12 -y
conda activate cdfuse
pip install -r requirements.txt -r requirements-dev.txt
```

Run the checks before opening a pull request:

```bash
pytest
ruff check .
streamlit run app.py    # exercise the change in the interface
```

## Design principles

Three rules shape this codebase. Please keep to them.

### 1. Dataset neutrality is not negotiable

CDFuse must not hard-code the conventions of any particular product. No assumed
accumulation window, no assumed timestamp offset, no special treatment of a
particular hour or dataset name.

Where such an adjustment is genuinely useful, expose it as an explicit control the
user sets — as the time-shift and hour-selection features already do — and document
what the user is responsible for deciding.

### 2. The library stays free of Streamlit

Everything under `cdfuse/` must be importable without Streamlit, so the pipeline can
be scripted and tested. `app.py` is the only place that may import `streamlit`, and
it should stay a presentation layer: read widgets, call the pipeline, render results.

### 3. Undefined is not zero

Where a metric is mathematically undefined — too few valid pairs, a zero
denominator — return a missing value. Never substitute 0, and never let a
divide-by-zero quietly produce `inf`. `cdfuse/metrics.py` shows the pattern.

## Adding a metric

1. Write the function in `cdfuse/metrics.py`, following the existing masking pattern
   (`_paired` gives you the aligned arrays and a per-cell count of valid pairs).
2. Register it in `METRIC_FUNCTIONS`.
3. Add an entry to `METRIC_CONFIG` in `cdfuse/config.py` with colour limits, a
   colormap, units, the perfect score, a description and interpretation guidance.
   Use `vmin`/`vmax` of `None` for unbounded metrics so the scale adapts.
4. If it is diverging around zero, add it to `DIVERGING_METRICS` in
   `cdfuse/plotting.py` so the colour scale stays symmetric.
5. Add tests: a perfect-score case, a known-offset case, and a case where the metric
   must be masked.
6. Mention it in the README table.

## Testing conventions

- Every metric needs at least one test with an analytically known answer.
- Every user-facing error path deserves a test asserting the message is helpful.
- Anything touching `.rio` belongs in `tests/test_geospatial.py`. That file exists
  because the accessor only attaches as a side effect of importing `rioxarray`, and
  removing that import breaks clipping and GeoTIFF export at runtime while every
  other test still passes.

## Style

`ruff` enforces the formatting and import rules; run `ruff check .` before pushing.
Beyond that: prefer clear names over short ones, and write comments that explain
*why* rather than restating *what*.

## Licence

Contributions are accepted under the [MIT Licence](LICENSE).
