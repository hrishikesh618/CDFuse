"""End-to-end tests for the comparison pipeline and the demo data."""

from __future__ import annotations

import pytest

from cdfuse.pipeline import ComparisonSettings, run_comparison
from cdfuse.preprocess import PreprocessError
from cdfuse.sample import make_demo_pair


@pytest.fixture(scope="module")
def demo_pair():
    return make_demo_pair(days=40, resolution=2.0)


def test_demo_pair_is_well_formed(demo_pair):
    candidate, reference = demo_pair
    for dataset in (candidate, reference):
        assert "value" in dataset.data_vars
        assert set(dataset["value"].dims) == {"time", "lat", "lon"}
    assert candidate["value"].shape == reference["value"].shape


def test_full_run_on_the_demo_pair(demo_pair):
    candidate, reference = demo_pair
    settings = ComparisonSettings(
        aggregation_level="None (use matched time steps)",
        metrics=["Correlation", "NSE", "KGE", "PBIAS", "RMSE", "MAE", "Bias"],
    )
    result = run_comparison(candidate["value"], reference["value"], settings)

    assert set(result.arrays) == set(settings.metrics)
    assert len(result.summary) == len(settings.metrics)
    assert result.time_steps == 40
    assert result.lat_cells > 0 and result.lon_cells > 0

    for metric_name, array in result.arrays.items():
        assert "time" not in array.dims, metric_name
        assert array.attrs["metric"] == metric_name


def test_demo_candidate_bias_is_detected(demo_pair):
    """The demo candidate is built with a known positive bias; PBIAS must see it."""
    candidate, reference = demo_pair
    settings = ComparisonSettings(
        aggregation_level="None (use matched time steps)",
        metrics=["PBIAS", "Bias", "Correlation"],
    )
    result = run_comparison(candidate["value"], reference["value"], settings)

    pbias = result.summary.set_index("Metric").loc["PBIAS", "Mean"]
    bias = result.summary.set_index("Metric").loc["Bias", "Mean"]
    correlation = result.summary.set_index("Metric").loc["Correlation", "Mean"]

    assert pbias > 5.0, "the +12% scaling should give a clearly positive PBIAS"
    assert bias > 0.0
    assert correlation > 0.8, "the two fields share a common signal"


def test_missing_data_block_reduces_coverage(demo_pair):
    """The demo candidate has a NaN corner, so coverage must be below 100%."""
    candidate, reference = demo_pair
    settings = ComparisonSettings(
        aggregation_level="None (use matched time steps)", metrics=["Correlation"]
    )
    result = run_comparison(candidate["value"], reference["value"], settings)
    coverage = result.summary.set_index("Metric").loc["Correlation", "Coverage %"]
    assert 0 < coverage < 100


def test_monthly_aggregation_reduces_time_steps(demo_pair):
    candidate, reference = demo_pair
    settings = ComparisonSettings(aggregation_level="Monthly", metrics=["Correlation"])
    result = run_comparison(candidate["value"], reference["value"], settings)
    assert result.time_steps < 40


def test_provenance_records_the_settings(demo_pair):
    candidate, reference = demo_pair
    settings = ComparisonSettings(
        aggregation_level="Daily", aggregation_method="Sum", metrics=["NSE"]
    )
    result = run_comparison(candidate["value"], reference["value"], settings)
    assert "Daily" in str(result.provenance["Aggregation"])
    assert "Sum" in str(result.provenance["Aggregation"])
    assert result.provenance["Metrics"] == "NSE"
    assert result.provenance["Candidate hour filter"] == "none"


def test_empty_metric_selection_is_rejected(demo_pair):
    candidate, reference = demo_pair
    with pytest.raises(PreprocessError, match="at least one metric"):
        run_comparison(candidate["value"], reference["value"], ComparisonSettings(metrics=[]))


def test_unknown_metric_is_rejected(demo_pair):
    candidate, reference = demo_pair
    with pytest.raises(PreprocessError, match="Unknown metric"):
        run_comparison(
            candidate["value"], reference["value"], ComparisonSettings(metrics=["NotAMetric"])
        )


def test_identical_inputs_score_perfectly(demo_pair):
    """Comparing the reference with itself is the pipeline's sanity check."""
    _, reference = demo_pair
    settings = ComparisonSettings(
        aggregation_level="None (use matched time steps)",
        metrics=["Correlation", "NSE", "KGE", "RMSE"],
    )
    result = run_comparison(reference["value"], reference["value"], settings)
    summary = result.summary.set_index("Metric")

    assert summary.loc["Correlation", "Mean"] == pytest.approx(1.0, abs=1e-6)
    assert summary.loc["NSE", "Mean"] == pytest.approx(1.0, abs=1e-6)
    assert summary.loc["KGE", "Mean"] == pytest.approx(1.0, abs=1e-6)
    assert summary.loc["RMSE", "Mean"] == pytest.approx(0.0, abs=1e-9)


def test_notes_explain_the_alignment(demo_pair):
    candidate, reference = demo_pair
    settings = ComparisonSettings(
        aggregation_level="None (use matched time steps)", metrics=["Correlation"]
    )
    result = run_comparison(candidate["value"], reference["value"], settings)
    assert any("Matched" in note for note in result.notes)


def test_time_shift_can_destroy_the_overlap(demo_pair):
    """A large shift moves the candidate off the reference period entirely."""
    candidate, reference = demo_pair
    settings = ComparisonSettings(
        candidate_shift_hours=24 * 365 * 5,
        aggregation_level="None (use matched time steps)",
        metrics=["Correlation"],
    )
    with pytest.raises(PreprocessError, match="share no time steps"):
        run_comparison(candidate["value"], reference["value"], settings)
