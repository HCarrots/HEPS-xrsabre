from __future__ import annotations

import numpy as np
import pytest

from xrs_compton_extraction.exceptions import DataValidationError
from xrs_compton_extraction.uncertainty import (
    BootstrapResult,
    NormalizationUncertainty,
    ScanRepeatability,
    bootstrap_statistic,
    combine_independent_uncertainties,
    multi_scan_repeatability,
    poisson_uncertainty,
    propagate_normalization_uncertainty,
)


def test_poisson_uncertainty_preserves_shape_and_is_read_only() -> None:
    counts = np.array([[0.0, 1.0], [4.0, 9.0]])
    result = poisson_uncertainty(counts)

    np.testing.assert_allclose(result, [[0, 1], [2, 3]])
    assert result.shape == counts.shape
    assert not result.flags.writeable
    counts[1, 1] = 100
    assert result[1, 1] == 3

    with pytest.raises(DataValidationError, match="non-negative"):
        poisson_uncertainty([1, -1])
    with pytest.raises(DataValidationError, match="NaN"):
        poisson_uncertainty([1, np.nan])


def test_combine_independent_uncertainties_supports_mapping_and_stack() -> None:
    mapping_result = combine_independent_uncertainties(
        {"statistical": [3.0, 0.0], "model": 4.0}
    )
    stack_result = combine_independent_uncertainties([[3.0, 0.0], [4.0, 4.0]])

    np.testing.assert_allclose(mapping_result, [5.0, 4.0])
    np.testing.assert_allclose(stack_result, [5.0, 4.0])
    assert not mapping_result.flags.writeable

    with pytest.raises(DataValidationError, match="must not be empty"):
        combine_independent_uncertainties({})
    with pytest.raises(DataValidationError, match="broadcast-compatible"):
        combine_independent_uncertainties({"a": [1, 2], "b": [1, 2, 3]})
    with pytest.raises(DataValidationError, match="non-negative"):
        combine_independent_uncertainties([[1, -1], [2, 2]])


def test_normalization_analytic_propagation_matches_hand_calculation() -> None:
    result = propagate_normalization_uncertainty(
        raw_counts=[100.0, 400.0],
        acquisition_time_s=2.0,
        i0=[10.0, 20.0],
        acquisition_time_uncertainty_s=0.2,
        i0_uncertainty=[1.0, 2.0],
    )

    assert isinstance(result, NormalizationUncertainty)
    np.testing.assert_allclose(result.normalized_intensity, [5.0, 10.0])
    np.testing.assert_allclose(result.counts_component, [0.5, 0.5])
    np.testing.assert_allclose(result.acquisition_time_component, [0.5, 1.0])
    np.testing.assert_allclose(result.i0_component, [0.5, 1.0])
    np.testing.assert_allclose(
        result.total_uncertainty,
        [np.sqrt(0.75), 1.5],
    )
    assert not result.total_uncertainty.flags.writeable


def test_normalization_allows_explicit_counts_uncertainty_and_zero_optional_terms() -> None:
    result = propagate_normalization_uncertainty(
        [10.0, 20.0],
        acquisition_time_s=[2.0, 4.0],
        i0=5.0,
        counts_uncertainty=[1.0, 2.0],
    )
    np.testing.assert_allclose(result.normalized_intensity, [1.0, 1.0])
    np.testing.assert_allclose(result.counts_component, [0.1, 0.1])
    np.testing.assert_allclose(result.acquisition_time_component, 0.0)
    np.testing.assert_allclose(result.i0_component, 0.0)
    np.testing.assert_allclose(result.total_uncertainty, [0.1, 0.1])


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"raw_counts": [1, -1], "acquisition_time_s": 1, "i0": 1}, "non-negative"),
        ({"raw_counts": [1, 2], "acquisition_time_s": 0, "i0": 1}, "positive"),
        ({"raw_counts": [1, 2], "acquisition_time_s": 1, "i0": [1, 2, 3]}, "broadcast"),
        (
            {
                "raw_counts": [1, 2],
                "acquisition_time_s": 1,
                "i0": 1,
                "i0_uncertainty": [0.1, np.inf],
            },
            "infinite",
        ),
    ],
)
def test_normalization_rejects_invalid_values_and_shapes(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(DataValidationError, match=message):
        propagate_normalization_uncertainty(**kwargs)  # type: ignore[arg-type]


def test_multi_scan_repeatability_reports_std_and_standard_error() -> None:
    result = multi_scan_repeatability([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    assert isinstance(result, ScanRepeatability)
    assert result.scan_count == 3
    np.testing.assert_allclose(result.mean, [3.0, 4.0])
    np.testing.assert_allclose(result.sample_standard_deviation, [2.0, 2.0])
    np.testing.assert_allclose(result.standard_error, 2 / np.sqrt(3))
    assert not result.standard_error.flags.writeable

    with pytest.raises(DataValidationError, match="at least two"):
        multi_scan_repeatability([[1.0, 2.0]])
    with pytest.raises(DataValidationError, match="numeric"):
        multi_scan_repeatability([[1.0, 2.0], [3.0]])


def test_bootstrap_statistic_is_reproducible_for_a_fixed_seed() -> None:
    samples = np.arange(20.0).reshape(5, 4)
    first = bootstrap_statistic(
        samples,
        statistic=np.mean,
        resample_count=50,
        seed=1234,
        confidence_level=0.90,
    )
    second = bootstrap_statistic(
        samples,
        statistic=np.mean,
        resample_count=50,
        seed=1234,
        confidence_level=0.90,
    )

    assert isinstance(first, BootstrapResult)
    np.testing.assert_array_equal(first.distribution, second.distribution)
    np.testing.assert_allclose(first.estimate, np.mean(samples, axis=0))
    assert first.distribution.shape == (50, 4)
    assert first.standard_error.shape == (4,)
    assert first.confidence_interval is not None
    lower, upper = first.confidence_interval
    assert np.all(lower <= upper)
    assert not first.distribution.flags.writeable


def test_bootstrap_supports_scalar_statistics() -> None:
    result = bootstrap_statistic(
        [1.0, 2.0, 3.0, 4.0],
        statistic=np.mean,
        resample_count=20,
        seed=7,
    )
    assert result.estimate.shape == ()
    assert result.distribution.shape == (20,)
    assert float(result.estimate) == pytest.approx(2.5)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"samples": [1.0], "resample_count": 10, "seed": 1}, "at least two"),
        ({"samples": [1.0, 2.0], "resample_count": 1, "seed": 1}, "at least two"),
        ({"samples": [1.0, 2.0], "resample_count": 10, "seed": True}, "seed"),
        (
            {
                "samples": [1.0, 2.0],
                "resample_count": 10,
                "seed": 1,
                "confidence_level": 1.0,
            },
            "strictly between",
        ),
    ],
)
def test_bootstrap_rejects_invalid_configuration(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(DataValidationError, match=message):
        bootstrap_statistic(**kwargs)  # type: ignore[arg-type]


def test_bootstrap_rejects_nonfinite_statistic_output() -> None:
    def bad_statistic(values: np.ndarray, *, axis: int) -> float:
        del values, axis
        return np.nan

    with pytest.raises(DataValidationError, match="NaN"):
        bootstrap_statistic(
            [1.0, 2.0],
            statistic=bad_statistic,
            resample_count=10,
            seed=1,
        )

