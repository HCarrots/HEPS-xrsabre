from __future__ import annotations

import numpy as np
import pytest

from xrs_compton_extraction.corrections import normalize_counts


def test_normalize_counts_and_poisson_uncertainty() -> None:
    result = normalize_counts(
        [100.0, 400.0],
        acquisition_time_s=2.0,
        i0=[10.0, 20.0],
        detector_efficiency=0.5,
    )
    np.testing.assert_allclose(result.intensity, [10.0, 20.0])
    np.testing.assert_allclose(result.statistical_uncertainty, [1.0, 1.0])
    assert result.metadata["count_variance_source"] == "poisson"


def test_normalize_counts_propagates_i0_uncertainty() -> None:
    result = normalize_counts([100.0], i0=[10.0], i0_uncertainty=[1.0])
    expected_variance = 100.0 / 100.0 + (10.0 * 1.0 / 10.0) ** 2
    np.testing.assert_allclose(result.statistical_uncertainty**2, [expected_variance])


def test_normalize_counts_propagates_all_explicit_factor_uncertainties() -> None:
    result = normalize_counts(
        [100.0],
        acquisition_time_s=2.0,
        i0=10.0,
        detector_efficiency=0.5,
        raw_count_variance=[4.0],
        acquisition_time_uncertainty_s=0.2,
        i0_uncertainty=1.0,
        detector_efficiency_uncertainty=0.05,
    )

    # I = 10; count, time, I0, and efficiency standard uncertainties are
    # respectively 0.2, 1, 1, and 1 in normalized-intensity units.
    np.testing.assert_allclose(result.intensity, [10.0])
    np.testing.assert_allclose(result.count_variance, [0.04])
    np.testing.assert_allclose(result.acquisition_time_variance_contribution, [1.0])
    np.testing.assert_allclose(result.i0_variance_contribution, [1.0])
    np.testing.assert_allclose(
        result.detector_efficiency_variance_contribution, [1.0]
    )
    np.testing.assert_allclose(result.statistical_uncertainty**2, [3.04])


@pytest.mark.parametrize("keyword,value", [("acquisition_time_s", 0), ("i0", 0), ("detector_efficiency", -1)])
def test_normalize_counts_rejects_nonpositive_factors(keyword: str, value: float) -> None:
    with pytest.raises(ValueError, match=keyword):
        normalize_counts([1.0, 2.0], **{keyword: value})


def test_normalize_counts_does_not_allow_negative_raw_counts() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        normalize_counts([1.0, -1.0])
