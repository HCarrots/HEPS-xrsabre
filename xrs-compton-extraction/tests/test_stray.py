from __future__ import annotations

import numpy as np
import pytest

from xrs_compton_extraction.corrections.stray import (
    estimate_constant_background,
    subtract_stray_background,
)


def test_weighted_constant_background_uses_only_requested_windows() -> None:
    x = np.arange(6.0)
    y = np.asarray([2.0, 2.0, 100.0, 100.0, 4.0, 4.0])
    sigma = np.asarray([1.0, 1.0, 1.0, 1.0, 2.0, 2.0])
    result = estimate_constant_background(
        x,
        y,
        fit_windows=((0.0, 1.0), (4.0, 5.0)),
        uncertainty=sigma,
    )
    assert result.level == pytest.approx(2.4)
    np.testing.assert_allclose(result.component, 2.4)
    np.testing.assert_array_equal(result.fit_mask, [True, True, False, False, True, True])


def test_stray_subtraction_retains_negative_values() -> None:
    result = subtract_stray_background([1.0, 2.0], [3.0, 1.0])
    np.testing.assert_array_equal(result, [-2.0, 1.0])


def test_empty_constant_fit_windows_are_rejected() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        estimate_constant_background([0.0, 1.0], [1.0, 1.0], fit_windows=())

