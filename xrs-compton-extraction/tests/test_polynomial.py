from __future__ import annotations

import numpy as np
import pytest

from xrs_compton_extraction.backgrounds.polynomial import fit_polynomial


def test_weighted_polynomial_recovers_background_outside_masked_edge() -> None:
    x = np.linspace(0.0, 100.0, 201)
    background = 3.0 + 0.2 * x + 0.01 * x**2
    observed = background.copy()
    observed[(x > 40.0) & (x < 60.0)] += 100.0
    result = fit_polynomial(
        x,
        observed,
        degree=2,
        fit_windows_ev=((0.0, 35.0), (65.0, 100.0)),
        sigma=np.ones_like(x),
    )
    assert result.success
    np.testing.assert_allclose(result.fitted_background, background, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(result.evaluate(x), background, rtol=1e-10, atol=1e-10)
    assert result.covariance.shape == (3, 3)


def test_polynomial_rejects_excessive_degree_and_too_few_points() -> None:
    x = np.arange(10.0)
    with pytest.raises(ValueError, match="degree"):
        fit_polynomial(x, x, degree=6, fit_windows_ev=((0.0, 9.0),))
    with pytest.raises(ValueError, match="more samples"):
        fit_polynomial(x, x, degree=3, fit_windows_ev=((0.0, 3.0),))


def test_condition_threshold_marks_numerically_unapproved_fit() -> None:
    x = np.linspace(0.0, 1.0, 20)
    result = fit_polynomial(
        x,
        x,
        degree=2,
        fit_windows_ev=((0.0, 1.0),),
        max_condition_number=1.01,
    )
    assert not result.success
    assert "condition number" in result.message

