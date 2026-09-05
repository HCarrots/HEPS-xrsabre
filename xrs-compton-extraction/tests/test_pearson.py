from __future__ import annotations

import numpy as np
import pytest

from xrs_compton_extraction.backgrounds import fit_pearson, pearson_background


def test_pearson_background_matches_definition() -> None:
    x = np.asarray([-1.0, 0.0, 1.0])
    actual = pearson_background(x, 2.0, 0.0, 3.0, 2.0)
    expected = 2.0 * (1.0 + 9.0 * x**2) ** -2.0
    np.testing.assert_allclose(actual, expected)


def test_weighted_fit_recovers_noise_free_curve() -> None:
    x = np.linspace(0.0, 100.0, 401)
    expected = np.asarray([120.0, 45.0, 0.035, 1.8])
    y = pearson_background(x, *expected)
    result = fit_pearson(
        x,
        y,
        sigma=np.full_like(x, 0.5),
        initial=[100.0, 40.0, 0.03, 1.5],
        loss="linear",
    )
    assert result.success
    np.testing.assert_allclose(list(result.parameters.values()), expected, rtol=1e-5, atol=1e-7)
    np.testing.assert_allclose(result.residual, 0.0, atol=1e-7)
    assert result.covariance.shape == (4, 4)


def test_fit_supports_multiple_windows_and_excludes_edge_region() -> None:
    x = np.linspace(0.0, 100.0, 201)
    y = pearson_background(x, 50.0, 50.0, 0.04, 1.4)
    y[(x > 40.0) & (x < 60.0)] += 100.0
    result = fit_pearson(
        x,
        y,
        sigma=np.ones_like(x),
        fit_windows_ev=[(0.0, 35.0), (65.0, 100.0)],
        initial=[45.0, 50.0, 0.03, 1.2],
        loss="linear",
    )
    assert not np.any(result.fit_mask[(x > 40.0) & (x < 60.0)])
    np.testing.assert_allclose(
        result.fitted_background,
        pearson_background(x, 50.0, 50.0, 0.04, 1.4),
        rtol=1e-4,
        atol=1e-5,
    )


def test_fit_rejects_invalid_window() -> None:
    x = np.linspace(0.0, 10.0, 20)
    with pytest.raises(ValueError, match="start < stop"):
        fit_pearson(x, np.ones_like(x), fit_windows_ev=[(5.0, 2.0)])

