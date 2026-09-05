from __future__ import annotations

import numpy as np
import pytest

from xrs_compton_extraction.corrections.elastic import fit_elastic_peak, gaussian_peak


def test_gaussian_peak_uses_height_parameterization() -> None:
    x = np.asarray([-1.0, 0.0, 1.0])
    peak = gaussian_peak(x, amplitude=10.0, center=0.0, sigma=1.0)
    assert peak[1] == pytest.approx(10.0)
    np.testing.assert_allclose(peak[[0, 2]], 10.0 * np.exp(-0.5))


def test_fit_elastic_peak_separates_linear_baseline() -> None:
    x = np.linspace(-10.0, 10.0, 401)
    peak = gaussian_peak(x, 100.0, 0.3, 1.2)
    baseline = 8.0 + 0.1 * x
    result = fit_elastic_peak(
        x,
        peak + baseline,
        fit_window=(-5.0, 5.0),
        uncertainty=np.ones_like(x),
        initial=(90.0, 0.0, 1.0, 8.0, 0.1),
        loss="linear",
    )
    assert result.success
    np.testing.assert_allclose(result.elastic_component, peak, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.local_baseline, baseline, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(result.residual, 0.0, atol=1e-6)


def test_fit_elastic_requires_explicit_valid_window() -> None:
    x = np.arange(10.0)
    with pytest.raises(ValueError, match="start < stop"):
        fit_elastic_peak(x, np.ones_like(x), fit_window=(5.0, 2.0))

