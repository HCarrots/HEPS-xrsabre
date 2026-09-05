from __future__ import annotations

import numpy as np
import pytest

from xrs_compton_extraction.corrections.absorption import (
    apply_transmission_correction,
    beer_lambert_transmission,
    slab_self_absorption_factor,
)


def test_beer_lambert_transmission_limits() -> None:
    np.testing.assert_allclose(beer_lambert_transmission([0.0, 1.0], 2.0), [1.0, np.exp(-2.0)])


def test_reflection_slab_matches_analytic_depth_average() -> None:
    factor = slab_self_absorption_factor(1.0, 2.0, 0.5, geometry="reflection")
    expected = (1.0 - np.exp(-1.5)) / 1.5
    assert float(factor) == pytest.approx(expected)


def test_transmission_slab_equal_coefficients_has_simple_limit() -> None:
    factor = slab_self_absorption_factor(2.0, 2.0, 0.5, geometry="transmission")
    assert float(factor) == pytest.approx(np.exp(-1.0))


def test_transmission_correction_propagates_factor_uncertainty() -> None:
    corrected, sigma = apply_transmission_correction(
        [10.0],
        [0.5],
        statistical_uncertainty=[2.0],
        transmission_uncertainty=[0.05],
    )
    np.testing.assert_allclose(corrected, [20.0])
    np.testing.assert_allclose(sigma, [np.sqrt(4.0**2 + 2.0**2)])


def test_absorption_rejects_nonphysical_values() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        beer_lambert_transmission(-1.0, 1.0)
    with pytest.raises(ValueError, match=r"\[0, 90\)"):
        slab_self_absorption_factor(1.0, 1.0, 1.0, exit_angle_from_normal_deg=90.0)
