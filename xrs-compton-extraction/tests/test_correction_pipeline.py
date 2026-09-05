from __future__ import annotations

import numpy as np
import pytest

from xrs_compton_extraction.corrections import correct_spectrum
from xrs_compton_extraction.data import XRSSpectrum
from xrs_compton_extraction.exceptions import DataValidationError


def _spectrum() -> XRSSpectrum:
    return XRSSpectrum(
        energy_eV=[1.0, 2.0],
        counts=[100.0, 400.0],
        energy_loss_eV=[10.0, 11.0],
        monitor=[10.0, 20.0],
        acquisition_time_s=2.0,
    )


def test_correction_chain_records_order_and_factors() -> None:
    result = correct_spectrum(
        _spectrum(),
        detector_efficiency=0.5,
        elastic_component=[1.0, 2.0],
        stray_background=1.0,
        path_transmission=0.5,
        self_absorption_factor=0.5,
        cross_section_correction=2.0,
    )
    np.testing.assert_allclose(result.normalized_intensity, [10.0, 20.0])
    np.testing.assert_allclose(result.corrected_intensity, [64.0, 136.0])
    assert set(result.correction_factors) == {
        "normalization",
        "path_absorption",
        "self_absorption",
        "cross_section",
    }
    assert result.metadata["negative_values_clipped"] is False


def test_correction_chain_propagates_background_uncertainty() -> None:
    result = correct_spectrum(
        _spectrum(),
        elastic_uncertainty=2.0,
        stray_uncertainty=3.0,
    )
    base_variance = np.asarray([0.25, 0.25])
    np.testing.assert_allclose(
        np.square(result.statistical_uncertainty), base_variance + 4.0 + 9.0
    )


def test_correction_chain_requires_requested_metadata() -> None:
    spectrum = XRSSpectrum(energy_eV=[1.0], counts=[1.0])
    with pytest.raises(DataValidationError, match="acquisition-time"):
        correct_spectrum(spectrum)


def test_correction_chain_rejects_nonphysical_attenuation() -> None:
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        correct_spectrum(_spectrum(), path_transmission=1.1)


def test_correction_chain_uses_supplied_raw_count_uncertainty() -> None:
    source = _spectrum()
    spectrum = XRSSpectrum(
        energy_eV=source.energy_eV,
        counts=source.counts,
        energy_loss_eV=source.energy_loss_eV,
        monitor=source.monitor,
        acquisition_time_s=source.acquisition_time_s,
        uncertainty=[2.0, 4.0],
    )
    result = correct_spectrum(spectrum)

    np.testing.assert_allclose(result.statistical_uncertainty, [0.1, 0.1])


def test_correction_chain_propagates_multiplicative_factor_uncertainty() -> None:
    result = correct_spectrum(
        _spectrum(),
        path_transmission=0.5,
        path_transmission_uncertainty=0.05,
        cross_section_correction=2.0,
        cross_section_correction_uncertainty=0.1,
    )

    # Before multiplicative corrections I=[5, 10] with sigma=[0.5, 0.5].
    # Divide by T=0.5, then multiply by C=2; each factor contribution is
    # evaluated at the stage where it is applied.
    expected_variance = (
        np.array([0.5, 0.5]) ** 2 / 0.5**2
        + (np.array([5.0, 10.0]) * 0.05 / 0.5**2) ** 2
    ) * 2.0**2 + (np.array([10.0, 20.0]) * 0.1) ** 2
    np.testing.assert_allclose(result.statistical_uncertainty**2, expected_variance)


def test_factor_uncertainty_requires_its_factor() -> None:
    with pytest.raises(ValueError, match="requires"):
        correct_spectrum(_spectrum(), path_transmission_uncertainty=0.01)
