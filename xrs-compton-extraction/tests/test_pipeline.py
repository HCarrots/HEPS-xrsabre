from __future__ import annotations

import numpy as np
import pytest

from xrs_compton_extraction.backgrounds import pearson_background
from xrs_compton_extraction.data import XRSSpectrum
from xrs_compton_extraction.exceptions import DataValidationError
from xrs_compton_extraction.pipeline import extract_pearson


def _spectrum(*, include_q: bool = True) -> tuple[XRSSpectrum, np.ndarray, np.ndarray]:
    energy_loss = np.linspace(0.0, 100.0, 401)
    background = pearson_background(energy_loss, 200.0, 50.0, 0.03, 1.7)
    edge = np.where((energy_loss >= 40.0) & (energy_loss <= 60.0), 25.0, 0.0)
    spectrum = XRSSpectrum(
        energy_eV=energy_loss,
        counts=background + edge,
        energy_loss_eV=energy_loss,
        q_au=2.0 if include_q else None,
        monitor=1.0,
        acquisition_time_s=1.0,
        scan_id="synthetic",
        analyzer_id="A1",
    )
    return spectrum, background, edge


def test_end_to_end_pearson_extraction_recovers_injected_edge() -> None:
    spectrum, background, edge = _spectrum()
    result = extract_pearson(
        spectrum,
        fit_windows_ev=((0.0, 35.0), (65.0, 100.0)),
        initial=(190.0, 50.0, 0.025, 1.5),
        loss="linear",
    )
    assert result.quality_grade == "Pass"
    assert result.background_model_name == "pearson"
    np.testing.assert_allclose(result.total_background, background, rtol=1e-4, atol=1e-4)
    np.testing.assert_allclose(result.extracted_edge, edge, rtol=1e-4, atol=1e-4)
    assert np.any(result.extracted_edge == 0.0)
    assert result.provenance["negative_values_clipped"] is False


def test_pipeline_requires_requested_monitor_and_time() -> None:
    energy = np.linspace(0.0, 10.0, 21)
    spectrum = XRSSpectrum(
        energy_eV=energy,
        counts=np.ones_like(energy),
        energy_loss_eV=energy,
        q_au=1.0,
    )
    with pytest.raises(DataValidationError, match="acquisition-time"):
        extract_pearson(spectrum, fit_windows_ev=((0.0, 10.0),))


def test_pipeline_requires_q_but_accepts_explicit_value() -> None:
    spectrum, _, _ = _spectrum(include_q=False)
    kwargs = {
        "fit_windows_ev": ((0.0, 35.0), (65.0, 100.0)),
        "initial": (190.0, 50.0, 0.025, 1.5),
        "loss": "linear",
    }
    with pytest.raises(DataValidationError, match="requires q"):
        extract_pearson(spectrum, **kwargs)
    result = extract_pearson(spectrum, q_au=2.0, **kwargs)
    np.testing.assert_allclose(result.q_au, 2.0)


def test_pipeline_never_clips_negative_extraction() -> None:
    spectrum, _, _ = _spectrum()
    counts = np.array(spectrum.raw_counts, copy=True)
    counts[200] = 0.0
    altered = XRSSpectrum(
        energy_eV=spectrum.energy_ev,
        counts=counts,
        energy_loss_eV=spectrum.energy_loss_ev,
        q_au=2.0,
        monitor=1.0,
        acquisition_time_s=1.0,
    )
    result = extract_pearson(
        altered,
        fit_windows_ev=((0.0, 35.0), (65.0, 100.0)),
        initial=(190.0, 50.0, 0.025, 1.5),
        loss="linear",
    )
    assert np.min(result.extracted_edge) < 0.0
    assert result.risk_metrics["negative_area_fraction"] > 0.0


def test_pipeline_uses_explicit_raw_count_uncertainty() -> None:
    source, _, _ = _spectrum()
    spectrum = XRSSpectrum(
        energy_eV=source.energy_eV,
        counts=source.counts,
        energy_loss_eV=source.energy_loss_eV,
        q_au=source.q_au,
        monitor=source.monitor,
        acquisition_time_s=source.acquisition_time_s,
        uncertainty=np.full(len(source), 3.0),
    )
    result = extract_pearson(
        spectrum,
        fit_windows_ev=((0.0, 35.0), (65.0, 100.0)),
        initial=(190.0, 50.0, 0.025, 1.5),
        loss="linear",
    )

    np.testing.assert_allclose(result.statistical_uncertainty, 3.0)
