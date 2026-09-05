from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from xrs_compton_extraction.data import ExtractionResult, XRSSpectrum
from xrs_compton_extraction.plotting import (
    plot_extraction,
    plot_residuals,
    plot_spectrum,
)

plt.switch_backend("Agg")


def _extraction() -> ExtractionResult:
    energy = np.arange(5.0)
    background = np.ones(5)
    edge = np.asarray([-1.0, 0.0, 1.0, 2.0, 3.0])
    corrected = background + edge
    return ExtractionResult(
        energy_loss_eV=energy,
        raw_counts=np.maximum(corrected, 0.0),
        q_au=2.0,
        normalized_intensity=corrected,
        corrected_intensity=corrected,
        valence_background=background,
        total_background=background,
        extracted_edge=edge,
        fit_residual=edge,
        statistical_uncertainty=np.full(5, 0.2),
        background_model_name="pearson",
        quality_grade="Warning",
        software_version="test",
    )


def test_plot_spectrum_labels_energy_and_counts() -> None:
    spectrum = XRSSpectrum(
        energy_eV=[1.0, 2.0],
        counts=[3.0, 4.0],
        energy_loss_eV=[10.0, 11.0],
        q_au=2.0,
        analyzer_id="A1",
    )
    figure, axis = plt.subplots()
    assert plot_spectrum(spectrum, axis=axis) is axis
    assert axis.get_xlabel() == "Energy loss (eV)"
    assert axis.get_ylabel() == "Counts"
    assert "A1" in axis.get_title()
    plt.close(figure)


def test_extraction_and_residual_plots_preserve_negative_values() -> None:
    result = _extraction()
    extraction_axis = plot_extraction(result)
    residual_axis = plot_residuals(result)
    plotted_edge = extraction_axis.lines[2].get_ydata()
    np.testing.assert_array_equal(plotted_edge, result.extracted_edge)
    np.testing.assert_array_equal(residual_axis.lines[1].get_ydata(), result.fit_residual)
    plt.close(extraction_axis.figure)
    plt.close(residual_axis.figure)
