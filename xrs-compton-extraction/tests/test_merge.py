from __future__ import annotations

import numpy as np
import pytest

from xrs_compton_extraction.data import XRSDataset, XRSSpectrum
from xrs_compton_extraction.exceptions import DataValidationError
from xrs_compton_extraction.merge import merge_dataset, merge_spectra


def _spectrum(
    scan_id: str,
    energy: list[float] | np.ndarray,
    counts: list[float] | np.ndarray,
    *,
    uncertainty: list[float] | np.ndarray | None = None,
    analyzer_id: str = "A-1",
    roi_id: str = "ROI-1",
) -> XRSSpectrum:
    return XRSSpectrum(
        energy_eV=energy,
        energy_loss_eV=energy,
        counts=counts,
        uncertainty=uncertainty,
        scan_id=scan_id,
        analyzer_id=analyzer_id,
        roi_id=roi_id,
    )


def test_equal_weight_merge_and_repeatability() -> None:
    first = _spectrum("1", [0, 1, 2], [10, 20, 30])
    second = _spectrum("2", [0, 1, 2], [20, 30, 40])

    result = merge_spectra((first, second), weighting="equal")

    np.testing.assert_allclose(result.spectrum.counts, [15, 25, 35])
    np.testing.assert_allclose(
        result.diagnostics.repeatability_std,
        np.full(3, np.sqrt(50.0)),
    )
    np.testing.assert_allclose(result.spectrum.uncertainty, [5, 5, 5])
    assert result.diagnostics.source_scan_ids == ("1", "2")
    assert result.diagnostics.interpolation is None
    assert not result.diagnostics.repeatability_std.flags.writeable


def test_inverse_variance_weighting_and_uncertainty() -> None:
    first = _spectrum("1", [0, 1], [10, 10], uncertainty=[1, 1])
    second = _spectrum("2", [0, 1], [20, 20], uncertainty=[2, 2])

    result = merge_spectra((first, second), weighting="inverse_variance")

    np.testing.assert_allclose(result.spectrum.counts, [12, 12])
    np.testing.assert_allclose(result.spectrum.uncertainty, [np.sqrt(0.8)] * 2)
    assert result.diagnostics.reduced_chi_square == pytest.approx(20.0)


def test_inverse_variance_requires_complete_positive_uncertainty() -> None:
    first = _spectrum("1", [0, 1], [1, 2], uncertainty=[1, 1])
    missing = _spectrum("missing", [0, 1], [2, 3])
    with pytest.raises(DataValidationError, match="missing for: missing"):
        merge_spectra((first, missing), weighting="inverse_variance")

    zero = _spectrum("zero", [0, 1], [2, 3], uncertainty=[0, 1])
    with pytest.raises(DataValidationError, match="strictly positive uncertainties"):
        merge_spectra((first, zero), weighting="inverse_variance")


def test_coordinate_mismatch_never_interpolates_implicitly() -> None:
    first = _spectrum("1", [0, 1, 2], [1, 2, 3])
    second = _spectrum("2", [0, 0.5, 1, 1.5, 2], [1, 2, 3, 4, 5])

    with pytest.raises(DataValidationError, match="interpolation='linear' explicitly"):
        merge_spectra((first, second))


def test_explicit_linear_interpolation_uses_target_grid() -> None:
    first = _spectrum("1", [0, 1, 2], [0, 1, 2])
    second = _spectrum("2", [0, 0.5, 1, 1.5, 2], [0, 1, 2, 3, 4])

    result = merge_spectra(
        (first, second),
        interpolation="linear",
        target_coordinate_eV=[0, 1, 2],
    )

    np.testing.assert_allclose(result.spectrum.energy_loss_eV, [0, 1, 2])
    np.testing.assert_allclose(result.spectrum.counts, [0, 1.5, 3])
    assert result.diagnostics.interpolation == "linear"
    assert result.diagnostics.coordinate_max_abs_deviation_eV == (0.0, None)


def test_interpolation_rejects_extrapolation_instead_of_truncating() -> None:
    first = _spectrum("1", [0, 1, 2], [1, 2, 3])
    second = _spectrum("2", [0.5, 1, 1.5], [2, 3, 4])

    with pytest.raises(DataValidationError, match="Extrapolation and silent truncation"):
        merge_spectra((first, second), interpolation="linear")


def test_descending_coordinates_are_supported_when_monotonic() -> None:
    first = _spectrum("1", [2, 1, 0], [2, 1, 0])
    second = _spectrum("2", [2, 1.5, 1, 0.5, 0], [4, 3, 2, 1, 0])

    result = merge_spectra((first, second), interpolation="linear")

    np.testing.assert_allclose(result.spectrum.energy_loss_eV, [2, 1, 0])
    np.testing.assert_allclose(result.spectrum.counts, [3, 1.5, 0])


def test_non_monotonic_or_duplicate_coordinates_are_rejected() -> None:
    invalid = _spectrum("bad", [0, 1, 1], [1, 2, 3])
    with pytest.raises(DataValidationError, match="strictly monotonic"):
        merge_spectra((invalid,))


def test_only_repeated_scans_of_same_channel_can_be_merged() -> None:
    first = _spectrum("1", [0, 1], [1, 2], analyzer_id="A-1")
    second = _spectrum("2", [0, 1], [2, 3], analyzer_id="A-2")

    with pytest.raises(DataValidationError, match="one analyzer/ROI channel"):
        merge_spectra((first, second))


def test_centroid_shift_is_reported_as_drift_metric() -> None:
    energy = np.arange(5.0)
    first = _spectrum("1", energy, [0, 1, 4, 1, 0])
    shifted = _spectrum("2", energy, [0, 0, 1, 4, 1])

    diagnostics = merge_spectra((first, shifted)).diagnostics

    assert diagnostics.centroid_shifts_eV[0] == pytest.approx(0.0)
    assert diagnostics.centroid_shifts_eV[1] == pytest.approx(1.0)
    assert diagnostics.relative_repeatability_rms > 0


def test_merge_dataset_delegates_and_preserves_channel_ids() -> None:
    first = _spectrum("1", [0, 1], [1, 3])
    second = _spectrum("2", [0, 1], [3, 5])
    dataset = XRSDataset(spectra=(first, second))

    result = merge_dataset(dataset, output_scan_id="combined")

    assert result.spectrum.scan_id == "combined"
    assert result.spectrum.analyzer_id == "A-1"
    assert result.spectrum.roi_id == "ROI-1"
    np.testing.assert_allclose(result.spectrum.counts, [2, 4])


@pytest.mark.parametrize("interpolation", [None, "linear"])
def test_energy_loss_merge_preserves_photon_energies(interpolation: str | None) -> None:
    loss = np.array([100.0, 101.0, 102.0])
    scans = tuple(
        XRSSpectrum(
            energy_eV=loss + scattered,
            energy_loss_eV=loss,
            incident_energy_ev=loss + scattered,
            scattered_energy_ev=scattered,
            counts=[10, 20, 30],
        )
        for scattered in (9000.0, 9100.0)
    )
    merged = merge_spectra(scans, interpolation=interpolation).spectrum
    np.testing.assert_allclose(merged.energy_eV, loss + 9050.0)
    np.testing.assert_allclose(merged.incident_energy_ev, loss + 9050.0)
    np.testing.assert_allclose(merged.scattered_energy_ev, 9050.0)
    np.testing.assert_allclose(merged.energy_loss_eV, loss)
