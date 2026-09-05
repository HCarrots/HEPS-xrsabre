"""UI-independent orchestration for the first low-q extraction pipeline."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .backgrounds import fit_pearson
from .corrections import normalize_counts
from .data import ExtractionResult, XRSSpectrum
from .exceptions import DataValidationError
from .geometry import inverse_angstrom_to_au

FloatArray = NDArray[np.float64]


def _pearson_model_uncertainty(
    energy_loss_ev: FloatArray,
    parameters: Mapping[str, float],
    covariance: FloatArray,
) -> FloatArray:
    """Propagate the local Pearson parameter covariance to every energy point."""

    beta1 = parameters["beta1"]
    beta2 = parameters["beta2"]
    beta3 = parameters["beta3"]
    beta4 = parameters["beta4"]
    delta = energy_loss_ev - beta2
    base = 1.0 + np.square(beta3 * delta)
    base_power = np.power(base, -beta4)
    derivatives = np.column_stack(
        (
            base_power,
            2.0 * beta1 * beta4 * beta3**2 * delta * np.power(base, -beta4 - 1.0),
            -2.0 * beta1 * beta4 * beta3 * delta**2 * np.power(base, -beta4 - 1.0),
            -beta1 * base_power * np.log(base),
        )
    )
    variance = np.einsum("ij,jk,ik->i", derivatives, covariance, derivatives)
    return np.sqrt(np.maximum(variance, 0.0))


def _negative_area_fraction(x: FloatArray, values: FloatArray) -> float:
    total = float(np.trapezoid(np.abs(values), x=x))
    if total == 0.0:
        return 0.0
    negative = float(np.trapezoid(np.clip(-values, 0.0, None), x=x))
    return max(0.0, min(1.0, negative / total))


def extract_pearson(
    spectrum: XRSSpectrum,
    *,
    fit_windows_ev: Sequence[tuple[float, float]],
    q_au: ArrayLike | float | None = None,
    normalize_acquisition_time: bool = True,
    normalize_i0: bool = True,
    detector_efficiency: ArrayLike | float = 1.0,
    acquisition_time_uncertainty_s: ArrayLike | float | None = None,
    i0_uncertainty: ArrayLike | float | None = None,
    detector_efficiency_uncertainty: ArrayLike | float | None = None,
    initial: Mapping[str, float] | Sequence[float] | None = None,
    bounds: tuple[Sequence[float], Sequence[float]] | None = None,
    loss: str = "soft_l1",
) -> ExtractionResult:
    """Normalize one channel, fit a low-q Pearson background, and extract its edge.

    No energy range is guessed: callers must supply one or more background-only
    fit windows. Missing requested monitor/time metadata is an error. Negative
    extracted values are retained and quantified, never clipped.
    """

    if not isinstance(spectrum, XRSSpectrum):
        raise TypeError("spectrum must be an XRSSpectrum")
    if spectrum.metadata.get("intensity_kind") == "processed" and spectrum.uncertainty is None:
        raise DataValidationError(
            "processed intensity requires explicit uncertainty; Poisson variance cannot be inferred"
        )
    if spectrum.energy_loss_ev is None:
        raise DataValidationError("spectrum does not contain an energy-loss coordinate")
    if not fit_windows_ev:
        raise DataValidationError("fit_windows_ev must explicitly identify background-only data")

    acquisition_time = 1.0
    if normalize_acquisition_time:
        if spectrum.acquisition_time_s is None:
            raise DataValidationError(
                "acquisition-time normalization requested but spectrum has no acquisition_time_s"
            )
        acquisition_time = spectrum.acquisition_time_s
    elif acquisition_time_uncertainty_s is not None:
        raise ValueError(
            "acquisition_time_uncertainty_s requires acquisition-time normalization"
        )

    monitor = 1.0
    if normalize_i0:
        if spectrum.i0 is None:
            raise DataValidationError("I0 normalization requested but spectrum has no monitor/I0")
        monitor = spectrum.i0
    elif i0_uncertainty is not None:
        raise ValueError("i0_uncertainty requires I0 normalization")

    normalized = normalize_counts(
        spectrum.raw_counts,
        acquisition_time_s=acquisition_time,
        i0=monitor,
        detector_efficiency=detector_efficiency,
        raw_count_variance=None
        if spectrum.uncertainty is None
        else np.square(spectrum.uncertainty),
        acquisition_time_uncertainty_s=acquisition_time_uncertainty_s,
        i0_uncertainty=i0_uncertainty,
        detector_efficiency_uncertainty=detector_efficiency_uncertainty,
    )
    fit = fit_pearson(
        spectrum.energy_loss_ev,
        normalized.intensity,
        sigma=np.maximum(normalized.statistical_uncertainty, np.finfo(float).tiny),
        fit_windows_ev=fit_windows_ev,
        initial=initial,
        bounds=bounds,
        loss=loss,
    )

    q_values: ArrayLike | None = q_au if q_au is not None else spectrum.q_au
    if q_values is None and spectrum.q_inverse_angstrom is not None:
        q_values = inverse_angstrom_to_au(spectrum.q_inverse_angstrom)
    if q_values is None:
        raise DataValidationError(
            "Pearson extraction requires q_au or q_inverse_angstrom; pass q_au explicitly"
        )

    zero = np.zeros_like(normalized.intensity)
    extracted = normalized.intensity - fit.fitted_background
    model_uncertainty = _pearson_model_uncertainty(
        np.asarray(spectrum.energy_loss_ev),
        fit.parameters,
        fit.covariance,
    )
    negative_fraction = _negative_area_fraction(
        np.asarray(spectrum.energy_loss_ev), extracted
    )
    warnings: list[str] = []
    grade = "Pass"
    if not fit.success:
        grade = "Reject"
        warnings.append(f"Pearson optimizer did not converge: {fit.message}")

    source_identifier = spectrum.metadata.get("source_file") or spectrum.scan_id
    raw_identifiers = (str(source_identifier),) if source_identifier else ()

    return ExtractionResult(
        energy_loss_eV=spectrum.energy_loss_ev,
        raw_counts=spectrum.raw_counts,
        q_au=q_values,
        q_inverse_angstrom=spectrum.q_inverse_angstrom,
        normalized_intensity=normalized.intensity,
        corrected_intensity=normalized.intensity,
        elastic_component=zero,
        stray_background=zero,
        valence_background=fit.fitted_background,
        core_background=zero,
        constant_background=zero,
        total_background=fit.fitted_background,
        extracted_edge=extracted,
        fit_residual=fit.residual,
        statistical_uncertainty=normalized.statistical_uncertainty,
        model_uncertainty=model_uncertainty,
        background_model_name="pearson",
        fit_parameters=fit.parameters,
        parameter_covariance=fit.covariance,
        fit_windows=fit.fit_windows_ev,
        risk_metrics={
            "reduced_chi_square": fit.reduced_chi_square,
            "negative_area_fraction": negative_fraction,
        },
        warnings=warnings,
        quality_grade=grade,
        provenance={
            "normalization": normalized.metadata,
            "negative_values_clipped": False,
            "fit_loss": loss,
        },
        software_version="0.1.0.dev0",
        raw_data_identifiers=raw_identifiers,
    )


__all__ = ["extract_pearson"]
