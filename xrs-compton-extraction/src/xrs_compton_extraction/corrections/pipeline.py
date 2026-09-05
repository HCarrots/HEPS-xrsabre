"""Auditable composition of implemented intensity-correction primitives."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..data import CorrectionResult, XRSSpectrum
from ..exceptions import DataValidationError
from .normalization import normalize_counts

FloatArray = NDArray[np.float64]


def _series(value: ArrayLike | None, length: int, name: str, default: float) -> FloatArray:
    raw = default if value is None else value
    try:
        array = np.broadcast_to(np.asarray(raw, dtype=float), (length,))
    except ValueError as exc:
        raise ValueError(f"{name} is not broadcast-compatible with the spectrum") from exc
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return np.asarray(array)


def correct_spectrum(
    spectrum: XRSSpectrum,
    *,
    normalize_acquisition_time: bool = True,
    normalize_i0: bool = True,
    detector_efficiency: ArrayLike | float = 1.0,
    acquisition_time_uncertainty_s: ArrayLike | float | None = None,
    i0_uncertainty: ArrayLike | float | None = None,
    detector_efficiency_uncertainty: ArrayLike | float | None = None,
    elastic_component: ArrayLike | None = None,
    elastic_uncertainty: ArrayLike | None = None,
    stray_background: ArrayLike | None = None,
    stray_uncertainty: ArrayLike | None = None,
    path_transmission: ArrayLike | None = None,
    path_transmission_uncertainty: ArrayLike | None = None,
    self_absorption_factor: ArrayLike | None = None,
    self_absorption_uncertainty: ArrayLike | None = None,
    cross_section_correction: ArrayLike | None = None,
    cross_section_correction_uncertainty: ArrayLike | None = None,
) -> CorrectionResult:
    """Apply explicit corrections in a documented order.

    The order is normalization, subtraction of elastic/stray components, path
    and self-absorption correction, then cross-section correction. Optional
    quantities are identity operations; requested time/I0 normalization never
    guesses missing metadata.
    """

    if not isinstance(spectrum, XRSSpectrum):
        raise TypeError("spectrum must be an XRSSpectrum")
    if spectrum.metadata.get("intensity_kind") == "processed" and spectrum.uncertainty is None:
        raise DataValidationError(
            "processed intensity requires explicit uncertainty; Poisson variance cannot be inferred"
        )
    length = len(spectrum)
    if normalize_acquisition_time and spectrum.acquisition_time_s is None:
        raise DataValidationError(
            "acquisition-time normalization requested but acquisition_time_s is missing"
        )
    if normalize_i0 and spectrum.i0 is None:
        raise DataValidationError("I0 normalization requested but monitor/I0 is missing")
    if not normalize_acquisition_time and acquisition_time_uncertainty_s is not None:
        raise ValueError(
            "acquisition_time_uncertainty_s requires acquisition-time normalization"
        )
    if not normalize_i0 and i0_uncertainty is not None:
        raise ValueError("i0_uncertainty requires I0 normalization")
    normalized = normalize_counts(
        spectrum.raw_counts,
        acquisition_time_s=spectrum.acquisition_time_s
        if normalize_acquisition_time
        else 1.0,
        i0=spectrum.i0 if normalize_i0 else 1.0,
        detector_efficiency=detector_efficiency,
        raw_count_variance=None
        if spectrum.uncertainty is None
        else np.square(spectrum.uncertainty),
        acquisition_time_uncertainty_s=acquisition_time_uncertainty_s,
        i0_uncertainty=i0_uncertainty,
        detector_efficiency_uncertainty=detector_efficiency_uncertainty,
    )

    elastic = _series(elastic_component, length, "elastic_component", 0.0)
    stray = _series(stray_background, length, "stray_background", 0.0)
    elastic_sigma = _series(elastic_uncertainty, length, "elastic_uncertainty", 0.0)
    stray_sigma = _series(stray_uncertainty, length, "stray_uncertainty", 0.0)
    if np.any(elastic_sigma < 0) or np.any(stray_sigma < 0):
        raise ValueError("component uncertainties must be non-negative")

    intensity = normalized.intensity - elastic - stray
    variance = (
        np.square(normalized.statistical_uncertainty)
        + np.square(elastic_sigma)
        + np.square(stray_sigma)
    )
    factors: dict[str, FloatArray] = {
        "normalization": np.asarray(normalized.normalization_factor),
    }

    factor_uncertainties: dict[str, FloatArray] = {}
    for name, attenuation, attenuation_uncertainty in (
        (
            "path_absorption",
            path_transmission,
            path_transmission_uncertainty,
        ),
        (
            "self_absorption",
            self_absorption_factor,
            self_absorption_uncertainty,
        ),
    ):
        if attenuation is None:
            if attenuation_uncertainty is not None:
                raise ValueError(
                    f"{name} uncertainty requires the corresponding attenuation factor"
                )
            continue
        attenuation_array = _series(attenuation, length, name, 1.0)
        if np.any(attenuation_array <= 0) or np.any(attenuation_array > 1):
            raise ValueError(f"{name} attenuation factors must lie in (0, 1]")
        correction = 1.0 / attenuation_array
        previous_intensity = np.array(intensity, copy=True)
        intensity *= correction
        variance *= np.square(correction)
        if attenuation_uncertainty is not None:
            factor_sigma = _series(
                attenuation_uncertainty,
                length,
                f"{name}_uncertainty",
                0.0,
            )
            if np.any(factor_sigma < 0):
                raise ValueError(f"{name} uncertainty must be non-negative")
            variance += np.square(
                previous_intensity * factor_sigma / np.square(attenuation_array)
            )
            factor_uncertainties[name] = factor_sigma
        factors[name] = correction

    if cross_section_correction is not None:
        correction = _series(
            cross_section_correction, length, "cross_section_correction", 1.0
        )
        if np.any(correction <= 0):
            raise ValueError("cross_section_correction must be positive")
        previous_intensity = np.array(intensity, copy=True)
        intensity *= correction
        variance *= np.square(correction)
        if cross_section_correction_uncertainty is not None:
            correction_sigma = _series(
                cross_section_correction_uncertainty,
                length,
                "cross_section_correction_uncertainty",
                0.0,
            )
            if np.any(correction_sigma < 0):
                raise ValueError(
                    "cross_section_correction_uncertainty must be non-negative"
                )
            variance += np.square(previous_intensity * correction_sigma)
            factor_uncertainties["cross_section"] = correction_sigma
        factors["cross_section"] = correction
    elif cross_section_correction_uncertainty is not None:
        raise ValueError(
            "cross_section_correction_uncertainty requires cross_section_correction"
        )

    component_uncertainties: Mapping[str, FloatArray] = {
        "elastic": elastic_sigma,
        "stray": stray_sigma,
        **factor_uncertainties,
    }
    return CorrectionResult(
        raw_counts=spectrum.raw_counts,
        normalized_intensity=normalized.intensity,
        corrected_intensity=intensity,
        correction_factors=factors,
        statistical_uncertainty=np.sqrt(variance),
        component_uncertainties=component_uncertainties,
        metadata={
            "order": [
                "normalization",
                "elastic_subtraction",
                "stray_subtraction",
                "path_absorption",
                "self_absorption",
                "cross_section",
            ],
            "negative_values_clipped": False,
        },
    )


__all__ = ["correct_spectrum"]
