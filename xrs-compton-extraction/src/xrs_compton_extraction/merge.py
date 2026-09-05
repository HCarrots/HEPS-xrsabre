"""Explicit, uncertainty-aware merging of repeated XRS scans."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .data import XRSDataset, XRSSpectrum
from .exceptions import DataValidationError

CoordinateName = Literal["energy_loss", "energy"]
Weighting = Literal["equal", "inverse_variance"]
Interpolation = Literal["linear"] | None


@dataclass(frozen=True, slots=True)
class MergeDiagnostics:
    """Auditable repeatability and drift metrics from a scan merge."""

    source_scan_ids: tuple[str, ...]
    coordinate_name: CoordinateName
    weighting: Weighting
    interpolation: Interpolation
    coordinate_max_abs_deviation_eV: tuple[float | None, ...]
    centroid_shifts_eV: tuple[float | None, ...]
    repeatability_std: NDArray[np.float64]
    repeatability_rms: float
    relative_repeatability_rms: float
    reduced_chi_square: float | None

    def __post_init__(self) -> None:
        repeatability = np.array(self.repeatability_std, dtype=float, copy=True)
        if repeatability.ndim != 1 or not np.all(np.isfinite(repeatability)):
            raise DataValidationError("repeatability_std must be a finite one-dimensional array")
        if np.any(repeatability < 0):
            raise DataValidationError("repeatability_std must be non-negative")
        repeatability.setflags(write=False)
        object.__setattr__(self, "repeatability_std", repeatability)


@dataclass(frozen=True, slots=True)
class MergeResult:
    """Merged spectrum together with metrics needed for quality control."""

    spectrum: XRSSpectrum
    diagnostics: MergeDiagnostics


def merge_dataset(
    dataset: XRSDataset,
    **kwargs: object,
) -> MergeResult:
    """Merge all spectra in one dataset as repeated scans of one channel."""

    if not isinstance(dataset, XRSDataset):
        raise TypeError("dataset must be an XRSDataset")
    return merge_spectra(dataset.spectra, **kwargs)


def merge_spectra(
    spectra: Sequence[XRSSpectrum],
    *,
    coordinate: CoordinateName = "energy_loss",
    interpolation: Interpolation = None,
    target_coordinate_eV: ArrayLike | None = None,
    weighting: Weighting = "equal",
    coordinate_rtol: float = 1e-9,
    coordinate_atol_eV: float = 1e-8,
    output_scan_id: str | None = None,
) -> MergeResult:
    """Merge repeated scans without implicit interpolation or truncation.

    By default all coordinate arrays must have identical shapes and agree
    within the configured tolerances.  Setting ``interpolation='linear'`` is an
    explicit opt-in; the first spectrum's grid is then used unless
    ``target_coordinate_eV`` is supplied.  Extrapolation is always rejected.

    ``inverse_variance`` weighting requires a strictly positive uncertainty
    array on every input spectrum.  ``equal`` weighting can operate without
    uncertainties and uses between-scan repeatability for the merged standard
    error in that case.
    """

    items = tuple(spectra)
    if not items:
        raise DataValidationError("spectra must contain at least one XRSSpectrum")
    if not all(isinstance(item, XRSSpectrum) for item in items):
        raise DataValidationError("spectra must contain only XRSSpectrum objects")
    if coordinate not in ("energy_loss", "energy"):
        raise ValueError("coordinate must be 'energy_loss' or 'energy'")
    if interpolation not in (None, "linear"):
        raise ValueError("interpolation must be None or 'linear'")
    if weighting not in ("equal", "inverse_variance"):
        raise ValueError("weighting must be 'equal' or 'inverse_variance'")
    if coordinate_rtol < 0 or coordinate_atol_eV < 0:
        raise ValueError("coordinate tolerances must be non-negative")

    _require_same_channel(items)
    axes = tuple(_coordinate(item, coordinate) for item in items)
    for index, axis in enumerate(axes):
        _validate_axis(axis, f"spectrum {index} {coordinate} coordinate")

    if target_coordinate_eV is None:
        target = np.array(axes[0], dtype=float, copy=True)
    else:
        target = np.asarray(target_coordinate_eV, dtype=float)
        if target.ndim != 1 or target.size == 0 or not np.all(np.isfinite(target)):
            raise DataValidationError(
                "target_coordinate_eV must be a non-empty finite one-dimensional array"
            )
        target = np.array(target, dtype=float, copy=True)
        _validate_axis(target, "target_coordinate_eV")

    if interpolation is None:
        for index, axis in enumerate(axes):
            if axis.shape != target.shape or not np.allclose(
                axis,
                target,
                rtol=coordinate_rtol,
                atol=coordinate_atol_eV,
            ):
                raise DataValidationError(
                    f"Spectrum {index} coordinate does not match the target grid. "
                    "Pass interpolation='linear' explicitly to resample; coordinates "
                    "are never interpolated or truncated implicitly."
                )
    else:
        for index, axis in enumerate(axes):
            _require_no_extrapolation(
                axis,
                target,
                source_label=f"spectrum {index}",
                atol=coordinate_atol_eV,
            )

    aligned_counts = np.stack(
        [
            _align_series(axis, item.counts, target, interpolation)
            for axis, item in zip(axes, items, strict=True)
        ]
    )
    aligned_variances = _aligned_variances(
        items,
        axes,
        target,
        interpolation,
        require_all=weighting == "inverse_variance",
    )

    if weighting == "inverse_variance":
        if aligned_variances is None:  # pragma: no cover - guarded above
            raise DataValidationError(
                "inverse_variance weighting requires uncertainty on every spectrum"
            )
        if np.any(aligned_variances <= 0):
            raise DataValidationError(
                "inverse_variance weighting requires strictly positive uncertainties"
            )
        raw_weights = 1.0 / aligned_variances
        weight_fractions = raw_weights / np.sum(raw_weights, axis=0)
        merged_counts = np.sum(weight_fractions * aligned_counts, axis=0)
        merged_uncertainty = np.sqrt(1.0 / np.sum(raw_weights, axis=0))
    else:
        weight_fractions = np.full_like(
            aligned_counts, 1.0 / len(items), dtype=float
        )
        merged_counts = np.mean(aligned_counts, axis=0)
        if aligned_variances is not None:
            merged_uncertainty = np.sqrt(np.sum(aligned_variances, axis=0)) / len(items)
        elif len(items) > 1:
            merged_uncertainty = np.std(aligned_counts, axis=0, ddof=1) / np.sqrt(len(items))
        else:
            merged_uncertainty = None

    repeatability_std = (
        np.std(aligned_counts, axis=0, ddof=1)
        if len(items) > 1
        else np.zeros(target.size, dtype=float)
    )
    residuals = aligned_counts - merged_counts
    repeatability_rms = float(np.sqrt(np.mean(np.square(residuals))))
    signal_rms = float(np.sqrt(np.mean(np.square(merged_counts))))
    relative_repeatability_rms = (
        repeatability_rms / signal_rms if signal_rms > 0 else 0.0
    )
    reduced_chi_square = _reduced_chi_square(
        aligned_counts, merged_counts, aligned_variances
    )

    coordinate_deviations = tuple(
        _coordinate_deviation(axis, axes[0]) for axis in axes
    )
    centroids = tuple(_centroid(target, counts) for counts in aligned_counts)
    reference_centroid = centroids[0]
    centroid_shifts = tuple(
        None
        if value is None or reference_centroid is None
        else float(value - reference_centroid)
        for value in centroids
    )

    q_inverse_angstrom = _merge_optional_field(
        items,
        "q_inverse_angstrom",
        axes,
        target,
        interpolation,
        weight_fractions,
    )
    q_au = _merge_optional_field(
        items,
        "q_au",
        axes,
        target,
        interpolation,
        weight_fractions,
    )
    monitor = _merge_optional_field(
        items,
        "monitor",
        axes,
        target,
        interpolation,
        weight_fractions,
    )
    acquisition_time = _merge_optional_field(
        items,
        "acquisition_time_s",
        axes,
        target,
        interpolation,
        weight_fractions,
    )
    merged_energy_loss = (
        target
        if coordinate == "energy_loss"
        else _merge_optional_field(
            items,
            "energy_loss_eV",
            axes,
            target,
            interpolation,
            weight_fractions,
        )
    )

    source_scan_ids = tuple(item.scan_id for item in items)
    diagnostics = MergeDiagnostics(
        source_scan_ids=source_scan_ids,
        coordinate_name=coordinate,
        weighting=weighting,
        interpolation=interpolation,
        coordinate_max_abs_deviation_eV=coordinate_deviations,
        centroid_shifts_eV=centroid_shifts,
        repeatability_std=repeatability_std,
        repeatability_rms=repeatability_rms,
        relative_repeatability_rms=relative_repeatability_rms,
        reduced_chi_square=reduced_chi_square,
    )
    spectrum = XRSSpectrum(
        energy_eV=(
            target if coordinate == "energy" else _merge_optional_field(
                items, "energy_eV", axes, target, interpolation, weight_fractions
            )
        ),
        incident_energy_ev=_merge_optional_field(
            items, "incident_energy_ev", axes, target, interpolation, weight_fractions
        ),
        scattered_energy_ev=_merge_optional_field(
            items, "scattered_energy_ev", axes, target, interpolation, weight_fractions
        ),
        counts=merged_counts,
        energy_loss_eV=merged_energy_loss,
        q_inverse_angstrom=q_inverse_angstrom,
        q_au=q_au,
        monitor=monitor,
        acquisition_time_s=acquisition_time,
        uncertainty=merged_uncertainty,
        scan_id=output_scan_id or _merged_scan_id(source_scan_ids),
        analyzer_id=items[0].analyzer_id,
        roi_id=items[0].roi_id,
        metadata={
            "merge": {
                "source_scan_ids": source_scan_ids,
                "coordinate": coordinate,
                "weighting": weighting,
                "interpolation": interpolation,
                "coordinate_max_abs_deviation_eV": coordinate_deviations,
                "centroid_shifts_eV": centroid_shifts,
                "repeatability_rms": repeatability_rms,
                "relative_repeatability_rms": relative_repeatability_rms,
                "reduced_chi_square": reduced_chi_square,
            }
        },
    )
    return MergeResult(spectrum=spectrum, diagnostics=diagnostics)


def _coordinate(spectrum: XRSSpectrum, name: CoordinateName) -> NDArray[np.float64]:
    if name == "energy":
        return np.asarray(spectrum.energy_eV, dtype=float)
    if spectrum.energy_loss_eV is None:
        raise DataValidationError(
            f"Spectrum {spectrum.scan_id!r} has no energy_loss_eV coordinate"
        )
    return np.asarray(spectrum.energy_loss_eV, dtype=float)


def _require_same_channel(spectra: tuple[XRSSpectrum, ...]) -> None:
    analyzers = {item.analyzer_id for item in spectra}
    rois = {item.roi_id for item in spectra}
    if len(analyzers) != 1 or len(rois) != 1:
        raise DataValidationError(
            "Only repeated scans from one analyzer/ROI channel can be merged; "
            f"got analyzers={sorted(analyzers)!r}, rois={sorted(rois)!r}."
        )


def _validate_axis(axis: NDArray[np.float64], label: str) -> None:
    if axis.ndim != 1 or axis.size == 0 or not np.all(np.isfinite(axis)):
        raise DataValidationError(f"{label} must be a non-empty finite one-dimensional array")
    differences = np.diff(axis)
    if not (np.all(differences > 0) or np.all(differences < 0)):
        raise DataValidationError(f"{label} must be strictly monotonic with no duplicates")


def _require_no_extrapolation(
    source: NDArray[np.float64],
    target: NDArray[np.float64],
    *,
    source_label: str,
    atol: float,
) -> None:
    source_min, source_max = float(np.min(source)), float(np.max(source))
    target_min, target_max = float(np.min(target)), float(np.max(target))
    if target_min < source_min - atol or target_max > source_max + atol:
        raise DataValidationError(
            f"Target grid [{target_min:g}, {target_max:g}] eV requires extrapolation "
            f"outside {source_label} range [{source_min:g}, {source_max:g}] eV. "
            "Extrapolation and silent truncation are not allowed."
        )


def _align_series(
    source_axis: NDArray[np.float64],
    values: ArrayLike,
    target: NDArray[np.float64],
    interpolation: Interpolation,
) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        return np.full(target.size, float(array), dtype=float)
    if array.ndim != 1 or array.size != source_axis.size:
        raise DataValidationError(
            "A point-wise merge field must be scalar or match its source coordinate"
        )
    if interpolation is None:
        return np.array(array, dtype=float, copy=True)
    source_x, source_y = _ascending(source_axis, array)
    target_x, reverse = _ascending_target(target)
    result = np.interp(target_x, source_x, source_y)
    return result[::-1] if reverse else result


def _align_variance(
    source_axis: NDArray[np.float64],
    uncertainty: ArrayLike,
    target: NDArray[np.float64],
    interpolation: Interpolation,
) -> NDArray[np.float64]:
    sigma = np.asarray(uncertainty, dtype=float)
    if sigma.ndim == 0:
        return np.full(target.size, float(sigma) ** 2, dtype=float)
    if sigma.ndim != 1 or sigma.size != source_axis.size:
        raise DataValidationError(
            "Uncertainty must be scalar or match its source coordinate"
        )
    variance = np.square(sigma)
    if interpolation is None:
        return variance
    if source_axis.size == 1:
        # A one-point grid can only map to the same one-point target; the
        # no-extrapolation check has already rejected every other case.
        return np.full(target.size, float(variance[0]), dtype=float)

    source_x, source_variance = _ascending(source_axis, variance)
    target_x, reverse = _ascending_target(target)
    right = np.searchsorted(source_x, target_x, side="right")
    right = np.clip(right, 1, source_x.size - 1)
    left = right - 1
    denominator = source_x[right] - source_x[left]
    fraction = (target_x - source_x[left]) / denominator
    result = (
        np.square(1.0 - fraction) * source_variance[left]
        + np.square(fraction) * source_variance[right]
    )
    return result[::-1] if reverse else result


def _aligned_variances(
    spectra: tuple[XRSSpectrum, ...],
    axes: tuple[NDArray[np.float64], ...],
    target: NDArray[np.float64],
    interpolation: Interpolation,
    *,
    require_all: bool,
) -> NDArray[np.float64] | None:
    available = tuple(item.uncertainty is not None for item in spectra)
    if require_all and not all(available):
        missing = [
            item.scan_id or str(index)
            for index, item in enumerate(spectra)
            if item.uncertainty is None
        ]
        raise DataValidationError(
            "inverse_variance weighting requires uncertainty on every spectrum; "
            f"missing for: {', '.join(missing)}"
        )
    if not all(available):
        return None
    return np.stack(
        [
            _align_variance(axis, item.uncertainty, target, interpolation)
            for axis, item in zip(axes, spectra, strict=True)
        ]
    )


def _merge_optional_field(
    spectra: tuple[XRSSpectrum, ...],
    field_name: str,
    axes: tuple[NDArray[np.float64], ...],
    target: NDArray[np.float64],
    interpolation: Interpolation,
    weight_fractions: NDArray[np.float64],
) -> NDArray[np.float64] | None:
    values = tuple(getattr(item, field_name) for item in spectra)
    if not all(value is not None for value in values):
        return None
    aligned = np.stack(
        [
            _align_series(axis, value, target, interpolation)
            for axis, value in zip(axes, values, strict=True)
        ]
    )
    return np.sum(weight_fractions * aligned, axis=0)


def _ascending(
    axis: NDArray[np.float64], values: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if axis[0] < axis[-1]:
        return axis, values
    return axis[::-1], values[::-1]


def _ascending_target(
    target: NDArray[np.float64],
) -> tuple[NDArray[np.float64], bool]:
    if target[0] < target[-1]:
        return target, False
    return target[::-1], True


def _coordinate_deviation(
    axis: NDArray[np.float64], reference: NDArray[np.float64]
) -> float | None:
    if axis.shape != reference.shape:
        return None
    return float(np.max(np.abs(axis - reference)))


def _centroid(
    coordinate: NDArray[np.float64], intensity: NDArray[np.float64]
) -> float | None:
    total = float(np.sum(intensity))
    if total <= 0:
        return None
    return float(np.sum(coordinate * intensity) / total)


def _reduced_chi_square(
    aligned_counts: NDArray[np.float64],
    merged_counts: NDArray[np.float64],
    aligned_variances: NDArray[np.float64] | None,
) -> float | None:
    scan_count, point_count = aligned_counts.shape
    if aligned_variances is None or scan_count < 2:
        return None
    if np.any(aligned_variances <= 0):
        return None
    degrees_of_freedom = point_count * (scan_count - 1)
    value = np.sum(np.square(aligned_counts - merged_counts) / aligned_variances)
    return float(value / degrees_of_freedom)


def _merged_scan_id(source_scan_ids: tuple[str, ...]) -> str:
    labels = tuple(value or f"scan-{index}" for index, value in enumerate(source_scan_ids))
    return "merged[" + ",".join(labels) + "]"


__all__ = [
    "MergeDiagnostics",
    "MergeResult",
    "merge_dataset",
    "merge_spectra",
]
