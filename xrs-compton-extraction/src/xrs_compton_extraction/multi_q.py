"""Auditable comparison and averaging of independently extracted q channels.

The routines in this module operate on :class:`ExtractionResult` objects.  A
coordinate mismatch is an error by default.  Linear interpolation must be
requested explicitly, and it is never allowed to extrapolate beyond any used
channel's measured energy-loss range.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .constants import AU_WAVEVECTOR_INVERSE_ANGSTROM
from .data import ExtractionResult
from .exceptions import DataValidationError

Weighting = Literal["equal", "inverse_variance"]
Interpolation = Literal["linear"] | None


@dataclass(frozen=True, slots=True)
class MultiQDiagnostics:
    """Diagnostics aligned with the used and rejected channel labels."""

    weighting: Weighting
    interpolation: Interpolation
    coordinate_max_abs_deviation_eV: tuple[float | None, ...]
    q_mean_au: tuple[float, ...]
    q_mean_inverse_angstrom: tuple[float, ...]
    rms_deviation_from_average: tuple[float, ...]
    negative_point_fraction: tuple[float, ...]
    average_negative_point_fraction: float
    reduced_chi_square: float | None
    effective_channel_count: NDArray[np.float64]
    rejection_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        effective_count = _readonly_array(
            self.effective_channel_count,
            "effective_channel_count",
            ndim=1,
            nonnegative=True,
        )
        object.__setattr__(self, "effective_channel_count", effective_count)


@dataclass(frozen=True, slots=True)
class MultiQResult:
    """Aligned single-channel arrays and their uncertainty-aware average.

    ``single_channel_*`` and ``weights`` have shape ``(channel, energy)`` and
    contain used channels only.  Every original result, including rejected
    channels, remains available in ``source_results``.
    """

    energy_loss_eV: NDArray[np.float64]
    single_channel_edges: NDArray[np.float64]
    single_channel_statistical_uncertainties: NDArray[np.float64]
    single_channel_model_uncertainties: NDArray[np.float64]
    q_au: NDArray[np.float64]
    q_inverse_angstrom: NDArray[np.float64]
    weights: NDArray[np.float64]
    average_edge: NDArray[np.float64]
    statistical_uncertainty: NDArray[np.float64]
    model_uncertainty: NDArray[np.float64]
    total_uncertainty: NDArray[np.float64]
    all_channels: tuple[str, ...]
    used_channels: tuple[str, ...]
    rejected_channels: tuple[str, ...]
    source_results: tuple[ExtractionResult, ...]
    diagnostics: MultiQDiagnostics

    def __post_init__(self) -> None:
        energy = _readonly_array(self.energy_loss_eV, "energy_loss_eV", ndim=1)
        channel_count = len(self.used_channels)
        shape = (channel_count, energy.size)
        for name in (
            "single_channel_edges",
            "single_channel_statistical_uncertainties",
            "single_channel_model_uncertainties",
            "q_au",
            "q_inverse_angstrom",
            "weights",
        ):
            value = _readonly_array(
                getattr(self, name),
                name,
                ndim=2,
                nonnegative=name
                in {
                    "single_channel_statistical_uncertainties",
                    "single_channel_model_uncertainties",
                    "q_au",
                    "q_inverse_angstrom",
                    "weights",
                },
            )
            if value.shape != shape:
                raise DataValidationError(f"{name} has shape {value.shape}, expected {shape}")
            object.__setattr__(self, name, value)
        for name in (
            "average_edge",
            "statistical_uncertainty",
            "model_uncertainty",
            "total_uncertainty",
        ):
            value = _readonly_array(
                getattr(self, name),
                name,
                ndim=1,
                nonnegative=name != "average_edge",
            )
            if value.shape != energy.shape:
                raise DataValidationError(
                    f"{name} has shape {value.shape}, expected {energy.shape}"
                )
            object.__setattr__(self, name, value)
        if len(self.all_channels) != len(self.source_results):
            raise DataValidationError(
                "all_channels and source_results must have the same length"
            )
        if len(set(self.all_channels)) != len(self.all_channels):
            raise DataValidationError("channel labels must be unique")
        if set(self.used_channels) & set(self.rejected_channels):
            raise DataValidationError("used_channels and rejected_channels must be disjoint")
        if tuple(self.used_channels) + tuple(self.rejected_channels) == ():
            raise DataValidationError("at least one channel is required")
        if not np.allclose(np.sum(self.weights, axis=0), 1.0):
            raise DataValidationError("weights must sum to one at every energy point")
        expected_total = np.hypot(
            self.statistical_uncertainty, self.model_uncertainty
        )
        if not np.allclose(self.total_uncertainty, expected_total):
            raise DataValidationError(
                "total_uncertainty must combine statistical and model terms in quadrature"
            )
        object.__setattr__(self, "energy_loss_eV", energy)

    @property
    def energy_loss_ev(self) -> NDArray[np.float64]:
        """PEP 8 alias for :attr:`energy_loss_eV`."""

        return self.energy_loss_eV


def average_multi_q(
    results: Sequence[ExtractionResult],
    *,
    channel_labels: Sequence[str] | None = None,
    excluded_channels: Sequence[str | int] = (),
    reject_quality_grade: bool = True,
    weighting: Weighting = "equal",
    interpolation: Interpolation = None,
    target_energy_loss_eV: ArrayLike | None = None,
    coordinate_rtol: float = 1e-9,
    coordinate_atol_eV: float = 1e-8,
) -> MultiQResult:
    """Compare and average extracted signals from several momentum transfers.

    Statistical and model uncertainties are propagated independently as
    independent channel errors, using the same normalized weights.  For
    ``inverse_variance`` weighting, the weights are calculated from the sum of
    statistical and model variances.  No signal or uncertainty values are
    clipped.

    Results graded ``Reject`` are omitted by default.  Additional channels can
    be excluded by their label or zero-based input index.  Rejection choices
    and reasons are retained in the returned diagnostics.
    """

    items = tuple(results)
    if not items:
        raise DataValidationError("results must contain at least one ExtractionResult")
    if not all(isinstance(item, ExtractionResult) for item in items):
        raise DataValidationError(
            "results must contain only ExtractionResult objects"
        )
    labels = _channel_labels(channel_labels, len(items))
    if weighting not in ("equal", "inverse_variance"):
        raise ValueError("weighting must be 'equal' or 'inverse_variance'")
    if interpolation not in (None, "linear"):
        raise ValueError("interpolation must be None or 'linear'")
    if (
        isinstance(coordinate_rtol, bool)
        or isinstance(coordinate_atol_eV, bool)
        or not np.isfinite(coordinate_rtol)
        or not np.isfinite(coordinate_atol_eV)
        or coordinate_rtol < 0
        or coordinate_atol_eV < 0
    ):
        raise ValueError("coordinate tolerances must be finite and non-negative")

    excluded_indices = _excluded_indices(excluded_channels, labels)
    rejection_by_index: dict[int, str] = {
        index: "explicitly excluded" for index in excluded_indices
    }
    if reject_quality_grade:
        for index, item in enumerate(items):
            if item.quality_grade == "Reject" and index not in rejection_by_index:
                rejection_by_index[index] = "quality_grade=Reject"
    used_indices = tuple(
        index for index in range(len(items)) if index not in rejection_by_index
    )
    if not used_indices:
        raise DataValidationError("all multi-q channels were rejected")
    used_results = tuple(items[index] for index in used_indices)
    used_labels = tuple(labels[index] for index in used_indices)
    rejected_indices = tuple(
        index for index in range(len(items)) if index in rejection_by_index
    )
    rejected_labels = tuple(labels[index] for index in rejected_indices)

    axes = tuple(np.asarray(item.energy_loss_eV, dtype=float) for item in used_results)
    for label, axis in zip(used_labels, axes, strict=True):
        _validate_axis(axis, f"channel {label!r} energy_loss_eV")
    if target_energy_loss_eV is None:
        target = np.array(axes[0], dtype=float, copy=True)
    else:
        target = np.asarray(target_energy_loss_eV, dtype=float)
        _validate_axis(target, "target_energy_loss_eV")
        target = np.array(target, dtype=float, copy=True)

    deviations: list[float | None] = []
    for label, axis in zip(used_labels, axes, strict=True):
        deviation = (
            float(np.max(np.abs(axis - target)))
            if axis.shape == target.shape
            else None
        )
        deviations.append(deviation)
        if interpolation is None:
            if axis.shape != target.shape or not np.allclose(
                axis,
                target,
                rtol=coordinate_rtol,
                atol=coordinate_atol_eV,
            ):
                raise DataValidationError(
                    f"Channel {label!r} energy grid does not match the target grid. "
                    "Pass interpolation='linear' explicitly to resample; implicit "
                    "interpolation or truncation is not allowed."
                )
        else:
            _require_no_extrapolation(axis, target, label)

    edges = np.stack(
        [
            _align_values(axis, item.extracted_edge, target, interpolation)
            for axis, item in zip(axes, used_results, strict=True)
        ]
    )
    statistical_variances = np.stack(
        [
            _align_variance(
                axis, item.statistical_uncertainty, target, interpolation
            )
            for axis, item in zip(axes, used_results, strict=True)
        ]
    )
    model_variances = np.stack(
        [
            _align_variance(axis, item.model_uncertainty, target, interpolation)
            for axis, item in zip(axes, used_results, strict=True)
        ]
    )
    total_variances = statistical_variances + model_variances

    if weighting == "inverse_variance":
        if np.any(total_variances <= 0):
            raise DataValidationError(
                "inverse_variance weighting requires strictly positive total "
                "uncertainty for every used channel and energy point"
            )
        raw_weights = 1.0 / total_variances
        weights = raw_weights / np.sum(raw_weights, axis=0)
    else:
        weights = np.full(edges.shape, 1.0 / len(used_results), dtype=float)

    average = np.sum(weights * edges, axis=0)
    statistical_uncertainty = np.sqrt(
        np.sum(np.square(weights) * statistical_variances, axis=0)
    )
    model_uncertainty = np.sqrt(
        np.sum(np.square(weights) * model_variances, axis=0)
    )
    total_uncertainty = np.hypot(statistical_uncertainty, model_uncertainty)

    q_au = np.stack(
        [
            _q_in_both_units(axis, item, target, interpolation)[0]
            for axis, item in zip(axes, used_results, strict=True)
        ]
    )
    q_inverse_angstrom = q_au * AU_WAVEVECTOR_INVERSE_ANGSTROM
    deviations_from_average = edges - average
    rms_deviations = tuple(
        float(np.sqrt(np.mean(np.square(row)))) for row in deviations_from_average
    )
    negative_fractions = tuple(float(np.mean(row < 0.0)) for row in edges)
    reduced_chi_square = _reduced_chi_square(
        edges, average, total_variances
    )
    effective_channel_count = 1.0 / np.sum(np.square(weights), axis=0)
    diagnostics = MultiQDiagnostics(
        weighting=weighting,
        interpolation=interpolation,
        coordinate_max_abs_deviation_eV=tuple(deviations),
        q_mean_au=tuple(float(np.mean(row)) for row in q_au),
        q_mean_inverse_angstrom=tuple(
            float(np.mean(row)) for row in q_inverse_angstrom
        ),
        rms_deviation_from_average=rms_deviations,
        negative_point_fraction=negative_fractions,
        average_negative_point_fraction=float(np.mean(average < 0.0)),
        reduced_chi_square=reduced_chi_square,
        effective_channel_count=effective_channel_count,
        rejection_reasons=tuple(
            rejection_by_index[index] for index in rejected_indices
        ),
    )
    return MultiQResult(
        energy_loss_eV=target,
        single_channel_edges=edges,
        single_channel_statistical_uncertainties=np.sqrt(statistical_variances),
        single_channel_model_uncertainties=np.sqrt(model_variances),
        q_au=q_au,
        q_inverse_angstrom=q_inverse_angstrom,
        weights=weights,
        average_edge=average,
        statistical_uncertainty=statistical_uncertainty,
        model_uncertainty=model_uncertainty,
        total_uncertainty=total_uncertainty,
        all_channels=labels,
        used_channels=used_labels,
        rejected_channels=rejected_labels,
        source_results=items,
        diagnostics=diagnostics,
    )


def _channel_labels(labels: Sequence[str] | None, count: int) -> tuple[str, ...]:
    if labels is None:
        return tuple(f"channel-{index}" for index in range(count))
    normalized = tuple(labels)
    if len(normalized) != count:
        raise DataValidationError(
            f"channel_labels has length {len(normalized)}, expected {count}"
        )
    if any(not isinstance(label, str) or not label.strip() for label in normalized):
        raise DataValidationError("channel_labels must contain non-empty strings")
    normalized = tuple(label.strip() for label in normalized)
    if len(set(normalized)) != len(normalized):
        raise DataValidationError("channel_labels must be unique")
    return normalized


def _excluded_indices(
    excluded: Sequence[str | int], labels: tuple[str, ...]
) -> set[int]:
    indices: set[int] = set()
    for value in excluded:
        if isinstance(value, bool):
            raise DataValidationError(
                "excluded_channels entries must be labels or integer indices"
            )
        if isinstance(value, int):
            if value < 0 or value >= len(labels):
                raise DataValidationError(
                    f"excluded channel index {value} is out of range"
                )
            indices.add(value)
        elif isinstance(value, str):
            if value not in labels:
                raise DataValidationError(f"unknown excluded channel label {value!r}")
            indices.add(labels.index(value))
        else:
            raise DataValidationError(
                "excluded_channels entries must be labels or integer indices"
            )
    return indices


def _validate_axis(axis: NDArray[np.float64], name: str) -> None:
    if axis.ndim != 1 or axis.size == 0 or not np.all(np.isfinite(axis)):
        raise DataValidationError(
            f"{name} must be a non-empty finite one-dimensional array"
        )
    difference = np.diff(axis)
    if not (np.all(difference > 0) or np.all(difference < 0)):
        raise DataValidationError(f"{name} must be strictly monotonic")


def _require_no_extrapolation(
    source: NDArray[np.float64], target: NDArray[np.float64], label: str
) -> None:
    source_min = float(np.min(source))
    source_max = float(np.max(source))
    target_min = float(np.min(target))
    target_max = float(np.max(target))
    if target_min < source_min or target_max > source_max:
        raise DataValidationError(
            f"Target grid [{target_min:g}, {target_max:g}] eV requires "
            f"extrapolation beyond channel {label!r} range "
            f"[{source_min:g}, {source_max:g}] eV. Extrapolation is not allowed."
        )


def _ascending(
    coordinate: NDArray[np.float64], values: NDArray[np.float64]
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if coordinate[0] < coordinate[-1]:
        return coordinate, values
    return coordinate[::-1], values[::-1]


def _ascending_target(
    target: NDArray[np.float64],
) -> tuple[NDArray[np.float64], bool]:
    if target[0] < target[-1]:
        return target, False
    return target[::-1], True


def _align_values(
    source: NDArray[np.float64],
    values: ArrayLike,
    target: NDArray[np.float64],
    interpolation: Interpolation,
) -> NDArray[np.float64]:
    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        return np.full(target.size, float(array), dtype=float)
    if array.ndim != 1 or array.size != source.size:
        raise DataValidationError(
            "a point-wise multi-q field must be scalar or match its source grid"
        )
    if interpolation is None:
        return np.array(array, dtype=float, copy=True)
    source_x, source_y = _ascending(source, array)
    target_x, reverse = _ascending_target(target)
    aligned = np.interp(target_x, source_x, source_y)
    return aligned[::-1] if reverse else aligned


def _align_variance(
    source: NDArray[np.float64],
    uncertainty: ArrayLike,
    target: NDArray[np.float64],
    interpolation: Interpolation,
) -> NDArray[np.float64]:
    sigma = np.asarray(uncertainty, dtype=float)
    if sigma.ndim == 0:
        return np.full(target.size, float(sigma) ** 2, dtype=float)
    if sigma.ndim != 1 or sigma.size != source.size:
        raise DataValidationError(
            "a point-wise uncertainty must be scalar or match its source grid"
        )
    variance = np.square(sigma)
    if interpolation is None:
        return variance
    if source.size == 1:
        return np.full(target.size, float(variance[0]), dtype=float)
    source_x, source_variance = _ascending(source, variance)
    target_x, reverse = _ascending_target(target)
    right = np.searchsorted(source_x, target_x, side="right")
    right = np.clip(right, 1, source_x.size - 1)
    left = right - 1
    fraction = (target_x - source_x[left]) / (
        source_x[right] - source_x[left]
    )
    aligned = (
        np.square(1.0 - fraction) * source_variance[left]
        + np.square(fraction) * source_variance[right]
    )
    return aligned[::-1] if reverse else aligned


def _q_in_both_units(
    source: NDArray[np.float64],
    result: ExtractionResult,
    target: NDArray[np.float64],
    interpolation: Interpolation,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    if result.q_au is not None:
        q_au = _align_values(source, result.q_au, target, interpolation)
        return q_au, q_au * AU_WAVEVECTOR_INVERSE_ANGSTROM
    if result.q_inverse_angstrom is None:  # pragma: no cover - model invariant
        raise DataValidationError("an ExtractionResult must contain a q coordinate")
    q_inverse = _align_values(
        source, result.q_inverse_angstrom, target, interpolation
    )
    return q_inverse / AU_WAVEVECTOR_INVERSE_ANGSTROM, q_inverse


def _reduced_chi_square(
    edges: NDArray[np.float64],
    average: NDArray[np.float64],
    variances: NDArray[np.float64],
) -> float | None:
    channel_count, point_count = edges.shape
    if channel_count < 2 or np.any(variances <= 0):
        return None
    degrees_of_freedom = point_count * (channel_count - 1)
    return float(np.sum(np.square(edges - average) / variances) / degrees_of_freedom)


def _readonly_array(
    values: ArrayLike,
    name: str,
    *,
    ndim: int,
    nonnegative: bool = False,
) -> NDArray[np.float64]:
    try:
        array = np.array(values, dtype=float, copy=True)
    except (TypeError, ValueError) as exc:
        raise DataValidationError(f"{name} must be a numeric array") from exc
    if array.ndim != ndim or array.size == 0 or not np.all(np.isfinite(array)):
        raise DataValidationError(
            f"{name} must be a non-empty finite {ndim}-dimensional array"
        )
    if nonnegative and np.any(array < 0):
        raise DataValidationError(f"{name} must be non-negative")
    array.setflags(write=False)
    return array


__all__ = [
    "Interpolation",
    "MultiQDiagnostics",
    "MultiQResult",
    "Weighting",
    "average_multi_q",
]
