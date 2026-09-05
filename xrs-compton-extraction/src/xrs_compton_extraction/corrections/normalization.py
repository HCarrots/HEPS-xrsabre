"""Acquisition-time, incident-monitor, and detector-efficiency normalization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def _readonly_float_array(value: ArrayLike, *, name: str) -> FloatArray:
    array = np.array(value, dtype=np.float64, copy=True)
    if array.ndim == 0:
        array = array.reshape(1)
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class NormalizationResult:
    """Result of a normalization operation, including propagated variances."""

    intensity: FloatArray
    statistical_uncertainty: FloatArray
    normalization_factor: FloatArray
    count_variance: FloatArray
    acquisition_time_variance_contribution: FloatArray
    i0_variance_contribution: FloatArray
    detector_efficiency_variance_contribution: FloatArray
    metadata: dict[str, Any]

    def __post_init__(self) -> None:
        names = (
            "intensity",
            "statistical_uncertainty",
            "normalization_factor",
            "count_variance",
            "acquisition_time_variance_contribution",
            "i0_variance_contribution",
            "detector_efficiency_variance_contribution",
        )
        arrays = []
        for name in names:
            value = _readonly_float_array(getattr(self, name), name=name)
            object.__setattr__(self, name, value)
            arrays.append(value)
        if len({item.shape for item in arrays}) != 1:
            raise ValueError("normalization result arrays must have identical shapes")
        object.__setattr__(self, "metadata", dict(self.metadata))


def normalize_counts(
    raw_counts: ArrayLike,
    *,
    acquisition_time_s: ArrayLike | float = 1.0,
    i0: ArrayLike | float = 1.0,
    detector_efficiency: ArrayLike | float = 1.0,
    raw_count_variance: ArrayLike | None = None,
    acquisition_time_uncertainty_s: ArrayLike | float | None = None,
    i0_uncertainty: ArrayLike | float | None = None,
    detector_efficiency_uncertainty: ArrayLike | float | None = None,
) -> NormalizationResult:
    """Normalize detector counts and propagate independent statistical errors.

    The returned intensity is ``counts / (time * i0 * efficiency)``. When no
    count variance is supplied, Poisson variance equal to the raw counts is used.
    Every supplied uncertainty is interpreted as one standard deviation in the
    same units as its corresponding quantity. Omitted normalization-factor
    uncertainties contribute exactly zero; no relative error is guessed.
    """

    counts = np.asarray(raw_counts, dtype=np.float64)
    if counts.ndim == 0:
        counts = counts.reshape(1)
    if not np.all(np.isfinite(counts)):
        raise ValueError("raw_counts must contain only finite values")
    if np.any(counts < 0):
        raise ValueError("raw_counts must be non-negative for Poisson propagation")

    try:
        counts_b, time_b, i0_b, efficiency_b = np.broadcast_arrays(
            counts,
            np.asarray(acquisition_time_s, dtype=np.float64),
            np.asarray(i0, dtype=np.float64),
            np.asarray(detector_efficiency, dtype=np.float64),
        )
    except ValueError as exc:
        raise ValueError("normalization inputs are not broadcast-compatible") from exc

    for name, value in (
        ("acquisition_time_s", time_b),
        ("i0", i0_b),
        ("detector_efficiency", efficiency_b),
    ):
        if not np.all(np.isfinite(value)) or np.any(value <= 0):
            raise ValueError(f"{name} must contain finite values greater than zero")

    denominator = time_b * i0_b * efficiency_b
    intensity = counts_b / denominator

    if raw_count_variance is None:
        count_variance_raw = counts_b
        variance_source = "poisson"
    else:
        try:
            count_variance_raw = np.broadcast_to(
                np.asarray(raw_count_variance, dtype=np.float64), counts_b.shape
            )
        except ValueError as exc:
            raise ValueError("raw_count_variance is not broadcast-compatible") from exc
        if not np.all(np.isfinite(count_variance_raw)) or np.any(count_variance_raw < 0):
            raise ValueError("raw_count_variance must be finite and non-negative")
        variance_source = "provided"

    count_variance = count_variance_raw / np.square(denominator)

    def factor_variance_contribution(
        uncertainty: ArrayLike | float | None,
        factor: FloatArray,
        name: str,
    ) -> FloatArray:
        if uncertainty is None:
            return np.zeros_like(intensity)
        try:
            sigma = np.broadcast_to(
                np.asarray(uncertainty, dtype=np.float64), counts_b.shape
            )
        except ValueError as exc:
            raise ValueError(f"{name} is not broadcast-compatible") from exc
        if not np.all(np.isfinite(sigma)) or np.any(sigma < 0):
            raise ValueError(f"{name} must be finite and non-negative")
        return np.square(intensity * sigma / factor)

    time_contribution = factor_variance_contribution(
        acquisition_time_uncertainty_s,
        time_b,
        "acquisition_time_uncertainty_s",
    )
    i0_contribution = factor_variance_contribution(
        i0_uncertainty,
        i0_b,
        "i0_uncertainty",
    )
    efficiency_contribution = factor_variance_contribution(
        detector_efficiency_uncertainty,
        efficiency_b,
        "detector_efficiency_uncertainty",
    )

    total_variance = (
        count_variance
        + time_contribution
        + i0_contribution
        + efficiency_contribution
    )
    return NormalizationResult(
        intensity=intensity,
        statistical_uncertainty=np.sqrt(total_variance),
        normalization_factor=1.0 / denominator,
        count_variance=count_variance,
        acquisition_time_variance_contribution=time_contribution,
        i0_variance_contribution=i0_contribution,
        detector_efficiency_variance_contribution=efficiency_contribution,
        metadata={
            "formula": "counts / (acquisition_time_s * i0 * detector_efficiency)",
            "count_variance_source": variance_source,
        },
    )
