"""Uncertainty propagation utilities for XRS intensity processing.

The functions here make independence assumptions explicit.  They never infer
correlations between inputs and never repair invalid measurements silently.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .exceptions import DataValidationError

FloatArray = NDArray[np.float64]


def _numeric_array(
    value: ArrayLike,
    name: str,
    *,
    minimum_ndim: int = 0,
    nonnegative: bool = False,
    positive: bool = False,
) -> FloatArray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise DataValidationError(f"{name} must be a numeric array") from exc
    if array.ndim < minimum_ndim:
        raise DataValidationError(f"{name} must have at least {minimum_ndim} dimension(s)")
    if array.size == 0:
        raise DataValidationError(f"{name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise DataValidationError(f"{name} contains NaN or infinite values")
    if nonnegative and np.any(array < 0):
        raise DataValidationError(f"{name} must contain only non-negative values")
    if positive and np.any(array <= 0):
        raise DataValidationError(f"{name} must contain only positive values")
    return array


def _readonly(value: ArrayLike) -> FloatArray:
    result = np.array(value, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _broadcast_to_measurement(
    value: ArrayLike,
    name: str,
    shape: tuple[int, ...],
    *,
    nonnegative: bool = False,
    positive: bool = False,
) -> FloatArray:
    array = _numeric_array(
        value,
        name,
        nonnegative=nonnegative,
        positive=positive,
    )
    try:
        broadcast = np.broadcast_to(array, shape)
    except ValueError as exc:
        raise DataValidationError(
            f"{name} shape {array.shape} cannot be broadcast to measurement shape {shape}"
        ) from exc
    return np.asarray(broadcast, dtype=np.float64)


def poisson_uncertainty(counts: ArrayLike) -> FloatArray:
    """Return the Poisson standard deviation ``sqrt(counts)``."""

    values = _numeric_array(counts, "counts", nonnegative=True)
    return _readonly(np.sqrt(values))


def combine_independent_uncertainties(
    components: Mapping[str, ArrayLike] | ArrayLike,
) -> FloatArray:
    """Combine independent standard uncertainties in quadrature.

    ``components`` can be a mapping of named arrays or an array whose leading
    axis indexes independent components.  Scalar components broadcast normally.
    """

    if isinstance(components, Mapping):
        if not components:
            raise DataValidationError("components must not be empty")
        names = tuple(components)
        if any(not isinstance(name, str) or not name for name in names):
            raise DataValidationError("component names must be non-empty strings")
        arrays = [
            _numeric_array(value, f"components[{name!r}]", nonnegative=True)
            for name, value in components.items()
        ]
        try:
            broadcast = np.broadcast_arrays(*arrays)
        except ValueError as exc:
            shapes = [array.shape for array in arrays]
            raise DataValidationError(
                f"component shapes are not broadcast-compatible: {shapes}"
            ) from exc
        variance = np.zeros(broadcast[0].shape, dtype=np.float64)
        for array in broadcast:
            variance += np.square(array)
        return _readonly(np.sqrt(variance))

    array = _numeric_array(
        components,
        "components",
        minimum_ndim=1,
        nonnegative=True,
    )
    if array.shape[0] == 0:  # pragma: no cover - caught by empty check
        raise DataValidationError("components must contain at least one component")
    return _readonly(np.sqrt(np.sum(np.square(array), axis=0)))


@dataclass(frozen=True, slots=True)
class NormalizationUncertainty:
    """Intensity and separate analytic uncertainty contributions."""

    normalized_intensity: FloatArray
    total_uncertainty: FloatArray
    counts_component: FloatArray
    acquisition_time_component: FloatArray
    i0_component: FloatArray

    def __post_init__(self) -> None:
        expected_shape: tuple[int, ...] | None = None
        for name in (
            "normalized_intensity",
            "total_uncertainty",
            "counts_component",
            "acquisition_time_component",
            "i0_component",
        ):
            array = _numeric_array(
                getattr(self, name),
                name,
                nonnegative=name != "normalized_intensity",
            )
            if expected_shape is None:
                expected_shape = array.shape
            elif array.shape != expected_shape:
                raise DataValidationError(
                    f"{name} shape {array.shape} does not match {expected_shape}"
                )
            object.__setattr__(self, name, _readonly(array))


def propagate_normalization_uncertainty(
    raw_counts: ArrayLike,
    acquisition_time_s: ArrayLike,
    i0: ArrayLike,
    *,
    counts_uncertainty: ArrayLike | None = None,
    acquisition_time_uncertainty_s: ArrayLike | None = None,
    i0_uncertainty: ArrayLike | None = None,
) -> NormalizationUncertainty:
    """Propagate independent errors through ``counts / (time * i0)``.

    If ``counts_uncertainty`` is omitted, Poisson uncertainty is used.  Omitted
    acquisition-time or incident-monitor uncertainties contribute exactly zero.
    Scalars and arrays broadcast to the shape of ``raw_counts``; values that
    would expand that measurement shape are rejected.
    """

    counts = _numeric_array(raw_counts, "raw_counts", nonnegative=True)
    shape = counts.shape
    time = _broadcast_to_measurement(
        acquisition_time_s,
        "acquisition_time_s",
        shape,
        positive=True,
    )
    monitor = _broadcast_to_measurement(i0, "i0", shape, positive=True)
    sigma_counts = (
        np.sqrt(counts)
        if counts_uncertainty is None
        else _broadcast_to_measurement(
            counts_uncertainty,
            "counts_uncertainty",
            shape,
            nonnegative=True,
        )
    )
    sigma_time = (
        np.zeros(shape, dtype=np.float64)
        if acquisition_time_uncertainty_s is None
        else _broadcast_to_measurement(
            acquisition_time_uncertainty_s,
            "acquisition_time_uncertainty_s",
            shape,
            nonnegative=True,
        )
    )
    sigma_i0 = (
        np.zeros(shape, dtype=np.float64)
        if i0_uncertainty is None
        else _broadcast_to_measurement(
            i0_uncertainty,
            "i0_uncertainty",
            shape,
            nonnegative=True,
        )
    )

    denominator = time * monitor
    normalized = counts / denominator
    counts_component = sigma_counts / denominator
    time_component = counts * sigma_time / (np.square(time) * monitor)
    i0_component = counts * sigma_i0 / (time * np.square(monitor))
    total = combine_independent_uncertainties(
        {
            "counts": counts_component,
            "acquisition_time": time_component,
            "i0": i0_component,
        }
    )
    return NormalizationUncertainty(
        normalized_intensity=normalized,
        total_uncertainty=total,
        counts_component=counts_component,
        acquisition_time_component=time_component,
        i0_component=i0_component,
    )


@dataclass(frozen=True, slots=True)
class ScanRepeatability:
    """Mean, sample spread, and standard error across repeated scans."""

    mean: FloatArray
    sample_standard_deviation: FloatArray
    standard_error: FloatArray
    scan_count: int

    def __post_init__(self) -> None:
        if isinstance(self.scan_count, bool) or not isinstance(self.scan_count, int):
            raise DataValidationError("scan_count must be an integer")
        if self.scan_count < 2:
            raise DataValidationError("scan_count must be at least two")
        expected_shape: tuple[int, ...] | None = None
        for name in ("mean", "sample_standard_deviation", "standard_error"):
            array = _numeric_array(
                getattr(self, name),
                name,
                nonnegative=name != "mean",
            )
            if expected_shape is None:
                expected_shape = array.shape
            elif array.shape != expected_shape:
                raise DataValidationError(
                    f"{name} shape {array.shape} does not match {expected_shape}"
                )
            object.__setattr__(self, name, _readonly(array))


def multi_scan_repeatability(scans: ArrayLike) -> ScanRepeatability:
    """Calculate sample standard deviation and standard error across scans.

    The leading axis is the scan axis.  All scans must already be aligned to the
    same point grid; ragged or differently shaped input is rejected by NumPy
    conversion and the explicit dimensionality checks.
    """

    values = _numeric_array(scans, "scans", minimum_ndim=2)
    scan_count = values.shape[0]
    if scan_count < 2:
        raise DataValidationError("scans must contain at least two repeated scans")
    mean = np.mean(values, axis=0)
    sample_std = np.std(values, axis=0, ddof=1)
    standard_error = sample_std / math.sqrt(scan_count)
    return ScanRepeatability(
        mean=mean,
        sample_standard_deviation=sample_std,
        standard_error=standard_error,
        scan_count=scan_count,
    )


@dataclass(frozen=True, slots=True)
class BootstrapResult:
    """Reproducible bootstrap distribution and its standard uncertainty."""

    estimate: FloatArray
    standard_error: FloatArray
    distribution: FloatArray
    seed: int
    resample_count: int
    confidence_interval: tuple[FloatArray, FloatArray] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.seed, bool) or not isinstance(self.seed, int):
            raise DataValidationError("seed must be an integer")
        if isinstance(self.resample_count, bool) or not isinstance(self.resample_count, int):
            raise DataValidationError("resample_count must be an integer")
        if self.resample_count < 2:
            raise DataValidationError("resample_count must be at least two")
        estimate = _numeric_array(self.estimate, "estimate")
        error = _numeric_array(self.standard_error, "standard_error", nonnegative=True)
        distribution = _numeric_array(
            self.distribution,
            "distribution",
            minimum_ndim=1,
        )
        if distribution.shape[0] != self.resample_count:
            raise DataValidationError(
                "distribution leading dimension must equal resample_count"
            )
        if distribution.shape[1:] != estimate.shape or error.shape != estimate.shape:
            raise DataValidationError(
                "estimate, standard_error, and bootstrap distribution shapes are inconsistent"
            )
        interval = None
        if self.confidence_interval is not None:
            if len(self.confidence_interval) != 2:
                raise DataValidationError("confidence_interval must contain lower and upper arrays")
            lower = _numeric_array(self.confidence_interval[0], "confidence_interval.lower")
            upper = _numeric_array(self.confidence_interval[1], "confidence_interval.upper")
            if lower.shape != estimate.shape or upper.shape != estimate.shape:
                raise DataValidationError(
                    "confidence interval bounds must match estimate shape"
                )
            if np.any(lower > upper):
                raise DataValidationError(
                    "confidence interval lower bounds must not exceed upper bounds"
                )
            interval = (_readonly(lower), _readonly(upper))
        object.__setattr__(self, "estimate", _readonly(estimate))
        object.__setattr__(self, "standard_error", _readonly(error))
        object.__setattr__(self, "distribution", _readonly(distribution))
        object.__setattr__(self, "confidence_interval", interval)


BootstrapStatistic = Callable[..., ArrayLike]


def bootstrap_statistic(
    samples: ArrayLike,
    *,
    statistic: BootstrapStatistic = np.mean,
    resample_count: int,
    seed: int,
    confidence_level: float | None = None,
) -> BootstrapResult:
    """Bootstrap a NumPy-style statistic by resampling the leading axis.

    ``statistic`` is called as ``statistic(array, axis=0)`` and must remove the
    scan axis.  Both ``seed`` and ``resample_count`` are mandatory so analysis
    configurations fully determine the result.
    """

    values = _numeric_array(samples, "samples", minimum_ndim=1)
    observation_count = values.shape[0]
    if observation_count < 2:
        raise DataValidationError("samples must contain at least two observations")
    if not callable(statistic):
        raise DataValidationError("statistic must be callable")
    if isinstance(resample_count, bool) or not isinstance(resample_count, int):
        raise DataValidationError("resample_count must be an integer")
    if resample_count < 2:
        raise DataValidationError("resample_count must be at least two")
    if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)):
        raise DataValidationError("seed must be an integer")
    normalized_seed = int(seed)
    if confidence_level is not None:
        confidence_level = float(confidence_level)
        if not math.isfinite(confidence_level) or not 0 < confidence_level < 1:
            raise DataValidationError("confidence_level must lie strictly between 0 and 1")

    try:
        estimate = np.asarray(statistic(values, axis=0), dtype=np.float64)
    except Exception as exc:
        raise DataValidationError("statistic failed on the original samples") from exc
    if not np.all(np.isfinite(estimate)):
        raise DataValidationError("statistic returned NaN or infinite estimate values")

    rng = np.random.default_rng(normalized_seed)
    distribution = np.empty((resample_count, *estimate.shape), dtype=np.float64)
    for index in range(resample_count):
        selection = rng.integers(0, observation_count, size=observation_count)
        resampled = values[selection]
        try:
            result = np.asarray(statistic(resampled, axis=0), dtype=np.float64)
        except Exception as exc:
            raise DataValidationError(
                f"statistic failed on bootstrap resample {index}"
            ) from exc
        if result.shape != estimate.shape:
            raise DataValidationError(
                "statistic returned an inconsistent shape during bootstrap"
            )
        if not np.all(np.isfinite(result)):
            raise DataValidationError(
                f"statistic returned NaN or infinite values on resample {index}"
            )
        distribution[index] = result

    standard_error = np.std(distribution, axis=0, ddof=1)
    interval = None
    if confidence_level is not None:
        tail = (1.0 - confidence_level) / 2.0
        lower, upper = np.quantile(distribution, [tail, 1.0 - tail], axis=0)
        interval = (lower, upper)
    return BootstrapResult(
        estimate=estimate,
        standard_error=standard_error,
        distribution=distribution,
        seed=normalized_seed,
        resample_count=resample_count,
        confidence_interval=interval,
    )


__all__ = [
    "BootstrapResult",
    "NormalizationUncertainty",
    "ScanRepeatability",
    "bootstrap_statistic",
    "combine_independent_uncertainties",
    "multi_scan_repeatability",
    "poisson_uncertainty",
    "propagate_normalization_uncertainty",
]
