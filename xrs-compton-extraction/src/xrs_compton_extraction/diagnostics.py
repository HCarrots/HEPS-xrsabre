"""Quality-control metrics for extracted XRS spectra.

This module intentionally defines metric calculations separately from grading.
No scientific thresholds are embedded here: callers must supply every threshold
used to produce a :class:`~xrs_compton_extraction.data.QualityReport`.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .data import QualityReport
from .exceptions import DataValidationError

FloatArray = NDArray[np.float64]


def _vector(
    values: ArrayLike,
    name: str,
    *,
    minimum_length: int = 1,
    finite: bool = True,
) -> FloatArray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise DataValidationError(f"{name} must be a numeric one-dimensional array") from exc
    if array.ndim != 1:
        raise DataValidationError(f"{name} must be one-dimensional")
    if len(array) < minimum_length:
        raise DataValidationError(
            f"{name} must contain at least {minimum_length} value(s)"
        )
    if finite and not np.all(np.isfinite(array)):
        raise DataValidationError(f"{name} contains NaN or infinite values")
    return array


def _matching_vectors(
    first: ArrayLike,
    second: ArrayLike,
    first_name: str,
    second_name: str,
    *,
    minimum_length: int = 1,
) -> tuple[FloatArray, FloatArray]:
    first_array = _vector(first, first_name, minimum_length=minimum_length)
    second_array = _vector(second, second_name, minimum_length=minimum_length)
    if first_array.shape != second_array.shape:
        raise DataValidationError(
            f"{second_name} shape {second_array.shape} does not match "
            f"{first_name} shape {first_array.shape}"
        )
    return first_array, second_array


def _strict_coordinate(values: ArrayLike, name: str, minimum_length: int) -> FloatArray:
    coordinate = _vector(values, name, minimum_length=minimum_length)
    differences = np.diff(coordinate)
    if not (np.all(differences > 0) or np.all(differences < 0)):
        raise DataValidationError(f"{name} must be strictly monotonic")
    return coordinate


def pre_edge_residual_mean(residual: ArrayLike) -> float:
    """Arithmetic mean of residual samples selected by the caller as pre-edge."""

    values = _vector(residual, "residual")
    return float(np.mean(values))


def pre_edge_residual_std(residual: ArrayLike, *, ddof: int = 1) -> float:
    """Standard deviation of caller-selected pre-edge residuals."""

    if isinstance(ddof, bool) or not isinstance(ddof, int) or ddof < 0:
        raise DataValidationError("ddof must be a non-negative integer")
    values = _vector(residual, "residual", minimum_length=ddof + 1)
    return float(np.std(values, ddof=ddof))


def reduced_chi_square(
    residual: ArrayLike,
    uncertainty: ArrayLike,
    *,
    fitted_parameter_count: int,
) -> float:
    """Return ``sum((residual / sigma)**2) / (N - parameter_count)``."""

    values, sigma = _matching_vectors(
        residual,
        uncertainty,
        "residual",
        "uncertainty",
    )
    if np.any(sigma <= 0):
        raise DataValidationError("uncertainty must contain only positive values")
    if isinstance(fitted_parameter_count, bool) or not isinstance(
        fitted_parameter_count, int
    ):
        raise DataValidationError("fitted_parameter_count must be an integer")
    degrees_of_freedom = len(values) - fitted_parameter_count
    if fitted_parameter_count < 0 or degrees_of_freedom <= 0:
        raise DataValidationError(
            "fitted_parameter_count must be non-negative and below the sample count"
        )
    return float(np.sum(np.square(values / sigma)) / degrees_of_freedom)


def residual_curvature_rms(coordinate: ArrayLike, residual: ArrayLike) -> float:
    """RMS numerical second derivative of a residual curve.

    This is an absolute, unit-carrying diagnostic.  Its threshold therefore has
    to be selected for the coordinate and intensity units of the analysis.
    """

    x = _strict_coordinate(coordinate, "coordinate", minimum_length=3)
    _, values = _matching_vectors(x, residual, "coordinate", "residual", minimum_length=3)
    first_derivative = np.gradient(values, x)
    second_derivative = np.gradient(first_derivative, x)
    return float(np.sqrt(np.mean(np.square(second_derivative))))


def negative_area_fraction(coordinate: ArrayLike, intensity: ArrayLike) -> float:
    """Integrated negative magnitude divided by integrated absolute magnitude.

    The result lies in ``[0, 1]``.  A zero curve has a fraction of zero.
    """

    x = _strict_coordinate(coordinate, "coordinate", minimum_length=2)
    _, y = _matching_vectors(x, intensity, "coordinate", "intensity", minimum_length=2)
    if x[0] > x[-1]:
        x = x[::-1]
        y = y[::-1]
    negative_area = float(np.trapezoid(np.clip(-y, 0.0, None), x))
    absolute_area = float(np.trapezoid(np.abs(y), x))
    return 0.0 if absolute_area == 0 else negative_area / absolute_area


def fit_window_sensitivity(window_results: ArrayLike, *, ddof: int = 1) -> float:
    """Sample standard deviation of scalar results from alternate fit windows."""

    return pre_edge_residual_std(window_results, ddof=ddof)


def background_model_difference(model_backgrounds: ArrayLike) -> float:
    """Maximum pairwise RMS difference among background-model curves."""

    try:
        models = np.asarray(model_backgrounds, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise DataValidationError("model_backgrounds must be a numeric 2-D array") from exc
    if models.ndim != 2 or models.shape[0] < 2 or models.shape[1] == 0:
        raise DataValidationError(
            "model_backgrounds must contain at least two non-empty model curves"
        )
    if not np.all(np.isfinite(models)):
        raise DataValidationError("model_backgrounds contains NaN or infinite values")
    maximum = 0.0
    for left in range(models.shape[0] - 1):
        differences = models[left + 1 :] - models[left]
        pairwise_rms = np.sqrt(np.mean(np.square(differences), axis=1))
        maximum = max(maximum, float(np.max(pairwise_rms)))
    return maximum


def adjacent_q_continuity(current: ArrayLike, adjacent: ArrayLike) -> float:
    """RMS point-wise difference between aligned neighboring-q spectra."""

    current_values, adjacent_values = _matching_vectors(
        current,
        adjacent,
        "current",
        "adjacent",
    )
    return float(np.sqrt(np.mean(np.square(current_values - adjacent_values))))


def target_edge_integral(
    energy_loss_ev: ArrayLike,
    extracted_edge: ArrayLike,
    *,
    integration_window: tuple[float, float],
) -> float:
    """Integrate the target edge over an explicit inclusive energy window."""

    x = _strict_coordinate(energy_loss_ev, "energy_loss_ev", minimum_length=2)
    _, y = _matching_vectors(
        x,
        extracted_edge,
        "energy_loss_ev",
        "extracted_edge",
        minimum_length=2,
    )
    if len(integration_window) != 2:
        raise DataValidationError("integration_window must contain two bounds")
    lower = float(integration_window[0])
    upper = float(integration_window[1])
    if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
        raise DataValidationError(
            "integration_window bounds must be finite and strictly increasing"
        )
    selected = (x >= lower) & (x <= upper)
    if np.count_nonzero(selected) < 2:
        raise DataValidationError(
            "integration_window must contain at least two measured points"
        )
    selected_x = x[selected]
    selected_y = y[selected]
    if selected_x[0] > selected_x[-1]:
        selected_x = selected_x[::-1]
        selected_y = selected_y[::-1]
    return float(np.trapezoid(selected_y, selected_x))


def nan_inf_count(*arrays: ArrayLike) -> int:
    """Count all NaN and infinite entries across the supplied arrays."""

    if not arrays:
        raise DataValidationError("at least one array is required")
    count = 0
    for index, values in enumerate(arrays):
        try:
            array = np.asarray(values, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise DataValidationError(f"arrays[{index}] must be numeric") from exc
        if array.size == 0:
            raise DataValidationError(f"arrays[{index}] must not be empty")
        count += int(np.count_nonzero(~np.isfinite(array)))
    return count


def compute_quality_metrics(
    *,
    pre_edge_energy_loss_ev: ArrayLike,
    pre_edge_residual: ArrayLike,
    pre_edge_uncertainty: ArrayLike,
    fitted_parameter_count: int,
    energy_loss_ev: ArrayLike,
    extracted_edge: ArrayLike,
    target_edge_window: tuple[float, float],
    fit_window_integrals: ArrayLike,
    model_backgrounds: ArrayLike,
    adjacent_q_edge: ArrayLike,
    arrays_to_check: Sequence[ArrayLike] = (),
) -> dict[str, float]:
    """Compute the ten minimum channel-QC metrics as an ungraded mapping.

    Inputs used by numerical metrics must be finite.  Additional raw/intermediate
    arrays containing non-finite values may be passed through ``arrays_to_check``
    so those values are counted without being silently repaired.
    """

    checked_arrays: tuple[ArrayLike, ...] = (
        pre_edge_energy_loss_ev,
        pre_edge_residual,
        pre_edge_uncertainty,
        energy_loss_ev,
        extracted_edge,
        model_backgrounds,
        adjacent_q_edge,
        *arrays_to_check,
    )
    return {
        "pre_edge_residual_mean": pre_edge_residual_mean(pre_edge_residual),
        "pre_edge_residual_std": pre_edge_residual_std(pre_edge_residual),
        "reduced_chi_square": reduced_chi_square(
            pre_edge_residual,
            pre_edge_uncertainty,
            fitted_parameter_count=fitted_parameter_count,
        ),
        "residual_curvature_rms": residual_curvature_rms(
            pre_edge_energy_loss_ev, pre_edge_residual
        ),
        "negative_area_fraction": negative_area_fraction(
            energy_loss_ev, extracted_edge
        ),
        "fit_window_sensitivity": fit_window_sensitivity(fit_window_integrals),
        "background_model_difference": background_model_difference(model_backgrounds),
        "adjacent_q_continuity": adjacent_q_continuity(
            extracted_edge, adjacent_q_edge
        ),
        "target_edge_integral": target_edge_integral(
            energy_loss_ev,
            extracted_edge,
            integration_window=target_edge_window,
        ),
        "nan_inf_count": float(nan_inf_count(*checked_arrays)),
    }


@dataclass(frozen=True, slots=True)
class QualityThreshold:
    """Explicit two-level decision rule for one named metric."""

    warning: float
    reject: float
    direction: Literal["max", "min"]
    absolute: bool = False

    def __post_init__(self) -> None:
        for name in ("warning", "reject"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0:
                raise DataValidationError(f"{name} threshold must be finite and non-negative")
            object.__setattr__(self, name, value)
        if self.direction not in {"max", "min"}:
            raise DataValidationError("direction must be 'max' or 'min'")
        if not isinstance(self.absolute, bool):
            raise DataValidationError("absolute must be a bool")
        if self.direction == "max" and self.reject < self.warning:
            raise DataValidationError(
                "a max-direction reject threshold must not be below warning"
            )
        if self.direction == "min" and self.reject > self.warning:
            raise DataValidationError(
                "a min-direction reject threshold must not exceed warning"
            )

    def classify(self, value: float) -> Literal["Pass", "Warning", "Reject"]:
        tested = abs(value) if self.absolute else value
        if self.direction == "max":
            if tested >= self.reject:
                return "Reject"
            if tested >= self.warning:
                return "Warning"
        else:
            if tested <= self.reject:
                return "Reject"
            if tested <= self.warning:
                return "Warning"
        return "Pass"


def build_quality_report(
    metrics: Mapping[str, float],
    thresholds: Mapping[str, QualityThreshold],
    *,
    recommended_actions: Iterable[str] = (),
    anomalous_indices: Iterable[int] = (),
    metadata: Mapping[str, Any] | None = None,
) -> QualityReport:
    """Grade metrics using only caller-supplied thresholds.

    There are deliberately no default thresholds.  The threshold mapping must
    contain exactly the same names as the metric mapping, which prevents a new
    metric from being silently omitted from quality grading.
    """

    if not metrics:
        raise DataValidationError("metrics must not be empty")
    if not thresholds:
        raise DataValidationError("thresholds must be supplied explicitly")
    metric_names = set(metrics)
    threshold_names = set(thresholds)
    if metric_names != threshold_names:
        missing = sorted(metric_names - threshold_names)
        unknown = sorted(threshold_names - metric_names)
        raise DataValidationError(
            f"threshold keys must exactly match metric keys; missing={missing}, unknown={unknown}"
        )

    normalized_metrics: dict[str, float] = {}
    flattened_thresholds: dict[str, float] = {}
    reasons: list[str] = []
    overall_rank = 0
    rank = {"Pass": 0, "Warning": 1, "Reject": 2}
    for name, raw_value in metrics.items():
        if not isinstance(name, str) or not name:
            raise DataValidationError("metric names must be non-empty strings")
        value = float(raw_value)
        if not math.isfinite(value):
            raise DataValidationError(f"metric {name!r} must be finite")
        threshold = thresholds[name]
        if not isinstance(threshold, QualityThreshold):
            raise DataValidationError(
                f"thresholds[{name!r}] must be a QualityThreshold"
            )
        grade = threshold.classify(value)
        normalized_metrics[name] = value
        flattened_thresholds[f"{name}.warning"] = threshold.warning
        flattened_thresholds[f"{name}.reject"] = threshold.reject
        overall_rank = max(overall_rank, rank[grade])
        if grade != "Pass":
            transformed = abs(value) if threshold.absolute else value
            reasons.append(
                f"{name}={transformed:g} triggered {grade} "
                f"({threshold.direction}, warning={threshold.warning:g}, "
                f"reject={threshold.reject:g})"
            )

    grade_for_rank = {0: "Pass", 1: "Warning", 2: "Reject"}
    return QualityReport(
        grade=grade_for_rank[overall_rank],
        metrics=normalized_metrics,
        thresholds=flattened_thresholds,
        reasons=reasons or ("all configured quality thresholds passed",),
        anomalous_indices=tuple(anomalous_indices),
        recommended_actions=tuple(recommended_actions),
        metadata={} if metadata is None else metadata,
    )


__all__ = [
    "QualityThreshold",
    "adjacent_q_continuity",
    "background_model_difference",
    "build_quality_report",
    "compute_quality_metrics",
    "fit_window_sensitivity",
    "nan_inf_count",
    "negative_area_fraction",
    "pre_edge_residual_mean",
    "pre_edge_residual_std",
    "reduced_chi_square",
    "residual_curvature_rms",
    "target_edge_integral",
]
