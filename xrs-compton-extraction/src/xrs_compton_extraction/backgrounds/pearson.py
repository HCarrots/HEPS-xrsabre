"""Bounded, weighted Pearson background fitting for low-q XRS spectra."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares

FloatArray = NDArray[np.float64]
PARAMETER_NAMES = ("beta1", "beta2", "beta3", "beta4")


def pearson_background(
    energy_loss_ev: ArrayLike,
    beta1: float,
    beta2: float,
    beta3: float,
    beta4: float,
) -> FloatArray:
    """Evaluate the four-parameter Pearson background from Sternemann et al."""

    x = np.asarray(energy_loss_ev, dtype=np.float64)
    parameters = np.asarray([beta1, beta2, beta3, beta4], dtype=np.float64)
    if not np.all(np.isfinite(parameters)):
        raise ValueError("Pearson parameters must be finite")
    if beta1 < 0:
        raise ValueError("beta1 must be non-negative")
    if beta3 <= 0 or beta4 <= 0:
        raise ValueError("beta3 and beta4 must be greater than zero")
    with np.errstate(over="raise", invalid="raise"):
        try:
            return beta1 * np.power(1.0 + np.square(beta3 * (x - beta2)), -beta4)
        except FloatingPointError as exc:
            raise ValueError("Pearson evaluation overflowed for the supplied parameters") from exc


@dataclass(frozen=True, slots=True)
class PearsonFitResult:
    """Immutable result returned by :func:`fit_pearson`."""

    parameters: Mapping[str, float]
    covariance: FloatArray
    fitted_background: FloatArray
    residual: FloatArray
    fit_mask: NDArray[np.bool_]
    success: bool
    message: str
    cost: float
    reduced_chi_square: float
    loss: str
    fit_windows_ev: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", dict(self.parameters))
        object.__setattr__(self, "fit_windows_ev", tuple(self.fit_windows_ev))
        for name in ("covariance", "fitted_background", "residual", "fit_mask"):
            value = np.array(getattr(self, name), copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)


def _window_mask(x: FloatArray, windows: Sequence[tuple[float, float]] | None) -> tuple[NDArray[np.bool_], tuple[tuple[float, float], ...]]:
    if windows is None:
        normalized = ((float(np.min(x)), float(np.max(x))),)
    else:
        normalized_items: list[tuple[float, float]] = []
        for start, stop in windows:
            start_f, stop_f = float(start), float(stop)
            if not np.isfinite(start_f) or not np.isfinite(stop_f) or start_f >= stop_f:
                raise ValueError("each fit window must contain finite start < stop")
            normalized_items.append((start_f, stop_f))
        if not normalized_items:
            raise ValueError("at least one fit window is required")
        normalized = tuple(normalized_items)
    mask = np.zeros(x.shape, dtype=bool)
    for start, stop in normalized:
        mask |= (x >= start) & (x <= stop)
    return mask, normalized


def _initial_guess(x: FloatArray, y: FloatArray) -> FloatArray:
    peak_index = int(np.argmax(y))
    span = max(float(np.ptp(x)), np.finfo(float).eps)
    return np.asarray(
        [max(float(y[peak_index]), np.finfo(float).eps), float(x[peak_index]), 2.0 / span, 1.5],
        dtype=np.float64,
    )


def fit_pearson(
    energy_loss_ev: ArrayLike,
    intensity: ArrayLike,
    *,
    sigma: ArrayLike | None = None,
    fit_windows_ev: Sequence[tuple[float, float]] | None = None,
    initial: Mapping[str, float] | Sequence[float] | None = None,
    bounds: tuple[Sequence[float], Sequence[float]] | None = None,
    loss: str = "soft_l1",
    max_nfev: int = 10_000,
) -> PearsonFitResult:
    """Fit a Pearson background with weights, bounds, and a robust loss.

    Covariance is the local linearized estimate based on the optimizer Jacobian.
    With a robust loss it is diagnostic rather than a complete model uncertainty;
    window and model sensitivity are handled separately by later pipeline stages.
    """

    x = np.asarray(energy_loss_ev, dtype=np.float64)
    y = np.asarray(intensity, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape:
        raise ValueError("energy_loss_ev and intensity must be equal-length 1-D arrays")
    if x.size < 5:
        raise ValueError("at least five samples are required for a four-parameter fit")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("fit inputs must contain only finite values")
    if np.any(np.diff(x) <= 0):
        raise ValueError("energy_loss_ev must be strictly increasing")

    if sigma is None:
        sigma_array = np.ones_like(y)
    else:
        try:
            sigma_array = np.broadcast_to(np.asarray(sigma, dtype=np.float64), y.shape)
        except ValueError as exc:
            raise ValueError("sigma is not broadcast-compatible with intensity") from exc
        if not np.all(np.isfinite(sigma_array)) or np.any(sigma_array <= 0):
            raise ValueError("sigma must contain finite values greater than zero")

    fit_mask, normalized_windows = _window_mask(x, fit_windows_ev)
    if np.count_nonzero(fit_mask) <= len(PARAMETER_NAMES):
        raise ValueError("fit windows must select more than four samples")
    x_fit, y_fit, sigma_fit = x[fit_mask], y[fit_mask], sigma_array[fit_mask]

    if initial is None:
        initial_array = _initial_guess(x_fit, y_fit)
    elif isinstance(initial, Mapping):
        missing = set(PARAMETER_NAMES).difference(initial)
        if missing:
            raise ValueError(f"initial is missing parameters: {sorted(missing)}")
        initial_array = np.asarray([initial[name] for name in PARAMETER_NAMES], dtype=np.float64)
    else:
        initial_array = np.asarray(initial, dtype=np.float64)
    if initial_array.shape != (4,) or not np.all(np.isfinite(initial_array)):
        raise ValueError("initial must contain four finite parameter values")

    if bounds is None:
        epsilon = np.finfo(float).eps
        lower = np.asarray([0.0, float(np.min(x)), epsilon, epsilon])
        upper = np.asarray([np.inf, float(np.max(x)), np.inf, 50.0])
    else:
        lower, upper = (np.asarray(item, dtype=np.float64) for item in bounds)
        if lower.shape != (4,) or upper.shape != (4,) or np.any(lower >= upper):
            raise ValueError("bounds must be two length-four sequences with lower < upper")
    if np.any(initial_array < lower) or np.any(initial_array > upper):
        raise ValueError("initial parameters must lie within bounds")

    def weighted_residual(parameters: FloatArray) -> FloatArray:
        return (
            pearson_background(x_fit, *parameters) - y_fit
        ) / sigma_fit

    optimized = least_squares(
        weighted_residual,
        initial_array,
        bounds=(lower, upper),
        loss=loss,
        max_nfev=max_nfev,
        x_scale="jac",
    )
    fitted = pearson_background(x, *optimized.x)
    residual = y - fitted
    degrees_of_freedom = max(x_fit.size - len(PARAMETER_NAMES), 1)
    weighted_linear_residual = (fitted[fit_mask] - y_fit) / sigma_fit
    chi_square = float(np.dot(weighted_linear_residual, weighted_linear_residual))
    reduced_chi_square = chi_square / degrees_of_freedom

    jacobian = np.asarray(optimized.jac, dtype=np.float64)
    covariance = np.linalg.pinv(jacobian.T @ jacobian)
    covariance *= reduced_chi_square

    return PearsonFitResult(
        parameters=dict(zip(PARAMETER_NAMES, map(float, optimized.x), strict=True)),
        covariance=covariance,
        fitted_background=fitted,
        residual=residual,
        fit_mask=fit_mask,
        success=bool(optimized.success),
        message=str(optimized.message),
        cost=float(optimized.cost),
        reduced_chi_square=reduced_chi_square,
        loss=loss,
        fit_windows_ev=normalized_windows,
    )

