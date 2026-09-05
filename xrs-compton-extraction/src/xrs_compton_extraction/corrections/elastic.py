"""Elastic-line estimation with a Gaussian peak and local linear baseline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import least_squares

FloatArray = NDArray[np.float64]


def gaussian_peak(
    coordinate: ArrayLike, amplitude: float, center: float, sigma: float
) -> FloatArray:
    """Return a Gaussian whose ``amplitude`` is its height, not its area."""

    x = np.asarray(coordinate, dtype=np.float64)
    parameters = np.asarray([amplitude, center, sigma], dtype=np.float64)
    if not np.all(np.isfinite(parameters)):
        raise ValueError("Gaussian parameters must be finite")
    if amplitude < 0 or sigma <= 0:
        raise ValueError("Gaussian amplitude must be non-negative and sigma positive")
    return amplitude * np.exp(-0.5 * np.square((x - center) / sigma))


@dataclass(frozen=True, slots=True)
class ElasticFitResult:
    """Elastic component and local-baseline diagnostics on the full coordinate."""

    elastic_component: FloatArray
    local_baseline: FloatArray
    residual: FloatArray
    fit_mask: NDArray[np.bool_]
    parameters: dict[str, float]
    covariance: FloatArray
    success: bool
    message: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", dict(self.parameters))
        for name in ("elastic_component", "local_baseline", "residual", "fit_mask", "covariance"):
            value = np.array(getattr(self, name), copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)


def fit_elastic_peak(
    coordinate: ArrayLike,
    intensity: ArrayLike,
    *,
    fit_window: tuple[float, float],
    uncertainty: ArrayLike | None = None,
    initial: ArrayLike | None = None,
    loss: str = "soft_l1",
) -> ElasticFitResult:
    """Fit an elastic Gaussian while retaining a separate linear baseline."""

    x = np.asarray(coordinate, dtype=np.float64)
    y = np.asarray(intensity, dtype=np.float64)
    if x.ndim != 1 or y.shape != x.shape or x.size < 6:
        raise ValueError("coordinate and intensity must be equal-length 1-D arrays with >=6 points")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("coordinate and intensity must be finite")
    if np.any(np.diff(x) <= 0):
        raise ValueError("coordinate must be strictly increasing")
    start, stop = map(float, fit_window)
    if not np.isfinite(start) or not np.isfinite(stop) or start >= stop:
        raise ValueError("fit_window must contain finite start < stop")
    mask = (x >= start) & (x <= stop)
    if np.count_nonzero(mask) <= 5:
        raise ValueError("fit_window must select more than five points")

    if uncertainty is None:
        sigma_y = np.ones_like(y)
    else:
        sigma_y = np.broadcast_to(np.asarray(uncertainty, dtype=np.float64), y.shape)
        if not np.all(np.isfinite(sigma_y)) or np.any(sigma_y <= 0):
            raise ValueError("uncertainty must be finite and positive")

    x_fit, y_fit, sigma_fit = x[mask], y[mask], sigma_y[mask]
    origin = float(np.mean(x_fit))
    span = float(np.ptp(x_fit))
    if initial is None:
        offset = float(np.median(np.r_[y_fit[:2], y_fit[-2:]]))
        guess = np.asarray(
            [
                max(float(np.max(y_fit)) - offset, np.finfo(float).eps),
                float(x_fit[np.argmax(y_fit)]),
                max(span / 10.0, np.finfo(float).eps),
                offset,
                0.0,
            ]
        )
    else:
        guess = np.asarray(initial, dtype=np.float64)
        if guess.shape != (5,) or not np.all(np.isfinite(guess)):
            raise ValueError("initial must contain amplitude, center, sigma, offset, slope")

    epsilon = np.finfo(float).eps
    lower = np.asarray([0.0, start, epsilon, -np.inf, -np.inf])
    upper = np.asarray([np.inf, stop, max(span, epsilon), np.inf, np.inf])
    if np.any(guess < lower) or np.any(guess > upper):
        raise ValueError("initial elastic parameters lie outside physical bounds")

    def model(parameters: FloatArray, values: FloatArray) -> FloatArray:
        amplitude, center, peak_sigma, offset, slope = parameters
        return gaussian_peak(values, amplitude, center, peak_sigma) + offset + slope * (
            values - origin
        )

    optimized = least_squares(
        lambda values: (model(values, x_fit) - y_fit) / sigma_fit,
        guess,
        bounds=(lower, upper),
        loss=loss,
        x_scale="jac",
        max_nfev=10_000,
    )
    amplitude, center, peak_sigma, offset, slope = map(float, optimized.x)
    elastic = gaussian_peak(x, amplitude, center, peak_sigma)
    baseline = offset + slope * (x - origin)
    residual = y - elastic - baseline
    degrees_of_freedom = max(x_fit.size - 5, 1)
    chi_square = float(np.sum(np.square(residual[mask] / sigma_fit)))
    covariance = np.linalg.pinv(optimized.jac.T @ optimized.jac) * (
        chi_square / degrees_of_freedom
    )
    return ElasticFitResult(
        elastic_component=elastic,
        local_baseline=baseline,
        residual=residual,
        fit_mask=mask,
        parameters={
            "amplitude": amplitude,
            "center": center,
            "sigma": peak_sigma,
            "offset": offset,
            "slope": slope,
        },
        covariance=covariance,
        success=bool(optimized.success),
        message=str(optimized.message),
    )


__all__ = ["ElasticFitResult", "fit_elastic_peak", "gaussian_peak"]

