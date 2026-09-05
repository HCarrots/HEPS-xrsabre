"""Condition-aware weighted polynomial background fitting."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class PolynomialFitResult:
    """Polynomial fit in a normalized coordinate ``z=(x-origin)/scale``."""

    coefficients: FloatArray
    coordinate_origin_ev: float
    coordinate_scale_ev: float
    covariance: FloatArray
    fitted_background: FloatArray
    residual: FloatArray
    fit_mask: NDArray[np.bool_]
    reduced_chi_square: float
    condition_number: float
    success: bool
    message: str
    fit_windows_ev: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        for name in (
            "coefficients",
            "covariance",
            "fitted_background",
            "residual",
            "fit_mask",
        ):
            value = np.array(getattr(self, name), copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        object.__setattr__(self, "fit_windows_ev", tuple(self.fit_windows_ev))

    @property
    def parameters(self) -> dict[str, float]:
        return {
            f"coefficient_{index}": float(value)
            for index, value in enumerate(self.coefficients)
        }

    def evaluate(self, energy_loss_ev: ArrayLike) -> FloatArray:
        x = np.asarray(energy_loss_ev, dtype=float)
        z = (x - self.coordinate_origin_ev) / self.coordinate_scale_ev
        return np.polynomial.polynomial.polyval(z, self.coefficients)


def fit_polynomial(
    energy_loss_ev: ArrayLike,
    intensity: ArrayLike,
    *,
    degree: int,
    fit_windows_ev: Sequence[tuple[float, float]],
    sigma: ArrayLike | None = None,
    maximum_degree: int = 5,
    max_condition_number: float = 1.0e12,
) -> PolynomialFitResult:
    """Fit a low-order polynomial over explicit background-only windows."""

    if isinstance(degree, bool) or not isinstance(degree, (int, np.integer)):
        raise TypeError("degree must be an integer")
    degree = int(degree)
    if degree < 0 or degree > maximum_degree:
        raise ValueError(f"degree must lie within [0, {maximum_degree}]")
    if not np.isfinite(max_condition_number) or max_condition_number <= 1:
        raise ValueError("max_condition_number must be finite and greater than one")
    x = np.asarray(energy_loss_ev, dtype=float)
    y = np.asarray(intensity, dtype=float)
    if x.ndim != 1 or y.shape != x.shape or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("energy_loss_ev and intensity must be equal-length finite 1-D arrays")
    if np.any(np.diff(x) <= 0):
        raise ValueError("energy_loss_ev must be strictly increasing")
    if not fit_windows_ev:
        raise ValueError("fit_windows_ev must not be empty")
    mask = np.zeros(x.shape, dtype=bool)
    normalized_windows: list[tuple[float, float]] = []
    for start, stop in fit_windows_ev:
        start, stop = float(start), float(stop)
        if not np.isfinite(start) or not np.isfinite(stop) or start >= stop:
            raise ValueError("each fit window must contain finite start < stop")
        mask |= (x >= start) & (x <= stop)
        normalized_windows.append((start, stop))
    selected_count = int(np.count_nonzero(mask))
    if selected_count <= degree + 1:
        raise ValueError("fit windows must select more samples than fitted coefficients")

    if sigma is None:
        sigma_array = np.ones_like(y)
    else:
        sigma_array = np.broadcast_to(np.asarray(sigma, dtype=float), y.shape)
        if not np.all(np.isfinite(sigma_array)) or np.any(sigma_array <= 0):
            raise ValueError("sigma must be finite and positive")
    x_fit = x[mask]
    origin = float(np.mean(x_fit))
    scale = float(np.ptp(x_fit) / 2.0)
    if scale == 0:
        raise ValueError("selected energy coordinates do not span a finite interval")
    z = (x - origin) / scale
    design = np.polynomial.polynomial.polyvander(z[mask], degree)
    weighted_design = design / sigma_array[mask, None]
    weighted_y = y[mask] / sigma_array[mask]
    coefficients, _, rank, singular_values = np.linalg.lstsq(
        weighted_design, weighted_y, rcond=None
    )
    if rank != degree + 1:
        raise ValueError("polynomial design matrix is rank deficient")
    condition_number = float(singular_values[0] / singular_values[-1])
    fitted = np.polynomial.polynomial.polyval(z, coefficients)
    residual = y - fitted
    degrees_of_freedom = max(selected_count - degree - 1, 1)
    reduced_chi_square = float(
        np.sum(np.square(residual[mask] / sigma_array[mask])) / degrees_of_freedom
    )
    covariance = np.linalg.pinv(weighted_design.T @ weighted_design)
    covariance *= reduced_chi_square
    success = condition_number <= max_condition_number
    message = (
        "fit converged"
        if success
        else f"condition number {condition_number:.6g} exceeds {max_condition_number:.6g}"
    )
    return PolynomialFitResult(
        coefficients=coefficients,
        coordinate_origin_ev=origin,
        coordinate_scale_ev=scale,
        covariance=covariance,
        fitted_background=fitted,
        residual=residual,
        fit_mask=mask,
        reduced_chi_square=reduced_chi_square,
        condition_number=condition_number,
        success=success,
        message=message,
        fit_windows_ev=tuple(normalized_windows),
    )


__all__ = ["PolynomialFitResult", "fit_polynomial"]

