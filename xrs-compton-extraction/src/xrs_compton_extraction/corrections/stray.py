"""Constant or independently measured stray-background operations."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class ConstantBackgroundResult:
    level: float
    standard_uncertainty: float
    component: FloatArray
    fit_mask: NDArray[np.bool_]

    def __post_init__(self) -> None:
        for name in ("component", "fit_mask"):
            value = np.array(getattr(self, name), copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)


def estimate_constant_background(
    coordinate: ArrayLike,
    intensity: ArrayLike,
    *,
    fit_windows: Sequence[tuple[float, float]],
    uncertainty: ArrayLike | None = None,
) -> ConstantBackgroundResult:
    """Estimate a constant by a weighted mean over explicit fit windows."""

    x = np.asarray(coordinate, dtype=np.float64)
    y = np.asarray(intensity, dtype=np.float64)
    if x.ndim != 1 or y.shape != x.shape or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("coordinate and intensity must be equal-length finite 1-D arrays")
    if not fit_windows:
        raise ValueError("fit_windows must not be empty")
    mask = np.zeros(x.shape, dtype=bool)
    for start, stop in fit_windows:
        if not np.isfinite(start) or not np.isfinite(stop) or start >= stop:
            raise ValueError("each fit window must contain finite start < stop")
        mask |= (x >= start) & (x <= stop)
    if not np.any(mask):
        raise ValueError("fit_windows do not select any samples")
    if uncertainty is None:
        weights = np.ones(np.count_nonzero(mask), dtype=float)
    else:
        sigma = np.broadcast_to(np.asarray(uncertainty, dtype=float), y.shape)
        if not np.all(np.isfinite(sigma)) or np.any(sigma <= 0):
            raise ValueError("uncertainty must be finite and positive")
        weights = 1.0 / np.square(sigma[mask])
    level = float(np.average(y[mask], weights=weights))
    standard_uncertainty = float(np.sqrt(1.0 / np.sum(weights)))
    return ConstantBackgroundResult(
        level=level,
        standard_uncertainty=standard_uncertainty,
        component=np.full_like(y, level),
        fit_mask=mask,
    )


def subtract_stray_background(
    intensity: ArrayLike, background: ArrayLike, *, scale: float = 1.0
) -> FloatArray:
    """Subtract a supplied background without clipping negative values."""

    observed = np.asarray(intensity, dtype=float)
    measured = np.asarray(background, dtype=float)
    if not np.isfinite(scale) or scale < 0:
        raise ValueError("scale must be finite and non-negative")
    try:
        observed, measured = np.broadcast_arrays(observed, measured)
    except ValueError as exc:
        raise ValueError("intensity and background are not broadcast-compatible") from exc
    if not np.all(np.isfinite(observed)) or not np.all(np.isfinite(measured)):
        raise ValueError("intensity and background must be finite")
    return observed - scale * measured


__all__ = [
    "ConstantBackgroundResult",
    "estimate_constant_background",
    "subtract_stray_background",
]

