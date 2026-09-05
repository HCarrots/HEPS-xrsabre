"""Explicit Beer–Lambert and homogeneous-slab absorption corrections."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def beer_lambert_transmission(
    linear_attenuation_inverse_length: ArrayLike, path_length: ArrayLike
) -> FloatArray:
    """Return ``exp(-mu * path)`` when ``mu`` and path use reciprocal units."""

    mu, path = np.broadcast_arrays(
        np.asarray(linear_attenuation_inverse_length, dtype=float),
        np.asarray(path_length, dtype=float),
    )
    if not np.all(np.isfinite(mu)) or not np.all(np.isfinite(path)):
        raise ValueError("attenuation coefficients and path lengths must be finite")
    if np.any(mu < 0) or np.any(path < 0):
        raise ValueError("attenuation coefficients and path lengths must be non-negative")
    return np.exp(-mu * path)


def slab_self_absorption_factor(
    incident_attenuation_inverse_length: ArrayLike,
    scattered_attenuation_inverse_length: ArrayLike,
    thickness_length: float,
    *,
    incident_angle_from_normal_deg: float = 0.0,
    exit_angle_from_normal_deg: float = 0.0,
    geometry: Literal["transmission", "reflection"] = "transmission",
) -> FloatArray:
    """Return the depth-averaged attenuation of a homogeneous flat slab.

    Angles are explicitly measured from the surface normal. In transmission
    geometry the scattered ray exits the opposite face; in reflection geometry
    it returns through the entrance face. The returned number is an attenuation
    factor in ``(0, 1]``; corrected intensity is obtained by dividing by it.
    """

    if not np.isfinite(thickness_length) or thickness_length <= 0:
        raise ValueError("thickness_length must be finite and positive")
    angles = np.asarray(
        [incident_angle_from_normal_deg, exit_angle_from_normal_deg], dtype=float
    )
    if not np.all(np.isfinite(angles)) or np.any(angles < 0) or np.any(angles >= 90):
        raise ValueError("angles from the normal must be finite and in [0, 90) degrees")
    mu_in, mu_out = np.broadcast_arrays(
        np.asarray(incident_attenuation_inverse_length, dtype=float),
        np.asarray(scattered_attenuation_inverse_length, dtype=float),
    )
    if not np.all(np.isfinite(mu_in)) or not np.all(np.isfinite(mu_out)):
        raise ValueError("attenuation coefficients must be finite")
    if np.any(mu_in < 0) or np.any(mu_out < 0):
        raise ValueError("attenuation coefficients must be non-negative")
    a = mu_in / np.cos(np.deg2rad(incident_angle_from_normal_deg))
    b = mu_out / np.cos(np.deg2rad(exit_angle_from_normal_deg))
    thickness = float(thickness_length)
    if geometry == "reflection":
        optical_depth = (a + b) * thickness
        factor = np.ones_like(optical_depth, dtype=float)
        nonzero = optical_depth != 0
        factor[nonzero] = -np.expm1(-optical_depth[nonzero]) / optical_depth[nonzero]
    elif geometry == "transmission":
        delta = (a - b) * thickness
        close = np.isclose(delta, 0.0, atol=1e-12, rtol=1e-10)
        factor = np.empty_like(delta, dtype=float)
        factor[close] = np.exp(-a[close] * thickness)
        factor[~close] = (
            np.exp(-b[~close] * thickness) - np.exp(-a[~close] * thickness)
        ) / delta[~close]
    else:
        raise ValueError("geometry must be 'transmission' or 'reflection'")
    if np.any(factor <= 0) or not np.all(np.isfinite(factor)):
        raise ValueError("self-absorption factor underflowed; parameters are not numerically usable")
    return factor


def apply_transmission_correction(
    intensity: ArrayLike,
    transmission: ArrayLike,
    *,
    statistical_uncertainty: ArrayLike | None = None,
    transmission_uncertainty: ArrayLike | None = None,
) -> tuple[FloatArray, FloatArray | None]:
    """Divide by transmission and propagate independent first-order errors."""

    signal, factor = np.broadcast_arrays(
        np.asarray(intensity, dtype=float), np.asarray(transmission, dtype=float)
    )
    if not np.all(np.isfinite(signal)) or not np.all(np.isfinite(factor)) or np.any(factor <= 0):
        raise ValueError("intensity must be finite and transmission finite and positive")
    corrected = signal / factor
    if statistical_uncertainty is None and transmission_uncertainty is None:
        return corrected, None
    signal_sigma = (
        np.zeros_like(signal)
        if statistical_uncertainty is None
        else np.broadcast_to(np.asarray(statistical_uncertainty, dtype=float), signal.shape)
    )
    factor_sigma = (
        np.zeros_like(factor)
        if transmission_uncertainty is None
        else np.broadcast_to(np.asarray(transmission_uncertainty, dtype=float), factor.shape)
    )
    if (
        not np.all(np.isfinite(signal_sigma))
        or not np.all(np.isfinite(factor_sigma))
        or np.any(signal_sigma < 0)
        or np.any(factor_sigma < 0)
    ):
        raise ValueError("uncertainties must be finite and non-negative")
    variance = np.square(signal_sigma / factor) + np.square(signal * factor_sigma / factor**2)
    return corrected, np.sqrt(variance)


__all__ = [
    "apply_transmission_correction",
    "beer_lambert_transmission",
    "slab_self_absorption_factor",
]

