"""Explicit photon-scattering kinematic-factor corrections."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

FloatArray = NDArray[np.float64]


def klein_nishina_shape(
    incident_energy_ev: ArrayLike,
    scattered_energy_ev: ArrayLike,
    scattering_angle_deg: ArrayLike,
) -> FloatArray:
    """Return the dimensionless unpolarized Klein–Nishina angular shape.

    This is the differential cross section divided by the squared classical
    electron radius. It is exposed as an optional kinematic correction, not as a
    substitute for the full XRS double-differential cross-section model.
    """

    incident, scattered, angle = np.broadcast_arrays(
        np.asarray(incident_energy_ev, dtype=float),
        np.asarray(scattered_energy_ev, dtype=float),
        np.asarray(scattering_angle_deg, dtype=float),
    )
    if not np.all(np.isfinite(incident)) or not np.all(np.isfinite(scattered)):
        raise ValueError("photon energies must be finite")
    if np.any(incident <= 0) or np.any(scattered <= 0):
        raise ValueError("photon energies must be positive")
    if not np.all(np.isfinite(angle)) or np.any(angle < 0) or np.any(angle > 180):
        raise ValueError("scattering_angle_deg must lie within [0, 180]")
    ratio = scattered / incident
    angle_rad = np.deg2rad(angle)
    shape = 0.5 * ratio**2 * (
        ratio + 1.0 / ratio - np.square(np.sin(angle_rad))
    )
    if np.any(shape <= 0) or not np.all(np.isfinite(shape)):
        raise ValueError("calculated cross-section shape is not finite and positive")
    return shape


def relative_cross_section_correction(
    cross_section_shape: ArrayLike, *, reference_shape: float | None = None
) -> FloatArray:
    """Return ``reference / shape`` for explicitly supplied positive values."""

    shape = np.asarray(cross_section_shape, dtype=float)
    if shape.size == 0 or not np.all(np.isfinite(shape)) or np.any(shape <= 0):
        raise ValueError("cross_section_shape must contain finite positive values")
    reference = float(np.max(shape)) if reference_shape is None else float(reference_shape)
    if not np.isfinite(reference) or reference <= 0:
        raise ValueError("reference_shape must be finite and positive")
    return reference / shape


__all__ = ["klein_nishina_shape", "relative_cross_section_correction"]

