"""Vectorized photon kinematics and atomic-unit coordinate conversions.

Public energies are expressed in eV, photon wavevectors in inverse angstrom,
and scattering angles in degrees unless ``angle_unit="rad"`` is requested.
The longitudinal electron momentum coordinate ``p_z`` and its associated
momentum transfer are dimensionless atomic-unit values.
"""

from __future__ import annotations

from typing import Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .constants import (
    BOHR_RADIUS_ANGSTROM,
    COMPTON_PEAK_COEFFICIENT_EV,
    HARTREE_ENERGY_EV,
    HBAR_C_EV_ANGSTROM,
)

NumericResult: TypeAlias = float | NDArray[np.float64]


def _as_finite_float_array(value: ArrayLike, name: str) -> NDArray[np.float64]:
    """Convert *value* to a finite float array with a useful error message."""

    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real numeric scalar or array") from exc
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _broadcast(*named_arrays: tuple[str, NDArray[np.float64]]) -> tuple[NDArray[np.float64], ...]:
    """Broadcast named arrays and replace NumPy's terse shape error."""

    try:
        return tuple(np.broadcast_arrays(*(array for _, array in named_arrays)))
    except ValueError as exc:
        names = ", ".join(name for name, _ in named_arrays)
        raise ValueError(f"{names} could not be broadcast to a common shape") from exc


def _result(value: NDArray[np.float64]) -> NumericResult:
    """Return a Python float for scalar input, otherwise an ndarray."""

    if value.ndim == 0:
        return float(value)
    return np.asarray(value, dtype=np.float64)


def energy_loss(incident_energy_eV: ArrayLike, scattered_energy_eV: ArrayLike) -> NumericResult:
    """Return the energy transfer ``incident_energy - scattered_energy`` in eV.

    Photon energies must be positive and may be any mutually broadcastable
    scalar/array combination.  Negative energy transfer itself is permitted so
    that energy-gain spectra are not silently discarded.
    """

    incident = _as_finite_float_array(incident_energy_eV, "incident_energy_eV")
    scattered = _as_finite_float_array(scattered_energy_eV, "scattered_energy_eV")
    incident, scattered = _broadcast(
        ("incident_energy_eV", incident),
        ("scattered_energy_eV", scattered),
    )
    if np.any(incident <= 0.0):
        raise ValueError("incident_energy_eV must be strictly positive")
    if np.any(scattered <= 0.0):
        raise ValueError("scattered_energy_eV must be strictly positive")
    return _result(incident - scattered)


def photon_wavenumber(photon_energy_eV: ArrayLike) -> NumericResult:
    """Convert photon energy in eV to wavevector magnitude in inverse angstrom."""

    energy = _as_finite_float_array(photon_energy_eV, "photon_energy_eV")
    if np.any(energy <= 0.0):
        raise ValueError("photon_energy_eV must be strictly positive")
    return _result(energy / HBAR_C_EV_ANGSTROM)


def momentum_transfer(
    incident_energy_eV: ArrayLike,
    scattered_energy_eV: ArrayLike,
    scattering_angle: ArrayLike,
    *,
    angle_unit: Literal["deg", "rad"] = "deg",
) -> NumericResult:
    r"""Return momentum-transfer magnitude in inverse angstrom.

    The calculation uses

    .. math:: q = \sqrt{k_i^2 + k_f^2 - 2 k_i k_f \cos(\theta)},

    where ``scattering_angle`` is the full angle between the incident and
    scattered photon directions (the symbol in the implementation is not a
    crystallographic half-angle).  The default input unit is degrees.
    """

    incident = _as_finite_float_array(incident_energy_eV, "incident_energy_eV")
    scattered = _as_finite_float_array(scattered_energy_eV, "scattered_energy_eV")
    angle = _as_finite_float_array(scattering_angle, "scattering_angle")
    incident, scattered, angle = _broadcast(
        ("incident_energy_eV", incident),
        ("scattered_energy_eV", scattered),
        ("scattering_angle", angle),
    )

    if np.any(incident <= 0.0):
        raise ValueError("incident_energy_eV must be strictly positive")
    if np.any(scattered <= 0.0):
        raise ValueError("scattered_energy_eV must be strictly positive")

    if angle_unit == "deg":
        if np.any((angle < 0.0) | (angle > 180.0)):
            raise ValueError("scattering_angle must be within [0, 180] degrees")
        angle_rad = np.deg2rad(angle)
    elif angle_unit == "rad":
        if np.any((angle < 0.0) | (angle > np.pi)):
            raise ValueError("scattering_angle must be within [0, pi] radians")
        angle_rad = angle
    else:
        raise ValueError("angle_unit must be either 'deg' or 'rad'")

    # Use the equivalent, cancellation-resistant form.  It gives exactly zero
    # for equal energies in forward scattering, rather than a tiny negative
    # radicand caused by subtraction of nearly equal floating-point values.
    k_i = incident / HBAR_C_EV_ANGSTROM
    k_f = scattered / HBAR_C_EV_ANGSTROM
    q_squared = (k_i - k_f) ** 2 + 4.0 * k_i * k_f * np.sin(angle_rad / 2.0) ** 2
    return _result(np.sqrt(np.maximum(q_squared, 0.0)))


def inverse_angstrom_to_au(q_inverse_angstrom: ArrayLike) -> NumericResult:
    """Convert wavevector from inverse angstrom to atomic units (``a0^-1``)."""

    q = _as_finite_float_array(q_inverse_angstrom, "q_inverse_angstrom")
    return _result(q * BOHR_RADIUS_ANGSTROM)


def au_to_inverse_angstrom(q_au: ArrayLike) -> NumericResult:
    """Convert a wavevector in atomic units to inverse angstrom."""

    q = _as_finite_float_array(q_au, "q_au")
    return _result(q / BOHR_RADIUS_ANGSTROM)


def energy_loss_to_pz(energy_loss_eV: ArrayLike, q_au: ArrayLike) -> NumericResult:
    r"""Map energy loss to longitudinal electron momentum in atomic units.

    ``energy_loss_eV`` is converted to Hartree before applying
    :math:`p_z = \omega/q - q/2`.  Momentum transfer is a magnitude and must be
    strictly positive.
    """

    loss = _as_finite_float_array(energy_loss_eV, "energy_loss_eV")
    q = _as_finite_float_array(q_au, "q_au")
    loss, q = _broadcast(("energy_loss_eV", loss), ("q_au", q))
    if np.any(q <= 0.0):
        raise ValueError("q_au must be strictly positive")
    omega_hartree = loss / HARTREE_ENERGY_EV
    return _result(omega_hartree / q - q / 2.0)


def pz_to_energy_loss(pz_au: ArrayLike, q_au: ArrayLike) -> NumericResult:
    r"""Map longitudinal momentum in atomic units back to energy loss in eV.

    This is the algebraic inverse of :func:`energy_loss_to_pz`:
    :math:`\omega = q(p_z + q/2)`.
    """

    pz = _as_finite_float_array(pz_au, "pz_au")
    q = _as_finite_float_array(q_au, "q_au")
    pz, q = _broadcast(("pz_au", pz), ("q_au", q))
    if np.any(q <= 0.0):
        raise ValueError("q_au must be strictly positive")
    omega_hartree = q * (pz + q / 2.0)
    return _result(omega_hartree * HARTREE_ENERGY_EV)


def compton_peak_energy(q_au: ArrayLike) -> NumericResult:
    """Return the non-relativistic Compton recoil energy in eV.

    The project-specified rounded expression ``13.6057 * q_au**2`` is used
    exactly.  ``q_au`` is a magnitude, so negative inputs are rejected.
    """

    q = _as_finite_float_array(q_au, "q_au")
    if np.any(q < 0.0):
        raise ValueError("q_au must be non-negative")
    return _result(COMPTON_PEAK_COEFFICIENT_EV * q**2)


__all__ = [
    "au_to_inverse_angstrom",
    "compton_peak_energy",
    "energy_loss",
    "energy_loss_to_pz",
    "inverse_angstrom_to_au",
    "momentum_transfer",
    "photon_wavenumber",
    "pz_to_energy_loss",
]
