from __future__ import annotations

import numpy as np
import pytest

from xrs_compton_extraction.constants import (
    BOHR_RADIUS_ANGSTROM,
    HARTREE_ENERGY_EV,
    HBAR_C_EV_ANGSTROM,
)
from xrs_compton_extraction.geometry import (
    au_to_inverse_angstrom,
    compton_peak_energy,
    energy_loss,
    energy_loss_to_pz,
    inverse_angstrom_to_au,
    momentum_transfer,
    photon_wavenumber,
    pz_to_energy_loss,
)


def test_energy_loss_scalar_and_broadcast() -> None:
    assert energy_loss(10_000.0, 9_900.0) == pytest.approx(100.0)
    result = energy_loss(np.array([10_000.0, 10_100.0]), 9_900.0)
    np.testing.assert_allclose(result, [100.0, 200.0])


@pytest.mark.parametrize(
    ("function", "args", "message"),
    [
        (energy_loss, (0.0, 1.0), "incident_energy_eV"),
        (energy_loss, (1.0, -1.0), "scattered_energy_eV"),
        (photon_wavenumber, ([1.0, np.nan],), "finite"),
        (momentum_transfer, (10_000.0, 9_900.0, 181.0), r"\[0, 180\]"),
        (energy_loss_to_pz, (10.0, 0.0), "strictly positive"),
        (pz_to_energy_loss, (0.0, -1.0), "strictly positive"),
        (compton_peak_energy, (-1.0,), "non-negative"),
    ],
)
def test_invalid_physical_inputs_raise(function, args, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        function(*args)


def test_non_numeric_input_has_clear_error() -> None:
    with pytest.raises(TypeError, match="real numeric"):
        photon_wavenumber("not-an-energy")


def test_photon_wavenumber_uses_hbar_c() -> None:
    energy = np.array([HBAR_C_EV_ANGSTROM, 2.0 * HBAR_C_EV_ANGSTROM])
    np.testing.assert_allclose(photon_wavenumber(energy), [1.0, 2.0], rtol=1e-14)


def test_momentum_transfer_limiting_geometries() -> None:
    energy = 10_000.0
    k = energy / HBAR_C_EV_ANGSTROM
    assert momentum_transfer(energy, energy, 0.0) == 0.0
    assert momentum_transfer(energy, energy, 180.0) == pytest.approx(2.0 * k)
    assert momentum_transfer(energy, energy, 60.0) == pytest.approx(k)


def test_momentum_transfer_accepts_radians_and_broadcasts() -> None:
    angles_degrees = np.array([0.0, 60.0, 180.0])
    in_degrees = momentum_transfer(10_000.0, 9_500.0, angles_degrees)
    in_radians = momentum_transfer(
        np.full(3, 10_000.0),
        9_500.0,
        np.deg2rad(angles_degrees),
        angle_unit="rad",
    )
    np.testing.assert_allclose(in_degrees, in_radians, rtol=1e-14, atol=1e-14)


def test_incompatible_broadcast_shapes_are_reported() -> None:
    with pytest.raises(ValueError, match="broadcast"):
        momentum_transfer(np.ones(2), np.ones(3), 90.0)


def test_inverse_angstrom_atomic_unit_round_trip() -> None:
    q_inverse_angstrom = np.array([-2.0, 0.0, 1.0, 4.5])
    q_au = inverse_angstrom_to_au(q_inverse_angstrom)
    np.testing.assert_allclose(q_au, q_inverse_angstrom * BOHR_RADIUS_ANGSTROM)
    np.testing.assert_allclose(au_to_inverse_angstrom(q_au), q_inverse_angstrom)


def test_energy_loss_pz_round_trip_and_recoil_zero() -> None:
    q_au = np.array([0.5, 1.0, 2.0, 4.0])
    loss_eV = np.array([-5.0, 0.0, 50.0, 500.0])
    pz = energy_loss_to_pz(loss_eV, q_au)
    np.testing.assert_allclose(pz_to_energy_loss(pz, q_au), loss_eV, rtol=1e-14, atol=1e-14)

    exact_recoil = 0.5 * HARTREE_ENERGY_EV * q_au**2
    np.testing.assert_allclose(energy_loss_to_pz(exact_recoil, q_au), 0.0, atol=1e-15)


def test_compton_peak_uses_required_rounded_coefficient() -> None:
    q_au = np.array([0.0, 1.0, 2.0, 3.5])
    np.testing.assert_array_equal(compton_peak_energy(q_au), 13.6057 * q_au**2)
