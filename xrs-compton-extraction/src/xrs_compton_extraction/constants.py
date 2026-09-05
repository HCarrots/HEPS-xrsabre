"""Physical constants used by :mod:`xrs_compton_extraction`.

The values below use the 2022 CODATA recommended values unless an exact SI
definition is noted.  Keeping the source and units beside each value avoids
untraceable ``magic numbers`` in the numerical routines.
"""

from __future__ import annotations

import math

# Exact SI definitions (BIPM SI Brochure, 9th edition).
ELEMENTARY_CHARGE_C: float = 1.602_176_634e-19
PLANCK_CONSTANT_J_S: float = 6.626_070_15e-34
SPEED_OF_LIGHT_M_S: float = 299_792_458.0

HBAR_J_S: float = PLANCK_CONSTANT_J_S / (2.0 * math.pi)

# 2022 CODATA recommended values.
BOHR_RADIUS_M: float = 5.291_772_105_44e-11
HARTREE_ENERGY_J: float = 4.359_744_722_206e-18

ANGSTROM_M: float = 1.0e-10
BOHR_RADIUS_ANGSTROM: float = BOHR_RADIUS_M / ANGSTROM_M
HARTREE_ENERGY_EV: float = HARTREE_ENERGY_J / ELEMENTARY_CHARGE_C

# Derived from the exact SI constants above.  E = (hbar*c) k.
HBAR_C_EV_ANGSTROM: float = (
    HBAR_J_S * SPEED_OF_LIGHT_M_S / ELEMENTARY_CHARGE_C / ANGSTROM_M
)

# One atomic unit of wavevector is 1/a_0.
AU_WAVEVECTOR_INVERSE_ANGSTROM: float = 1.0 / BOHR_RADIUS_ANGSTROM

# Project requirement / Sternemann-style non-relativistic recoil expression:
# omega_C = 13.6057 * q_au**2 eV.  This rounded coefficient is intentionally
# retained rather than replacing it by HARTREE_ENERGY_EV / 2.
COMPTON_PEAK_COEFFICIENT_EV: float = 13.6057

__all__ = [
    "ANGSTROM_M",
    "AU_WAVEVECTOR_INVERSE_ANGSTROM",
    "BOHR_RADIUS_ANGSTROM",
    "BOHR_RADIUS_M",
    "COMPTON_PEAK_COEFFICIENT_EV",
    "ELEMENTARY_CHARGE_C",
    "HARTREE_ENERGY_EV",
    "HARTREE_ENERGY_J",
    "HBAR_C_EV_ANGSTROM",
    "HBAR_J_S",
    "PLANCK_CONSTANT_J_S",
    "SPEED_OF_LIGHT_M_S",
]
