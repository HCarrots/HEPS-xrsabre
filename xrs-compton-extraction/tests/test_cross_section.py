from __future__ import annotations

import numpy as np
import pytest

from xrs_compton_extraction.corrections.cross_section import (
    klein_nishina_shape,
    relative_cross_section_correction,
)


def test_klein_nishina_reduces_to_thomson_shape_at_equal_energy() -> None:
    angles = np.asarray([0.0, 90.0, 180.0])
    actual = klein_nishina_shape(10_000.0, 10_000.0, angles)
    expected = 0.5 * (1.0 + np.square(np.cos(np.deg2rad(angles))))
    np.testing.assert_allclose(actual, expected)


def test_relative_cross_section_is_normalized_to_largest_by_default() -> None:
    correction = relative_cross_section_correction([1.0, 2.0, 4.0])
    np.testing.assert_allclose(correction, [4.0, 2.0, 1.0])


def test_cross_section_rejects_nonphysical_energy() -> None:
    with pytest.raises(ValueError, match="positive"):
        klein_nishina_shape(0.0, 1.0, 90.0)

