from __future__ import annotations

import pytest

from xrs_compton_extraction.q_groups import classify_q_band, group_q_channels


def test_q_groups_use_strict_nine_inverse_angstrom_cutoff() -> None:
    assert classify_q_band(8.999) == "low_q"
    assert classify_q_band(9.0) == "boundary"
    assert classify_q_band(9.001) == "mid_high_q"


def test_q_groups_are_deterministic_and_validate_inputs() -> None:
    assert group_q_channels(
        ("B", "A", "C"), {"A": 8.0, "B": 10.0, "C": 9.0}
    ) == {"low_q": ("A",), "mid_high_q": ("B",), "boundary": ("C",)}
    with pytest.raises(ValueError, match="missing"):
        group_q_channels(("A",), {})
    with pytest.raises(ValueError, match="finite"):
        classify_q_band(float("nan"))
