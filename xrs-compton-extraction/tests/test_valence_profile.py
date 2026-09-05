from dataclasses import replace

import numpy as np
import pytest

from xrs_compton_extraction.backgrounds.valence_profile import (
    ValenceReferenceCandidate,
    build_valence_profile,
    map_valence_profile,
    select_reference_candidate,
)
from xrs_compton_extraction.constants import HARTREE_ENERGY_EV
from xrs_compton_extraction.geometry import pz_to_energy_loss


def candidate():
    pz = np.linspace(-3, 3, 601)
    profile = np.exp(-pz**2)
    profile *= 4 / np.trapezoid(profile, pz)
    q = 8.0
    return ValenceReferenceCandidate(
        "reference", pz_to_energy_loss(pz, q), profile / (q * HARTREE_ENERGY_EV),
        q, np.zeros(pz.size, dtype=bool), quality_scores={"ia": 0.9},
    )


def build(item, **kwargs):
    return build_valence_profile(
        [item], score_weights={"ia": 1}, valence_electron_count=4,
        normalization_convention="full_symmetric", **kwargs,
    )


def test_jacobian_round_trip_and_transfer_conserve_electron_count():
    original = candidate()
    profile = build(original)
    mapped = map_valence_profile(profile, original.energy_loss_ev, target_q_au=8)
    np.testing.assert_allclose(mapped.intensity_per_ev, original.corrected_intensity, rtol=1e-9)
    assert np.trapezoid(profile.profile, profile.pz_au) == pytest.approx(4)
    transferred = map_valence_profile(profile, pz_to_energy_loss(profile.pz_au, 5), target_q_au=5)
    assert np.trapezoid(transferred.intensity_per_ev, transferred.energy_loss_ev) == pytest.approx(4)
    assert not profile.profile.flags.writeable


def test_reference_selection_is_explicit_and_auditable():
    a = candidate()
    b = replace(a, channel_id="better", quality_scores={"ia": 1.0})
    selection = select_reference_candidate([a, b], score_weights={"ia": 1})
    assert selection.selected_channel_id == "better"
    assert selection.candidate_scores["reference"] == pytest.approx(0.9)


def test_mask_filling_and_gap_rejection():
    original = candidate()
    mask = np.zeros(len(original.energy_loss_ev), dtype=bool)
    mask[290:311] = True
    corrupted = np.array(original.corrected_intensity)
    corrupted[mask] = 100
    item = replace(original, corrected_intensity=corrupted, contamination_mask=mask)
    with pytest.raises(ValueError, match="gaps"):
        build(item)
    result = build(item, masked_region_policy="linear")
    assert result.diagnostics["interpolated_sample_count"] == 21
    assert result.profile.max() < 3
    mask[0] = True
    with pytest.raises(ValueError, match="extrapolation"):
        build(replace(item, contamination_mask=mask), masked_region_policy="linear")


def test_negative_values_remain_and_outside_support_is_rejected():
    original = candidate()
    values = np.array(original.corrected_intensity)
    values[0:20] = -0.001
    values[-20:] = -0.001
    profile = build(replace(original, corrected_intensity=values))
    assert np.min(profile.profile) < 0
    with pytest.raises(ValueError, match="extrapolation"):
        map_valence_profile(profile, pz_to_energy_loss([-4, 0, 4], 5), target_q_au=5)


def test_asymmetry_fit_and_smoothing():
    original = candidate()
    pz = np.linspace(-3, 3, 601)
    asymmetric = replace(original, corrected_intensity=original.corrected_intensity * (1 + 0.05 * pz))
    profile = build(asymmetric, asymmetry_mode="fit_first_order", max_fractional_asymmetry=0.5, gaussian_sigma_pz_au=0.05)
    assert profile.diagnostics["asymmetry_coefficient_per_au"] == pytest.approx(0.05)
    np.testing.assert_allclose(profile.profile, profile.profile[::-1])
