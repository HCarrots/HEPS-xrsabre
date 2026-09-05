from __future__ import annotations

import pytest

from xrs_compton_extraction.backgrounds.selection import (
    ModelSelectionFeatures,
    ModelSelectionPolicy,
    select_background_models,
)
from xrs_compton_extraction.geometry import compton_peak_energy

POLICY = ModelSelectionPolicy(
    high_risk_overlap_ratio_max=1.0,
    minimum_signal_to_noise=5.0,
    minimum_impulse_approximation_score=0.7,
    minimum_fit_window_fraction=0.2,
)


def test_overlap_uses_edge_peak_distance_normalized_by_width() -> None:
    q = 2.0
    peak = float(compton_peak_energy(q))
    features = ModelSelectionFeatures(q, peak + 10.0, 5.0, 20.0, 0.9, 0.5)
    assert features.overlap_ratio == pytest.approx(2.0)


def test_safe_impulse_channel_selects_compton_profile() -> None:
    features = ModelSelectionFeatures(2.0, 100.0, 10.0, 20.0, 0.9, 0.5)
    result = select_background_models(features, POLICY)
    assert result.primary_model == "compton_profile"
    assert result.candidate_models == ("compton_profile",)
    assert not result.requires_model_ensemble


def test_low_impulse_score_selects_pearson_without_fixed_q_cutoff() -> None:
    features = ModelSelectionFeatures(10.0, 5_000.0, 10.0, 20.0, 0.2, 0.5)
    result = select_background_models(features, POLICY)
    assert result.primary_model == "pearson"


def test_overlap_or_contamination_requires_multiple_models() -> None:
    q = 2.0
    features = ModelSelectionFeatures(
        q,
        float(compton_peak_energy(q)),
        20.0,
        20.0,
        0.9,
        0.5,
        other_edge_contamination=True,
    )
    result = select_background_models(features, POLICY)
    assert result.risk_level == "high"
    assert set(result.candidate_models) == {"pearson", "compton_profile", "polynomial"}
    assert result.requires_model_ensemble

