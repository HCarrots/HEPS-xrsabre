"""Explainable, element-independent background-model selection features."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ..geometry import compton_peak_energy

ModelName = Literal["pearson", "compton_profile", "polynomial"]


@dataclass(frozen=True, slots=True)
class ModelSelectionFeatures:
    q_au: float
    target_edge_energy_ev: float
    valence_width_ev: float
    signal_to_noise: float
    impulse_approximation_score: float
    available_fit_window_fraction: float
    other_edge_contamination: bool = False

    def __post_init__(self) -> None:
        numeric = {
            "q_au": self.q_au,
            "target_edge_energy_ev": self.target_edge_energy_ev,
            "valence_width_ev": self.valence_width_ev,
            "signal_to_noise": self.signal_to_noise,
            "impulse_approximation_score": self.impulse_approximation_score,
            "available_fit_window_fraction": self.available_fit_window_fraction,
        }
        if not all(np.isfinite(value) for value in numeric.values()):
            raise ValueError("all model-selection features must be finite")
        if self.q_au <= 0 or self.target_edge_energy_ev <= 0 or self.valence_width_ev <= 0:
            raise ValueError("q, target-edge energy, and valence width must be positive")
        if self.signal_to_noise < 0:
            raise ValueError("signal_to_noise must be non-negative")
        if not 0 <= self.impulse_approximation_score <= 1:
            raise ValueError("impulse_approximation_score must lie within [0, 1]")
        if not 0 <= self.available_fit_window_fraction <= 1:
            raise ValueError("available_fit_window_fraction must lie within [0, 1]")

    @property
    def compton_peak_energy_ev(self) -> float:
        return float(compton_peak_energy(self.q_au))

    @property
    def overlap_ratio(self) -> float:
        """Return edge/Compton separation divided by valence-profile width."""

        return abs(self.target_edge_energy_ev - self.compton_peak_energy_ev) / self.valence_width_ev


@dataclass(frozen=True, slots=True)
class ModelSelectionPolicy:
    """Caller-approved thresholds; none are element-independent q cutoffs."""

    high_risk_overlap_ratio_max: float
    minimum_signal_to_noise: float
    minimum_impulse_approximation_score: float
    minimum_fit_window_fraction: float

    def __post_init__(self) -> None:
        values = (
            self.high_risk_overlap_ratio_max,
            self.minimum_signal_to_noise,
            self.minimum_impulse_approximation_score,
            self.minimum_fit_window_fraction,
        )
        if not all(np.isfinite(value) for value in values) or any(value < 0 for value in values):
            raise ValueError("model-selection thresholds must be finite and non-negative")
        if self.minimum_impulse_approximation_score > 1:
            raise ValueError("minimum_impulse_approximation_score must not exceed one")
        if self.minimum_fit_window_fraction > 1:
            raise ValueError("minimum_fit_window_fraction must not exceed one")


@dataclass(frozen=True, slots=True)
class ModelSelectionResult:
    primary_model: ModelName
    candidate_models: tuple[ModelName, ...]
    risk_level: Literal["low", "high"]
    overlap_ratio: float
    compton_peak_energy_ev: float
    reasons: tuple[str, ...]

    @property
    def requires_model_ensemble(self) -> bool:
        return len(self.candidate_models) > 1


def select_background_models(
    features: ModelSelectionFeatures, policy: ModelSelectionPolicy
) -> ModelSelectionResult:
    """Choose candidates using overlap and data quality, never a fixed q threshold."""

    if not isinstance(features, ModelSelectionFeatures) or not isinstance(
        policy, ModelSelectionPolicy
    ):
        raise TypeError("features and policy must use their typed model-selection objects")
    reasons: list[str] = []
    high_risk = False
    if features.overlap_ratio <= policy.high_risk_overlap_ratio_max:
        high_risk = True
        reasons.append("target edge overlaps the valence Compton region")
    if features.signal_to_noise < policy.minimum_signal_to_noise:
        high_risk = True
        reasons.append("signal-to-noise is below the approved threshold")
    if features.available_fit_window_fraction < policy.minimum_fit_window_fraction:
        high_risk = True
        reasons.append("available background-only fit range is too limited")
    if features.other_edge_contamination:
        high_risk = True
        reasons.append("another absorption edge contaminates the candidate fit region")

    impulse_applicable = (
        features.impulse_approximation_score
        >= policy.minimum_impulse_approximation_score
    )
    primary: ModelName = "compton_profile" if impulse_applicable else "pearson"
    if not impulse_applicable:
        reasons.append("impulse approximation score favors a low-q Pearson model")

    if high_risk:
        candidates: tuple[ModelName, ...] = tuple(
            dict.fromkeys((primary, "pearson", "compton_profile", "polynomial"))
        )
        reasons.append("high-risk channel requires a multi-model comparison")
        risk_level: Literal["low", "high"] = "high"
    else:
        candidates = (primary,)
        reasons.append(f"data-quality features support {primary}")
        risk_level = "low"
    return ModelSelectionResult(
        primary_model=primary,
        candidate_models=candidates,
        risk_level=risk_level,
        overlap_ratio=features.overlap_ratio,
        compton_peak_energy_ev=features.compton_peak_energy_ev,
        reasons=tuple(reasons),
    )


__all__ = [
    "ModelSelectionFeatures",
    "ModelSelectionPolicy",
    "ModelSelectionResult",
    "select_background_models",
]

