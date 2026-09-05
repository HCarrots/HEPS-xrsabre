"""Background models for XRS edge extraction."""

from .base import BackgroundModel
from .core_profile import (
    CoreAsymmetry,
    CoreProfileResult,
    ProfileSource,
    TargetShell,
    XraylibProfileSource,
    build_core_profile,
)
from .dabax_profile import DabaxProfileSource
from .pearson import PearsonFitResult, fit_pearson, pearson_background
from .polynomial import PolynomialFitResult, fit_polynomial
from .selection import (
    ModelSelectionFeatures,
    ModelSelectionPolicy,
    ModelSelectionResult,
    select_background_models,
)
from .valence_profile import (
    ValenceProfileResult,
    ValenceReferenceCandidate,
    build_valence_profile,
    map_valence_profile,
    select_reference_candidate,
)

__all__ = [
    "BackgroundModel",
    "CoreAsymmetry",
    "CoreProfileResult",
    "DabaxProfileSource",
    "ModelSelectionFeatures",
    "ModelSelectionPolicy",
    "ModelSelectionResult",
    "PearsonFitResult",
    "PolynomialFitResult",
    "ProfileSource",
    "TargetShell",
    "ValenceProfileResult",
    "ValenceReferenceCandidate",
    "XraylibProfileSource",
    "build_core_profile",
    "build_valence_profile",
    "fit_pearson",
    "fit_polynomial",
    "map_valence_profile",
    "pearson_background",
    "select_background_models",
    "select_reference_candidate",
]
