"""Composable experimental-intensity corrections."""

from .absorption import (
    apply_transmission_correction,
    beer_lambert_transmission,
    slab_self_absorption_factor,
)
from .cross_section import klein_nishina_shape, relative_cross_section_correction
from .elastic import ElasticFitResult, fit_elastic_peak, gaussian_peak
from .normalization import NormalizationResult, normalize_counts
from .pipeline import correct_spectrum
from .stray import (
    ConstantBackgroundResult,
    estimate_constant_background,
    subtract_stray_background,
)

__all__ = [
    "ConstantBackgroundResult",
    "ElasticFitResult",
    "NormalizationResult",
    "apply_transmission_correction",
    "beer_lambert_transmission",
    "correct_spectrum",
    "estimate_constant_background",
    "fit_elastic_peak",
    "gaussian_peak",
    "klein_nishina_shape",
    "normalize_counts",
    "relative_cross_section_correction",
    "slab_self_absorption_factor",
    "subtract_stray_background",
]
