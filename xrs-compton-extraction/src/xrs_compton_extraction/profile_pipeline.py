"""Extraction with explicit atomic/empirical Compton-profile templates.

Templates use density per eV. Their common experimental intensity scale and an
optional constant are fitted only in caller-selected background-only windows.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import lsq_linear

from .backgrounds.core_profile import CoreProfileResult
from .backgrounds.valence_profile import ValenceProfileResult, map_valence_profile
from .constants import HARTREE_ENERGY_EV
from .corrections import correct_spectrum
from .data import CorrectionResult, ExtractionResult, XRSSpectrum
from .diagnostics import negative_area_fraction
from .geometry import energy_loss_to_pz


def extract_compton_profile(
    spectrum: XRSSpectrum,
    *,
    q_au: float,
    core_profile: CoreProfileResult,
    valence_profile: ValenceProfileResult,
    fit_windows_ev: Sequence[tuple[float, float]],
    correction: CorrectionResult | None = None,
    correction_options: Mapping[str, Any] | None = None,
    fit_constant: bool = True,
    template_uncertainty_per_ev: ArrayLike | None = None,
) -> ExtractionResult:
    """Fit a shared nonnegative profile amplitude and extract the target edge.

    The core grid must correspond to this spectrum's pz grid. A supplied
    CorrectionResult must have the same raw counts as the spectrum. Correction
    uncertainties and template uncertainties are treated as independent.
    """
    if spectrum.energy_loss_ev is None:
        raise ValueError("spectrum requires energy-loss coordinates")
    if not np.isscalar(q_au) or not np.isfinite(q_au) or q_au <= 0:
        raise ValueError("q_au must be a finite positive channel scalar")
    energy = np.asarray(spectrum.energy_loss_ev)
    if np.any(np.diff(energy) <= 0):
        raise ValueError("energy loss must be strictly increasing")
    pz = np.asarray(energy_loss_to_pz(energy, q_au))
    if core_profile.pz_au.shape != pz.shape or not np.allclose(core_profile.pz_au, pz, rtol=1e-10, atol=1e-10):
        raise ValueError("core profile pz grid does not match the spectrum and q")
    if correction is not None and correction_options is not None:
        raise ValueError("supply correction or correction_options, not both")
    if correction is None:
        correction = correct_spectrum(spectrum, **dict(correction_options or {}))
    if not np.array_equal(correction.raw_counts, spectrum.raw_counts):
        raise ValueError("correction raw counts do not match the spectrum")
    if correction.statistical_uncertainty is None:
        raise ValueError("correction requires statistical uncertainty for weighted fitting")
    sigma = np.asarray(correction.statistical_uncertainty)
    y = np.asarray(correction.corrected_intensity)
    core = core_profile.total_profile / (q_au * HARTREE_ENERGY_EV)
    valence = map_valence_profile(valence_profile, energy, target_q_au=q_au).intensity_per_ev
    template = core + valence
    mask = np.zeros(energy.size, dtype=bool)
    windows = []
    for start, stop in fit_windows_ev:
        if not np.isfinite(start) or not np.isfinite(stop) or start >= stop:
            raise ValueError("fit windows require finite start < stop")
        windows.append((float(start), float(stop)))
        mask |= (energy >= start) & (energy <= stop)
    parameter_count = 2 if fit_constant else 1
    if np.count_nonzero(mask) <= parameter_count:
        raise ValueError("fit windows must select more points than fitted parameters")
    if np.any(sigma[mask] <= 0):
        raise ValueError("selected fit samples require positive statistical uncertainty")
    design = np.column_stack((template, np.ones_like(template))) if fit_constant else template[:, None]
    weighted = design[mask] / sigma[mask, None]
    if np.linalg.matrix_rank(weighted) < parameter_count:
        raise ValueError("profile amplitude and constant are not identifiable in fit windows")
    fit = lsq_linear(weighted, y[mask] / sigma[mask], bounds=(
        [0.0, -np.inf] if fit_constant else [0.0], np.full(parameter_count, np.inf),
    ))
    scale = float(fit.x[0])
    constant = float(fit.x[1]) if fit_constant else 0.0
    background = design @ fit.x
    residual = y - background
    reduced_chi = float(np.sum((residual[mask] / sigma[mask])**2) / (np.count_nonzero(mask) - parameter_count))
    # Input sigma is absolute. Do not force parameter variance to zero for an
    # exactly fitted synthetic curve, or rescale known uncertainties by chi2.
    _, singular, vt = np.linalg.svd(weighted, full_matrices=False)
    covariance = (vt.T / singular**2) @ vt
    model_variance = np.einsum("ij,jk,ik->i", design, covariance, design)
    warnings = list(correction.warnings)
    if template_uncertainty_per_ev is not None:
        template_sigma = np.broadcast_to(np.asarray(template_uncertainty_per_ev, dtype=float), energy.shape)
        if not np.all(np.isfinite(template_sigma)) or np.any(template_sigma < 0):
            raise ValueError("template uncertainty must be finite and non-negative")
        model_variance += (scale * template_sigma)**2
    else:
        warnings.append("Atomic and empirical profile uncertainty was not supplied; model uncertainty covers fitted parameters only.")
    warnings.append("Quality thresholds and experimental validation have not been applied.")
    if np.any(fit.active_mask):
        warnings.append("Amplitude reached its bound; local covariance is approximate.")
    parameters = {"profile_scale": scale}
    if fit_constant:
        parameters["constant"] = constant
    return ExtractionResult(
        energy_loss_eV=energy, raw_counts=spectrum.raw_counts, q_au=q_au,
        normalized_intensity=correction.normalized_intensity, corrected_intensity=y,
        valence_background=scale * valence, core_background=scale * core,
        constant_background=np.full_like(y, constant), extracted_edge=residual,
        fit_residual=residual, statistical_uncertainty=sigma,
        model_uncertainty=np.sqrt(np.maximum(model_variance, 0)),
        background_model_name="compton_profile", fit_parameters=parameters,
        parameter_covariance=covariance, fit_windows=windows,
        quality_grade="Warning" if fit.success else "Reject", warnings=warnings,
        risk_metrics={"reduced_chi_square": reduced_chi, "negative_area_fraction": negative_area_fraction(energy, residual)},
        provenance={
            "core_source": dict(core_profile.source_provenance),
            "core_components": {key: dict(value) for key, value in core_profile.component_metadata.items()},
            "excluded_target": core_profile.excluded_target,
            "core_resolution_sigma_au": core_profile.resolution_sigma_au,
            "valence_profile": dict(valence_profile.provenance),
            "correction": correction.to_dict(),
            "background_coordinate_stage": "after correction and elastic/stray subtraction",
            "profile_mapping": "J(pz) / (q_au * Hartree_eV)",
            "negative_values_clipped": False,
        }, software_version="0.1.0.dev0",
        raw_data_identifiers=tuple(
            str(value) for value in (spectrum.metadata.get("source_file") or spectrum.scan_id,)
            if value
        ),
    )


__all__ = ["extract_compton_profile"]
