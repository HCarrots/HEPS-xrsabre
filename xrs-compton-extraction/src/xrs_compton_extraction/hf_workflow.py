"""Explicit, unweighted HF matching and target-preserving extraction.

Inspired by the locally inspected XRStools workflow, independently implemented.
Fixed-q impulse approximation and finite-table f-sum normalization are explicit
model assumptions. This module neither imports XRStools nor invents errors.
"""

from dataclasses import dataclass

import numpy as np
from scipy.optimize import lsq_linear

from .constants import HARTREE_ENERGY_EV as EH
from .exploratory import fit_mask
from .geometry import energy_loss_to_pz


@dataclass
class HFEnergyProfiles:
    energy_ev: np.ndarray
    total: np.ndarray
    core: np.ndarray
    valence: np.ndarray
    target: np.ndarray
    other_core: np.ndarray
    components: dict
    metadata: dict


def _grid(energy):
    energy = np.asarray(energy, dtype=float)
    if (energy.ndim != 1 or energy.size < 3 or not np.isfinite(energy).all()
            or np.any(np.diff(energy) <= 0)):
        raise ValueError("energy must be a finite increasing grid with at least 3 points")
    return energy


def build_hf_energy(energy_ev, q_au, source, composition, *, target,
                    valence_cutoff_ev, integration_points=16001):
    """Threshold each shell and enforce its first moment on full table support.

    For constant q, integral E*S(E)dE = N*EH*q**2/2 (S per eV).
    Integrate from each binding threshold to the table's positive pz endpoint,
    NOT to the experimental scan endpoint. No extrapolated tail is assumed.
    Shells whose binding threshold exceeds table support are recorded as absent.
    """
    energy = _grid(energy_ev)
    if not np.isscalar(q_au) or not np.isfinite(q_au) or q_au <= 0:
        raise ValueError("q must be a finite positive scalar")
    if not np.isfinite(valence_cutoff_ev) or valence_cutoff_ev < 0:
        raise ValueError("explicit valence cutoff must be finite and nonnegative")
    if not isinstance(integration_points, int) or integration_points < 101:
        raise ValueError("integration_points must be an integer >= 101")
    if not composition or len(target) != 2:
        raise ValueError("explicit composition and target (element, Shell_N) required")
    target_id = f"{source.resolve_element(target[0]).symbol}:{source.shell_label(target[1])}"
    components, shell_metadata = {}, {}
    core = np.zeros_like(energy)
    valence = np.zeros_like(energy)
    target_values = None
    pz = energy_loss_to_pz(energy, q_au)
    valence_count = 0.0
    symbols = set()
    for element, amount in composition.items():
        symbol = source.resolve_element(element).symbol
        if symbol in symbols or not np.isfinite(amount) or amount <= 0:
            raise ValueError("composition must have unique elements with positive weights")
        symbols.add(symbol)
        pmax = source.momentum_support_au(element)[1]
        max_energy = (pmax+q_au/2)*q_au*EH
        for shell in source.available_shells(element):
            key = f"{symbol}:{shell}"
            binding = source.binding_energy_ev(element, shell)
            electrons = source.electron_occupancy(element, shell)*amount
            is_valence = binding < valence_cutoff_ev
            if key == target_id and is_valence:
                raise ValueError("target cannot be classified as valence")
            values = np.zeros_like(energy)
            factor = None
            if binding < max_energy:
                lower_pz = binding/(q_au*EH)-q_au/2
                # Restrict every query to tabulated momentum support.
                if lower_pz < -pmax:
                    raise ValueError("positive energy onset lies outside the momentum table")
                integration_pz = np.linspace(lower_pz, pmax, integration_points)
                integration_energy = (integration_pz+q_au/2)*q_au*EH
                shape = source.partial_profile(element, shell, integration_pz)/(q_au*EH)
                first_moment = np.trapezoid(integration_energy*shape, integration_energy)
                if not np.isfinite(first_moment) or first_moment <= 0:
                    raise ValueError(f"unusable first moment: {key}")
                factor = float(electrons*EH*q_au**2/2/first_moment)
                active = energy >= binding
                values[active] = source.partial_profile(element, shell, pz[active])/(q_au*EH)*factor
            components[key] = values
            shell_metadata[key] = {
                "binding_energy_eV": binding, "electron_count": electrons,
                "kind": "valence" if is_valence else "core",
                "fsum_scale": factor, "normalization_upper_energy_eV": max_energy,
                "status": "finite_table_fsum" if factor is not None else "threshold_outside_table_support",
            }
            if is_valence:
                valence += values
                valence_count += electrons
            else:
                core += values
            if key == target_id:
                target_values = values
    if target_values is None:
        raise ValueError("target absent from composition/table")
    return HFEnergyProfiles(energy.copy(), core+valence, core, valence,
        target_values, core-target_values, components, {
            "source": dict(source.provenance), "target": target_id,
            "q_au": float(q_au), "q_model": "constant channel q approximation",
            "valence_cutoff_ev": float(valence_cutoff_ev), "valence_electron_count": valence_count,
            "normalization": "finite-table first moment N*Hartree_eV*q_au^2/2",
            "integration_points": integration_points, "shells": shell_metadata,
            "tails_extrapolated": False, "energy_axis_shift_eV": 0.0,
        })


def _masked_integral(energy, values, mask):
    """Integrate contiguous observed segments without bridging protected gaps."""
    indices = np.flatnonzero(mask)
    groups = np.split(indices, np.flatnonzero(np.diff(indices) != 1)+1)
    return float(sum(np.trapezoid(values[g], energy[g]) for g in groups if len(g) >= 2))


def match_hf_scale(energy_ev, intensity, template, *, windows, edge,
                   prenorm_windows, fit_linear=True):
    """Same-support area pre-match, then y_pre = a*HF + b0+b1*(E-Ec).

    Returns (y_pre-linear)/a on the model density scale. Missing template points
    are excluded, never filled. The target protection applies to both steps.
    """
    energy = _grid(energy_ev)
    y, template = np.asarray(intensity, dtype=float), np.asarray(template, dtype=float)
    if y.shape != energy.shape or template.shape != energy.shape or not np.isfinite(y).all():
        raise ValueError("intensity/template shape mismatch or invalid intensity")
    if len(edge) != 2 or not np.isfinite(edge).all() or edge[0] >= edge[1]:
        raise ValueError("invalid target protection window")
    available = np.isfinite(template)
    pre_mask = fit_mask(energy, prenorm_windows, edge) & available
    data_area = _masked_integral(energy, y, pre_mask)
    hf_area = _masked_integral(energy, template, pre_mask)
    if data_area <= 0 or hf_area <= 0 or not np.isfinite([data_area, hf_area]).all():
        raise ValueError("pre-normalization requires positive finite same-support areas")
    pre_scale = hf_area/data_area
    fit = fit_mask(energy, windows, edge) & available
    nparams = 3 if fit_linear else 2
    if fit.sum() <= nparams:
        raise ValueError("insufficient fitting support")
    center = float(np.mean(energy[fit]))
    span = float(np.ptp(energy[fit]))
    columns = [template, np.ones_like(energy)]
    if fit_linear:
        columns.append((energy-center)/span)
    design = np.column_stack(columns)
    # Scale columns numerically without altering the physical fitted model.
    norms = np.linalg.norm(design[fit], axis=0)
    if np.any(norms == 0) or np.linalg.matrix_rank(design[fit]/norms) < nparams:
        raise ValueError("HF scale and baseline are unidentifiable")
    solution = lsq_linear(design[fit]/norms, pre_scale*y[fit],
                         bounds=([0]+[-np.inf]*(nparams-1), [np.inf]*nparams))
    parameters = solution.x/norms
    amplitude = float(parameters[0])
    if not solution.success or amplitude <= 1e-12:
        raise ValueError("HF matching failed or reached zero scale")
    baseline = design[:, 1:] @ parameters[1:]
    calibrated = (pre_scale*y-baseline)/amplitude
    residual = calibrated-template
    return {
        "calibrated": calibrated, "scaled_intensity": pre_scale*y/amplitude,
        "baseline": baseline/amplitude, "fit_mask": fit, "pre_mask": pre_mask,
        "available": available, "fit_residual": residual,
        "metadata": {
            "area_pre_scale": pre_scale, "data_area": data_area, "hf_area": hf_area,
            "fitted_hf_scale": amplitude, "raw_to_hf_scale": pre_scale/amplitude,
            "baseline_intercept_pre": float(parameters[1]),
            "baseline_slope_pre_per_eV": float(parameters[2]/span) if fit_linear else 0.0,
            "baseline_center_eV": center, "fit_linear": bool(fit_linear),
            "windows_ev": list(windows), "prenorm_windows_ev": list(prenorm_windows),
            "protected_window_ev": list(edge), "fit_rmse_per_eV": float(np.sqrt(np.mean(residual[fit]**2))),
            "statistical_uncertainty": None, "parameter_covariance": None,
        },
    }


def extract_hf_target(energy_ev, intensity, hf, mapped_valence, *, windows,
                      edge, prenorm_windows, fit_linear=True):
    """Fit core+valence, subtract OTHER core+valence+baseline; retain target."""
    energy = _grid(energy_ev)
    if not np.array_equal(hf.energy_ev, energy):
        raise ValueError("HF energy grid mismatch")
    mapped = np.asarray(mapped_valence, dtype=float)
    if mapped.shape != energy.shape:
        raise ValueError("mapped valence must match the energy grid")
    matched = match_hf_scale(energy, intensity, hf.core+mapped, windows=windows,
        edge=edge, prenorm_windows=prenorm_windows, fit_linear=fit_linear)
    background = hf.other_core+mapped+matched["baseline"]
    extracted = matched["scaled_intensity"]-background
    return {
        "model": "hf_target_preserving", "normalized_intensity": matched["scaled_intensity"],
        "raw_intensity": np.asarray(intensity), "background": background,
        "residual": extracted, "target_hf": hf.target,
        "core_background": hf.other_core, "valence_background": mapped,
        "linear_background": matched["baseline"],
        "model_fit_residual": extracted-hf.target,
        "available": matched["available"], "fit_mask": matched["fit_mask"],
        "parameters": matched["metadata"], "statistical_uncertainty": None,
        "parameter_covariance": None, "model_uncertainty": None,
        "fit_rmse": matched["metadata"]["fit_rmse_per_eV"],
        "intensity_unit": "HF-model density per eV; exploratory, not absolute calibration",
        "result_semantics": "target core retained; residual is not target-minus-HF",
        "status": "exploratory_unweighted; calibration and model not experimentally validated",
    }
