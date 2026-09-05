"""Small, unweighted notebook helpers; never manufacture statistical errors.

The public weighted pipelines are intentionally unchanged. Missing mapped
support is represented by NaN plus an explicit availability mask.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear

from .backgrounds import fit_pearson
from .backgrounds.valence_profile import build_valence_profile, map_valence_profile
from .constants import HARTREE_ENERGY_EV
from .geometry import energy_loss_to_pz, inverse_angstrom_to_au, pz_to_energy_loss
from .io import TextMapping, load_text_channels


def prompt_pz(energy_ev, q_au):
    """Prompt convention q/2 - omega/q; energy is in eV."""
    return -energy_loss_to_pz(energy_ev, q_au)


def prompt_energy(pz, q_au):
    return pz_to_energy_loss(-np.asarray(pz), q_au)


def prompt_coordinates(package_pz, *values):
    """Reflect the coordinate and sort *all* paired arrays together."""
    pz = -np.asarray(package_pz)
    order = np.argsort(pz)
    if any(np.asarray(v).shape != pz.shape for v in values):
        raise ValueError("paired arrays must have matching shapes")
    return (pz[order], *(np.asarray(v)[order] for v in values))


def read_crystals(source, fit_results, run_info):
    """Read processed channels and join by label, never by row position."""
    paths = [Path(p) for p in (source, fit_results, run_info)]
    hashes = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    with paths[0].open(encoding="utf-8-sig") as stream:
        columns = next(csv.reader(stream, delimiter="\t"))
    if len(columns) != len(set(columns)):
        raise ValueError("duplicate intensity labels")
    with paths[1].open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream, delimiter="\t"))
    labels = [r["crystal"] for r in rows]
    if len(labels) != len(set(labels)):
        raise ValueError("duplicate crystal metadata")
    lookup = {r["crystal"]: r for r in rows}
    missing = set(columns[1:]) - lookup.keys()
    if missing:
        raise ValueError(f"missing crystal metadata: {sorted(missing)}")
    dataset = load_text_channels(paths[0], [
        TextMapping(columns[0], c, "energy_loss", "eV", delimiter="\t",
                    analyzer_id=c, intensity_kind="processed") for c in columns[1:]
    ])
    records = []
    for spectrum in dataset.spectra:
        row = dict(lookup[spectrum.analyzer_id])
        for key in ("center_eV", "fwhm_eV", "q_ave", "dq_ave", "q_range", "dq_range"):
            row[key] = float(row[key])
            if not np.isfinite(row[key]):
                raise ValueError(f"nonfinite {key}: {spectrum.analyzer_id}")
        if row["q_ave"] <= 0 or row["fwhm_eV"] <= 0:
            raise ValueError("q and resolution must be positive")
        row["q_au"] = float(inverse_angstrom_to_au(row["q_ave"]))
        row["all_zero"] = bool(np.all(spectrum.counts == 0))
        row["negative_points"] = int(np.count_nonzero(spectrum.counts < 0))
        row["group"] = ("low_q" if row["q_ave"] < 9 else
                        "mid_high_q" if row["q_ave"] > 9 else "boundary")
        records.append(row)
    return dataset, records, {"input_sha256": hashes,
                             "run_info": json.loads(paths[2].read_text(encoding="utf-8-sig")),
                             "q_convention": "q_ave inverse angstrom, constant per channel approximation",
                             "intensity": "processed arbitrary units; no repeated I0 normalization",
                             "unused_metadata_labels": sorted(set(labels) - set(columns[1:]))}


def atomic_audit(source):
    """Finite-grid diagnostics, not automatic scientific approval of the table."""
    grid = np.concatenate(([0.0], np.geomspace(1e-5, 100, 1500)))
    rows, summaries = [], []
    for element, z in (("Ho", 67), ("B", 5)):
        total = source.total_profile(element, grid)
        summed = np.zeros_like(grid)
        electrons = 0.0
        for shell in source.available_shells(element):
            occupancy = source.electron_occupancy(element, shell)
            profile = source.partial_profile(element, shell, grid)
            summed += occupancy * profile
            electrons += occupancy
            rows.append({"element": element, "shell": source.shell_label(shell),
                         "occupancy": occupancy,
                         "binding_energy_eV": (source.binding_energy_ev(element, shell)
                                               if hasattr(source, "binding_energy_ev") else None),
                         "partial_full_integral_to_100_au": float(2 * np.trapezoid(profile, grid))})
        summaries.append({"element": element, "atomic_number": z,
                          "available_occupancy_sum": electrons,
                          "electron_count_difference": electrons - z,
                          "relative_profile_l2_difference": float(np.linalg.norm(summed-total) / np.linalg.norm(total)),
                          "total_full_integral_to_100_au": float(2*np.trapezoid(total, grid)),
                          "status": "requires explicit table/occupation review"})
    return rows, summaries


def hf_missing(parameters):
    required = ("shells_by_element", "valence_electron_count", "exclude_target",
                "reference_scale", "reference_constant")
    missing = [key for key in required if parameters.get(key) is None]
    if not parameters.get("atomic_review_passed", False):
        missing.append("atomic_review_passed")
    if not parameters.get("atomic_review_note"):
        missing.append("atomic_review_note")
    if not missing:
        if set(parameters["shells_by_element"]) != {"Ho", "B"}:
            raise ValueError("explicit Ho and B shell lists required")
        for key in ("reference_scale", "valence_electron_count"):
            if not np.isfinite(parameters[key]) or parameters[key] <= 0:
                raise ValueError(f"{key} must be finite and positive")
        if not np.isfinite(parameters["reference_constant"]):
            raise ValueError("reference_constant must be finite")
    return missing


def reference_scores(dataset, records, *, edge=(111, 211), minimum_energy=20, tail=(230, 700)):
    """Equal scores: relative q, clean fraction, sampled clean tail coverage.

    No SNR proxy is invented. Ties use crystal labels. Scores only screen
    candidates; profile support is validated again after core subtraction.
    """
    rows = []
    by_label = {s.analyzer_id: s for s in dataset.spectra}
    eligible = [r for r in records if r["q_ave"] > 9 and not r["all_zero"]]
    maximum_q = max((r["q_ave"] for r in eligible), default=1)
    for record in eligible:
        energy = by_label[record["crystal"]].energy_loss_ev
        retained = energy >= minimum_energy
        clean = retained & ~((energy >= edge[0]) & (energy <= edge[1]))
        tail_points = clean & (energy >= tail[0]) & (energy <= tail[1])
        step = float(np.median(np.diff(energy)))
        coverage = min(1.0, float(np.count_nonzero(tail_points) * step / (tail[1]-tail[0])))
        components = [record["q_ave"]/maximum_q, float(clean.sum()/max(retained.sum(), 1)), coverage]
        rows.append({"crystal": record["crystal"], "q_score": components[0],
                     "clean_fraction": components[1], "tail_coverage": components[2],
                     "score": float(np.mean(components))})
    return sorted(rows, key=lambda r: (-r["score"], r["crystal"]))


def valence_stages(candidate, electron_count, sigma=0.05):
    if candidate.intensity_convention != "density_per_ev":
        raise ValueError("notebook stages require density_per_ev")
    options = {"score_weights": {"uncontaminated_fraction": 1},
               "valence_electron_count": electron_count,
               "normalization_convention": "full_symmetric", "masked_region_policy": "linear"}
    sym = build_valence_profile([candidate], **options)
    smooth = build_valence_profile([candidate], gaussian_sigma_pz_au=sigma, **options)
    raw = sym.source_residual * candidate.q_au * HARTREE_ENERGY_EV
    pz_raw, raw, mask = prompt_coordinates(sym.source_pz_au, raw, sym.contamination_mask)
    pz, symmetric, corrected = prompt_coordinates(sym.pz_au, sym.profile, smooth.profile)
    return smooth, {"pz_raw": pz_raw, "raw": raw, "mask": mask,
                    "pz": pz, "symmetric": symmetric, "corrected": corrected,
                    "sym_normalization_scale": sym.diagnostics["normalization_scale"],
                    "smooth_normalization_scale": smooth.diagnostics["normalization_scale"]}


def map_available(profile, energy, q):
    energy = np.asarray(energy, dtype=float)
    pz = energy_loss_to_pz(energy, q)
    available = np.zeros(energy.shape, dtype=bool)
    for start, stop in profile.support_intervals_pz_au:
        available |= (pz >= start) & (pz <= stop)
    mapped = np.full(energy.shape, np.nan)
    if available.sum() >= 2:
        mapped[available] = map_valence_profile(profile, energy[available], target_q_au=q).intensity_per_ev
    else:
        available[:] = False
    return mapped, available


def fit_mask(energy, windows, edge):
    energy = np.asarray(energy)
    mask = np.zeros(energy.shape, dtype=bool)
    for start, stop in windows:
        if not np.isfinite([start, stop]).all() or start >= stop:
            raise ValueError("invalid fit window")
        mask |= (energy >= start) & (energy <= stop)
    return mask & ~((energy >= edge[0]) & (energy <= edge[1]))


def _result(y, background, mask, parameters, model):
    residual = y - background
    return {"model": model, "background": background, "residual": residual,
            "available": np.isfinite(background), "fit_mask": mask,
            "parameters": parameters, "statistical_uncertainty": None,
            "parameter_covariance": None, "model_uncertainty": None,
            "fit_rmse": float(np.sqrt(np.mean(residual[mask]**2))),
            "status": "exploratory_unweighted; not scientifically validated"}


def fit_unweighted_pearson(energy, intensity, windows, edge):
    energy, y = np.asarray(energy), np.asarray(intensity)
    mask = fit_mask(energy, windows, edge)
    if mask.sum() <= 4:
        raise ValueError("insufficient background points")
    fit = fit_pearson(energy[mask], y[mask], loss="linear")
    if not fit.success:
        raise ValueError(f"Pearson optimizer failed: {fit.message}")
    from .backgrounds.pearson import pearson_background
    background = pearson_background(energy, **fit.parameters)
    # Deliberately discard legacy covariance/chi-square from sigma-free fitting.
    return _result(y, background, mask, dict(fit.parameters), "pearson")


def fit_unweighted_templates(energy, intensity, core, valence, windows, edge):
    energy, y, core, valence = map(np.asarray, (energy, intensity, core, valence))
    if any(a.shape != energy.shape for a in (y, core, valence)):
        raise ValueError("template shapes must match energy")
    if not np.isfinite(y).all():
        raise ValueError("intensity must be finite")
    available = np.isfinite(core) & np.isfinite(valence)
    mask = fit_mask(energy, windows, edge) & available
    design = np.column_stack((core+valence, np.ones_like(energy)))
    if mask.sum() <= 2 or np.linalg.matrix_rank(design[mask]) < 2:
        raise ValueError("insufficient support or unidentifiable scale/constant")
    fit = lsq_linear(design[mask], y[mask], bounds=([0, -np.inf], [np.inf, np.inf]))
    if not fit.success:
        raise ValueError(f"template optimizer failed: {fit.message}")
    background = np.full(y.shape, np.nan)
    background[available] = design[available] @ fit.x
    result = _result(y, background, mask, {"scale": float(fit.x[0]), "constant": float(fit.x[1])}, "compton_profile")
    result.update(core_background=fit.x[0]*core, valence_background=fit.x[0]*valence)
    return result


def save_result(directory, label, energy, intensity, result, *, edge, provenance):
    """Per-channel data and auditable parameters; plots are made by notebook."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    fields = [energy, intensity, result["background"], result["residual"],
              result["available"], result["fit_mask"]]
    unit = "per_eV" if "intensity_unit" in result else "au"
    header = f"energy_loss_eV,intensity_{unit},background_{unit},residual_{unit},available,fit_mask"
    for name in ("core_background", "valence_background", "linear_background",
                 "target_hf", "model_fit_residual", "raw_intensity"):
        if name in result:
            header += "," + name
            fields.append(result[name])
    np.savetxt(directory / f"{label}.csv", np.column_stack(fields), delimiter=",", header=header, comments="")
    metadata = {k: v for k, v in result.items() if not isinstance(v, np.ndarray)}
    metadata.update(provenance=provenance, edge_protection_ev=edge, negative_values_clipped=False)
    (directory / f"{label}.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
