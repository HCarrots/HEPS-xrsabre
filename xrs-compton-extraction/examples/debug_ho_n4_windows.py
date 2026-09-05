"""Scan explicit candidate windows for the Ho N4 processed-data audit.

This tool is diagnostic only. It does not extract an edge or manufacture
uncertainties. Pearson is evaluated as an exploratory low-q comparison; a poor
high-q score is evidence to use profile templates, not a reason to widen windows.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from xrs_compton_extraction.backgrounds import fit_pearson
from xrs_compton_extraction.q_groups import group_q_channels


def _read(source: Path, fit_results: Path) -> tuple[np.ndarray, list[str], np.ndarray, dict[str, float]]:
    with source.open(encoding="utf-8-sig", newline="") as stream:
        reader = csv.reader(stream, delimiter="\t")
        header = next(reader)
    table = np.loadtxt(source, delimiter="\t", skiprows=1)
    fit = np.genfromtxt(fit_results, delimiter="\t", names=True, dtype=None, encoding="utf-8")
    q = {str(row[0]): float(row[14]) for row in fit}
    return table[:, 0], header[1:], table[:, 1:], q


def scan_windows(
    source: Path,
    fit_results: Path,
    output: Path,
    *,
    threshold: float = 9.0,
    n4_center_ev: float = 161.0,
    pre_start_ev: float = 20.0,
    pre_ends: tuple[float, ...] = (80.0, 90.0, 100.0),
    post_starts: tuple[float, ...] = (210.0, 230.0, 250.0),
    post_end_ev: float = 700.0,
) -> dict:
    energy, labels, intensities, q = _read(source, fit_results)
    groups = group_q_channels(labels, q, threshold=threshold)
    exclusion = (n4_center_ev - 50.0, n4_center_ev + 50.0)
    results: dict[str, list[dict[str, object]]] = {}
    for band, members in groups.items():
        if band == "boundary" or not members:
            continue
        indices = [labels.index(label) for label in members if np.any(intensities[:, labels.index(label)])]
        if not indices:
            continue
        median = np.median(intensities[:, indices], axis=1)
        rows = []
        for pre_end in pre_ends:
            for post_start in post_starts:
                windows = ((pre_start_ev, pre_end), (post_start, min(post_end_ev, float(energy[-1]))))
                try:
                    fit = fit_pearson(energy, median, fit_windows_ev=windows, loss="linear")
                    outside = (energy > exclusion[0]) & (energy < exclusion[1])
                    rows.append({
                        "windows_ev": windows,
                        "reduced_chi_square": fit.reduced_chi_square,
                        "outside_exclusion_rms": float(np.sqrt(np.mean(np.square(fit.residual[outside])))),
                        "success": fit.success,
                        "parameters": dict(fit.parameters),
                    })
                except (ValueError, FloatingPointError) as exc:
                    rows.append({"windows_ev": windows, "error": str(exc)})
        results[band] = rows
    report = {
        "purpose": "Ho N4 fit-window diagnostic; not an edge extraction",
        "n4_reference_energy_ev": n4_center_ev,
        "excluded_structure_window_ev": exclusion,
        "q_threshold_inverse_angstrom": threshold,
        "q_groups": groups,
        "results": results,
        "interpretation": [
            "Theoretical Ho N4 reference is about 161 eV; inspect the energy calibration before fixing a target window.",
            "Use the lowest stable low_q residual/chi-square candidate only as a starting window.",
            "Do not use Pearson for mid_high_q when its residual is structurally large; use core/valence templates.",
            "No uncertainty or covariance is present in the processed input, so scores are unweighted diagnostics.",
        ],
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "window-scan.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--fit-results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=9.0)
    parser.add_argument("--n4-center", type=float, default=161.0)
    parser.add_argument("--pre-start", type=float, default=20.0)
    args = parser.parse_args()
    report = scan_windows(args.source, args.fit_results, args.output,
                           threshold=args.threshold, n4_center_ev=args.n4_center,
                           pre_start_ev=args.pre_start)
    for band, rows in report["results"].items():
        valid = [row for row in rows if "reduced_chi_square" in row]
        best = min(valid, key=lambda row: row["outside_exclusion_rms"]) if valid else None
        print(f"{band}: {len(valid)} candidates; exploratory best={best['windows_ev'] if best else 'none'}")
