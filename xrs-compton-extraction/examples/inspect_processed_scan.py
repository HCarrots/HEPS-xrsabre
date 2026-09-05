"""Audit a processed wide TSV without assuming a noise model or fitting an edge.

Optional fit-results and run-info files are supplied explicitly, never discovered
or imported from the producing software. All input files remain unchanged.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np

from xrs_compton_extraction.geometry import inverse_angstrom_to_au
from xrs_compton_extraction.io import TextMapping, load_text_channels


def inspect_scan(source: Path, output: Path, *, fit_results: Path | None = None,
                 run_info: Path | None = None) -> dict:
    with source.open(encoding="utf-8-sig", newline="") as stream:
        header = next(csv.reader(stream, delimiter="\t"))
    if header[0] != "Energy Transfer (eV)":
        raise ValueError("expected an explicitly labelled Energy Transfer (eV) first column")
    dataset = load_text_channels(source, [
        TextMapping(header[0], name, "energy_loss", "eV", delimiter="\t",
                    analyzer_id=name, intensity_kind="processed")
        for name in header[1:]
    ])
    provenance = {}
    for path in (source, fit_results, run_info):
        if path is not None:
            provenance[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
    fit_rows = {}
    if fit_results is not None:
        with fit_results.open(encoding="utf-8-sig", newline="") as stream:
            for row in csv.DictReader(stream, delimiter="\t"):
                if row["crystal"] in fit_rows:
                    raise ValueError(f"duplicate fit-results crystal: {row['crystal']}")
                fit_rows[row["crystal"]] = row
        missing = set(header[1:]) - set(fit_rows)
        if missing:
            raise ValueError(f"missing fit-results channels: {sorted(missing)}")
    run = json.loads(run_info.read_text(encoding="utf-8-sig")) if run_info else {}
    energy = dataset.spectra[0].energy_loss_ev
    if not np.all(np.diff(energy) > 0):
        raise ValueError("energy loss must be strictly increasing")
    channels = []
    for spectrum in dataset.spectra:
        y = spectrum.counts
        row = fit_rows.get(spectrum.analyzer_id)
        item = {
            "channel": spectrum.analyzer_id,
            "minimum": float(y.min()), "maximum": float(y.max()),
            "zero_count": int(np.count_nonzero(y == 0)),
            "negative_count": int(np.count_nonzero(y < 0)),
            "all_zero": bool(np.all(y == 0)),
            "uncertainty_available": spectrum.uncertainty is not None,
        }
        if row:
            q = float(row["q_ave"])
            item.update(q_mean_inverse_angstrom=q, q_mean_au=inverse_angstrom_to_au(q),
                        q_span_inverse_angstrom=float(row["q_range"]),
                        elastic_center_ev=float(row["center_eV"]),
                        upstream_selected=row["selected"] == "True")
        channels.append(item)
    report = {
        "purpose": "input audit only; not a validated edge extraction",
        "source_sha256": provenance,
        "channel_count": len(channels), "energy_point_count": len(energy),
        "energy_range_ev": [float(energy[0]), float(energy[-1])],
        "energy_steps_ev": np.unique(np.round(np.diff(energy), 8)).tolist(),
        "nonfinite_count": 0,  # strict loader rejects nonfinite input
        "upstream_config": run.get("config", {}),
        "channels": channels,
        "warnings": [
            "Processed intensities must not be reinterpreted as raw Poisson counts.",
            "No uncertainty columns: weighted extraction is blocked pending an explicit noise model.",
            "Filtering/interpolation can correlate neighboring samples; no covariance was supplied.",
            "q_mean is a summary, not an energy-resolved q calibration or a fit input.",
            "All-zero channels are retained and flagged, not silently removed.",
        ],
    }
    # Do not allow audit outputs to overwrite any input, even under renamed paths.
    targets = [output / name for name in ("inspection.json", "inspection.md", "overview.png")]
    if {target.resolve() for target in targets} & {Path(path) for path in provenance}:
        raise ValueError("output paths overlap input files")
    output.mkdir(parents=True, exist_ok=True)
    (output / "inspection.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    zeros = [item["channel"] for item in channels if item["all_zero"]]
    lines = ["# Processed scan inspection", "", report["purpose"], "",
             f"- Channels: {len(channels)}; energy points: {len(energy)}.",
             f"- Energy range: {energy[0]:g} to {energy[-1]:g} eV.",
             f"- All-zero channels: {', '.join(zeros) or 'none'}.",
             "- Detailed channel statistics and input SHA-256 hashes: inspection.json.", "",
             "## Limitations", "", *[f"- {warning}" for warning in report["warnings"]], "",
             "## Overview", "", "![Overview](overview.png)", ""]
    (output / "inspection.md").write_text("\n".join(lines), encoding="utf-8")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    modules = sorted({s.analyzer_id.split("-")[0] for s in dataset.spectra})
    figure, axes = plt.subplots(len(modules), 1, figsize=(12, 3 * len(modules)),
                                sharex=True, squeeze=False, layout="constrained")
    for module, ax in zip(modules, axes[:, 0], strict=True):
        for spectrum in dataset.spectra:
            if spectrum.analyzer_id.split("-")[0] == module:
                ax.plot(energy, spectrum.counts, lw=0.8, label=spectrum.analyzer_id)
        ax.set_yscale("symlog", linthresh=1)
        ax.set_ylabel("Processed intensity")
        ax.set_title(f"{module} | symlog display; zeros retained")
        ax.legend(ncol=5, fontsize=8)
        ax.grid(alpha=0.2)
    axes[-1, 0].set_xlabel("Energy transfer (eV)")
    figure.savefig(output / "overview.png", dpi=130)
    plt.close(figure)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fit-results", type=Path)
    parser.add_argument("--run-info", type=Path)
    args = parser.parse_args()
    result = inspect_scan(args.source, args.output, fit_results=args.fit_results, run_info=args.run_info)
    print(f"Read {result['channel_count']} channels x {result['energy_point_count']} points; report: {args.output}")
