"""Summarize extracted target responses in their common input intensity units."""

import csv
import json
from pathlib import Path

import numpy as np


def _mean_columns(values):
    valid = np.isfinite(values)
    count = valid.sum(axis=1)
    mean = np.divide(np.where(valid, values, 0).sum(axis=1), count,
                     out=np.full(values.shape[0], np.nan), where=count > 0)
    return mean, count


def average_target_exports(directory, records, *, target_window):
    """Read current successful exports; undo HF scaling before averaging.

    Module means give each available crystal equal weight at each energy.
    Two overall means distinguish equal module and equal crystal weighting.
    No uncertainty bars are inferred from crystal-to-crystal differences.
    """
    directory = Path(directory)
    start, stop = target_window
    if not np.isfinite([start, stop]).all() or start >= stop:
        raise ValueError("target window must be finite and increasing")
    with (directory / "channel-status.csv").open(encoding="utf-8-sig", newline="") as stream:
        status_rows = list(csv.DictReader(stream))
    statuses = {r["crystal"]: r for r in status_rows}
    if len(statuses) != len(status_rows):
        raise ValueError("duplicate channel status")
    labels = [r["crystal"] for r in records]
    if len(set(labels)) != len(labels):
        raise ValueError("duplicate crystal records")
    energy = None
    by_module = {}
    included, excluded, warnings = [], [], []
    for record in records:
        label, module = record["crystal"], record["module"]
        if not module:
            raise ValueError(f"missing module for {label}")
        by_module.setdefault(module, [])
        status = statuses.get(label, {})
        if record["all_zero"] or status.get("status") != "exploratory":
            excluded.append({"crystal": label, "module": module,
                             "reason": "all_zero" if record["all_zero"] else status.get("reason", "not successful")})
            continue
        metadata = json.loads((directory / f"{label}.json").read_text(encoding="utf-8"))
        data = np.genfromtxt(directory / f"{label}.csv", delimiter=",", names=True)
        full_energy = np.atleast_1d(data["energy_loss_eV"])
        if not np.isfinite(full_energy).all() or np.any(np.diff(full_energy) <= 0):
            raise ValueError(f"invalid energy grid: {label}")
        mask = (full_energy >= start) & (full_energy <= stop)
        selected_energy = full_energy[mask]
        if selected_energy.size < 2:
            raise ValueError(f"insufficient target-window coverage: {label}")
        if energy is None:
            energy = selected_energy
        elif not np.array_equal(energy, selected_energy):
            raise ValueError("target energy grids differ; explicit alignment is required")
        model = metadata["model"]
        if model == "hf_target_preserving":
            scale = metadata["parameters"]["raw_to_hf_scale"]
            if not np.isfinite(scale) or scale <= 0:
                raise ValueError(f"invalid raw-to-HF scale: {label}")
            response = np.atleast_1d(data["residual_per_eV"])[mask]/scale
        elif model == "pearson":
            scale = 1.0
            response = np.atleast_1d(data["residual_au"])[mask].copy()
        else:
            raise ValueError(f"unsupported averaging units/model: {model}")
        available = np.atleast_1d(data["available"])[mask] == 1
        response[~available | ~np.isfinite(response)] = np.nan
        if not np.isfinite(response).any():
            excluded.append({"crystal": label, "module": module, "reason": "no available target samples"})
            continue
        by_module[module].append(response)
        included.append({"crystal": label, "module": module, "model": model,
                         "raw_to_hf_scale": float(scale),
                         "available_target_points": int(np.isfinite(response).sum())})
        input_column = "raw_intensity" if model == "hf_target_preserving" else "intensity_au"
        if input_column in data.dtype.names:
            input_peak = float(np.max(np.abs(np.atleast_1d(data[input_column])[mask])))
            response_peak = float(np.nanmax(np.abs(response)))
            if input_peak > 0 and response_peak > 10*input_peak:
                warnings.append({"crystal": label, "module": module,
                    "reason": "target residual exceeds 10 times input peak; inspect background fit",
                    "peak_ratio": response_peak/input_peak,
                    "action": "retained in mean; diagnostic only"})
    if energy is None or not included:
        raise ValueError("no valid extracted target channels")
    modules = {}
    all_channels = []
    for module, curves in by_module.items():
        values = np.column_stack(curves) if curves else np.empty((len(energy), 0))
        mean, counts = _mean_columns(values)
        modules[module] = {"mean": mean, "n_crystals": counts,
                           "total_crystals": len(curves)}
        all_channels.extend(curves)
    equal_modules, module_count = _mean_columns(np.column_stack([m["mean"] for m in modules.values()]))
    equal_crystals, crystal_count = _mean_columns(np.column_stack(all_channels))
    return {"energy_ev": energy, "modules": modules,
            "module_equal_mean": equal_modules, "crystal_equal_mean": equal_crystals,
            "n_modules": module_count, "n_crystals": crystal_count,
            "metadata": {"target_window_ev": [start, stop], "included": included,
                "excluded": excluded, "warnings": warnings, "units": "input processed intensity (a.u.)",
                "hf_conversion": "extracted HF density divided by saved raw_to_hf_scale",
                "primary_weighting": "equal module means, available samples only",
                "comparison_weighting": "equal valid crystals, available samples only",
                "negative_values_clipped": False, "statistical_uncertainty": None,
                "note": "descriptive cross-q average; no detector-efficiency or absolute calibration implied"}}


def export_target_averages(result, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    energy = result["energy_ev"]
    for module, values in result["modules"].items():
        np.savetxt(directory / f"{module}-target-mean.csv",
                   np.column_stack((energy, values["mean"], values["n_crystals"])),
                   delimiter=",", comments="", header="energy_loss_eV,mean_input_au,n_crystals")
    np.savetxt(directory / "all-modules-target-mean.csv",
               np.column_stack((energy, result["module_equal_mean"], result["crystal_equal_mean"],
                                result["n_modules"], result["n_crystals"])),
               delimiter=",", comments="",
               header="energy_loss_eV,module_equal_mean_input_au,crystal_equal_mean_input_au,n_modules,n_crystals")
    (directory / "averaging-metadata.json").write_text(
        json.dumps(result["metadata"], indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
