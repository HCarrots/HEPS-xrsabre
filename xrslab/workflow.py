"""Reproducible, quality-gated XRS analysis workflow.

The notebook and command-line entry point both call this module.  Low-level
array operations remain in :mod:`xrslab.analysis`; this module owns workflow
configuration, validation, quality control, scan merging and provenance.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import sys
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Sequence

import h5py
import numpy as np
import pandas as pd
from scipy.optimize import OptimizeWarning, curve_fit

from . import math_func
from .analysis import (
    DEFAULT_MODULE_ANGLES,
    FitSummary,
    RoiCollection,
    RoiMasks,
    RoiSums,
    ScanBatch,
    build_roi_masks,
    load_roi_collection,
    load_scan_batch,
    resolve_motor_pv,
    sum_roi_spectra,
    sum_roi_spectra_batch,
)
from xrsabre.paths import WorkspacePaths, load_workspace
from .xrs_roi import XRSRoi, q_calc


@dataclass(frozen=True)
class AnalysisConfig:
    """Portable configuration for one XRS analysis."""

    element: str = "Ho"
    elastic_scan_ids: tuple[int, ...] = (57,)
    xrs_scan_ids: tuple[int, ...] = (59,)
    analysis_name: str = "XRS_analysis"
    elastic_i0_pv: str = "D_SiC_I_A"
    xrs_i0_pv: str = "D_SiC_I_A"
    elastic_energy_pv: str = "M_DCM_B5Energy_readback"
    xrs_energy_pv: str = "M_DCM_B5Energy_readback"
    energy_pv_alternatives: tuple[str, ...] = ("M_DCM_B5Link_Energy_readback",)
    roi_filename: str = "roi_60crystals-H.txt"
    minipix_roi_filename: str = "roi_minipix-v3.txt"
    modules: tuple[str, ...] = ("VB", "HB", "HL", "VD", "VU")
    excluded_rois: tuple[str, ...] = ()
    q_range: tuple[float, float] = (0.0, 10.0)
    auto_adjust_rois: bool = True
    roi_size: int = 30
    filter_value: float = 0.15
    use_filter: bool = True
    divide_i0_elastic: bool = True
    divide_i0_xrs: bool = True
    correct_i0_glitches: bool = True
    i0_glitch_threshold: float = 0.7
    elastic_center_range_kev: tuple[float, float] = (9.67, 9.69)
    max_fwhm_ev: float = 2.0
    min_r_squared: float = 0.8
    energy_step_ev: float = 0.2
    module_offsets: tuple[tuple[str, float, float], ...] = (
        ("VU", 5.0, 10.0),
        ("VD", -10.0, 20.0),
    )

    def validate(self) -> None:
        if not self.element.strip():
            raise ValueError("element cannot be empty")
        if not self.analysis_name.strip():
            raise ValueError("analysis_name cannot be empty")
        if not self.elastic_scan_ids or not self.xrs_scan_ids:
            raise ValueError("elastic_scan_ids and xrs_scan_ids cannot be empty")
        if len(set(self.elastic_scan_ids)) != len(self.elastic_scan_ids):
            raise ValueError("elastic_scan_ids contains duplicates")
        if len(set(self.xrs_scan_ids)) != len(self.xrs_scan_ids):
            raise ValueError("xrs_scan_ids contains duplicates")
        if not 0 <= self.filter_value <= 1:
            raise ValueError("filter_value must be between 0 and 1")
        if not 0 < self.i0_glitch_threshold < 1:
            raise ValueError("i0_glitch_threshold must be between 0 and 1")
        if self.roi_size <= 0 or self.energy_step_ev <= 0:
            raise ValueError("roi_size and energy_step_ev must be positive")
        if self.elastic_center_range_kev[0] >= self.elastic_center_range_kev[1]:
            raise ValueError("elastic_center_range_kev must be increasing")
        if self.q_range[0] >= self.q_range[1]:
            raise ValueError("q_range must be increasing")

    def to_dict(self) -> dict[str, object]:
        return _jsonable(asdict(self))


@dataclass(frozen=True)
class QcApproval:
    """Explicit operator decision required before formal export."""

    approved: bool = False
    excluded_scans: tuple[int, ...] = ()
    excluded_rois: tuple[str, ...] = ()
    note: str = ""


@dataclass
class PreparedAnalysis:
    """All calibrated, normalised products required for QC and finalisation."""

    config: AnalysisConfig
    workspace: WorkspacePaths
    input_files: dict[str, Path]
    input_table: pd.DataFrame
    elastic_batch: ScanBatch
    xrs_batch: ScanBatch
    elastic_energy: np.ndarray
    xrs_energy: list[np.ndarray]
    elastic_i0_original: list[np.ndarray]
    elastic_i0_corrected: list[np.ndarray]
    xrs_i0_original: list[np.ndarray]
    xrs_i0_corrected: list[np.ndarray]
    scan_qc: pd.DataFrame
    roi_collection: RoiCollection
    lambda_masks: RoiMasks
    minipix_masks: RoiMasks
    elastic_sums: RoiSums
    minipix_elastic_sums: RoiSums
    xrs_sums: list[RoiSums]
    minipix_xrs_sums: list[RoiSums]
    fit_summary: FitSummary


@dataclass
class QCReport:
    """Machine-readable quality report shown before operator approval."""

    roi_table: pd.DataFrame
    scan_table: pd.DataFrame
    summary: dict[str, object]

    def to_dict(self) -> dict[str, object]:
        return {
            "summary": _jsonable(self.summary),
            "roi_records": _jsonable(self.roi_table.to_dict(orient="records")),
            "scan_records": _jsonable(self.scan_table.to_dict(orient="records")),
        }


@dataclass
class AnalysisResult:
    """Final or provisional spectra created from a QC decision."""

    approved: bool
    approval_note: str
    energy_transfer: np.ndarray
    intensity_sum: np.ndarray
    intensity_mean: np.ndarray
    scan_coverage: np.ndarray
    roi_coverage: np.ndarray
    roi_spectra: pd.DataFrame
    module_spectra: pd.DataFrame
    fit_table: pd.DataFrame
    selected_roi_ids: list[str]
    used_scan_ids: list[int]
    excluded_scan_ids: list[int]
    manually_excluded_rois: list[str]
    prepared: PreparedAnalysis = field(repr=False)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _scan_directories(raw_directory: Path, scan_id: int) -> list[Path]:
    prefix = f"{int(scan_id)}_"
    return sorted(
        path for path in raw_directory.iterdir()
        if path.is_dir() and path.name.startswith(prefix)
    )


def _find_scan_file(raw_directory: Path, scan_id: int) -> Path:
    matches = _scan_directories(raw_directory, scan_id)
    if not matches:
        raise FileNotFoundError(f"No directory found for scan {scan_id} in {raw_directory}")
    if len(matches) > 1:
        raise ValueError(
            f"Multiple directories found for scan {scan_id}: "
            f"{[path.name for path in matches]}"
        )
    files = sorted(matches[0].glob("*.nxs"))
    if len(files) != 1:
        raise FileNotFoundError(
            f"Expected exactly one .nxs file for scan {scan_id} in {matches[0]}, "
            f"found {len(files)}"
        )
    return files[0]


def _inspect_scan_file(
    path: Path,
    scan_id: int,
    kind: str,
    monitor_pv: str,
    energy_candidates: Sequence[str],
) -> dict[str, object]:
    try:
        handle = h5py.File(path, "r")
    except OSError as exc:
        raise ValueError(f"Cannot open {kind} scan {scan_id}: {path}: {exc}") from exc
    with handle:
        if "entry" not in handle:
            raise KeyError(f"{kind} scan {scan_id} has no /entry group")
        entry = handle["entry"]
        if "data" not in entry or "instrument" not in entry:
            raise KeyError(f"{kind} scan {scan_id} requires /entry/data and /entry/instrument")
        data = entry["data"]
        instrument = entry["instrument"]
        missing = [name for name in (monitor_pv, "lambda", "minipix") if name not in data]
        if missing:
            raise KeyError(f"{kind} scan {scan_id} is missing data fields {missing}")
        energy_pv = next((name for name in energy_candidates if name in instrument), None)
        if energy_pv is None:
            raise KeyError(
                f"{kind} scan {scan_id} has none of the energy PVs {tuple(energy_candidates)}"
            )
        monitor = data[monitor_pv]
        points = int(monitor.shape[0]) if monitor.ndim == 1 else -1
        if points <= 0:
            raise ValueError(f"{kind} scan {scan_id} monitor {monitor_pv!r} is not 1D")
        if instrument[energy_pv].shape != (points,):
            raise ValueError(f"{kind} scan {scan_id} energy and monitor lengths differ")
        shapes = {}
        for detector in ("lambda", "minipix"):
            shape = tuple(data[detector].shape)
            if len(shape) != 3 or points not in (shape[0], shape[-1]):
                raise ValueError(
                    f"{kind} scan {scan_id} detector {detector!r} has incompatible shape {shape}"
                )
            shapes[detector] = shape
        return {
            "kind": kind,
            "scan_id": int(scan_id),
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "points": points,
            "energy_pv": energy_pv,
            "lambda_shape": shapes["lambda"],
            "minipix_shape": shapes["minipix"],
        }


def _preflight(config: AnalysisConfig, raw_directory: Path) -> tuple[dict[str, Path], pd.DataFrame]:
    if not raw_directory.is_dir():
        raise FileNotFoundError(f"Raw data directory not found: {raw_directory}")
    files: dict[str, Path] = {}
    records = []
    groups = (
        ("elastic", config.elastic_scan_ids, config.elastic_i0_pv, config.elastic_energy_pv),
        ("xrs", config.xrs_scan_ids, config.xrs_i0_pv, config.xrs_energy_pv),
    )
    for kind, scan_ids, monitor_pv, energy_pv in groups:
        for scan_id in scan_ids:
            path = _find_scan_file(raw_directory, scan_id)
            key = f"{kind}:{scan_id}"
            files[key] = path
            records.append(_inspect_scan_file(
                path,
                scan_id,
                kind,
                monitor_pv,
                (energy_pv, *config.energy_pv_alternatives),
            ))
    return files, pd.DataFrame(records)


def _correct_monitor(values: Sequence[float], threshold: float, enabled: bool):
    original = np.asarray(values, dtype=float).copy()
    corrected = original.copy()
    corrected_indices: list[int] = []
    invalid_indices = np.flatnonzero(~np.isfinite(original) | (original <= 0)).tolist()
    if enabled:
        for index in range(1, len(corrected) - 1):
            left, middle, right = corrected[index - 1:index + 2]
            if not np.all(np.isfinite((left, middle, right))):
                continue
            if left <= 0 or middle <= 0 or right <= 0:
                continue
            if middle / left < threshold and middle / right < threshold:
                corrected[index] = (left + right) / 2
                corrected_indices.append(index)
    corrected[~np.isfinite(corrected) | (corrected <= 0)] = np.nan
    return original, corrected, corrected_indices, invalid_indices


def _scan_axis_first(stack: np.ndarray, point_count: int) -> np.ndarray:
    array = np.asarray(stack, dtype=float)
    if array.ndim != 3:
        raise ValueError(f"Expected a 3D detector stack, got {array.shape}")
    if array.shape[0] == point_count:
        return array
    if array.shape[-1] == point_count:
        return np.moveaxis(array, -1, 0)
    raise ValueError(f"Detector shape {array.shape} does not contain scan length {point_count}")


def _normalise_loaded_batch(
    batch: ScanBatch,
    scan_ids: Sequence[int],
    monitor_pv: str,
    *,
    divide_by_monitor: bool,
    correct_glitches: bool,
    glitch_threshold: float,
    kind: str,
) -> tuple[ScanBatch, list[np.ndarray], list[np.ndarray], list[dict[str, object]]]:
    original_list, corrected_list, records = [], [], []
    lambda_data, minipix_data, corrected_frames = [], [], []
    if batch.scan_count != len(scan_ids):
        raise ValueError(
            f"Loaded {batch.scan_count} {kind} scans for requested IDs {list(scan_ids)}"
        )
    for index, (scan_id, frame) in enumerate(zip(scan_ids, batch.det1d_data)):
        if monitor_pv not in frame:
            raise KeyError(f"{kind} scan {scan_id} is missing monitor {monitor_pv!r}")
        original, corrected, changed, invalid = _correct_monitor(
            frame[monitor_pv].to_numpy(), glitch_threshold, correct_glitches
        )
        original_list.append(original)
        corrected_list.append(corrected)
        frame_copy = frame.copy()
        frame_copy[monitor_pv] = corrected
        corrected_frames.append(frame_copy)

        stacks = []
        for source in (batch.lambda_data[index], batch.minipix_data[index]):
            stack = _scan_axis_first(source, len(corrected))
            if divide_by_monitor:
                valid = np.isfinite(corrected) & (corrected > 0)
                np.divide(
                    stack,
                    corrected[:, None, None],
                    out=stack,
                    where=valid[:, None, None],
                )
                stack[~valid] = np.nan
            stacks.append(stack)
        lambda_data.append(stacks[0])
        minipix_data.append(stacks[1])
        records.append({
            "kind": kind,
            "scan_id": int(scan_id),
            "points": len(corrected),
            "corrected_i0_points": len(changed),
            "invalid_i0_points": len(invalid),
            "corrected_indices": changed,
            "invalid_indices": invalid,
            "valid_fraction": float(np.isfinite(corrected).mean()),
        })
    normalised = ScanBatch(
        mot_data=batch.mot_data,
        det1d_data=corrected_frames,
        det2d_data=[],
        roi_data=batch.roi_data,
        lambda_data=lambda_data,
        minipix_data=minipix_data,
    )
    return normalised, original_list, corrected_list, records


def _resolve_roi_directory(workspace: WorkspacePaths, config: AnalysisConfig) -> Path:
    directory = workspace.roi
    if (directory / config.roi_filename).is_file() and (
        directory / config.minipix_roi_filename
    ).is_file():
        return directory
    raise FileNotFoundError(
        f"Could not find ROI files {config.roi_filename!r} and "
        f"{config.minipix_roi_filename!r} in {directory}"
    )


def _validate_roi_collection(collection: RoiCollection) -> None:
    for detector, items in (
        ("lambda", collection.rois),
        ("minipix", collection.minipix_rois),
    ):
        names = [item.name for item in items]
        if any(not name for name in names):
            raise ValueError(f"{detector} ROI labels cannot be empty")
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"Duplicate {detector} ROI labels: {duplicates}")


def _roi_entries(collection: RoiCollection):
    return [
        *((f"lambda:{item.name}", "lambda", item) for item in collection.rois),
        *((f"minipix:{item.name}", "minipix", item) for item in collection.minipix_rois),
    ]


def _fit_elastic(
    config: AnalysisConfig,
    energy: np.ndarray,
    collection: RoiCollection,
    lambda_sums: RoiSums,
    minipix_sums: RoiSums,
    lambda_masks: RoiMasks,
    minipix_masks: RoiMasks,
) -> FitSummary:
    spectra = list(lambda_sums.filtered if config.use_filter else lambda_sums.raw)
    spectra += list(minipix_sums.filtered if config.use_filter else minipix_sums.raw)
    pixel_counts = [*lambda_masks.pixel_counts, *minipix_masks.pixel_counts]
    entries = _roi_entries(collection)
    if len(entries) != len(spectra):
        raise ValueError("Elastic spectra and ROI counts differ")

    rows, coefficients, bad = [], [], []
    fit_low = max(float(np.nanmin(energy)), config.elastic_center_range_kev[0])
    fit_high = min(float(np.nanmax(energy)), config.elastic_center_range_kev[1])
    if fit_low >= fit_high:
        raise ValueError("Elastic scan does not overlap elastic_center_range_kev")
    width_upper = max(float(np.nanmax(energy) - np.nanmin(energy)), 1e-5)

    for (roi_id, detector, item), values, pixel_count in zip(entries, spectra, pixel_counts):
        y = np.asarray(values, dtype=float)
        finite = np.isfinite(energy) & np.isfinite(y)
        x_fit, y_fit = energy[finite], y[finite]
        reasons: list[str] = []
        coeff = np.full(4, np.nan)
        r_squared = np.nan
        if pixel_count <= 0:
            reasons.append("empty_mask")
        if x_fit.size < 4:
            reasons.append("insufficient_finite_points")
        elif np.nanmax(y_fit) - np.nanmin(y_fit) <= max(1.0, abs(np.nanmax(y_fit))) * 1e-12:
            reasons.append("zero_or_constant_signal")
        if not reasons:
            background = float(np.nanmin(y_fit))
            amplitude = max(float(np.nanmax(y_fit) - background), np.finfo(float).eps)
            p0 = [amplitude, float(x_fit[np.nanargmax(y_fit)]), 0.001, background]
            p0[1] = float(np.clip(p0[1], fit_low, fit_high))
            p0[2] = float(np.clip(p0[2], 1e-6, width_upper))
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", OptimizeWarning)
                    coeff, _ = curve_fit(
                        math_func.lorentzian,
                        x_fit,
                        y_fit,
                        p0=p0,
                        bounds=(
                            [0.0, fit_low, 1e-6, -np.inf],
                            [np.inf, fit_high, width_upper, np.inf],
                        ),
                        maxfev=20_000,
                    )
                residuals = y_fit - math_func.lorentzian(x_fit, *coeff)
                ss_res = float(np.sum(residuals ** 2))
                ss_tot = float(np.sum((y_fit - np.mean(y_fit)) ** 2))
                r_squared = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
            except (RuntimeError, ValueError, TypeError, OptimizeWarning, FloatingPointError):
                reasons.append("fit_failed")
                coeff = np.full(4, np.nan)

        width_ev = 2 * abs(float(coeff[2])) * 1000 if np.isfinite(coeff[2]) else np.nan
        if not np.all(np.isfinite(coeff)) or not np.isfinite(r_squared):
            if "fit_failed" not in reasons and "zero_or_constant_signal" not in reasons:
                reasons.append("nonfinite_fit")
        else:
            if not config.elastic_center_range_kev[0] <= coeff[1] <= config.elastic_center_range_kev[1]:
                reasons.append("center_out_of_range")
            if width_ev > config.max_fwhm_ev:
                reasons.append("fwhm_above_threshold")
            if r_squared < config.min_r_squared:
                reasons.append("r_squared_below_threshold")
            if coeff[0] <= 0:
                reasons.append("nonpositive_height")
        item.elastic_energy = float(coeff[1]) if np.isfinite(coeff[1]) else np.nan
        if reasons:
            bad.append(roi_id)
        coefficients.append(np.asarray(coeff, dtype=float))
        rows.append({
            "roi_id": roi_id,
            "detector": detector,
            "crystal": item.name,
            "module": (item.name or "").split("-", 1)[0],
            "x1": item.x1,
            "x2": item.x2,
            "y1": item.y1,
            "y2": item.y2,
            "mask_pixels": int(pixel_count),
            "center": coeff[1] * 1000 if np.isfinite(coeff[1]) else np.nan,
            "width": width_ev,
            "height": coeff[0],
            "background": coeff[3],
            "r-square": r_squared,
            "automatic_exclusion_reasons": ";".join(reasons),
        })
    return FitSummary(pd.DataFrame(rows), bad, coefficients)


def _add_q_and_selection(config: AnalysisConfig, prepared_fit: FitSummary, collection, energy):
    table = prepared_fit.table.copy()
    q_columns = {name: [] for name in ("q_ave", "dq_ave", "q_range", "dq_range")}
    selected_modules = set(config.modules)
    configured_exclusions = set(config.excluded_rois)
    entries = _roi_entries(collection)
    for row_index, ((roi_id, _detector, item), row) in enumerate(zip(entries, table.to_dict("records"))):
        reasons = [value for value in str(row["automatic_exclusion_reasons"]).split(";") if value]
        values = (np.nan, np.nan, np.nan, np.nan)
        if np.isfinite(row["center"]):
            try:
                q_values = q_calc(
                    DEFAULT_MODULE_ANGLES,
                    item.name,
                    float(row["center"]) / 1000,
                    np.asarray(energy, dtype=float),
                )
                item.q, item.dq, item.q_ave, item.dq_ave, item.q_range, item.dq_range = q_values
                values = tuple(float(value) for value in q_values[2:])
            except (ValueError, KeyError, FloatingPointError):
                reasons.append("q_calculation_failed")
        module = row["module"]
        if selected_modules and module not in selected_modules:
            reasons.append("module_not_selected")
        if np.isfinite(values[0]) and not config.q_range[0] <= values[0] <= config.q_range[1]:
            reasons.append("q_out_of_range")
        if roi_id in configured_exclusions or item.name in configured_exclusions:
            reasons.append("configured_exclusion")
        for name, value in zip(q_columns, values):
            q_columns[name].append(value)
        table.loc[row_index, "automatic_exclusion_reasons"] = ";".join(dict.fromkeys(reasons))
    for name, values in q_columns.items():
        table[name] = values
    table["automatic_accepted"] = table["automatic_exclusion_reasons"].eq("")
    return FitSummary(
        table,
        table.loc[~table["automatic_accepted"], "roi_id"].tolist(),
        prepared_fit.coefficients,
    )


def prepare_analysis(
    config: AnalysisConfig,
    workspace: WorkspacePaths | None = None,
) -> PreparedAnalysis:
    """Validate inputs and prepare calibrated products for operator QC."""
    config.validate()
    resolved_workspace = load_workspace() if workspace is None else workspace
    raw_directory = resolved_workspace.raw / config.element
    roi_directory = _resolve_roi_directory(resolved_workspace, config)
    files, input_table = _preflight(config, raw_directory)

    elastic_raw = load_scan_batch(
        config.elastic_scan_ids,
        raw_directory,
        monitor_pv=config.elastic_i0_pv,
        divide_by_monitor=False,
        mute=True,
    )
    elastic_batch, elastic_i0_original, elastic_i0_corrected, elastic_records = (
        _normalise_loaded_batch(
            elastic_raw,
            config.elastic_scan_ids,
            config.elastic_i0_pv,
            divide_by_monitor=config.divide_i0_elastic,
            correct_glitches=config.correct_i0_glitches,
            glitch_threshold=config.i0_glitch_threshold,
            kind="elastic",
        )
    )
    elastic_energy_pv = resolve_motor_pv(
        elastic_batch.mot_data,
        config.elastic_energy_pv,
        config.energy_pv_alternatives,
    )
    if elastic_batch.scan_count != 1:
        raise ValueError("Exactly one elastic scan is currently supported")
    elastic_energy = elastic_batch.mot_data[0][elastic_energy_pv].to_numpy(dtype=float)

    offsets = {name: (dx, dy) for name, dx, dy in config.module_offsets}
    roi_collection = load_roi_collection(
        roi_directory,
        config.roi_filename,
        config.minipix_roi_filename,
        module_offsets=offsets,
    )
    _validate_roi_collection(roi_collection)
    lambda_masks = build_roi_masks(
        elastic_batch.lambda_data[0],
        roi_collection.rois,
        filter_value=config.filter_value,
        auto_adjust=config.auto_adjust_rois,
        roi_size=config.roi_size,
        x_shift=roi_collection.x_shift,
        y_shift=roi_collection.y_shift,
        x_expand=roi_collection.x_expand,
        y_expand=roi_collection.y_expand,
    )
    minipix_masks = build_roi_masks(
        elastic_batch.minipix_data[0],
        roi_collection.minipix_rois,
        filter_value=config.filter_value,
        auto_adjust=config.auto_adjust_rois,
        roi_size=config.roi_size,
        x_shift=roi_collection.minipix_x_shift,
        y_shift=roi_collection.minipix_y_shift,
        x_expand=roi_collection.minipix_x_expand,
        y_expand=roi_collection.minipix_y_expand,
    )
    elastic_sums = sum_roi_spectra(
        elastic_batch.lambda_data[0], roi_collection.rois, lambda_masks.masks
    )
    minipix_elastic_sums = sum_roi_spectra(
        elastic_batch.minipix_data[0], roi_collection.minipix_rois, minipix_masks.masks
    )
    fit_summary = _fit_elastic(
        config,
        elastic_energy,
        roi_collection,
        elastic_sums,
        minipix_elastic_sums,
        lambda_masks,
        minipix_masks,
    )

    xrs_raw = load_scan_batch(
        config.xrs_scan_ids,
        raw_directory,
        monitor_pv=config.xrs_i0_pv,
        divide_by_monitor=False,
        mute=True,
    )
    xrs_batch, xrs_i0_original, xrs_i0_corrected, xrs_records = _normalise_loaded_batch(
        xrs_raw,
        config.xrs_scan_ids,
        config.xrs_i0_pv,
        divide_by_monitor=config.divide_i0_xrs,
        correct_glitches=config.correct_i0_glitches,
        glitch_threshold=config.i0_glitch_threshold,
        kind="xrs",
    )
    xrs_energy_pv = resolve_motor_pv(
        xrs_batch.mot_data,
        config.xrs_energy_pv,
        config.energy_pv_alternatives,
    )
    xrs_energy = [frame[xrs_energy_pv].to_numpy(dtype=float) for frame in xrs_batch.mot_data]
    fit_summary = _add_q_and_selection(
        config, fit_summary, roi_collection, xrs_energy[0]
    )
    xrs_sums = sum_roi_spectra_batch(
        xrs_batch.lambda_data, roi_collection.rois, lambda_masks.masks
    )
    minipix_xrs_sums = sum_roi_spectra_batch(
        xrs_batch.minipix_data, roi_collection.minipix_rois, minipix_masks.masks
    )
    scan_qc = pd.DataFrame([*elastic_records, *xrs_records])
    return PreparedAnalysis(
        config=config,
        workspace=resolved_workspace,
        input_files=files,
        input_table=input_table,
        elastic_batch=elastic_batch,
        xrs_batch=xrs_batch,
        elastic_energy=elastic_energy,
        xrs_energy=xrs_energy,
        elastic_i0_original=elastic_i0_original,
        elastic_i0_corrected=elastic_i0_corrected,
        xrs_i0_original=xrs_i0_original,
        xrs_i0_corrected=xrs_i0_corrected,
        scan_qc=scan_qc,
        roi_collection=roi_collection,
        lambda_masks=lambda_masks,
        minipix_masks=minipix_masks,
        elastic_sums=elastic_sums,
        minipix_elastic_sums=minipix_elastic_sums,
        xrs_sums=xrs_sums,
        minipix_xrs_sums=minipix_xrs_sums,
        fit_summary=fit_summary,
    )


def build_qc_report(prepared: PreparedAnalysis) -> QCReport:
    """Build structured scan and ROI diagnostics for notebook review."""
    roi_table = prepared.fit_summary.table.copy()
    scan_table = prepared.scan_qc.copy()
    accepted = int(roi_table["automatic_accepted"].sum())
    summary = {
        "element": prepared.config.element,
        "elastic_scan_ids": list(prepared.config.elastic_scan_ids),
        "xrs_scan_ids": list(prepared.config.xrs_scan_ids),
        "roi_total": len(roi_table),
        "roi_accepted": accepted,
        "roi_excluded": len(roi_table) - accepted,
        "xrs_scan_count": len(prepared.config.xrs_scan_ids),
        "i0_corrected_points": int(scan_table["corrected_i0_points"].sum()),
        "i0_invalid_points": int(scan_table["invalid_i0_points"].sum()),
        "approval_required": True,
    }
    return QCReport(roi_table=roi_table, scan_table=scan_table, summary=summary)


def _matches_exclusion(roi_id: str, crystal: str, exclusions: set[str]) -> bool:
    return roi_id in exclusions or crystal in exclusions


def _interpolate_without_extrapolation(x, y, target):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 2:
        return np.full_like(target, np.nan, dtype=float)
    x, y = x[finite], y[finite]
    order = np.argsort(x)
    x, y = x[order], y[order]
    unique_x, unique_indices = np.unique(x, return_index=True)
    y = y[unique_indices]
    if unique_x.size < 2:
        return np.full_like(target, np.nan, dtype=float)
    return np.interp(target, unique_x, y, left=np.nan, right=np.nan)


def finalize_analysis(
    prepared: PreparedAnalysis,
    approval: QcApproval,
) -> AnalysisResult:
    """Create provisional or approved spectra using coverage-aware averaging."""
    config = prepared.config
    requested_scans = list(config.xrs_scan_ids)
    unknown_scans = sorted(set(approval.excluded_scans) - set(requested_scans))
    if unknown_scans:
        raise ValueError(f"Cannot exclude unrequested XRS scans: {unknown_scans}")
    excluded_scans = set(approval.excluded_scans)
    used_indices = [i for i, scan_id in enumerate(requested_scans) if scan_id not in excluded_scans]
    if not used_indices:
        raise ValueError("All XRS scans were excluded")

    table = prepared.fit_summary.table.copy()
    manual_exclusions = set(approval.excluded_rois)
    final_selected = []
    final_reasons = []
    for row in table.to_dict("records"):
        reasons = [
            value for value in str(row["automatic_exclusion_reasons"]).split(";") if value
        ]
        if _matches_exclusion(row["roi_id"], row["crystal"], manual_exclusions):
            reasons.append("manual_exclusion")
        final_reasons.append(";".join(dict.fromkeys(reasons)))
        final_selected.append(not reasons)
    table["final_exclusion_reasons"] = final_reasons
    table["final_selected"] = final_selected
    selected_indices = np.flatnonzero(table["final_selected"].to_numpy()).tolist()
    if not selected_indices:
        raise ValueError("No ROI passed automatic and manual selection")

    entries = _roi_entries(prepared.roi_collection)
    centers = table["center"].to_numpy(dtype=float)
    roi_bounds = []
    for roi_index in selected_indices:
        lows, highs = [], []
        for scan_index in used_indices:
            energy_transfer = prepared.xrs_energy[scan_index] * 1000 - centers[roi_index]
            finite = energy_transfer[np.isfinite(energy_transfer)]
            if finite.size:
                lows.append(float(finite.min()))
                highs.append(float(finite.max()))
        if not lows:
            raise ValueError(f"No finite XRS energy for ROI {table.loc[roi_index, 'roi_id']}")
        roi_bounds.append((min(lows), max(highs)))
    low = math.ceil(max(value[0] for value in roi_bounds) / config.energy_step_ev) * config.energy_step_ev
    high = math.floor(min(value[1] for value in roi_bounds) / config.energy_step_ev) * config.energy_step_ev
    if high < low:
        raise ValueError(f"No common energy-transfer range: {low} > {high}")
    point_count = int(round((high - low) / config.energy_step_ev)) + 1
    target = low + np.arange(point_count, dtype=float) * config.energy_step_ev

    roi_curves, roi_scan_coverage = [], []
    roi_ids, modules = [], []
    for roi_index in selected_indices:
        roi_id, detector, _item = entries[roi_index]
        curves = []
        for scan_index in used_indices:
            x = prepared.xrs_energy[scan_index] * 1000 - centers[roi_index]
            if detector == "lambda":
                sums = prepared.xrs_sums[scan_index]
                local_index = roi_index
            else:
                sums = prepared.minipix_xrs_sums[scan_index]
                local_index = roi_index - len(prepared.roi_collection.rois)
            values = sums.filtered[local_index] if config.use_filter else sums.raw[local_index]
            curves.append(_interpolate_without_extrapolation(x, values, target))
        curve_stack = np.asarray(curves, dtype=float)
        coverage = np.sum(np.isfinite(curve_stack), axis=0)
        average = np.divide(
            np.nansum(curve_stack, axis=0),
            coverage,
            out=np.full_like(target, np.nan),
            where=coverage > 0,
        )
        roi_curves.append(average)
        roi_scan_coverage.append(coverage)
        roi_ids.append(roi_id)
        modules.append(str(table.loc[roi_index, "module"]))

    roi_matrix = np.asarray(roi_curves, dtype=float)
    coverage_matrix = np.asarray(roi_scan_coverage, dtype=int)
    roi_coverage = np.sum(np.isfinite(roi_matrix), axis=0)
    complete = roi_coverage == len(roi_ids)
    if not np.any(complete):
        raise ValueError("Selected ROIs have no common valid energy-transfer points")
    target = target[complete]
    roi_matrix = roi_matrix[:, complete]
    coverage_matrix = coverage_matrix[:, complete]
    roi_coverage = roi_coverage[complete]
    scan_coverage = np.min(coverage_matrix, axis=0)
    intensity_sum = np.sum(roi_matrix, axis=0)
    intensity_mean = intensity_sum / len(roi_ids)
    roi_spectra = pd.DataFrame({"Energy Transfer (eV)": target})
    for roi_id, curve in zip(roi_ids, roi_matrix):
        roi_spectra[roi_id] = curve
    module_spectra = pd.DataFrame({"Energy Transfer (eV)": target})
    for module in dict.fromkeys(modules):
        indices = [index for index, value in enumerate(modules) if value == module]
        module_spectra[f"{module}_sum"] = roi_matrix[indices].sum(axis=0)
        module_spectra[f"{module}_mean"] = roi_matrix[indices].mean(axis=0)
    return AnalysisResult(
        approved=approval.approved,
        approval_note=approval.note,
        energy_transfer=target,
        intensity_sum=intensity_sum,
        intensity_mean=intensity_mean,
        scan_coverage=scan_coverage,
        roi_coverage=roi_coverage,
        roi_spectra=roi_spectra,
        module_spectra=module_spectra,
        fit_table=table,
        selected_roi_ids=roi_ids,
        used_scan_ids=[requested_scans[index] for index in used_indices],
        excluded_scan_ids=sorted(excluded_scans),
        manually_excluded_rois=sorted(manual_exclusions),
        prepared=prepared,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unique_run_directory(root: Path, config: AnalysisConfig) -> Path:
    scan_label = "-".join(str(value) for value in config.elastic_scan_ids)
    xrs_label = "-".join(str(value) for value in config.xrs_scan_ids)
    base = f"{datetime.now():%Y%m%d-%H%M%S}_el{scan_label}_xrs{xrs_label}"
    candidate = root / config.element / config.analysis_name / base
    suffix = 1
    while candidate.exists():
        candidate = root / config.element / config.analysis_name / f"{base}_{suffix:02d}"
        suffix += 1
    candidate.mkdir(parents=True, exist_ok=False)
    return candidate


def _adjusted_roi_table(prepared: PreparedAnalysis) -> pd.DataFrame:
    tables = []
    for detector, source, items in (
        ("lambda", prepared.roi_collection.roi_table, prepared.roi_collection.rois),
        ("minipix", prepared.roi_collection.minipix_table, prepared.roi_collection.minipix_rois),
    ):
        table = source.copy()
        table.insert(0, "detector", detector)
        table.insert(1, "roi_id", [f"{detector}:{item.name}" for item in items])
        for index, item in enumerate(items):
            for column in ("x1", "x2", "y1", "y2"):
                table.loc[index, column] = getattr(item, column)
        tables.append(table)
    return pd.concat(tables, ignore_index=True, sort=False)


def _write_qc_figures(directory: Path, prepared: PreparedAnalysis, qc: QCReport, result):
    import matplotlib.pyplot as plt

    figures = directory / "figures"
    figures.mkdir()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    groups = (
        (prepared.elastic_i0_original, prepared.elastic_i0_corrected, "Elastic I0"),
        (prepared.xrs_i0_original, prepared.xrs_i0_corrected, "XRS I0"),
    )
    for axis, (originals, corrected, title) in zip(axes, groups):
        for index, (before, after) in enumerate(zip(originals, corrected)):
            axis.plot(before, alpha=0.45, label=f"scan {index} raw")
            axis.plot(after, linewidth=1, label=f"scan {index} corrected")
        axis.set_title(title)
        axis.set_xlabel("Point")
        axis.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(figures / "i0_qc.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    accepted = qc.roi_table["automatic_accepted"]
    axes[0].hist(qc.roi_table.loc[accepted, "r-square"].dropna(), bins=20)
    axes[0].axvline(prepared.config.min_r_squared, color="red", linewidth=1)
    axes[0].set_title("Accepted ROI R-squared")
    axes[1].hist(qc.roi_table.loc[accepted, "width"].dropna(), bins=20)
    axes[1].axvline(prepared.config.max_fwhm_ev, color="red", linewidth=1)
    axes[1].set_title("Accepted ROI FWHM (eV)")
    fig.tight_layout()
    fig.savefig(figures / "elastic_fit_qc.png", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    axes[0].plot(result.energy_transfer, result.intensity_sum)
    axes[0].set_ylabel("Intensity sum")
    axes[1].plot(result.energy_transfer, result.scan_coverage, label="scan coverage")
    axes[1].plot(result.energy_transfer, result.roi_coverage, label="ROI coverage")
    axes[1].set_xlabel("Energy Transfer (eV)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(figures / "spectrum_coverage.png", dpi=160)
    plt.close(fig)


def export_analysis(
    result: AnalysisResult,
    qc: QCReport,
    config: AnalysisConfig,
    workspace: WorkspacePaths | None = None,
) -> Path:
    """Write one immutable, provenance-rich run directory.

    Formal export is rejected unless the operator supplied an approved
    :class:`QcApproval` to :func:`finalize_analysis`.
    """
    if not result.approved:
        raise PermissionError("QC approval is required before formal export")
    prepared = result.prepared
    if config != prepared.config:
        raise ValueError("config does not match the prepared analysis")
    resolved_workspace = prepared.workspace if workspace is None else workspace
    if resolved_workspace != prepared.workspace:
        raise ValueError("workspace does not match the prepared analysis")
    output_path = _unique_run_directory(resolved_workspace.processed, config)

    spectrum = pd.DataFrame({
        "Energy Transfer (eV)": result.energy_transfer,
        "Intensity Sum": result.intensity_sum,
        "Intensity Mean": result.intensity_mean,
        "Scan Coverage": result.scan_coverage,
        "ROI Coverage": result.roi_coverage,
    })
    spectrum.to_csv(output_path / "spectrum.tsv", sep="\t", index=False)
    result.roi_spectra.to_csv(output_path / "roi_spectra.tsv", sep="\t", index=False)
    result.module_spectra.to_csv(output_path / "module_spectra.tsv", sep="\t", index=False)
    result.fit_table.to_csv(output_path / "qc_roi_fits.tsv", sep="\t", index=False)
    qc.scan_table.to_csv(output_path / "qc_scan_summary.tsv", sep="\t", index=False)
    _adjusted_roi_table(prepared).to_csv(
        output_path / "adjusted_rois.tsv", sep="\t", index=False
    )
    (output_path / "config.json").write_text(
        json.dumps(config.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (output_path / "qc_report.json").write_text(
        json.dumps(qc.to_dict(), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    provenance = {
        "created_at": datetime.now().astimezone().isoformat(),
        "workspace": {
            "name": resolved_workspace.name,
            "config_file": str(resolved_workspace.config_file),
            "config_sha256": resolved_workspace.config_sha256,
            "paths": resolved_workspace.as_dict(),
        },
        "python": sys.version,
        "platform": platform.platform(),
        "versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": __import__("scipy").__version__,
            "h5py": h5py.__version__,
        },
        "inputs": {
            key: {"path": str(path.resolve()), "sha256": _sha256(path)}
            for key, path in prepared.input_files.items()
        },
        "used_scan_ids": result.used_scan_ids,
        "excluded_scan_ids": result.excluded_scan_ids,
        "selected_roi_ids": result.selected_roi_ids,
        "automatic_exclusions": prepared.fit_summary.table.loc[
            ~prepared.fit_summary.table["automatic_accepted"],
            ["roi_id", "automatic_exclusion_reasons"],
        ].to_dict(orient="records"),
        "manual_exclusions": result.manually_excluded_rois,
        "approval_note": result.approval_note,
    }
    (output_path / "provenance.json").write_text(
        json.dumps(_jsonable(provenance), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    # One-release compatibility with the legacy filenames.
    legacy = config.analysis_name
    spectrum[["Energy Transfer (eV)", "Intensity Sum"]].rename(
        columns={"Intensity Sum": "Intensity"}
    ).to_csv(output_path / f"{legacy}_data.txt", sep="\t", index=False)
    result.roi_spectra.to_csv(output_path / f"{legacy}_all_data.txt", sep="\t", index=False)
    result.fit_table.to_csv(output_path / f"{legacy}_rois.txt", sep="\t", index=False)
    result.module_spectra.to_csv(output_path / f"{legacy}_modules.txt", sep="\t", index=False)
    (output_path / f"{legacy}_info.txt").write_text(
        "\n".join([
            f"element = {config.element}",
            f"elastic_scan_ids = {list(config.elastic_scan_ids)}",
            f"xrs_scan_ids = {list(config.xrs_scan_ids)}",
            f"selected_roi_ids = {result.selected_roi_ids}",
            f"approval_note = {result.approval_note}",
        ]) + "\n",
        encoding="utf-8",
    )
    _write_qc_figures(output_path, prepared, qc, result)
    return output_path


__all__ = [
    "AnalysisConfig",
    "QcApproval",
    "PreparedAnalysis",
    "QCReport",
    "AnalysisResult",
    "prepare_analysis",
    "build_qc_report",
    "finalize_analysis",
    "export_analysis",
]
