"""Reusable XRS analysis building blocks.

The notebook is intentionally kept as a presentation layer.  Data loading,
normalisation, ROI preparation, fitting and interpolation live here so the
same workflow can be used from scripts and tests without relying on notebook
state.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
import pandas as pd
from xrsabre.paths import WorkspacePaths

from . import math_func
from .xrs_roi import XRSRoi, data_fit, q_calc, read_h5

DEFAULT_MODULE_ANGLES = np.deg2rad(
    [145.6, 79.03, 25.0, 118.8, 58.85, 67.862]
)


@dataclass
class ScanBatch:
    """Raw and normalised data belonging to one or more scans."""

    mot_data: list[pd.DataFrame]
    det1d_data: list[pd.DataFrame]
    det2d_data: list[dict[str, np.ndarray]]
    roi_data: list[pd.DataFrame]
    lambda_data: list[np.ndarray]
    minipix_data: list[np.ndarray]

    @property
    def scan_count(self) -> int:
        return len(self.mot_data)


@dataclass
class RoiCollection:
    """ROI objects and their source tables."""

    rois: list[XRSRoi]
    minipix_rois: list[XRSRoi]
    roi_table: pd.DataFrame
    minipix_table: pd.DataFrame
    x_shift: list[float]
    y_shift: list[float]
    x_expand: list[float]
    y_expand: list[float]
    minipix_x_shift: list[float]
    minipix_y_shift: list[float]
    minipix_x_expand: list[float]
    minipix_y_expand: list[float]


@dataclass
class RoiMasks:
    """Masks and diagnostics generated for one detector family."""

    masks: list[np.ndarray]
    stacked_rois: list[np.ndarray]
    pixel_counts: list[int]


@dataclass
class RoiSums:
    """Unfiltered and masked spectra for a detector family."""

    raw: list[np.ndarray]
    filtered: list[np.ndarray]


@dataclass
class FitSummary:
    """Elastic peak fit results and rejected ROI names."""

    table: pd.DataFrame
    bad_rois: list[str]
    coefficients: list[np.ndarray]


@dataclass
class InterpolationResult:
    """Interpolated spectra and their aggregate/per-ROI products."""

    energy_transfer: np.ndarray
    total_intensity: np.ndarray
    per_roi: pd.DataFrame
    selected_names: list[str]
    selected_count: int


def _as_float_array(values: Iterable[float]) -> np.ndarray:
    return np.asarray(values, dtype=float)


def normalize_detector_array(
    detector: np.ndarray,
    monitor: Iterable[float] | None = None,
) -> np.ndarray:
    """Return a detector stack normalised by a monitor along its scan axis.

    Detector stacks are normally ``(scan, height, width)``.  The last-axis
    form is also accepted because some exported NeXus files use it.  Zero
    monitor values produce zero output instead of infinities.
    """
    array = np.asarray(detector)
    if array.ndim != 3:
        raise ValueError(f"Expected a 3D detector stack, got shape {array.shape}")
    result = array.astype(float, copy=False)
    if monitor is None:
        return result

    monitor_array = np.abs(_as_float_array(monitor)).reshape(-1)
    if array.shape[0] == monitor_array.size:
        scale = monitor_array[:, None, None]
    elif array.shape[-1] == monitor_array.size:
        scale = monitor_array[None, None, :]
    else:
        raise ValueError(
            "Monitor length does not match detector scan axis: "
            f"detector shape={array.shape}, monitor length={monitor_array.size}"
        )
    normalised = np.zeros_like(result, dtype=float)
    np.divide(result, scale, out=normalised, where=scale != 0)
    return normalised


def normalize_detector_data(
    detector_data_list: Sequence[dict[str, np.ndarray]],
    monitor_data_list: Sequence[Iterable[float]] | None = None,
    *,
    detector_names: Sequence[str] = ("lambda", "minipix"),
) -> dict[str, list[np.ndarray]]:
    """Normalise matching detector dictionaries and return them by name."""
    if monitor_data_list is not None and len(detector_data_list) != len(monitor_data_list):
        raise ValueError("detector_data_list and monitor_data_list must have equal length")

    output = {name: [] for name in detector_names}
    for index, detector_data in enumerate(detector_data_list):
        monitor = None if monitor_data_list is None else monitor_data_list[index]
        for name in detector_names:
            if name not in detector_data:
                raise KeyError(f"Detector {name!r} is missing in scan {index}")
            output[name].append(normalize_detector_array(detector_data[name], monitor))
    return output


def load_scan_batch(
    scan_ids: Sequence[int],
    path: str | os.PathLike[str],
    *,
    monitor_pv: str = "D_SiC_I_A",
    divide_by_monitor: bool = True,
    mute: bool = False,
) -> ScanBatch:
    """Read scans and return raw metadata plus normalised detector stacks."""
    raw = read_h5(scan_ids, os.fspath(path), use_roi=True, mute=mute)
    (
        _mot_names,
        _det1d_names,
        _det2d_names,
        _roi_names,
        mot_data,
        det1d_data,
        det2d_data,
        roi_data,
    ) = raw
    if not mot_data:
        raise ValueError(f"No scans found for IDs {list(scan_ids)!r} in {path}")

    monitors = []
    if divide_by_monitor:
        for index, data in enumerate(det1d_data):
            if monitor_pv not in data:
                raise KeyError(f"Monitor PV {monitor_pv!r} is missing in scan {index}")
            monitors.append(data[monitor_pv].to_numpy())
    normalised = normalize_detector_data(
        det2d_data,
        monitors if divide_by_monitor else None,
    )
    return ScanBatch(
        mot_data=mot_data,
        det1d_data=det1d_data,
        det2d_data=det2d_data,
        roi_data=roi_data,
        lambda_data=normalised["lambda"],
        minipix_data=normalised["minipix"],
    )


def normalize_scan_batch(
    batch: ScanBatch,
    monitor_pv: str = "D_SiC_I_A",
) -> ScanBatch:
    """Create a normalised copy of a batch after any raw-data corrections."""
    monitors = []
    for index, data in enumerate(batch.det1d_data):
        if monitor_pv not in data:
            raise KeyError(f"Monitor PV {monitor_pv!r} is missing in scan {index}")
        monitors.append(data[monitor_pv].to_numpy())
    normalised = normalize_detector_data(batch.det2d_data, monitors)
    return ScanBatch(
        mot_data=batch.mot_data,
        det1d_data=batch.det1d_data,
        det2d_data=batch.det2d_data,
        roi_data=batch.roi_data,
        lambda_data=normalised["lambda"],
        minipix_data=normalised["minipix"],
    )


def remove_i0_glitches(
    det1d_data_list: Sequence[pd.DataFrame],
    det2d_data_list: Sequence[dict[str, np.ndarray]],
    monitor_pv: str,
    *,
    threshold: float = 0.7,
    detector_names: Sequence[str] = ("lambda", "minipix"),
) -> tuple[list[pd.DataFrame], list[dict[str, np.ndarray]], list[int]]:
    """Correct isolated monitor dips and matching detector frames.

    The input objects are never mutated.  A dip is corrected only when the
    middle point is below ``threshold`` relative to both neighbours.
    """
    if len(det1d_data_list) != len(det2d_data_list):
        raise ValueError("det1d_data_list and det2d_data_list must have equal length")

    corrected_1d, corrected_2d, counts = [], [], []
    for scan_index, (det1d, det2d) in enumerate(zip(det1d_data_list, det2d_data_list)):
        if monitor_pv not in det1d:
            raise KeyError(f"Monitor PV {monitor_pv!r} is missing in scan {scan_index}")
        monitor = det1d[monitor_pv].to_numpy(dtype=float, copy=True)
        detector_copy = {name: np.array(values, copy=True) for name, values in det2d.items()}
        count = 0
        for index in range(max(0, monitor.size - 2)):
            left, middle, right = monitor[index:index + 3]
            if left == 0 or right == 0:
                continue
            if middle / left < threshold and middle / right < threshold:
                monitor[index + 1] = (left + right) / 2
                for name in detector_names:
                    if name in detector_copy:
                        detector_copy[name][index + 1] = (
                            detector_copy[name][index] + detector_copy[name][index + 2]
                        ) / 2
                count += 1
        frame = det1d.copy()
        frame[monitor_pv] = monitor
        corrected_1d.append(frame)
        corrected_2d.append(detector_copy)
        counts.append(count)
    return corrected_1d, corrected_2d, counts


def resolve_motor_pv(
    mot_data_list: Sequence[pd.DataFrame],
    preferred: str,
    alternatives: Sequence[str] = (),
) -> str:
    """Resolve a motor column consistently across a scan batch."""
    candidates = (preferred, *alternatives)
    for candidate in candidates:
        if mot_data_list and all(candidate in frame for frame in mot_data_list):
            return candidate
    available = sorted(set().union(*(frame.columns for frame in mot_data_list)))
    raise KeyError(f"None of {candidates!r} is available; columns={available!r}")


def _read_roi_table(directory: str | os.PathLike[str], filename: str) -> pd.DataFrame:
    path = Path(directory) / filename
    if not path.is_file():
        raise FileNotFoundError(f"ROI file not found: {path}")
    table = pd.read_csv(path, sep="\t")
    if table.columns.size and str(table.columns[0]).startswith("Unnamed"):
        table = table.iloc[:, 1:]
    required = {"roi_label", "x1", "x2", "y1", "y2"}
    missing = required - set(table.columns)
    if missing:
        raise ValueError(f"ROI file {path} is missing columns: {sorted(missing)}")
    return table


def _adjustment_column(table: pd.DataFrame, name: str) -> list[float]:
    if name not in table:
        return [0.0] * len(table)
    return table[name].fillna(0).astype(float).tolist()


def load_roi_collection(
    roi_directory: str | os.PathLike[str],
    roi_filename: str,
    minipix_filename: str,
    *,
    module_offsets: dict[str, tuple[float, float]] | None = None,
) -> RoiCollection:
    """Load standard and MiniPIX ROI files into :class:`XRSRoi` objects."""
    table = _read_roi_table(roi_directory, roi_filename)
    minipix_table = _read_roi_table(roi_directory, minipix_filename)
    if module_offsets is None:
        module_offsets = {"VU": (5, 10), "VD": (-10, 20)}

    def make_rois(source: pd.DataFrame, apply_offsets: bool) -> list[XRSRoi]:
        result = []
        for number, row in enumerate(source.itertuples(index=False)):
            name = str(getattr(row, "roi_label"))
            item = XRSRoi(
                getattr(row, "x1"), getattr(row, "x2"),
                getattr(row, "y1"), getattr(row, "y2"),
                number, name=name,
            )
            if apply_offsets:
                dx, dy = module_offsets.get(name.split("-", 1)[0], (0, 0))
                item.x_shift(dx)
                item.y_shift(dy)
            result.append(item)
        return result

    return RoiCollection(
        rois=make_rois(table, apply_offsets=True),
        minipix_rois=make_rois(minipix_table, apply_offsets=False),
        roi_table=table,
        minipix_table=minipix_table,
        x_shift=_adjustment_column(table, "x_shift"),
        y_shift=_adjustment_column(table, "y_shift"),
        x_expand=_adjustment_column(table, "x_expand"),
        y_expand=_adjustment_column(table, "y_expand"),
        minipix_x_shift=_adjustment_column(minipix_table, "x_shift"),
        minipix_y_shift=_adjustment_column(minipix_table, "y_shift"),
        minipix_x_expand=_adjustment_column(minipix_table, "x_expand"),
        minipix_y_expand=_adjustment_column(minipix_table, "y_expand"),
    )


def _roi_bounds(roi: XRSRoi, shape: tuple[int, int]) -> tuple[slice, slice]:
    height, width = shape
    y1, y2 = max(0, roi.y1), min(height, roi.y2)
    x1, x2 = max(0, roi.x1), min(width, roi.x2)
    return slice(y1, max(y1, y2)), slice(x1, max(x1, x2))


def _make_masks(
    image_stack: np.ndarray,
    rois: Sequence[XRSRoi],
    filter_value: float,
) -> RoiMasks:
    # Invalid monitor points are represented by NaN in the workflow.  They
    # must not poison the accumulated elastic image used for ROI calibration.
    stacked = np.nansum(np.asarray(image_stack), axis=0)
    masks, stacked_rois, pixel_counts = [], [], []
    for item in rois:
        y_slice, x_slice = _roi_bounds(item, stacked.shape)
        roi_image = np.zeros_like(stacked)
        roi_image[y_slice, x_slice] = stacked[y_slice, x_slice]
        maximum = np.nanmax(roi_image) if roi_image.size else 0.0
        mask = np.zeros(stacked.shape, dtype=bool)
        mask[y_slice, x_slice] = roi_image[y_slice, x_slice] > maximum * filter_value
        masks.append(mask)
        stacked_rois.append(roi_image)
        pixel_counts.append(int(mask.sum()))
    return RoiMasks(masks, stacked_rois, pixel_counts)


def _auto_adjust_rois(
    rois: Sequence[XRSRoi],
    stacked_rois: Sequence[np.ndarray],
    *,
    roi_size: int,
    x_shift: Sequence[float],
    y_shift: Sequence[float],
    x_expand: Sequence[float],
    y_expand: Sequence[float],
) -> None:
    for index, item in enumerate(rois):
        image = stacked_rois[index]
        if image.size and np.any(np.isfinite(image)) and np.nanmax(image) > 0:
            peak_y, peak_x = np.unravel_index(np.nanargmax(image), image.shape)
            item.x_shift(peak_x - item.x_center)
            item.y_shift(peak_y - item.y_center)
        item.set_x_width(roi_size)
        item.set_y_width(roi_size)
        item.x_expand(x_expand[index])
        item.y_expand(y_expand[index])
        item.x_shift(x_shift[index])
        item.y_shift(y_shift[index])


def build_roi_masks(
    image_stack: np.ndarray,
    rois: Sequence[XRSRoi],
    *,
    filter_value: float = 0.15,
    auto_adjust: bool = False,
    roi_size: int = 30,
    x_shift: Sequence[float] | None = None,
    y_shift: Sequence[float] | None = None,
    x_expand: Sequence[float] | None = None,
    y_expand: Sequence[float] | None = None,
) -> RoiMasks:
    """Build ROI masks, optionally auto-centering and resizing each ROI."""
    if not 0 <= filter_value <= 1:
        raise ValueError("filter_value must be between 0 and 1")
    if len(rois) == 0:
        return RoiMasks([], [], [])
    initial = _make_masks(image_stack, rois, filter_value)
    if auto_adjust:
        values = (
            x_shift if x_shift is not None else [0] * len(rois),
            y_shift if y_shift is not None else [0] * len(rois),
            x_expand if x_expand is not None else [0] * len(rois),
            y_expand if y_expand is not None else [0] * len(rois),
        )
        if any(len(value) != len(rois) for value in values):
            raise ValueError("ROI adjustment lists must match the ROI count")
        _auto_adjust_rois(
            rois,
            initial.stacked_rois,
            roi_size=roi_size,
            x_shift=values[0],
            y_shift=values[1],
            x_expand=values[2],
            y_expand=values[3],
        )
    return _make_masks(image_stack, rois, filter_value)


def sum_roi_spectra(
    image_stack: np.ndarray,
    rois: Sequence[XRSRoi],
    masks: Sequence[np.ndarray] | None = None,
) -> RoiSums:
    """Sum detector pixels for every ROI without copying the full stack."""
    stack = np.asarray(image_stack)
    if stack.ndim != 3:
        raise ValueError(f"Expected image_stack with shape (scan, y, x), got {stack.shape}")
    if masks is not None and len(masks) != len(rois):
        raise ValueError("masks and rois must have equal length")

    raw, filtered = [], []
    for index, item in enumerate(rois):
        y_slice, x_slice = _roi_bounds(item, stack.shape[1:])
        region = stack[:, y_slice, x_slice]
        raw_sum = region.sum(axis=(1, 2))
        raw.append(raw_sum)
        if masks is None:
            filtered.append(raw_sum.copy())
        else:
            local_mask = masks[index][y_slice, x_slice]
            filtered.append((region * local_mask).sum(axis=(1, 2)))
    return RoiSums(raw, filtered)


def sum_roi_spectra_batch(
    image_stacks: Sequence[np.ndarray],
    rois: Sequence[XRSRoi],
    masks: Sequence[np.ndarray] | None = None,
) -> list[RoiSums]:
    """Apply :func:`sum_roi_spectra` to a sequence of scans."""
    return [sum_roi_spectra(stack, rois, masks) for stack in image_stacks]


def fit_elastic_rois(
    energy: Iterable[float],
    rois: Sequence[XRSRoi],
    roi_sums: RoiSums,
    minipix_rois: Sequence[XRSRoi],
    minipix_sums: RoiSums,
    *,
    fit_type: Callable = math_func.lorentzian,
    use_filter: bool = True,
    energy_range: tuple[float, float] | None = None,
    width_threshold_ev: float | None = None,
    r_squared_threshold: float | None = None,
) -> FitSummary:
    """Fit all elastic ROI spectra and return a single stable result table."""
    x = _as_float_array(energy)
    all_rois = list(rois) + list(minipix_rois)
    all_sums = list(roi_sums.filtered if use_filter else roi_sums.raw)
    all_sums += list(minipix_sums.filtered if use_filter else minipix_sums.raw)
    if len(all_rois) != len(all_sums):
        raise ValueError("ROI and spectrum counts do not match")

    coefficients, rows, bad = [], [], []
    for item, y_values in zip(all_rois, all_sums):
        y = _as_float_array(y_values)
        if x.size != y.size:
            raise ValueError(f"Energy and spectrum lengths differ for ROI {item.name!r}")
        p0 = [float(np.nanmax(y)), float(x[np.nanargmax(y)]), 0.001, 0.0]
        coeff, r_squared, _, _ = data_fit(fit_type, x, y, p0)
        coeff = np.asarray(coeff, dtype=float)
        item.elastic_energy = float(coeff[1])
        width_ev = 2 * abs(float(coeff[2])) * 1000
        if (
            (r_squared_threshold is not None and r_squared < r_squared_threshold)
            or (energy_range is not None and not energy_range[0] <= coeff[1] <= energy_range[1])
            or coeff[0] < 0
            or (width_threshold_ev is not None and width_ev > width_threshold_ev)
        ):
            bad.append(item.name)
        coefficients.append(coeff)
        rows.append({
            "crystal": item.name,
            "x1": item.x1,
            "x2": item.x2,
            "y1": item.y1,
            "y2": item.y2,
            "center": coeff[1] * 1000,
            "width": width_ev,
            "height": coeff[0],
            "background": coeff[3],
            "r-square": r_squared,
        })
    return FitSummary(pd.DataFrame(rows), bad, coefficients)


def add_q_columns(
    fit_result: pd.DataFrame,
    rois: Sequence[XRSRoi],
    minipix_rois: Sequence[XRSRoi],
    energy: Iterable[float],
    module_angles: Iterable[float] = DEFAULT_MODULE_ANGLES,
) -> pd.DataFrame:
    """Calculate q metadata and append it to a fit result table."""
    energy_array = _as_float_array(energy)
    module_angles = _as_float_array(module_angles)
    for item in (*rois, *minipix_rois):
        item.q, item.dq, item.q_ave, item.dq_ave, item.q_range, item.dq_range = q_calc(
            module_angles, item.name, item.elastic_energy, energy_array
        )
    result = fit_result.copy()
    all_rois = [*rois, *minipix_rois]
    result["q_ave"] = [item.q_ave for item in all_rois]
    result["dq_ave"] = [item.dq_ave for item in all_rois]
    result["q_range"] = [item.q_range for item in all_rois]
    result["dq_range"] = [item.dq_range for item in all_rois]
    return result


def configure_roi_selection(
    rois: Sequence[XRSRoi],
    minipix_rois: Sequence[XRSRoi],
    *,
    modules: Iterable[str] | None = None,
    q_range: tuple[float, float] | None = None,
    excluded: Iterable[str] = (),
    bad_fit_names: Iterable[str] = (),
) -> list[XRSRoi]:
    """Set ``roi.add`` consistently and return the selected ROI objects."""
    module_set = set(modules or ())
    excluded_names = set(excluded) | set(bad_fit_names)
    selected = []
    for item in (*rois, *minipix_rois):
        module = (item.name or "").split("-", 1)[0]
        q_value = item.q_ave
        in_q_range = (
            q_range is None
            or (q_value is not None and q_range[0] <= q_value <= q_range[1])
        )
        item.add = (
            item.name not in excluded_names
            and (not module_set or module in module_set)
            and in_q_range
        )
        if item.add:
            selected.append(item)
    return selected


def _interpolate(x: Iterable[float], y: Iterable[float], target: np.ndarray) -> np.ndarray:
    x_array = _as_float_array(x)
    y_array = _as_float_array(y)
    order = np.argsort(x_array)
    return np.interp(target, x_array[order], y_array[order])


def interpolate_and_sum(
    energy_list: Sequence[Iterable[float]],
    roi_sums_list: Sequence[RoiSums],
    minipix_sums_list: Sequence[RoiSums],
    rois: Sequence[XRSRoi],
    minipix_rois: Sequence[XRSRoi],
    fit_result: pd.DataFrame,
    *,
    step_ev: float = 0.2,
    use_filter: bool = True,
) -> InterpolationResult:
    """Align elastic-shifted spectra on a common energy-transfer grid."""
    if step_ev <= 0:
        raise ValueError("step_ev must be positive")
    if len(energy_list) != len(roi_sums_list) or len(energy_list) != len(minipix_sums_list):
        raise ValueError("Energy and ROI sum batches must have equal length")
    all_rois = [*rois, *minipix_rois]
    if len(fit_result) != len(all_rois):
        raise ValueError("fit_result and ROI counts do not match")
    centers = fit_result["center"].to_numpy(dtype=float)
    entries = []
    lower, upper = [], []
    for index, item in enumerate(all_rois):
        for scan_index, energy in enumerate(energy_list):
            x = _as_float_array(energy) * 1000 - centers[index]
            lower.append(np.nanmin(x))
            upper.append(np.nanmax(x))
            if index < len(rois):
                sums = roi_sums_list[scan_index]
                values = sums.filtered[index] if use_filter else sums.raw[index]
            else:
                sums = minipix_sums_list[scan_index]
                values = sums.filtered[index - len(rois)] if use_filter else sums.raw[index - len(rois)]
            entries.append((scan_index, item, x, values))
    if not entries:
        raise ValueError("No ROI spectra available for interpolation")

    low = round(max(lower) / step_ev) * step_ev
    high = round(min(upper) / step_ev) * step_ev
    if high < low:
        raise ValueError(f"No common interpolation range: {low} > {high}")
    count = int(round((high - low) / step_ev))
    target = np.linspace(low, high, count + 1)
    total = np.zeros_like(target)
    per_roi = {item.name: np.zeros_like(target) for item in all_rois}
    selected_names = []
    for scan_index, item, x, values in entries:
        curve = _interpolate(x, values, target)
        if item.add:
            total += curve
            per_roi[item.name] += curve
            if item.name not in selected_names:
                selected_names.append(item.name)
    table = pd.DataFrame({"Energy Transfer (eV)": target, **per_roi})
    selected_count = sum(item.add for item in all_rois) * len(energy_list)
    return InterpolationResult(target, total, table, selected_names, selected_count)


def save_interpolation(
    workspace: WorkspacePaths,
    filename: str,
    result: InterpolationResult,
    fit_result: pd.DataFrame,
    *,
    info: dict[str, object] | None = None,
    save_individual: bool = False,
) -> Path:
    """Save legacy interpolation products below the configured reduced root."""
    relative = Path(filename)
    if relative.is_absolute() or len(relative.parts) != 1 or relative.name in {".", ".."}:
        raise ValueError("filename must be one relative directory name")
    output_path = workspace.reduced / relative
    output_path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({
        "Energy Transfer (eV)": result.energy_transfer,
        "Intensity": result.total_intensity,
    }).to_csv(output_path / f"{filename}_data.txt", sep="\t", index=False)
    result.per_roi.to_csv(output_path / f"{filename}_all_data.txt", sep="\t", index=False)
    fit_result.to_csv(output_path / f"{filename}_rois.txt", sep="\t", index=False)
    if save_individual:
        for roi_name in result.per_roi.columns:
            if roi_name == "Energy Transfer (eV)":
                continue
            safe_name = str(roi_name).replace("/", "_").replace("\\", "_")
            result.per_roi[["Energy Transfer (eV)", roi_name]].rename(
                columns={roi_name: "Intensity"}
            ).to_csv(output_path / f"{safe_name}_data.txt", sep="\t", index=False)
    if info is not None:
        with (output_path / f"{filename}_info.txt").open("w", encoding="utf-8") as stream:
            for key, value in info.items():
                stream.write(f"{key} = {value}\n")
    return output_path
