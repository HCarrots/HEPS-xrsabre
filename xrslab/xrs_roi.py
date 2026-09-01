import os
from pathlib import Path
from typing import cast

import h5py
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

try:
    from . import math_func
    from xrsabre.paths import WorkspacePaths, load_workspace
except ImportError:  # pragma: no cover - running as a standalone script
    import math_func
    from xrsabre.paths import WorkspacePaths, load_workspace


def _as_group(obj):
    """Narrow an h5py object to ``Group`` for static type checkers."""
    return cast(h5py.Group, obj)

_HC = 12.398419297617678  # keV*A
_CRYSTAL_DIAMETER = 100
_CRYSTAL_DISTANCE = 1070
_MODULES = ['VB', 'VU', 'VD', 'HB', 'HL', 'HR']
_REVERSES = [-1, -1, -1, 1, 1, -1]
_ANGLE_OFFSET = np.deg2rad(7)
_ROWS = {'A': -2, 'B': -1, 'C': 0, 'D': 1, 'E': 2}
_COLS = {'1': -1, '2': 0, '3': 1}


# ROI functions
class XRSRoi:

    def __init__(self, x1, x2, y1, y2, num, name=None, add=True,
                 q=None, dq=None, q_ave=None, dq_ave=None,
                 q_range=None, dq_range=None, elastic_energy=9.685):
        self.x1 = round(x1)
        self.x2 = round(x2)
        self.y1 = round(y1)
        self.y2 = round(y2)
        self.num = num
        self._refresh_geometry()
        self.name = name
        self.add = add
        self.q = q
        self.dq = dq
        self.q_ave = q_ave
        self.dq_ave = dq_ave
        self.q_range = q_range
        self.dq_range = dq_range
        self.elastic_energy = elastic_energy

    def _refresh_geometry(self):
        """Keep all derived geometry values in sync after a ROI mutation."""
        if self.x2 < self.x1 or self.y2 < self.y1:
            raise ValueError("ROI upper bounds must not be smaller than lower bounds")
        self.x_width = self.x2 - self.x1
        self.y_width = self.y2 - self.y1
        self.x_center = 0.5 * (self.x2 + self.x1)
        self.y_center = 0.5 * (self.y2 + self.y1)
        self.pixel_no = self.x_width * self.y_width

    def x_shift(self, shift):
        self.x1 = round(self.x1 + shift)
        self.x2 = round(self.x2 + shift)
        if self.x1 < 0:
            self.x1 = 0
        self._refresh_geometry()

    def y_shift(self, shift):
        self.y1 = round(self.y1 + shift)
        self.y2 = round(self.y2 + shift)
        if self.y1 < 0:
            self.y1 = 0
        self._refresh_geometry()

    def x_expand(self, expansion):
        self.x1 = round(self.x1 - expansion)
        self.x2 = round(self.x2 + expansion)
        if self.x1 < 0:
            self.x1 = 0
        self._refresh_geometry()

    def y_expand(self, expansion):
        self.y1 = round(self.y1 - expansion)
        self.y2 = round(self.y2 + expansion)
        if self.y1 < 0:
            self.y1 = 0
        self._refresh_geometry()

    def set_x_width(self, val):
        self.x1 = round(self.x_center - val / 2)
        self.x2 = round(self.x_center + val / 2)
        self._refresh_geometry()

    def set_y_width(self, val):
        self.y1 = round(self.y_center - val / 2)
        self.y2 = round(self.y_center + val / 2)
        self._refresh_geometry()


def roi_path(filename, workspace: WorkspacePaths | None = None):
    """Return the full path to a ROI definition file in the ROI directory."""
    resolved_workspace = load_workspace() if workspace is None else workspace
    return str(resolved_workspace.roi / filename)


def q_calc(module_angles, roi_name, ei, ef):
    parts = roi_name.split('-')
    if len(parts) < 2 or len(parts[1]) < 2:
        raise ValueError(f'Invalid ROI name: {roi_name!r}')
    module = parts[0]
    if module not in _MODULES:
        raise ValueError(f'Unknown module: {module!r}')
    if len(module_angles) < len(_MODULES):
        raise ValueError(f'Expected at least {len(_MODULES)} module angles')
    index = _MODULES.index(module)
    module_angles = np.asarray(module_angles, dtype=float)
    theta = (module_angles[index]
             + _ANGLE_OFFSET * _ROWS[parts[1][0]] * _REVERSES[index])
    try:
        delta = _ANGLE_OFFSET * _COLS[parts[1][1]]
    except KeyError as exc:
        raise ValueError(f'Invalid crystal position in ROI name: {roi_name!r}') from exc
    scattering_angle = np.arccos(np.clip(np.cos(theta) * np.cos(delta), -1.0, 1.0))
    ki = 2 * np.pi * ei / _HC
    kf = 2 * np.pi * ef / _HC
    q = np.sqrt(ki**2 + kf**2
                - 2 * ki * kf * np.cos(np.abs(scattering_angle)))
    if np.any(q <= 0):
        raise ValueError('q is zero or negative; check energies and ROI geometry')
    dq = (ki * kf * _CRYSTAL_DIAMETER
          * np.sin(np.abs(scattering_angle)) / q / _CRYSTAL_DISTANCE)
    q_ave = np.mean(q)
    dq_ave = np.mean(dq)
    q_range = np.max(q) - np.min(q)
    dq_range = np.max(dq) - np.min(dq)
    return q, dq, q_ave, dq_ave, q_range, dq_range


# Read the hdf5 file
def read_h5(scan_ids, path, use_roi=False, mute=False):
    if not os.path.isdir(path):
        raise FileNotFoundError(f'Raw scan directory not found: {path}')

    # Build the index once. The previous implementation scanned every
    # directory once per requested scan ID (O(number_of_ids * directories)).
    dir_list = sorted(
        directory for directory in os.listdir(path)
        if os.path.isdir(os.path.join(path, directory))
    )
    scan_dirs_by_id = {}
    for directory in dir_list:
        try:
            scan_id = int(directory.split('_', 1)[0])
        except (ValueError, IndexError):
            continue
        scan_dirs_by_id.setdefault(scan_id, []).append(directory)

    mot_list, det1d_list, det2d_list, roi_list = [], [], [], []
    mot_data_list, det1d_data_list = [], []
    det2d_data_list, roi_data_list = [], []
    scan_list = []

    for scan_id in scan_ids:
        matches = scan_dirs_by_id.get(int(scan_id), [])
        if not matches:
            if not mute:
                print(f'No file for scan {scan_id}!')
            continue
        scan_list.extend(matches)

    count = 0
    for scan in scan_list:
        scan_path = os.path.join(path, scan)
        nxs_files = sorted(
            f for f in os.listdir(scan_path) if f.lower().endswith('.nxs')
        )
        filename = (
            os.path.join(scan_path, nxs_files[0])
            if nxs_files
            else os.path.join(scan_path, scan + '.nxs')
        )
        try:
            f = h5py.File(filename, 'r')
        except OSError:
            continue
        with f:
            entry = _as_group(f['entry'])
            try:
                instrument = _as_group(entry['instrument'])
                mot_names = list(instrument.keys())
            except KeyError:
                instrument = None
                mot_names = []
            try:
                det_names = list(_as_group(entry['data']).keys())
            except KeyError:
                det_names = []
            try:
                roi_names = list(_as_group(entry['roi']).keys())
            except KeyError:
                roi_names = []
            mot_list.append(mot_names)
            roi_list.append(roi_names)
            det1d_names, det2d_names = [], []
            det2d_data = {}
            mot_data_dict, det1d_data_dict, roi_data_dict = {}, {}, {}
            if instrument is not None:
                for mot_name in mot_names:
                    dataset = instrument[mot_name]
                    if isinstance(dataset, h5py.Dataset):
                        mot_data_dict[mot_name] = np.asarray(dataset)
            for det_name in det_names:
                det = np.asarray(entry['data'][det_name])
                if det.ndim == 1:
                    det1d_names.append(det_name)
                    det1d_data_dict[det_name] = det
                elif det.ndim == 3:
                    det2d_names.append(det_name)
                    det2d_data[det_name] = det
            for roi_name in roi_names:
                dataset = entry['roi'][roi_name]
                if isinstance(dataset, h5py.Dataset):
                    roi_data_dict[roi_name] = np.asarray(dataset)
            mot_data = pd.DataFrame(mot_data_dict)
            det1d_data = pd.DataFrame(det1d_data_dict)
            roi_data = pd.DataFrame(roi_data_dict)
            det1d_list.append(det1d_names)
            det2d_list.append(det2d_names)
            mot_data_list.append(mot_data)
            det1d_data_list.append(det1d_data)
            det2d_data_list.append(det2d_data)
            roi_data_list.append(roi_data)
        if not mute:
            scan_parts = scan.split('_')
            print(f'({count}) Scan {scan_parts[0]} - {scan_parts[-1]}')
            print('Motors:', mot_names)
            print('1D detectors:', det1d_names)
            print('2D detectors:')
            for det2d_name in det2d_names:
                print(det2d_name, 'Shape:', det2d_data[det2d_name].shape)
            if use_roi:
                print('Number of ROIs:', len(roi_names))
        count += 1
    # Always return the same 8-tuple; roi_list / roi_data_list are populated
    # regardless of ``use_roi`` (which only controls the printed summary).
    return (mot_list, det1d_list, det2d_list, roi_list,
            mot_data_list, det1d_data_list, det2d_data_list,
            roi_data_list)


def data_fit(fit_type, x_raw, y_raw, p0, get_fwhm=False):
    x_raw = np.asarray(x_raw, dtype=float)
    y_raw = np.asarray(y_raw, dtype=float)
    if x_raw.ndim != 1 or y_raw.ndim != 1 or x_raw.size != y_raw.size:
        raise ValueError('x_raw and y_raw must be one-dimensional arrays of equal length')
    if x_raw.size == 0:
        raise ValueError('Cannot fit empty data')

    finite = np.isfinite(x_raw) & np.isfinite(y_raw)
    if not np.all(finite):
        x_raw, y_raw = x_raw[finite], y_raw[finite]
    if x_raw.size == 0:
        raise ValueError('Cannot fit data containing no finite values')

    p0 = np.asarray(p0, dtype=float)
    try:
        coeff, _ = curve_fit(fit_type, x_raw, y_raw, p0=p0, maxfev=10000)
    except (RuntimeError, ValueError, TypeError):
        coeff = p0
    residuals = y_raw - fit_type(x_raw, *coeff)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_raw - np.mean(y_raw))**2)
    r_squared = 1.0 if ss_res == 0 and ss_tot == 0 else (
        1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    )
    fwhm_raw = 0
    fwhm_fit = 0
    if get_fwhm and x_raw.size >= 3:
        x_interp = np.linspace(np.min(x_raw), np.max(x_raw), 1000)
        y_interp = fit_type(x_interp, *coeff)
        edge_count = min(10, x_raw.size // 2)
        baseline = (np.mean(y_raw[:edge_count]) + np.mean(y_raw[-edge_count:])) / 2
        signal_scale = np.max(np.abs(y_raw))
        if signal_scale and np.abs(baseline) / signal_scale < 0.5:
            try:
                fwhm_raw = math_func.calc_fwhm(x_raw, y_raw)
            except Exception:
                pass
            try:
                fwhm_fit = math_func.calc_fwhm(x_interp, y_interp)
            except Exception:
                pass
        else:
            x_diff = x_raw[1:] - (x_raw[1] - x_raw[0]) / 2
            y_diff = np.diff(y_raw)
            if y_raw[0] > y_raw[-1]:
                y_diff = -y_diff
            try:
                fwhm_raw = math_func.calc_fwhm(x_diff, y_diff)
            except Exception:
                pass
            x_diff_interp = (x_interp[1:]
                             - (x_interp[1] - x_interp[0]) / 2)
            y_diff_interp = np.diff(y_interp)
            if y_interp[0] > y_interp[-1]:
                y_diff_interp = -y_diff_interp
            try:
                fwhm_fit = math_func.calc_fwhm(x_diff_interp, y_diff_interp)
            except Exception:
                pass
    return coeff, r_squared, fwhm_raw, fwhm_fit


def save_data(title_list, data_list, workspace: WorkspacePaths, filename):
    """Save a table below the configured reduced-data directory."""
    if len(title_list) != len(data_list):
        raise ValueError('title_list and data_list must have the same length')
    relative = Path(filename)
    if relative.is_absolute() or not relative.name or ".." in relative.parts:
        raise ValueError('filename must stay within workspace.reduced')
    output = (workspace.reduced / relative).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset = pd.DataFrame({title: data for title, data in zip(title_list, data_list)})
    dataset.to_csv(output, sep='\t', mode='x', index=False)
