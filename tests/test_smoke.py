"""Smoke tests for XRSlab.

These tests are intentionally fast and verify that the package imports, the
path machinery resolves/overrides correctly, and the core numerical helpers
behave as expected. The optional real-data test is skipped when no raw data
is available.
"""

from pathlib import Path

import h5py
import numpy as np
import pytest

import xrslab
from xrsabre.paths import LegacyEnvironmentError, initialize_workspace, load_workspace
from xrslab import math_func, xrs_roi as roi
from xrsana.xrs_prediction import (
    analyzer,
    get_all_input_from_text,
    installation_dir,
    save_prediction_inp,
)


# --- package / version ------------------------------------------------------

def test_import_and_version():
    assert isinstance(xrslab.__version__, str)
    assert xrslab.__version__.count(".") >= 1


# --- path resolution --------------------------------------------------------

def test_project_root_is_detected():
    workspace = load_workspace()
    assert workspace.config_file.name == "xrsabre.toml"
    assert workspace.root == workspace.config_file.parent


def test_default_dirs_are_under_project_root():
    workspace = load_workspace()
    assert workspace.raw.is_relative_to(workspace.root)
    assert workspace.processed.is_relative_to(workspace.root)
    assert workspace.roi.is_relative_to(workspace.root)


def test_legacy_env_is_rejected(monkeypatch, tmp_path):
    monkeypatch.setenv("XRSLAB_RAW_DIR", str(tmp_path / "raw"))
    with pytest.raises(LegacyEnvironmentError, match="paths.raw"):
        load_workspace()


def test_packaged_scientific_resources_cannot_be_overridden(tmp_path):
    parsed = get_all_input_from_text(
        "#### analyzer\ndatabase_dir = '/untrusted/custom/resources'\n"
    )
    assert parsed["analyzer"]["database_dir"] == installation_dir
    assert analyzer(database_dir="/untrusted/custom/resources").database_dir == installation_dir
    saved = save_prediction_inp({}, "prediction.inp", tmp_path)
    assert "database_dir" not in Path(saved).read_text(encoding="utf-8")


# --- math functions ---------------------------------------------------------

@pytest.mark.parametrize("func", [
    math_func.gauss,
    math_func.lorentzian,
    math_func.pseudovoigt,
    math_func.gauss2,
    math_func.lorentzian2,
    math_func.pseudovoigt2,
])
def test_peak_models_return_arrays(func):
    x = np.linspace(9.0, 10.0, 100)
    if func in (math_func.pseudovoigt, math_func.pseudovoigt2):
        params = (0.5, 1.0, 9.5, 0.01, 0.0)
    else:
        params = (1.0, 9.5, 0.01, 0.0)
    y = func(x, *params)
    assert y.shape == x.shape
    assert np.all(np.isfinite(y))


def test_calc_fwhm():
    x = np.linspace(0, 10, 1001)
    y = math_func.gauss(x, 1.0, 5.0, 0.5, 0.0)
    fwhm = math_func.calc_fwhm(x, y)
    # FWHM = 2.3548 * sigma
    assert fwhm == pytest.approx(2.3548 * 0.5, rel=0.05)


def test_calc_fwhm_interpolates_crossings():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = np.array([0.0, 0.75, 1.0, 0.0])
    assert math_func.calc_fwhm(x, y) == pytest.approx(1.8333333)


# --- ROI --------------------------------------------------------------------

def test_roi_geometry():
    r = roi.XRSRoi(10, 20, 5, 15, 0, name="VB-C2")
    assert r.x_width == 10 and r.y_width == 10
    assert r.x_center == 15 and r.y_center == 10
    r.x_shift(5)
    assert r.x1 == 15 and r.x2 == 25
    r.set_x_width(20)
    assert r.x_width == 20
    r.x_expand(2)
    assert r.x_width == 24
    assert r.x_center == pytest.approx(20)


def test_roi_rejects_inverted_bounds():
    with pytest.raises(ValueError, match="upper bounds"):
        roi.XRSRoi(20, 10, 5, 15, 0)


def test_q_calc_returns_positive_scalars():
    angles = [np.deg2rad(a) for a in (145.6, 79.03, 25.0, 118.8, 58.85, 67.862)]
    q, dq, q_ave, dq_ave, q_range, dq_range = roi.q_calc(angles, "VB-C2", 9.685, 9.8)
    assert float(q) > 0
    assert float(dq) > 0
    assert float(q_ave) == pytest.approx(float(q))


def test_roi_path_helper():
    workspace = load_workspace()
    assert roi.roi_path("roi_60crystals-v3.txt", workspace) == str(
        workspace.roi / "roi_60crystals-v3.txt"
    )


def test_data_fit_recovers_gauss_params():
    x = np.linspace(9.0, 10.0, 400)
    y = math_func.gauss(x, 5.0, 9.5, 0.01, 1.0)
    coeff, rsq, fwhm_raw, fwhm_fit = roi.data_fit(
        math_func.gauss, x, y, p0=[5.0, 9.5, 0.01, 1.0], get_fwhm=True
    )
    assert coeff[1] == pytest.approx(9.5, abs=0.001)
    assert coeff[0] == pytest.approx(5.0, rel=0.05)
    assert rsq > 0.99


def test_data_fit_constant_signal_has_finite_r_squared():
    x = np.linspace(0, 1, 10)
    coeff, rsq, _, _ = roi.data_fit(
        math_func.linear, x, np.full_like(x, 2.0), p0=[0.0, 2.0]
    )
    assert np.all(np.isfinite(coeff))
    assert np.isfinite(rsq)


def test_save_data_writes_without_index(tmp_path):
    workspace = initialize_workspace(tmp_path, name="save-data")
    roi.save_data(["x", "y"], [[1, 2], [3, 4]], workspace, "exports/data.tsv")
    lines = (workspace.reduced / "exports" / "data.tsv").read_text().splitlines()
    assert lines == ["x\ty", "1\t3", "2\t4"]


def test_read_h5_indexes_scan_directories(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    for scan_id in (10, 2):
        dirname = f"{scan_id}_scan_20260425164344_1_{scan_id}_scan"
        scan_dir = raw / dirname
        scan_dir.mkdir()
        with h5py.File(scan_dir / "data.nxs", "w") as f:
            entry = f.create_group("entry")
            instrument = entry.create_group("instrument")
            instrument.create_dataset("energy", data=np.array([9.0, 9.1]))
            data = entry.create_group("data")
            data.create_dataset("monitor", data=np.array([1.0, 2.0]))
            data.create_dataset("lambda", data=np.ones((2, 2, 2)))
            entry.create_group("roi").create_dataset("r1", data=np.ones(2))

    result = roi.read_h5([2, 10], str(raw), mute=True)
    assert [frame["energy"].iloc[0] for frame in result[4]] == [9.0, 9.0]
    assert result[1] == [["monitor"], ["monitor"]]
    assert result[2] == [["lambda"], ["lambda"]]
    assert result[7][0]["r1"].iloc[0] == 1
