from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pytest

from xrsabre.paths import initialize_workspace
from xrslab import math_func
from xrslab.workflow import (
    AnalysisConfig,
    QcApproval,
    _correct_monitor,
    _interpolate_without_extrapolation,
    build_qc_report,
    export_analysis,
    finalize_analysis,
    prepare_analysis,
)


def _write_scan(root, scan_id, energy, monitor, lambda_stack, minipix_stack):
    directory = root / f"{scan_id}_scan_test"
    directory.mkdir(parents=True)
    path = directory / f"{scan_id}_scan_test.nxs"
    with h5py.File(path, "w") as handle:
        entry = handle.create_group("entry")
        instrument = entry.create_group("instrument")
        instrument.create_dataset("M_DCM_B5Energy_readback", data=energy)
        data = entry.create_group("data")
        data.create_dataset("D_SiC_I_A", data=monitor)
        data.create_dataset("lambda", data=lambda_stack)
        data.create_dataset("minipix", data=minipix_stack)
    return path


def _roi_table(names, coordinates):
    rows = []
    for name, (x1, x2, y1, y2) in zip(names, coordinates):
        rows.append({
            "roi_label": name,
            "x1": x1,
            "x2": x2,
            "y1": y1,
            "y2": y2,
            "x_shift": 0,
            "y_shift": 0,
            "x_expand": 0,
            "y_expand": 0,
        })
    return pd.DataFrame(rows)


def _synthetic_project(tmp_path, xrs_scan_ids=(59,)):
    workspace = initialize_workspace(tmp_path, name="synthetic")
    raw = workspace.raw / "Ho"
    processed = workspace.processed
    rois = workspace.roi
    raw.mkdir(parents=True)

    elastic_energy = np.linspace(9.67, 9.69, 41)
    elastic_i0 = np.full(elastic_energy.size, 2.0)
    peak = math_func.lorentzian(elastic_energy, 100.0, 9.68, 0.0005, 2.0)
    elastic_lambda = np.zeros((elastic_energy.size, 8, 8), dtype=float)
    elastic_minipix = np.zeros_like(elastic_lambda)
    elastic_lambda[:, 1:3, 1:3] = (peak * elastic_i0 / 4)[:, None, None]
    elastic_minipix[:, 1:3, 1:3] = (peak * elastic_i0 / 4)[:, None, None]
    _write_scan(raw, 57, elastic_energy, elastic_i0, elastic_lambda, elastic_minipix)

    xrs_energy = np.linspace(9.70, 9.74, 9)
    xrs_i0 = np.full(xrs_energy.size, 2.0)
    signal = np.linspace(1.0, 9.0, xrs_energy.size)
    xrs_lambda = np.zeros((xrs_energy.size, 8, 8), dtype=float)
    xrs_minipix = np.zeros_like(xrs_lambda)
    xrs_lambda[:, 1:3, 1:3] = (signal * xrs_i0 / 4)[:, None, None]
    xrs_minipix[:, 1:3, 1:3] = (signal * xrs_i0 / 4)[:, None, None]
    for scan_id in xrs_scan_ids:
        _write_scan(raw, scan_id, xrs_energy, xrs_i0, xrs_lambda, xrs_minipix)

    _roi_table(
        ["VB-A1", "VU-E1"],
        [(1, 3, 1, 3), (4, 6, 4, 6)],
    ).to_csv(rois / "lambda.tsv", sep="\t", index=False)
    _roi_table(["VD-A1"], [(1, 3, 1, 3)]).to_csv(
        rois / "minipix.tsv", sep="\t", index=False
    )
    config = AnalysisConfig(
        elastic_scan_ids=(57,),
        xrs_scan_ids=tuple(xrs_scan_ids),
        roi_filename="lambda.tsv",
        minipix_roi_filename="minipix.tsv",
        auto_adjust_rois=False,
        filter_value=0.1,
        max_fwhm_ev=5.0,
        min_r_squared=0.8,
        energy_step_ev=5.0,
        module_offsets=(),
    )
    return config, workspace


def test_monitor_correction_preserves_invalid_values_and_reports_indices():
    original, corrected, changed, invalid = _correct_monitor(
        [10.0, 1.0, 10.0, 0.0], 0.7, True
    )
    assert original.tolist() == [10.0, 1.0, 10.0, 0.0]
    assert corrected[:3].tolist() == [10.0, 10.0, 10.0]
    assert np.isnan(corrected[3])
    assert changed == [1]
    assert invalid == [3]


def test_interpolation_never_extrapolates():
    result = _interpolate_without_extrapolation(
        [1.0, 2.0], [10.0, 20.0], np.array([0.0, 1.5, 3.0])
    )
    assert np.isnan(result[0])
    assert result[1] == 15.0
    assert np.isnan(result[2])


def test_workflow_auto_excludes_bad_roi_and_requires_approval(tmp_path):
    config, workspace = _synthetic_project(tmp_path)
    prepared = prepare_analysis(config, workspace)
    assert prepared.workspace is workspace
    qc = build_qc_report(prepared)
    bad = qc.roi_table.set_index("roi_id")
    assert not bool(bad.loc["lambda:VU-E1", "automatic_accepted"])
    assert "empty_mask" in bad.loc["lambda:VU-E1", "automatic_exclusion_reasons"]

    provisional = finalize_analysis(prepared, QcApproval(approved=False))
    assert provisional.selected_roi_ids == ["lambda:VB-A1", "minipix:VD-A1"]
    assert np.isfinite(provisional.intensity_sum).all()
    with pytest.raises(PermissionError, match="approval"):
        export_analysis(provisional, qc, config, workspace)

    approved = finalize_analysis(
        prepared, QcApproval(approved=True, note="synthetic QC reviewed")
    )
    output = export_analysis(approved, qc, config, workspace)
    assert (output / "spectrum.tsv").is_file()
    assert (output / "provenance.json").is_file()
    assert (output / "figures" / "spectrum_coverage.png").is_file()
    provenance = __import__("json").loads((output / "provenance.json").read_text())
    assert provenance["workspace"]["name"] == "synthetic"
    assert provenance["workspace"]["config_sha256"] == workspace.config_sha256
    assert provenance["workspace"]["paths"]["processed"] == str(workspace.processed)


def test_repeated_identical_scans_do_not_change_intensity_scale(tmp_path):
    one_root = tmp_path / "one"
    two_root = tmp_path / "two"
    one_config, one_paths = _synthetic_project(one_root, (59,))
    two_config, two_paths = _synthetic_project(two_root, (59, 60))
    one = finalize_analysis(
        prepare_analysis(one_config, one_paths), QcApproval(approved=False)
    )
    two = finalize_analysis(
        prepare_analysis(two_config, two_paths), QcApproval(approved=False)
    )
    assert np.array_equal(one.energy_transfer, two.energy_transfer)
    assert np.allclose(one.intensity_sum, two.intensity_sum)
    assert np.all(two.scan_coverage == 2)
