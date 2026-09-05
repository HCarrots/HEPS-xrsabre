import json

import numpy as np
import pytest

from xrs_compton_extraction.module_averages import (
    average_target_exports,
    export_target_averages,
)


def prepare(tmp_path):
    records = []
    statuses = ["crystal,status,reason"]
    for name, module, values, scale in (
        ("A1", "A", [2, 4, -2], 2),
        ("A2", "A", [4, 6, -4], None),
        ("B1", "B", [9, 12, 15], None),
    ):
        records.append({"crystal": name, "module": module, "all_zero": False})
        statuses.append(f"{name},exploratory,")
        response = np.array(values)*(scale or 1)
        unit = "per_eV" if scale else "au"
        np.savetxt(tmp_path / f"{name}.csv", np.column_stack(([111, 112, 113], response, [1, 1, 1])),
                   delimiter=",", comments="", header=f"energy_loss_eV,residual_{unit},available")
        metadata = {"model": "hf_target_preserving" if scale else "pearson",
                    "parameters": {"raw_to_hf_scale": scale}}
        (tmp_path / f"{name}.json").write_text(json.dumps(metadata))
    records.append({"crystal": "B0", "module": "B", "all_zero": True})
    statuses.append("B0,excluded,zero")
    (tmp_path / "channel-status.csv").write_text("\n".join(statuses))
    return records


def test_unit_conversion_and_two_weightings(tmp_path):
    result = average_target_exports(tmp_path, prepare(tmp_path), target_window=(111, 113))
    np.testing.assert_allclose(result["modules"]["A"]["mean"], [3, 5, -3])
    np.testing.assert_allclose(result["module_equal_mean"], [6, 8.5, 6])
    np.testing.assert_allclose(result["crystal_equal_mean"], [5, 22/3, 3])
    assert result["metadata"]["excluded"][0]["crystal"] == "B0"
    export_target_averages(result, tmp_path / "averages")
    assert (tmp_path / "averages/all-modules-target-mean.csv").exists()


def test_unavailable_points_not_zero_filled(tmp_path):
    records = prepare(tmp_path)
    path = tmp_path / "B1.csv"
    path.write_text("energy_loss_eV,residual_au,available\n111,9,0\n112,12,1\n113,15,1\n")
    result = average_target_exports(tmp_path, records, target_window=(111, 113))
    assert result["n_modules"][0] == 1
    assert result["n_crystals"][0] == 2
    assert result["module_equal_mean"][0] == 3


def test_unknown_scale_or_grid_rejected(tmp_path):
    records = prepare(tmp_path)
    p = tmp_path / "A1.json"
    metadata = json.loads(p.read_text())
    metadata["parameters"]["raw_to_hf_scale"] = 0
    p.write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="scale"):
        average_target_exports(tmp_path, records, target_window=(111, 113))
    prepare(tmp_path)
    p = tmp_path / "B1.csv"
    p.write_text(p.read_text().replace("1.120000000000000000e+02", "112.1"))
    with pytest.raises(ValueError, match="grids"):
        average_target_exports(tmp_path, records, target_window=(111, 113))
