from dataclasses import replace

import matplotlib
import numpy as np
import pandas as pd
import pytest

from xrsabre.paths import initialize_workspace
from xrslab.analysis import load_roi_collection
from xrslab.roi_editor import ROI_COLUMNS, RoiEditor
from xrslab.workflow import AnalysisConfig


matplotlib.use("Agg", force=True)


def _table(*rows):
    records = []
    for name, x1, x2, y1, y2 in rows:
        records.append({
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
    return pd.DataFrame(records, columns=ROI_COLUMNS)


@pytest.fixture
def editor(tmp_path):
    workspace = initialize_workspace(tmp_path, name="roi-editor")
    config = AnalysisConfig(
        roi_filename="lambda_source.txt",
        minipix_roi_filename="minipix_source.txt",
    )
    tables = {
        "lambda": _table(("VB-A1", 1, 4, 2, 6)),
        "minipix": _table(("VB-A1", 0, 2, 0, 2)),
    }
    for detector, filename in (
        ("lambda", config.roi_filename),
        ("minipix", config.minipix_roi_filename),
    ):
        tables[detector].to_csv(workspace.roi / filename, sep="\t", index=False)
    return RoiEditor(
        config,
        workspace,
        {"lambda": np.zeros((10, 12)), "minipix": np.zeros((6, 8))},
        tables,
    )


def test_add_move_resize_delete_clear_undo_redo_and_reset(editor):
    assert editor.roi_directory == editor.workspace.roi
    added = editor.add_roi("lambda", "VU-B2", 10.8, 20.2, -3, 7.1)
    assert (added.x1, added.x2, added.y1, added.y2) == (10, 12, 0, 8)

    moved = editor.update_roi("lambda", "VU-B2", 4.2, 9.0, 1.1, 5.0)
    assert (moved.x1, moved.x2, moved.y1, moved.y2) == (4, 9, 1, 5)
    assert editor.undo()
    assert editor.boxes("lambda")[-1] == added
    assert editor.redo()
    assert editor.boxes("lambda")[-1] == moved

    editor.delete_roi("lambda", "VU-B2")
    assert [box.name for box in editor.boxes("lambda")] == ["VB-A1"]
    assert editor.undo()
    assert [box.name for box in editor.boxes("lambda")] == ["VB-A1", "VU-B2"]

    editor.clear_detector("lambda")
    assert editor.boxes("lambda") == ()
    editor.reset()
    assert editor.boxes("lambda")[0].name == "VB-A1"
    assert editor.dirty is False


def test_name_and_area_validation_is_per_detector(editor):
    with pytest.raises(ValueError, match="Duplicate lambda"):
        editor.add_roi("lambda", "VB-A1", 5, 7, 5, 7)
    editor.add_roi("minipix", "VU-B2", 1, 3, 1, 3)
    editor.add_roi("lambda", "VU-B2", 5, 7, 5, 7)

    with pytest.raises(ValueError, match="Invalid ROI label"):
        editor.add_roi("lambda", "not-an-roi", 1, 2, 1, 2)
    with pytest.raises(ValueError, match="1x1"):
        editor.add_roi("lambda", "VD-C3", -4, -1, 1, 2)


def test_to_tables_uses_integer_half_open_schema(editor):
    editor.update_roi("lambda", "VB-A1", 1.2, 4.1, 2.8, 6.2)
    table = editor.to_tables()["lambda"]
    assert list(table.columns) == list(ROI_COLUMNS)
    assert table.loc[0, ["x1", "x2", "y1", "y2"]].tolist() == [1, 5, 2, 7]
    assert table.loc[0, ["x_shift", "y_shift", "x_expand", "y_expand"]].tolist() == [0, 0, 0, 0]


def test_versioned_save_preserves_sources_and_disables_adjustment(editor):
    source_text = (editor.roi_directory / editor.config.roi_filename).read_text()
    editor.update_roi("lambda", "VB-A1", 3, 8, 4, 9)
    result = editor.save()

    assert result.counts == {"lambda": 1, "minipix": 1}
    assert all(path.is_file() for path in result.saved_paths.values())
    assert all("_manual_" in name for name in result.filenames.values())
    assert (editor.roi_directory / editor.config.roi_filename).read_text() == source_text
    assert result.config.auto_adjust_rois is False
    assert result.config.module_offsets == ()
    assert editor.last_result is result
    assert editor.dirty is False
    assert editor.can_undo is False

    loaded = load_roi_collection(
        editor.roi_directory,
        result.config.roi_filename,
        result.config.minipix_roi_filename,
        module_offsets=dict(result.config.module_offsets),
    )
    assert (loaded.rois[0].x1, loaded.rois[0].x2) == (3, 8)
    assert (loaded.rois[0].y1, loaded.rois[0].y2) == (4, 9)

    second = editor.save()
    assert second.saved_paths["lambda"] != result.saved_paths["lambda"]
    assert result.saved_paths["lambda"].is_file()


def test_widget_smoke_with_zero_nan_linear_and_log_images(editor):
    editor.images["lambda"][:] = np.nan
    editor.images["minipix"][:] = 0
    widget = editor.widget
    assert widget is not None
    editor._norm_control.value = "linear"
    editor._render()
    editor._norm_control.value = "log"
    editor._render()
    editor.close()


def test_empty_module_offsets_do_not_reenable_legacy_defaults(tmp_path):
    table = _table(("VU-A1", 10, 20, 30, 40))
    table.to_csv(tmp_path / "lambda.txt", sep="\t", index=False)
    table.to_csv(tmp_path / "minipix.txt", sep="\t", index=False)
    config = replace(
        AnalysisConfig(),
        roi_filename="lambda.txt",
        minipix_roi_filename="minipix.txt",
        module_offsets=(),
    )
    loaded = load_roi_collection(
        tmp_path,
        config.roi_filename,
        config.minipix_roi_filename,
        module_offsets=dict(config.module_offsets),
    )
    assert (loaded.rois[0].x1, loaded.rois[0].y1) == (10, 30)
