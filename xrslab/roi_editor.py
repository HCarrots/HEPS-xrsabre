"""Interactive rectangular ROI editor for the XRSlab Jupyter workflow.

The state and persistence layer is independent of Jupyter.  Widget and
Matplotlib imports are lazy so importing XRSlab or running the CLI does not
initialise a graphical backend.
"""

from __future__ import annotations

import copy
import math
import os
import re
import uuid
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from .analysis import build_roi_masks, load_roi_collection, load_scan_batch
from xrsabre.paths import WorkspacePaths, load_workspace
from .workflow import (
    AnalysisConfig,
    PreparedAnalysis,
    _normalise_loaded_batch,
    _resolve_roi_directory,
)


DETECTORS = ("lambda", "minipix")
MODULES = ("VB", "VU", "VD", "HB", "HL", "HR")
ROWS = ("A", "B", "C", "D", "E")
COLS = ("1", "2", "3")
ROI_LABEL_PATTERN = re.compile(r"^(VB|VU|VD|HB|HL|HR)-[A-E][1-3]$")
ROI_COLUMNS = (
    "roi_label",
    "x1",
    "x2",
    "y1",
    "y2",
    "x_shift",
    "y_shift",
    "x_expand",
    "y_expand",
)


@dataclass(frozen=True)
class RoiBox:
    """One integer, half-open rectangular ROI."""

    name: str
    x1: int
    x2: int
    y1: int
    y2: int

    @property
    def width(self) -> int:
        return self.x2 - self.x1

    @property
    def height(self) -> int:
        return self.y2 - self.y1

    @property
    def area(self) -> int:
        return self.width * self.height


@dataclass(frozen=True)
class RoiEditResult:
    """Versioned ROI files and the matching no-auto-adjust configuration."""

    saved_paths: dict[str, Path]
    filenames: dict[str, str]
    config: AnalysisConfig
    counts: dict[str, int]


def _canonical_table(boxes: list[RoiBox]) -> pd.DataFrame:
    rows = []
    for box in boxes:
        rows.append({
            "roi_label": box.name,
            "x1": box.x1,
            "x2": box.x2,
            "y1": box.y1,
            "y2": box.y2,
            "x_shift": 0,
            "y_shift": 0,
            "x_expand": 0,
            "y_expand": 0,
        })
    return pd.DataFrame(rows, columns=ROI_COLUMNS)


def _tables_from_items(lambda_items, minipix_items) -> dict[str, pd.DataFrame]:
    output = {}
    for detector, items in (
        ("lambda", lambda_items),
        ("minipix", minipix_items),
    ):
        output[detector] = _canonical_table([
            RoiBox(str(item.name), item.x1, item.x2, item.y1, item.y2)
            for item in items
        ])
    return output


class RoiEditor:
    """Editable ROI state with an optional Jupyter widget front end."""

    def __init__(
        self,
        config: AnalysisConfig,
        workspace: WorkspacePaths,
        images: Mapping[str, np.ndarray],
        tables: Mapping[str, pd.DataFrame],
        *,
        history_limit: int = 100,
    ):
        if history_limit < 1:
            raise ValueError("history_limit must be positive")
        self.config = config
        self.workspace = workspace
        self.roi_directory = workspace.roi
        self.images = self._validate_images(images)
        self.history_limit = int(history_limit)
        self._boxes = self._parse_tables(tables)
        self._validate_all()
        self._initial_snapshot = self._snapshot()
        self._history = [self._initial_snapshot]
        self._history_cursor = 0
        self.current_detector = "lambda"
        self.selected_name = self._boxes["lambda"][0].name if self._boxes["lambda"] else None
        self.last_result: RoiEditResult | None = None

        self._widget = None
        self._figure = None
        self._axis = None
        self._selector = None
        self._selector_suspended = False
        self._mode = "edit"
        self._syncing = False
        self._clear_armed = False
        self._canvas_connection = None

    @staticmethod
    def _validate_images(images: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        output = {}
        for detector in DETECTORS:
            if detector not in images:
                raise KeyError(f"Missing detector image {detector!r}")
            image = np.asarray(images[detector], dtype=float)
            if image.ndim != 2 or image.size == 0:
                raise ValueError(f"{detector} image must be a non-empty 2D array")
            output[detector] = image
        return output

    def _parse_tables(self, tables: Mapping[str, pd.DataFrame]) -> dict[str, list[RoiBox]]:
        output = {}
        for detector in DETECTORS:
            if detector not in tables:
                raise KeyError(f"Missing ROI table {detector!r}")
            table = tables[detector].copy()
            if table.columns.size and str(table.columns[0]).startswith("Unnamed"):
                table = table.iloc[:, 1:]
            required = {"roi_label", "x1", "x2", "y1", "y2"}
            missing = required - set(table.columns)
            if missing:
                raise ValueError(f"{detector} ROI table is missing {sorted(missing)}")
            boxes = []
            for row in table.itertuples(index=False):
                boxes.append(self._normalise_box(
                    detector,
                    str(getattr(row, "roi_label")),
                    getattr(row, "x1"),
                    getattr(row, "x2"),
                    getattr(row, "y1"),
                    getattr(row, "y2"),
                ))
            output[detector] = boxes
        return output

    @classmethod
    def from_config(
        cls,
        config: AnalysisConfig,
        workspace: WorkspacePaths | None = None,
        *,
        history_limit: int = 100,
    ) -> "RoiEditor":
        """Load only the elastic scan and initialise an editor from config."""
        config.validate()
        resolved_workspace = load_workspace() if workspace is None else workspace
        roi_directory = _resolve_roi_directory(resolved_workspace, config)
        raw_directory = resolved_workspace.raw / config.element
        elastic_raw = load_scan_batch(
            config.elastic_scan_ids,
            raw_directory,
            monitor_pv=config.elastic_i0_pv,
            divide_by_monitor=False,
            mute=True,
        )
        elastic_batch, _, _, _ = _normalise_loaded_batch(
            elastic_raw,
            config.elastic_scan_ids,
            config.elastic_i0_pv,
            divide_by_monitor=config.divide_i0_elastic,
            correct_glitches=config.correct_i0_glitches,
            glitch_threshold=config.i0_glitch_threshold,
            kind="elastic",
        )
        if elastic_batch.scan_count != 1:
            raise ValueError("ROI editing currently requires exactly one elastic scan")
        offsets = {name: (dx, dy) for name, dx, dy in config.module_offsets}
        collection = load_roi_collection(
            roi_directory,
            config.roi_filename,
            config.minipix_roi_filename,
            module_offsets=offsets,
        )
        build_roi_masks(
            elastic_batch.lambda_data[0],
            collection.rois,
            filter_value=config.filter_value,
            auto_adjust=config.auto_adjust_rois,
            roi_size=config.roi_size,
            x_shift=collection.x_shift,
            y_shift=collection.y_shift,
            x_expand=collection.x_expand,
            y_expand=collection.y_expand,
        )
        build_roi_masks(
            elastic_batch.minipix_data[0],
            collection.minipix_rois,
            filter_value=config.filter_value,
            auto_adjust=config.auto_adjust_rois,
            roi_size=config.roi_size,
            x_shift=collection.minipix_x_shift,
            y_shift=collection.minipix_y_shift,
            x_expand=collection.minipix_x_expand,
            y_expand=collection.minipix_y_expand,
        )
        images = {
            "lambda": np.nansum(elastic_batch.lambda_data[0], axis=0),
            "minipix": np.nansum(elastic_batch.minipix_data[0], axis=0),
        }
        tables = _tables_from_items(collection.rois, collection.minipix_rois)
        return cls(
            config,
            resolved_workspace,
            images,
            tables,
            history_limit=history_limit,
        )

    @classmethod
    def from_prepared(
        cls,
        prepared: PreparedAnalysis,
        *,
        history_limit: int = 100,
    ) -> "RoiEditor":
        """Initialise from the adjusted ROI and elastic images of a prepared run."""
        images = {
            "lambda": np.nansum(prepared.elastic_batch.lambda_data[0], axis=0),
            "minipix": np.nansum(prepared.elastic_batch.minipix_data[0], axis=0),
        }
        collection = prepared.roi_collection
        tables = _tables_from_items(
            copy.deepcopy(collection.rois),
            copy.deepcopy(collection.minipix_rois),
        )
        return cls(
            prepared.config,
            prepared.workspace,
            images,
            tables,
            history_limit=history_limit,
        )

    def _normalise_box(self, detector, name, x1, x2, y1, y2) -> RoiBox:
        if detector not in DETECTORS:
            raise ValueError(f"Unknown detector {detector!r}")
        if not ROI_LABEL_PATTERN.fullmatch(str(name)):
            raise ValueError(
                f"Invalid ROI label {name!r}; expected MODULE-[A-E][1-3], e.g. VB-A1"
            )
        height, width = self.images[detector].shape
        values = np.asarray([x1, x2, y1, y2], dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError("ROI coordinates must be finite")
        left = max(0, min(width, math.floor(min(float(x1), float(x2)))))
        right = max(0, min(width, math.ceil(max(float(x1), float(x2)))))
        top = max(0, min(height, math.floor(min(float(y1), float(y2)))))
        bottom = max(0, min(height, math.ceil(max(float(y1), float(y2)))))
        if right <= left or bottom <= top:
            raise ValueError("ROI must cover at least 1x1 pixel inside the image")
        return RoiBox(str(name), left, right, top, bottom)

    def _validate_all(self) -> None:
        for detector in DETECTORS:
            names = [box.name for box in self._boxes[detector]]
            duplicates = sorted({name for name in names if names.count(name) > 1})
            if duplicates:
                raise ValueError(f"Duplicate {detector} ROI labels: {duplicates}")
            for box in self._boxes[detector]:
                self._normalise_box(
                    detector, box.name, box.x1, box.x2, box.y1, box.y2
                )

    def _snapshot(self):
        return tuple(
            (detector, tuple(self._boxes[detector])) for detector in DETECTORS
        )

    def _restore_snapshot(self, snapshot) -> None:
        self._boxes = {detector: list(boxes) for detector, boxes in snapshot}
        names = [box.name for box in self._boxes[self.current_detector]]
        if self.selected_name not in names:
            self.selected_name = names[0] if names else None
        self._refresh_ui()

    def _record_history(self) -> None:
        snapshot = self._snapshot()
        if snapshot == self._history[self._history_cursor]:
            return
        self._history = self._history[:self._history_cursor + 1]
        self._history.append(snapshot)
        if len(self._history) > self.history_limit + 1:
            self._history.pop(0)
        self._history_cursor = len(self._history) - 1
        self.last_result = None
        self._clear_armed = False
        self._refresh_ui()

    @property
    def dirty(self) -> bool:
        return self._snapshot() != self._initial_snapshot

    @property
    def can_undo(self) -> bool:
        return self._history_cursor > 0

    @property
    def can_redo(self) -> bool:
        return self._history_cursor < len(self._history) - 1

    def boxes(self, detector: str) -> tuple[RoiBox, ...]:
        if detector not in DETECTORS:
            raise ValueError(f"Unknown detector {detector!r}")
        return tuple(self._boxes[detector])

    def add_roi(self, detector: str, name: str, x1, x2, y1, y2) -> RoiBox:
        if any(box.name == name for box in self._boxes.get(detector, ())):
            raise ValueError(f"Duplicate {detector} ROI label: {name}")
        box = self._normalise_box(detector, name, x1, x2, y1, y2)
        self._boxes[detector].append(box)
        self.current_detector = detector
        self.selected_name = box.name
        self._record_history()
        return box

    def update_roi(self, detector: str, name: str, x1, x2, y1, y2) -> RoiBox:
        box = self._normalise_box(detector, name, x1, x2, y1, y2)
        for index, current in enumerate(self._boxes[detector]):
            if current.name == name:
                self._boxes[detector][index] = box
                self.current_detector = detector
                self.selected_name = name
                self._record_history()
                return box
        raise KeyError(f"Unknown {detector} ROI {name!r}")

    def rename_roi(self, detector: str, old_name: str, new_name: str) -> RoiBox:
        if old_name != new_name and any(
            box.name == new_name for box in self._boxes[detector]
        ):
            raise ValueError(f"Duplicate {detector} ROI label: {new_name}")
        for index, current in enumerate(self._boxes[detector]):
            if current.name == old_name:
                box = self._normalise_box(
                    detector,
                    new_name,
                    current.x1,
                    current.x2,
                    current.y1,
                    current.y2,
                )
                self._boxes[detector][index] = box
                self.selected_name = new_name
                self._record_history()
                return box
        raise KeyError(f"Unknown {detector} ROI {old_name!r}")

    def delete_roi(self, detector: str, name: str) -> RoiBox:
        for index, box in enumerate(self._boxes[detector]):
            if box.name == name:
                removed = self._boxes[detector].pop(index)
                names = [item.name for item in self._boxes[detector]]
                self.selected_name = names[min(index, len(names) - 1)] if names else None
                self._record_history()
                return removed
        raise KeyError(f"Unknown {detector} ROI {name!r}")

    def clear_detector(self, detector: str) -> None:
        if detector not in DETECTORS:
            raise ValueError(f"Unknown detector {detector!r}")
        if not self._boxes[detector]:
            return
        self._boxes[detector] = []
        self.current_detector = detector
        self.selected_name = None
        self._record_history()

    def reset(self) -> None:
        if self._snapshot() == self._initial_snapshot:
            return
        self._boxes = {
            detector: list(boxes) for detector, boxes in self._initial_snapshot
        }
        self.selected_name = (
            self._boxes[self.current_detector][0].name
            if self._boxes[self.current_detector]
            else None
        )
        self._record_history()

    def undo(self) -> bool:
        if not self.can_undo:
            return False
        self._history_cursor -= 1
        self.last_result = None
        self._restore_snapshot(self._history[self._history_cursor])
        return True

    def redo(self) -> bool:
        if not self.can_redo:
            return False
        self._history_cursor += 1
        self.last_result = None
        self._restore_snapshot(self._history[self._history_cursor])
        return True

    def to_tables(self) -> dict[str, pd.DataFrame]:
        self._validate_all()
        return {
            detector: _canonical_table(list(self._boxes[detector]))
            for detector in DETECTORS
        }

    def _versioned_filenames(self) -> dict[str, str]:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        sources = {
            "lambda": self.config.roi_filename,
            "minipix": self.config.minipix_roi_filename,
        }
        counter = 0
        while True:
            suffix = "" if counter == 0 else f"_{counter:02d}"
            filenames = {}
            for detector, source in sources.items():
                path = Path(source)
                extension = path.suffix or ".txt"
                filenames[detector] = (
                    f"{path.stem}_manual_{stamp}{suffix}{extension}"
                )
            if not any((self.roi_directory / name).exists() for name in filenames.values()):
                return filenames
            counter += 1

    def save(self) -> RoiEditResult:
        """Atomically write versioned ROI files and return an updated config."""
        tables = self.to_tables()
        if not self.roi_directory.is_dir():
            raise FileNotFoundError(f"ROI directory not found: {self.roi_directory}")
        filenames = self._versioned_filenames()
        final_paths = {
            detector: self.roi_directory / filename
            for detector, filename in filenames.items()
        }
        temporary_paths = {
            detector: self.roi_directory / f".{filename}.{uuid.uuid4().hex}.tmp"
            for detector, filename in filenames.items()
        }
        committed: list[Path] = []
        try:
            for detector in DETECTORS:
                tables[detector].to_csv(
                    temporary_paths[detector], sep="\t", index=False
                )
                check = pd.read_csv(temporary_paths[detector], sep="\t")
                if list(check.columns) != list(ROI_COLUMNS):
                    raise ValueError(f"Saved {detector} ROI schema validation failed")
                if len(check) != len(tables[detector]):
                    raise ValueError(f"Saved {detector} ROI row count validation failed")
            for detector in DETECTORS:
                os.replace(temporary_paths[detector], final_paths[detector])
                committed.append(final_paths[detector])
        except Exception:
            for path in temporary_paths.values():
                path.unlink(missing_ok=True)
            for path in committed:
                path.unlink(missing_ok=True)
            raise

        edited_config = replace(
            self.config,
            roi_filename=filenames["lambda"],
            minipix_roi_filename=filenames["minipix"],
            auto_adjust_rois=False,
            module_offsets=(),
        )
        result = RoiEditResult(
            saved_paths=final_paths,
            filenames=filenames,
            config=edited_config,
            counts={detector: len(tables[detector]) for detector in DETECTORS},
        )
        self.last_result = result
        self._initial_snapshot = self._snapshot()
        self._history = [self._initial_snapshot]
        self._history_cursor = 0
        self._set_status(
            "已保存：" + ", ".join(filenames.values()), success=True
        )
        self._refresh_buttons()
        return result

    @property
    def widget(self):
        self._ensure_widget()
        return self._widget

    def display(self):
        """Display the editor in a Jupyter output cell and return its widget."""
        from IPython.display import display

        display(self.widget)
        return self.widget

    def _ensure_widget(self) -> None:
        if self._widget is not None:
            return
        try:
            import ipywidgets as widgets
            import matplotlib
            import matplotlib.pyplot as plt
            from IPython.display import display
            from matplotlib.colors import LogNorm, Normalize
            from matplotlib.patches import Rectangle
            from matplotlib.widgets import RectangleSelector
        except ImportError as exc:
            raise RuntimeError(
                "The ROI editor requires ipywidgets and ipympl. Run `pixi install` "
                "and use `%matplotlib widget` in Jupyter."
            ) from exc

        self._widgets_module = widgets
        self._plt = plt
        self._matplotlib = matplotlib
        self._Normalize = Normalize
        self._LogNorm = LogNorm
        self._Rectangle = Rectangle
        self._RectangleSelector = RectangleSelector

        with plt.ioff():
            self._figure, self._axis = plt.subplots(figsize=(8, 7))
        self._detector_control = widgets.ToggleButtons(
            options=DETECTORS, value=self.current_detector, description="探测器"
        )
        self._norm_control = widgets.ToggleButtons(
            options=(("对数", "log"), ("线性", "linear")),
            value="log",
            description="色标",
        )
        self._contrast_control = widgets.FloatRangeSlider(
            value=(1.0, 99.8),
            min=0.0,
            max=100.0,
            step=0.1,
            description="百分位",
            continuous_update=False,
            readout_format=".1f",
            layout=widgets.Layout(width="420px"),
        )
        self._labels_control = widgets.Checkbox(value=True, description="显示标签")
        self._roi_control = widgets.Dropdown(description="当前 ROI")
        self._module_control = widgets.Dropdown(options=MODULES, value="VB", description="模块")
        self._row_control = widgets.Dropdown(options=ROWS, value="A", description="行")
        self._col_control = widgets.Dropdown(options=COLS, value="1", description="列")
        self._new_button = widgets.Button(description="新增/绘制", icon="plus")
        self._delete_button = widgets.Button(description="删除", icon="trash")
        self._clear_button = widgets.Button(description="清空探测器", icon="eraser")
        self._undo_button = widgets.Button(description="撤销", icon="undo")
        self._redo_button = widgets.Button(description="重做", icon="repeat")
        self._reset_button = widgets.Button(description="恢复初始", icon="refresh")
        self._save_button = widgets.Button(
            description="另存版本", icon="save", button_style="success"
        )
        self._status = widgets.HTML()

        canvas = self._figure.canvas
        if isinstance(canvas, widgets.Widget):
            canvas_widget = canvas
        else:
            canvas_widget = widgets.Output()
            with canvas_widget:
                display(self._figure)
        controls = widgets.VBox([
            widgets.HBox([self._detector_control, self._norm_control, self._labels_control]),
            self._contrast_control,
            widgets.HBox([self._roi_control, self._delete_button]),
            widgets.HBox([
                self._module_control,
                self._row_control,
                self._col_control,
                self._new_button,
            ]),
            widgets.HBox([
                self._undo_button,
                self._redo_button,
                self._reset_button,
                self._clear_button,
                self._save_button,
            ]),
            self._status,
        ])
        self._widget = widgets.VBox([controls, canvas_widget])
        self._bind_widget_events()
        self._canvas_connection = canvas.mpl_connect(
            "button_press_event", self._on_canvas_click
        )
        self._render()
        backend = matplotlib.get_backend().lower()
        if "ipympl" not in backend and "widget" not in backend:
            self._set_status(
                "当前不是交互后端；请先在 Notebook 执行 `%matplotlib widget`。",
                error=True,
            )

    def _bind_widget_events(self) -> None:
        self._detector_control.observe(self._on_detector_change, names="value")
        self._norm_control.observe(lambda change: self._render(), names="value")
        self._contrast_control.observe(lambda change: self._render(), names="value")
        self._labels_control.observe(lambda change: self._render(), names="value")
        self._roi_control.observe(self._on_roi_change, names="value")
        self._new_button.on_click(self._on_new_clicked)
        self._delete_button.on_click(self._on_delete_clicked)
        self._clear_button.on_click(self._on_clear_clicked)
        self._undo_button.on_click(lambda button: self.undo())
        self._redo_button.on_click(lambda button: self.redo())
        self._reset_button.on_click(lambda button: self.reset())
        self._save_button.on_click(self._on_save_clicked)

    def _on_detector_change(self, change) -> None:
        if self._syncing:
            return
        self.current_detector = change["new"]
        names = [box.name for box in self._boxes[self.current_detector]]
        self.selected_name = names[0] if names else None
        self._mode = "edit"
        self._clear_armed = False
        self._render()

    def _on_roi_change(self, change) -> None:
        if self._syncing:
            return
        self.selected_name = change["new"]
        self._mode = "edit"
        self._render()

    def _new_roi_name(self) -> str:
        return (
            f"{self._module_control.value}-"
            f"{self._row_control.value}{self._col_control.value}"
        )

    def _on_new_clicked(self, _button) -> None:
        name = self._new_roi_name()
        if any(box.name == name for box in self._boxes[self.current_detector]):
            self._set_status(
                f"{self.current_detector} 中已存在 {name}", error=True
            )
            return
        self._mode = "add"
        self.selected_name = None
        self._clear_armed = False
        self._render()
        self._set_status(f"请在图像上拖拽绘制 {name}")

    def _on_delete_clicked(self, _button) -> None:
        if self.selected_name is None:
            self._set_status("请先选择 ROI", error=True)
            return
        name = self.selected_name
        self.delete_roi(self.current_detector, name)
        self._set_status(f"已删除 {name}；可撤销")

    def _on_clear_clicked(self, _button) -> None:
        if not self._clear_armed:
            self._clear_armed = True
            self._set_status("再次点击“清空探测器”以确认；该操作可撤销")
            return
        detector = self.current_detector
        self.clear_detector(detector)
        self._set_status(f"已清空 {detector}；可撤销")

    def _on_save_clicked(self, _button) -> None:
        try:
            self.save()
        except Exception as exc:
            self._set_status(f"保存失败：{exc}", error=True)

    def _on_canvas_click(self, event) -> None:
        if self._mode == "add" or event.inaxes is not self._axis:
            return
        if event.xdata is None or event.ydata is None:
            return
        candidates = [
            box for box in self._boxes[self.current_detector]
            if box.x1 <= event.xdata < box.x2 and box.y1 <= event.ydata < box.y2
        ]
        if candidates:
            selected = min(candidates, key=lambda box: box.area)
            if selected.name != self.selected_name:
                self.selected_name = selected.name
                self._render()

    def _on_rectangle(self, start, end) -> None:
        if self._selector_suspended:
            return
        if start.xdata is None or end.xdata is None:
            return
        try:
            if self._mode == "add":
                name = self._new_roi_name()
                self.add_roi(
                    self.current_detector,
                    name,
                    start.xdata,
                    end.xdata,
                    start.ydata,
                    end.ydata,
                )
                self._mode = "edit"
                self._set_status(f"已新增 {name}")
            elif self.selected_name is not None:
                name = self.selected_name
                self.update_roi(
                    self.current_detector,
                    name,
                    start.xdata,
                    end.xdata,
                    start.ydata,
                    end.ydata,
                )
                self._set_status(f"已更新 {name}")
        except Exception as exc:
            self._set_status(str(exc), error=True)
            self._render()

    def _create_selector(self) -> None:
        if self._selector is not None:
            self._selector.disconnect_events()
        self._selector = self._RectangleSelector(
            self._axis,
            self._on_rectangle,
            useblit=False,
            button=[1],
            minspanx=1,
            minspany=1,
            spancoords="data",
            interactive=True,
            drag_from_anywhere=True,
            props={"facecolor": "none", "edgecolor": "yellow", "linewidth": 1.5},
            handle_props={"markeredgecolor": "yellow"},
        )
        active = self._mode == "add" or self.selected_name is not None
        self._selector.set_active(active)
        self._selector.set_visible(active)
        if self._mode == "edit" and self.selected_name is not None:
            box = next(
                item for item in self._boxes[self.current_detector]
                if item.name == self.selected_name
            )
            self._selector_suspended = True
            self._selector.extents = (box.x1, box.x2, box.y1, box.y2)
            self._selector_suspended = False

    def _image_norm(self, image):
        finite = image[np.isfinite(image)]
        if self._norm_control.value == "log":
            finite = finite[finite > 0]
        if finite.size == 0:
            return None
        low_percentile, high_percentile = self._contrast_control.value
        low, high = np.nanpercentile(finite, [low_percentile, high_percentile])
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            low, high = float(np.nanmin(finite)), float(np.nanmax(finite))
        if high <= low:
            high = low + max(abs(low) * 1e-6, 1e-12)
        if self._norm_control.value == "log" and low > 0:
            return self._LogNorm(vmin=low, vmax=high)
        return self._Normalize(vmin=low, vmax=high)

    def _render(self) -> None:
        if self._widget is None:
            return
        image = self.images[self.current_detector]
        self._axis.clear()
        self._axis.imshow(image, norm=self._image_norm(image), origin="upper")
        self._axis.set_title(
            f"{self.current_detector} ROI editor ({image.shape[1]}×{image.shape[0]})"
        )
        for box in self._boxes[self.current_detector]:
            selected = box.name == self.selected_name
            color = "yellow" if selected else "red"
            self._axis.add_patch(self._Rectangle(
                (box.x1, box.y1),
                box.width,
                box.height,
                edgecolor=color,
                facecolor="none",
                linewidth=1.4 if selected else 0.7,
            ))
            if self._labels_control.value:
                self._axis.text(
                    box.x1,
                    box.y1,
                    box.name,
                    color=color,
                    fontsize=7,
                    verticalalignment="bottom",
                )
        self._axis.set_xlim(-0.5, image.shape[1] - 0.5)
        self._axis.set_ylim(image.shape[0] - 0.5, -0.5)
        self._create_selector()
        self._sync_controls()
        self._figure.canvas.draw_idle()

    def _sync_controls(self) -> None:
        if self._widget is None:
            return
        self._syncing = True
        try:
            self._detector_control.value = self.current_detector
            names = [box.name for box in self._boxes[self.current_detector]]
            self._roi_control.options = names
            self._roi_control.value = self.selected_name if self.selected_name in names else None
        finally:
            self._syncing = False
        self._refresh_buttons()

    def _refresh_buttons(self) -> None:
        if self._widget is None:
            return
        self._undo_button.disabled = not self.can_undo
        self._redo_button.disabled = not self.can_redo
        self._delete_button.disabled = self.selected_name is None
        self._save_button.description = "另存版本 *" if self.dirty else "另存版本"

    def _refresh_ui(self) -> None:
        if self._widget is not None:
            self._render()

    def _set_status(self, message: str, *, error=False, success=False) -> None:
        if self._widget is None:
            return
        color = "#b00020" if error else ("#137333" if success else "inherit")
        self._status.value = f"<span style='color:{color}'>{message}</span>"

    def close(self) -> None:
        """Disconnect callbacks and close the Matplotlib figure."""
        if self._selector is not None:
            self._selector.disconnect_events()
        if self._figure is not None and self._canvas_connection is not None:
            self._figure.canvas.mpl_disconnect(self._canvas_connection)
        if self._figure is not None:
            self._plt.close(self._figure)


__all__ = ["RoiBox", "RoiEditResult", "RoiEditor"]
