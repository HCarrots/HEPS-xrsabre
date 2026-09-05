from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from xrs_compton_extraction import XRSWorkbench
from xrs_compton_extraction.backgrounds import pearson_background


def _write_nexus(path: Path) -> None:
    energy = np.linspace(0.0, 100.0, 401)
    background = pearson_background(energy, 200.0, 50.0, 0.03, 1.7)
    edge = np.where((energy >= 40.0) & (energy <= 60.0), 25.0, 0.0)
    with h5py.File(path, "w") as handle:
        entry = handle.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        entry.attrs["default"] = "data"
        data = entry.create_group("data")
        data.attrs["NX_class"] = "NXdata"
        data.attrs["signal"] = "counts"
        data.attrs["axes"] = "energy_loss"
        axis = data.create_dataset("energy_loss", data=energy)
        axis.attrs["units"] = "eV"
        data.create_dataset("counts", data=background + edge)
        data.create_dataset("i0", data=np.ones_like(energy))
        time = data.create_dataset("count_time", data=1.0)
        time.attrs["units"] = "s"


def test_programmatic_workbench_load_extract_save(tmp_path: Path) -> None:
    source = tmp_path / "workbench.nxs"
    _write_nexus(source)
    app = XRSWorkbench()
    dataset = app.load(source)
    assert len(dataset.spectra) == 1
    assert app.session.status == "ready"

    result = app.run_pearson(
        fit_windows_ev=((0.0, 35.0), (65.0, 100.0)),
        q_au=2.0,
        initial=(190.0, 50.0, 0.025, 1.5),
        loss="linear",
    )
    assert result.quality_grade == "Pass"
    assert app.session.status == "complete"
    assert len(app.results) == 1

    config_path = app.save_config(tmp_path / "analysis.yaml")
    assert config_path.is_file()
    result_path = app.save_results(tmp_path / "output")
    assert (result_path / "manifest.json").is_file()


def test_window_parser_is_strict() -> None:
    assert XRSWorkbench._parse_windows("0:10, 20:30") == ((0.0, 10.0), (20.0, 30.0))
    with pytest.raises(ValueError, match="start < stop"):
        XRSWorkbench._parse_windows("10:0")
    with pytest.raises(ValueError, match="at least one"):
        XRSWorkbench._parse_windows("  ")


def test_workbench_widget_builds_when_optional_dependency_is_installed() -> None:
    widgets = pytest.importorskip("ipywidgets")
    app = XRSWorkbench()
    root = app.build()
    assert isinstance(root, widgets.VBox)
    assert app.build() is root


def test_widget_load_fit_export_and_reload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import matplotlib.pyplot as plt

    plt.switch_backend("Agg")
    monkeypatch.setattr(plt, "show", lambda: None)
    widgets = pytest.importorskip("ipywidgets")
    source = tmp_path / "interactive.nxs"
    _write_nexus(source)
    app = XRSWorkbench()
    root = app.build()
    pages = root.children[1]
    assert isinstance(pages, widgets.Tab)
    assert tuple(pages.get_title(i) for i in range(5)) == (
        "Data", "Correction", "Background", "Extraction", "Results"
    )
    data, _, background, extraction, results = pages.children
    path_input, load_button = data.children[0].children
    path_input.value = str(source)
    load_button.click()
    assert app.dataset is not None
    background.children[0].value = 2.0
    background.children[1].value = "0:35, 65:100"
    extraction.children[0].children[0].click()
    assert len(app.results) == 1, root.children[2].value
    output_input, export_button = results.children[0].children
    output_input.value = str(tmp_path / "exported")
    export_button.click()
    assert (tmp_path / "exported" / "report.md").is_file()
    load_button.click()
    assert not app.results
    plt.close("all")


def test_workbench_loads_wide_processed_text(tmp_path: Path) -> None:
    from xrs_compton_extraction.io import TextMapping

    source = tmp_path / "wide.tsv"
    source.write_text("e\tA\tB\n0\t1.2\t0\n1\t2.4\t0\n", encoding="utf-8")
    app = XRSWorkbench()
    dataset = app.load_text(source, mappings=[
        TextMapping("e", label, "energy_loss", "eV", analyzer_id=label, intensity_kind="processed")
        for label in ("A", "B")
    ])
    assert len(dataset.spectra) == 2
    assert app.session.status == "ready"
    with pytest.raises(ValueError, match="exactly one"):
        app.load_text(source)
