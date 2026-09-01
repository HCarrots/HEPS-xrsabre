"""Parse and execute the data-independent parts of the Jupyter workbench."""

from __future__ import annotations

import ast

import nbformat
import pytest

from xrsabre.paths import load_workspace

WORKSPACE = load_workspace()
MAIN_NOTEBOOK = WORKSPACE.notebooks / "XRS_DataAnalysis.ipynb"


def _notebooks():
    paths = sorted(WORKSPACE.notebooks.glob("*.ipynb"))
    assert paths, f"no notebooks found in {WORKSPACE.notebooks}"
    return paths


def _load(path):
    return nbformat.read(path, as_version=4)


def _code_cells(notebook):
    return [cell for cell in notebook.cells if cell.cell_type == "code"]


@pytest.mark.parametrize("path", _notebooks(), ids=lambda path: path.name)
def test_notebooks_are_clean_valid_and_parseable(path):
    notebook = _load(path)
    assert notebook.nbformat >= 4
    assert _code_cells(notebook)
    for index, cell in enumerate(_code_cells(notebook)):
        ast.parse(cell.source, filename=f"{path.name}:cell-{index}")
        assert cell.get("outputs", []) == []
        assert cell.get("execution_count") is None

    serialized = path.read_text(encoding="utf-8")
    for forbidden in (
        "C:\\\\Users\\\\",
        "/home/",
        "/tmp/",
        "load_paths",
        "XRSLAB_",
        "XRSA_",
        "xrsana.cli",
        "xrslab.cli",
    ):
        assert forbidden not in serialized


def test_analysis_notebook_uses_workspace_apis():
    notebook = _load(MAIN_NOTEBOOK)
    text = "\n".join(cell.source for cell in _code_cells(notebook))
    for expected in (
        "from xrsabre.paths import load_workspace",
        "from xrslab.workflow import",
        "RoiEditor.from_config(config, workspace)",
        "prepare_analysis(config, workspace)",
        "build_qc_report(prepared)",
        "finalize_analysis(prepared, approval)",
        "export_analysis(",
    ):
        assert expected in text
    assert "curve_fit" not in text


def test_workspace_setup_cells_execute_in_a_jupyter_kernel():
    pytest.importorskip("nbclient")
    from nbclient import NotebookClient

    notebook = _load(MAIN_NOTEBOOK)
    markers = ("from xrslab.workflow import", "# Analysis configuration")
    safe_cells = [
        cell
        for cell in notebook.cells
        if cell.cell_type == "code" and any(marker in cell.source for marker in markers)
    ]
    assert len(safe_cells) == 2

    filtered = nbformat.v4.new_notebook(cells=safe_cells, metadata=notebook.metadata)
    client = NotebookClient(
        filtered,
        timeout=120,
        kernel_name="python3",
        resources={"metadata": {"path": str(WORKSPACE.notebooks)}},
    )
    try:
        executed = client.execute()
    except Exception as exc:
        pytest.fail(f"Jupyter setup execution failed: {exc}")

    for cell in executed.cells:
        errors = [output for output in cell.get("outputs", []) if output.output_type == "error"]
        assert not errors
