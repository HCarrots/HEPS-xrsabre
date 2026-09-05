from __future__ import annotations

import ast
import json
from pathlib import Path


def test_workbench_notebook_contains_no_algorithm_definitions() -> None:
    root = Path(__file__).resolve().parents[1]
    notebook = json.loads(
        (root / "notebooks" / "XRS_Workbench.ipynb").read_text(encoding="utf-8")
    )
    code = "\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )
    tree = ast.parse(code)
    assert not any(
        isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
        for node in ast.walk(tree)
    )
    assert "xrs_processing" not in code
    assert "from xrs_compton_extraction import XRSWorkbench" in code
    assert all(
        cell.get("execution_count") is None and not cell.get("outputs")
        for cell in notebook["cells"]
        if cell["cell_type"] == "code"
    )

