from __future__ import annotations

import ast
from pathlib import Path


def test_package_does_not_import_xrs_processing() -> None:
    package_root = Path(__file__).resolve().parents[1] / "src" / "xrs_compton_extraction"
    violations: list[str] = []
    for path in package_root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            if any(name == "xrs_processing" or name.startswith("xrs_processing.") for name in names):
                violations.append(str(path.relative_to(package_root)))
    assert not violations, f"xrs_processing imports found: {violations}"

