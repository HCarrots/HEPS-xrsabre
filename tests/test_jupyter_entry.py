"""Acceptance tests for the JupyterLab-only application entry."""

from __future__ import annotations

import sys
import tomllib

from xrsabre import __version__
from xrsabre.jupyter import jupyterlab_command, launch_jupyterlab
from xrsabre.paths import check_workspace, load_workspace


def test_packages_and_workspace_are_ready_for_jupyter():
    import xrsana  # noqa: F401
    import xrslab  # noqa: F401

    workspace = load_workspace()
    assert __version__ == "0.2.0"
    assert workspace.notebooks.is_dir()
    assert not [item for item in check_workspace(workspace) if item.level == "error"]


def test_jupyter_command_is_rooted_at_configured_notebooks():
    workspace = load_workspace()
    command = jupyterlab_command(workspace, ["--no-browser"])
    assert command[:3] == [sys.executable, "-m", "jupyterlab"]
    assert f"--ServerApp.root_dir={workspace.notebooks}" in command
    assert command[-1] == "--no-browser"


def test_launcher_uses_the_same_workspace_object(monkeypatch):
    workspace = load_workspace()
    invocation = {}

    def fake_call(command, *, cwd):
        invocation["command"] = command
        invocation["cwd"] = cwd
        return 17

    monkeypatch.setattr("xrsabre.jupyter.subprocess.call", fake_call)
    assert launch_jupyterlab(["--no-browser"], workspace=workspace) == 17
    assert invocation["cwd"] == workspace.notebooks
    assert str(workspace.notebooks) in invocation["command"][3]


def test_no_command_line_application_entries_remain():
    workspace = load_workspace()
    project = tomllib.loads((workspace.root / "pyproject.toml").read_text(encoding="utf-8"))
    assert "scripts" not in project["project"]
    tasks = project["tool"]["pixi"]["tasks"]
    assert tasks["lab"] == "python -m xrsabre"
    assert not ({"xrsabre", "pipeline", "pipeline-preview", "browse-data"} & set(tasks))
    for relative in (
        "xrsabre/cli.py",
        "xrsana/cli.py",
        "xrslab/cli.py",
        "xrslab/paths.py",
        "xrsana/__main__.py",
        "xrslab/__main__.py",
        "workspace/scripts/xrs_script/run_pipeline.py",
    ):
        assert not (workspace.root / relative).exists(), relative
