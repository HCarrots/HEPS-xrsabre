"""JupyterLab launcher for a configured XRS beamtime workspace."""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Sequence

from .paths import WorkspaceConfigError, WorkspacePaths, load_workspace


def jupyterlab_command(
    workspace: WorkspacePaths,
    arguments: Sequence[str] = (),
) -> list[str]:
    """Build the sole application entry command for this workspace."""
    return [
        sys.executable,
        "-m",
        "jupyterlab",
        f"--ServerApp.root_dir={workspace.notebooks}",
        *arguments,
    ]


def launch_jupyterlab(
    arguments: Sequence[str] = (),
    *,
    workspace: WorkspacePaths | None = None,
) -> int:
    """Launch JupyterLab rooted at ``workspace.notebooks``."""
    resolved = workspace or load_workspace()
    if not resolved.notebooks.is_dir():
        raise WorkspaceConfigError(
            f"Configured notebooks directory is missing: {resolved.notebooks}"
        )
    return subprocess.call(
        jupyterlab_command(resolved, arguments),
        cwd=resolved.notebooks,
    )


__all__ = ["jupyterlab_command", "launch_jupyterlab"]
