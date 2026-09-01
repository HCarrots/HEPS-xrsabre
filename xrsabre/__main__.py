"""Launch the configured experiment workspace in JupyterLab."""

from __future__ import annotations

import sys

from .jupyter import launch_jupyterlab


if __name__ == "__main__":
    raise SystemExit(launch_jupyterlab(sys.argv[1:]))
