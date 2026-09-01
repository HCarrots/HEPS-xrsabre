"""Unified public entry point for the HEPS XRS analysis toolkits."""

from __future__ import annotations

__version__ = "0.2.0"

from .paths import WorkspacePaths, load_workspace

__all__ = ["WorkspacePaths", "__version__", "load_workspace"]
