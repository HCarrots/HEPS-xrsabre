"""Typed, reproducible workspace path management for HEPS-xrsabre."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tomllib
from dataclasses import dataclass
from importlib.resources import as_file, files
from pathlib import Path
from typing import Mapping

CONFIG_FILENAME = "xrsabre.toml"
CONFIG_ENV = "XRSABRE_CONFIG"
SCHEMA_VERSION = 1
PATH_KEYS = (
    "raw", "processed", "roi", "planning", "reduced", "notebooks",
    "scripts", "diagnostics",
)
LEGACY_ENVIRONMENT = {
    "XRSLAB_PROJECT_ROOT": "the directory containing xrsabre.toml",
    "XRSLAB_DATA_DIR": "paths.raw and paths.processed",
    "XRSLAB_RAW_DIR": "paths.raw",
    "XRSLAB_PROCESSED_DIR": "paths.processed",
    "XRSLAB_ROI_DIR": "paths.roi",
    "XRSLAB_PATHS_FILE": "XRSABRE_CONFIG or --config",
    "XRSA_ROOT": "the directory containing xrsabre.toml",
    "XRSA_DATA": "paths.planning, paths.reduced, and paths.scripts",
    "XRSA_RESOURCES": "built-in package resources (not configurable)",
    "XRSA_BROWSER_DATA": "paths.reduced",
    "XRSA_PLOT_DIR": "paths.diagnostics",
}


class WorkspaceError(RuntimeError):
    """Base class for workspace configuration failures."""


class WorkspaceNotFoundError(WorkspaceError):
    """Raised when no workspace configuration can be discovered."""


class WorkspaceConfigError(WorkspaceError):
    """Raised when a workspace file is malformed or unsafe."""


class LegacyEnvironmentError(WorkspaceConfigError):
    """Raised when removed path environment variables are still set."""


@dataclass(frozen=True, slots=True)
class WorkspacePaths:
    """All filesystem locations for one beamtime workspace."""

    config_file: Path
    config_sha256: str
    name: str
    root: Path
    raw: Path
    processed: Path
    roi: Path
    planning: Path
    reduced: Path
    notebooks: Path
    scripts: Path
    diagnostics: Path

    def as_dict(self) -> dict[str, str]:
        return {
            "root": str(self.root),
            **{key: str(getattr(self, key)) for key in PATH_KEYS},
        }


@dataclass(frozen=True, slots=True)
class PathCheck:
    level: str
    key: str
    path: Path
    message: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_legacy_environment(environment: Mapping[str, str]) -> None:
    found = [name for name in LEGACY_ENVIRONMENT if environment.get(name)]
    if found:
        migrations = ", ".join(
            f"{name} -> {LEGACY_ENVIRONMENT[name]}" for name in found
        )
        raise LegacyEnvironmentError(
            "Legacy path environment variables are no longer supported: " + migrations
        )


def discover_workspace_file(
    config_file: str | os.PathLike[str] | None = None,
    *,
    cwd: str | os.PathLike[str] | None = None,
    environment: Mapping[str, str] | None = None,
) -> Path:
    """Resolve config by explicit argument, XRSABRE_CONFIG, then upward search."""
    env = os.environ if environment is None else environment
    _check_legacy_environment(env)
    selected = config_file or env.get(CONFIG_ENV)
    if selected:
        path = Path(selected).expanduser()
        if not path.is_absolute():
            path = Path(cwd or Path.cwd()) / path
        path = path.resolve()
        if path.is_dir():
            path /= CONFIG_FILENAME
        if not path.is_file():
            raise WorkspaceNotFoundError(f"Workspace configuration not found: {path}")
        return path

    start = Path(cwd or Path.cwd()).expanduser().resolve()
    if start.is_file():
        start = start.parent
    for directory in (start, *start.parents):
        candidate = directory / CONFIG_FILENAME
        if candidate.is_file():
            return candidate
    raise WorkspaceNotFoundError(
        f"No {CONFIG_FILENAME} found from {start}. "
        "Create xrsabre.toml in the beamtime root or set XRSABRE_CONFIG."
    )


def _expect_table(data: object, name: str) -> dict[str, object]:
    if not isinstance(data, dict):
        raise WorkspaceConfigError(f"[{name}] must be a TOML table")
    return data


def _resolve_config_path(value: object, *, key: str, root: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise WorkspaceConfigError(f"paths.{key} must be a non-empty string")
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve(strict=False)


def _is_within(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _validate_path_relationships(workspace: WorkspacePaths) -> None:
    output_keys = ("processed", "planning", "reduced", "scripts", "diagnostics")
    for key in output_keys:
        output = getattr(workspace, key)
        if (
            output == workspace.raw
            or _is_within(output, workspace.raw)
            or _is_within(workspace.raw, output)
        ):
            raise WorkspaceConfigError(
                f"paths.{key} must not equal or be contained by paths.raw: {output}"
            )
    values: dict[Path, list[str]] = {}
    for key in PATH_KEYS:
        values.setdefault(getattr(workspace, key), []).append(key)
    duplicates = [names for names in values.values() if len(names) > 1]
    if duplicates:
        joined = "; ".join(", ".join(names) for names in duplicates)
        raise WorkspaceConfigError(f"Workspace paths must be distinct: {joined}")


def load_workspace(
    config_file: str | os.PathLike[str] | None = None,
    *,
    cwd: str | os.PathLike[str] | None = None,
    environment: Mapping[str, str] | None = None,
) -> WorkspacePaths:
    """Load and validate one ``xrsabre.toml`` workspace."""
    resolved = discover_workspace_file(config_file, cwd=cwd, environment=environment)
    try:
        with resolved.open("rb") as stream:
            data = tomllib.load(stream)
    except tomllib.TOMLDecodeError as exc:
        raise WorkspaceConfigError(f"Invalid TOML in {resolved}: {exc}") from exc

    unknown_top = set(data) - {"schema_version", "workspace", "paths"}
    if unknown_top:
        raise WorkspaceConfigError(f"Unknown top-level keys: {sorted(unknown_top)}")
    if data.get("schema_version") != SCHEMA_VERSION:
        raise WorkspaceConfigError(
            f"schema_version must be {SCHEMA_VERSION}, got {data.get('schema_version')!r}"
        )
    workspace_table = _expect_table(data.get("workspace"), "workspace")
    paths_table = _expect_table(data.get("paths"), "paths")
    if set(workspace_table) != {"name"}:
        raise WorkspaceConfigError(
            f"[workspace] requires only 'name'; got {sorted(workspace_table)}"
        )
    name = workspace_table.get("name")
    if not isinstance(name, str) or not name.strip():
        raise WorkspaceConfigError("workspace.name must be a non-empty string")
    missing = set(PATH_KEYS) - set(paths_table)
    unknown = set(paths_table) - set(PATH_KEYS)
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing {sorted(missing)}")
        if unknown:
            details.append(f"unknown {sorted(unknown)}")
        raise WorkspaceConfigError("Invalid [paths]: " + ", ".join(details))

    root = resolved.parent.resolve()
    path_values = {
        key: _resolve_config_path(paths_table[key], key=key, root=root)
        for key in PATH_KEYS
    }
    workspace = WorkspacePaths(
        config_file=resolved,
        config_sha256=_sha256(resolved),
        name=name.strip(),
        root=root,
        **path_values,
    )
    _validate_path_relationships(workspace)
    return workspace


def _nearest_existing_parent(path: Path) -> Path | None:
    current = path
    while not current.exists() and current != current.parent:
        current = current.parent
    return current if current.exists() else None


def check_workspace(workspace: WorkspacePaths) -> tuple[PathCheck, ...]:
    """Return read/write and layout diagnostics without changing the workspace."""
    checks: list[PathCheck] = []
    input_keys = {"raw", "roi", "notebooks"}
    for key in PATH_KEYS:
        path = getattr(workspace, key)
        if path.exists() and not path.is_dir():
            checks.append(PathCheck("error", key, path, "path exists but is not a directory"))
            continue
        if key in input_keys and not path.is_dir():
            checks.append(PathCheck("error", key, path, "required input directory is missing"))
            continue
        if path.is_dir():
            mode = os.R_OK | os.X_OK if key == "raw" else os.R_OK | os.W_OK | os.X_OK
            if not os.access(path, mode):
                permission = "readable" if key == "raw" else "readable and writable"
                checks.append(PathCheck("error", key, path, f"path is not {permission}"))
            else:
                checks.append(PathCheck("ok", key, path, "directory exists"))
            continue
        parent = _nearest_existing_parent(path)
        if parent is None or not os.access(parent, os.W_OK | os.X_OK):
            checks.append(PathCheck("error", key, path, "parent is not writable"))
        else:
            checks.append(PathCheck("warning", key, path, "will be created when needed"))
    return tuple(checks)


def workspace_toml(name: str) -> str:
    safe_name = name.strip()
    if not safe_name:
        raise ValueError("workspace name cannot be empty")
    quoted_name = json.dumps(safe_name, ensure_ascii=False)
    return f'''schema_version = 1

[workspace]
name = {quoted_name}

[paths]
raw = "workspace/data/raw"
processed = "workspace/data/processed"
roi = "workspace/data/ROI"
planning = "workspace/planning"
reduced = "workspace/reduced"
notebooks = "workspace/scripts/xrs_script"
scripts = "workspace/scripts"
diagnostics = "workspace/diagnostics"
'''


def initialize_workspace(
    directory: str | os.PathLike[str], *, name: str,
) -> WorkspacePaths:
    """Create a new workspace layout without overwriting existing files."""
    root = Path(directory).expanduser().resolve(strict=False)
    root.mkdir(parents=True, exist_ok=True)
    config_file = root / CONFIG_FILENAME
    if config_file.exists():
        raise FileExistsError(f"Workspace configuration already exists: {config_file}")
    config_file.write_text(workspace_toml(name), encoding="utf-8")
    workspace = load_workspace(config_file, environment={})
    for key in PATH_KEYS:
        getattr(workspace, key).mkdir(parents=True, exist_ok=True)
    templates = files("xrsabre").joinpath("templates")
    if templates.is_dir():
        with as_file(templates) as template_dir:
            for source in Path(template_dir).glob("*.ipynb"):
                target = workspace.notebooks / source.name
                if not target.exists():
                    shutil.copy2(source, target)
    return workspace


__all__ = [
    "CONFIG_ENV", "CONFIG_FILENAME", "LEGACY_ENVIRONMENT",
    "LegacyEnvironmentError", "PathCheck", "WorkspaceConfigError",
    "WorkspaceError", "WorkspaceNotFoundError",
    "WorkspacePaths", "check_workspace", "discover_workspace_file",
    "initialize_workspace", "load_workspace", "workspace_toml",
]
