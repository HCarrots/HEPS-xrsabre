from __future__ import annotations

import os
from dataclasses import FrozenInstanceError
from pathlib import Path

import pandas as pd
import pytest

from xrsabre.datasets import discover_reduced_datasets
from xrsabre.paths import (
    CONFIG_ENV,
    LegacyEnvironmentError,
    WorkspaceConfigError,
    WorkspaceNotFoundError,
    check_workspace,
    initialize_workspace,
    load_workspace,
    workspace_toml,
)
from xrsana.data_browser import ReducedDataBrowser


def _config(root: Path, name: str = "test") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "xrsabre.toml"
    path.write_text(workspace_toml(name), encoding="utf-8")
    return path


def test_precedence_explicit_then_environment_then_upward_search(tmp_path):
    explicit = _config(tmp_path / "explicit", "explicit")
    environment = _config(tmp_path / "environment", "environment")
    discovered = _config(tmp_path / "discovered", "discovered")
    nested = discovered.parent / "one" / "two"
    nested.mkdir(parents=True)

    env = {CONFIG_ENV: str(environment)}
    assert load_workspace(explicit, cwd=nested, environment=env).name == "explicit"
    assert load_workspace(cwd=nested, environment=env).name == "environment"
    assert load_workspace(cwd=nested, environment={}).name == "discovered"


def test_relative_paths_are_based_on_config_not_cwd(tmp_path):
    config = _config(tmp_path / "beamtime")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    workspace = load_workspace(config, cwd=elsewhere, environment={})
    assert workspace.raw == (config.parent / "workspace" / "data" / "raw").resolve()
    assert workspace.diagnostics == (config.parent / "workspace" / "diagnostics").resolve()


def test_absolute_and_home_paths_are_supported(tmp_path):
    config = _config(tmp_path / "beamtime")
    absolute = (tmp_path / "external-raw").resolve()
    text = config.read_text(encoding="utf-8")
    text = text.replace('raw = "workspace/data/raw"', f'raw = "{absolute.as_posix()}"')
    text = text.replace('roi = "workspace/data/ROI"', 'roi = "~/xrsabre-test-roi"')
    config.write_text(text, encoding="utf-8")
    workspace = load_workspace(config, environment={})
    assert workspace.raw == absolute
    assert workspace.roi == Path("~/xrsabre-test-roi").expanduser().resolve()


@pytest.mark.parametrize(
    "replacement, message",
    [
        (("schema_version = 1", "schema_version = 99"), "schema_version"),
        (("[workspace]", "unknown = 1\n\n[workspace]"), "Unknown top-level"),
        (("raw = \"workspace/data/raw\"", "raw_typo = \"workspace/data/raw\""), r"Invalid \[paths\]"),
    ],
)
def test_invalid_schema_and_unknown_keys(tmp_path, replacement, message):
    config = _config(tmp_path / "beamtime")
    config.write_text(
        config.read_text(encoding="utf-8").replace(*replacement),
        encoding="utf-8",
    )
    with pytest.raises(WorkspaceConfigError, match=message):
        load_workspace(config, environment={})


def test_output_may_not_overlap_raw_and_paths_are_frozen(tmp_path):
    config = _config(tmp_path / "beamtime")
    text = config.read_text(encoding="utf-8").replace(
        'processed = "workspace/data/processed"',
        'processed = "workspace/data/raw/output"',
    )
    config.write_text(text, encoding="utf-8")
    with pytest.raises(WorkspaceConfigError, match="paths.processed"):
        load_workspace(config, environment={})

    config.write_text(workspace_toml("frozen"), encoding="utf-8")
    workspace = load_workspace(config, environment={})
    with pytest.raises(FrozenInstanceError):
        workspace.name = "changed"  # type: ignore[misc]


def test_missing_config_and_legacy_environment_are_explicit(tmp_path):
    with pytest.raises(WorkspaceNotFoundError, match="Create xrsabre.toml"):
        load_workspace(cwd=tmp_path, environment={})
    with pytest.raises(LegacyEnvironmentError, match="XRSLAB_RAW_DIR -> paths.raw"):
        load_workspace(cwd=tmp_path, environment={"XRSLAB_RAW_DIR": "old"})


def test_init_creates_layout_notebooks_and_checks(tmp_path):
    workspace = initialize_workspace(tmp_path / "beamtime", name="GID33-test")
    assert workspace.config_file.is_file()
    assert workspace.reduced == (
        tmp_path / "beamtime" / "workspace" / "reduced"
    ).resolve()
    assert workspace.notebooks == (
        tmp_path / "beamtime" / "workspace" / "scripts" / "xrs_script"
    ).resolve()
    assert (workspace.notebooks / "XRS_DataAnalysis.ipynb").is_file()
    assert all(getattr(workspace, key).is_dir() for key in (
        "raw", "processed", "roi", "planning", "reduced", "notebooks",
        "scripts", "diagnostics",
    ))
    assert not [check for check in check_workspace(workspace) if check.level == "error"]
    with pytest.raises(FileExistsError):
        initialize_workspace(tmp_path / "beamtime", name="duplicate")


def test_check_reports_missing_and_unwritable_paths(tmp_path, monkeypatch):
    config = _config(tmp_path / "beamtime")
    workspace = load_workspace(config, environment={})
    missing = check_workspace(workspace)
    assert {item.key for item in missing if item.level == "error"} == {
        "raw", "roi", "notebooks"
    }
    workspace.raw.mkdir(parents=True)
    workspace.roi.mkdir(parents=True)
    workspace.notebooks.mkdir(parents=True)
    monkeypatch.setattr(os, "access", lambda *_args, **_kwargs: False)
    assert all(item.level == "error" for item in check_workspace(workspace))


def test_check_allows_read_only_raw_input(tmp_path, monkeypatch):
    workspace = initialize_workspace(tmp_path / "beamtime", name="read-only-raw")
    real_access = os.access

    def raw_is_read_only(path, mode):
        if Path(path) == workspace.raw:
            return not bool(mode & os.W_OK)
        return real_access(path, mode)

    monkeypatch.setattr(os, "access", raw_is_read_only)
    raw_check = next(item for item in check_workspace(workspace) if item.key == "raw")
    assert raw_check.level == "ok"


def _reduced_export(directory: Path, scan: str, crystal: str) -> None:
    directory.mkdir(parents=True)
    pd.DataFrame({"Energy Transfer (eV)": [1.0, 2.0], crystal: [3.0, 4.0]}).to_csv(
        directory / f"{scan}_all_data.txt", sep="\t", index=False
    )
    pd.DataFrame({
        "crystal": [crystal], "q_ave": [4.0], "center": [9680.0],
        "width": [1.0], "r-square": [0.99],
    }).to_csv(directory / f"{scan}_rois.txt", sep="\t", index=False)


def test_recursive_dataset_ids_and_browser_selection(tmp_path):
    workspace = initialize_workspace(tmp_path / "beamtime", name="browser")
    root = workspace.reduced
    _reduced_export(root / "Ho" / "run-1", "scan_a", "VB-A1")
    _reduced_export(root / "Ho" / "run-2", "scan_b", "VB-A2")
    datasets = discover_reduced_datasets(root)
    assert [item.dataset_id for item in datasets] == ["Ho/run-1", "Ho/run-2"]
    browser = ReducedDataBrowser(workspace, dataset_id="Ho/run-2")
    assert set(browser.datasets) == {"Ho/run-1", "Ho/run-2"}
    assert [record.crystal for record in browser.records] == ["VB-A2"]
    assert browser.filtered_records(dataset_id="Ho/run-1")[0].crystal == "VB-A1"
