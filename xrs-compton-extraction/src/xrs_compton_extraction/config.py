"""Versioned YAML/JSON persistence for :class:`AnalysisConfig`."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import fields
from pathlib import Path
from typing import Any

import yaml

from .data import AnalysisConfig, Geometry, Sample
from .exceptions import DataValidationError

_SUPPORTED_SUFFIXES = frozenset({".yaml", ".yml", ".json"})


def _load_document(path: Path) -> Mapping[str, Any]:
    if path.suffix.casefold() not in _SUPPORTED_SUFFIXES:
        raise ValueError("configuration path must end in .yaml, .yml, or .json")
    if not path.is_file():
        raise FileNotFoundError(f"configuration file does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        if path.suffix.casefold() == ".json":
            payload = json.load(handle)
        else:
            payload = yaml.safe_load(handle)
    if payload is None:
        payload = {}
    if not isinstance(payload, Mapping):
        raise DataValidationError("configuration root must be a mapping")
    return payload


def _flatten_workbench_document(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Accept the human-oriented section layout used by ``configs/default.yaml``."""

    if not any(key in payload for key in ("data", "selection", "corrections", "background", "quality", "output")):
        return dict(payload)

    result: dict[str, Any] = {}
    for key in ("schema_version", "sample", "geometry", "software_version", "analysis_time"):
        if key in payload:
            result[key] = payload[key]

    data = payload.get("data", {})
    selection = payload.get("selection", {})
    corrections = payload.get("corrections", {})
    background = payload.get("background", {})
    quality = payload.get("quality", {})
    output = payload.get("output", {})
    for name, section in (
        ("data", data),
        ("selection", selection),
        ("corrections", corrections),
        ("background", background),
        ("quality", quality),
        ("output", output),
    ):
        if not isinstance(section, Mapping):
            raise DataValidationError(f"configuration section {name!r} must be a mapping")

    result.update(
        {
            "data_path": data.get("path"),
            "files": data.get("files", ()),
            "scan_ids": data.get("scan_ids", ()),
            "raw_data_identifiers": data.get("raw_data_identifiers", ()),
            "roi_ids": selection.get("roi_ids", ()),
            "analyzer_ids": selection.get("analyzer_ids", ()),
            "target_edge": selection.get("target_edge"),
            "target_edge_energy_eV": selection.get(
                "target_edge_energy_ev", selection.get("target_edge_energy_eV")
            ),
            "background_model": background.get("model", "auto"),
            "fit_windows": background.get(
                "fit_windows_ev", background.get("fit_windows", ())
            ),
            "core_normalization_windows": background.get(
                "core_normalization_windows_ev",
                background.get("core_normalization_windows", ()),
            ),
            "model_parameters": background.get("parameters", {}),
            "parameter_bounds": background.get("parameter_bounds", {}),
            "smoothing_sigma": background.get("smoothing_sigma"),
            "correction_flags": dict(corrections),
            "risk_thresholds": quality.get("risk_thresholds", {}),
        }
    )
    metadata = dict(payload.get("metadata", {}))
    if output:
        metadata["output"] = dict(output)
    extra_quality = {key: value for key, value in quality.items() if key != "risk_thresholds"}
    if extra_quality:
        metadata["quality"] = extra_quality
    if data.get("nexus_mapping") is not None:
        metadata["nexus_mapping"] = data["nexus_mapping"]
    result["metadata"] = metadata
    return {key: value for key, value in result.items() if value is not None}


def analysis_config_from_dict(payload: Mapping[str, Any]) -> AnalysisConfig:
    """Validate a YAML/JSON-compatible mapping as an :class:`AnalysisConfig`."""

    values = _flatten_workbench_document(payload)
    if "target_edge_energy_ev" in values and "target_edge_energy_eV" not in values:
        values["target_edge_energy_eV"] = values.pop("target_edge_energy_ev")
    if isinstance(values.get("sample"), Mapping):
        values["sample"] = Sample(**dict(values["sample"]))
    if isinstance(values.get("geometry"), Mapping):
        values["geometry"] = Geometry(**dict(values["geometry"]))
    allowed = {item.name for item in fields(AnalysisConfig)}
    unknown = sorted(set(values).difference(allowed))
    if unknown:
        raise DataValidationError(f"unknown configuration fields: {unknown}")
    return AnalysisConfig(**values)


def load_config(path: str | Path) -> AnalysisConfig:
    """Load and validate an analysis configuration."""

    return analysis_config_from_dict(_load_document(Path(path).expanduser()))


def save_config(config: AnalysisConfig, path: str | Path) -> Path:
    """Atomically save an analysis configuration as YAML or JSON."""

    if not isinstance(config, AnalysisConfig):
        raise TypeError("config must be an AnalysisConfig")
    target = Path(path).expanduser()
    if target.suffix.casefold() not in _SUPPORTED_SUFFIXES:
        raise ValueError("configuration path must end in .yaml, .yml, or .json")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.tmp")
    payload = config.to_dict()
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        if target.suffix.casefold() == ".json":
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
            handle.write("\n")
        else:
            yaml.safe_dump(payload, handle, allow_unicode=True, sort_keys=False)
    temporary.replace(target)
    return target.resolve()


__all__ = ["analysis_config_from_dict", "load_config", "save_config"]

