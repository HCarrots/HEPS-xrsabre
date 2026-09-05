"""Portable result export without hidden notebook state."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from ..config import save_config
from ..data import AnalysisConfig, ExtractionResult

_SAFE_STEM = re.compile(r"[^A-Za-z0-9_.-]+")


def _safe_stem(value: str) -> str:
    stem = _SAFE_STEM.sub("_", value).strip("._")
    if not stem:
        raise ValueError("result names must contain at least one filename-safe character")
    return stem


def _result_columns(result: ExtractionResult) -> tuple[list[str], np.ndarray]:
    columns = {
        "energy_loss_eV": result.energy_loss_ev,
        "q_au": result.q_au,
        "q_inverse_angstrom": result.q_inverse_angstrom,
        "raw_counts": result.raw_counts,
        "normalized_intensity": result.normalized_intensity,
        "corrected_intensity": result.corrected_intensity,
        "elastic_component": result.elastic_component,
        "stray_background": result.stray_background,
        "valence_background": result.valence_background,
        "core_background": result.core_background,
        "constant_background": result.constant_background,
        "total_background": result.total_background,
        "extracted_edge": result.extracted_edge,
        "fit_residual": result.fit_residual,
        "statistical_uncertainty": result.statistical_uncertainty,
        "model_uncertainty": result.model_uncertainty,
        "total_uncertainty": result.total_uncertainty,
    }
    present = {name: value for name, value in columns.items() if value is not None}
    return list(present), np.column_stack(tuple(present.values()))


def save_results(
    results: ExtractionResult | Mapping[str, ExtractionResult],
    output_directory: str | Path,
    *,
    config: AnalysisConfig | None = None,
    figures: Mapping[str, Any] | None = None,
) -> Path:
    """Save extraction arrays, metadata, optional configuration, and figures."""

    output = Path(output_directory).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    result_map = {"extraction": results} if isinstance(results, ExtractionResult) else dict(results)
    if not result_map or not all(isinstance(value, ExtractionResult) for value in result_map.values()):
        raise TypeError("results must be an ExtractionResult or a non-empty mapping of them")
    stems = [_safe_stem(str(name)) for name in result_map]
    if len({stem.casefold() for stem in stems}) != len(stems):
        raise ValueError("result names collide after filename normalization")

    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "created_at": datetime.now(UTC).isoformat(),
        "results": {},
        "figures": {},
    }
    for name, result in result_map.items():
        stem = _safe_stem(str(name))
        headers, values = _result_columns(result)
        data_name = f"{stem}.csv"
        np.savetxt(
            output / data_name,
            values,
            delimiter=",",
            header=",".join(headers),
            comments="",
        )
        metadata = result.to_dict()
        for header in headers:
            metadata.pop(header, None)
        metadata.pop("energy_loss_eV", None)
        metadata_name = f"{stem}.metadata.json"
        (output / metadata_name).write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        manifest["results"][str(name)] = {
            "data": data_name,
            "metadata": metadata_name,
            "columns": headers,
        }

    if config is not None:
        config_path = save_config(config, output / "analysis.yaml")
        manifest["config"] = config_path.name
    for name, figure in (figures or {}).items():
        stem = _safe_stem(str(name))
        filename = f"{stem}.png"
        figure.savefig(output / filename, dpi=150, bbox_inches="tight")
        manifest["figures"][str(name)] = filename

    report_lines = ["# XRS extraction report", "", f"Created: {manifest['created_at']}", ""]
    for name, result in result_map.items():
        report_lines.extend([
            f"## {name}", "", f"Model: {result.background_model_name}",
            f"Quality: {result.quality_grade}", f"Software: {result.software_version}", "",
            "Metrics:", "", *[f"- {key}: {value:g}" for key, value in result.risk_metrics.items()],
            "", "Warnings:", "", *[f"- {warning}" for warning in result.warnings], "",
        ])
    (output / "report.md").write_text("\n".join(report_lines), encoding="utf-8")
    manifest["report"] = "report.md"

    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return output.resolve()


__all__ = ["save_results"]
