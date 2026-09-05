from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from xrs_compton_extraction.data import AnalysisConfig, ExtractionResult
from xrs_compton_extraction.io import save_results


def _result() -> ExtractionResult:
    energy = np.arange(5.0)
    background = np.full(5, 2.0)
    corrected = np.arange(5.0) + background
    return ExtractionResult(
        energy_loss_eV=energy,
        raw_counts=corrected,
        q_au=1.5,
        normalized_intensity=corrected,
        corrected_intensity=corrected,
        valence_background=background,
        total_background=background,
        extracted_edge=corrected - background,
        fit_residual=corrected - background,
        statistical_uncertainty=np.ones(5),
        background_model_name="pearson",
        fit_parameters={"beta1": 1.0},
        parameter_covariance=[[1.0]],
        quality_grade="Pass",
        software_version="test",
    )


def test_save_results_writes_numeric_metadata_config_and_manifest(tmp_path: Path) -> None:
    output = save_results(
        {"分析器 A1": _result()},
        tmp_path / "结果 输出",
        config=AnalysisConfig(background_model="pearson"),
    )
    assert (output / "A1.csv").is_file()
    assert (output / "A1.metadata.json").is_file()
    assert (output / "analysis.yaml").is_file()
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["results"]["分析器 A1"]["data"] == "A1.csv"
    loaded = np.loadtxt(output / "A1.csv", delimiter=",", skiprows=1)
    assert loaded.shape[0] == 5

