from __future__ import annotations

from pathlib import Path

import pytest

from xrs_compton_extraction.config import load_config, save_config
from xrs_compton_extraction.data import AnalysisConfig, Geometry, Sample
from xrs_compton_extraction.exceptions import DataValidationError


def test_yaml_config_round_trip_with_unicode_path(tmp_path: Path) -> None:
    config = AnalysisConfig(
        data_path=tmp_path / "含 空格" / "scan.nxs",
        scan_ids=("42",),
        sample=Sample(name="test sample", composition={"Si": 1.0}),
        geometry=Geometry(scattering_angle_deg=120.0),
        target_edge="Si L",
        target_edge_energy_eV=100.0,
        fit_windows=((20.0, 60.0), (140.0, 180.0)),
        background_model="pearson",
        correction_flags={"normalize_i0": True},
    )
    path = save_config(config, tmp_path / "配置.yaml")
    loaded = load_config(path)
    assert loaded.to_dict() == config.to_dict()


def test_json_config_round_trip(tmp_path: Path) -> None:
    config = AnalysisConfig(background_model="polynomial")
    path = save_config(config, tmp_path / "analysis.json")
    assert load_config(path).to_dict() == config.to_dict()


def test_human_oriented_default_config_loads() -> None:
    root = Path(__file__).resolve().parents[1]
    config = load_config(root / "configs" / "default.yaml")
    assert config.schema_version == "1.0"
    assert config.background_model == "pearson"
    assert config.correction_flags["normalize_i0"] is True
    assert config.metadata["output"]["directory"] == "output"


def test_unknown_config_field_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("schema_version: '1.0'\nmade_up: true\n", encoding="utf-8")
    with pytest.raises(DataValidationError, match="unknown configuration fields"):
        load_config(path)

