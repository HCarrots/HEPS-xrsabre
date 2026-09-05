from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from xrs_compton_extraction.data import (
    ROI,
    AnalysisConfig,
    AnalysisSession,
    Analyzer,
    BackgroundResult,
    CorrectionResult,
    ExtractionResult,
    Geometry,
    QualityReport,
    Sample,
    Scan,
    XRSDataset,
    XRSSpectrum,
)
from xrs_compton_extraction.exceptions import DataValidationError


def spectrum(**overrides: object) -> XRSSpectrum:
    values: dict[str, object] = {
        "energy_eV": [10_000.0, 10_010.0, 10_020.0],
        "counts": [100, 121, 144],
        "energy_loss_eV": [50.0, 60.0, 70.0],
        "q_inverse_angstrom": 5.5,
        "q_au": 2.91,
        "monitor": [10.0, 10.0, 10.0],
        "acquisition_time_s": 2.0,
        "uncertainty": [10.0, 11.0, 12.0],
        "scan_id": "scan-1",
        "analyzer_id": "analyzer-1",
        "roi_id": "roi-1",
        "metadata": {"source_file": Path("数据/sample 1.nxs")},
    }
    values.update(overrides)
    return XRSSpectrum(**values)  # type: ignore[arg-type]


def extraction(**overrides: object) -> ExtractionResult:
    values: dict[str, object] = {
        "energy_loss_eV": [1.0, 2.0, 3.0],
        "raw_counts": [30.0, 40.0, 50.0],
        "q_au": 2.0,
        "normalized_intensity": [15.0, 20.0, 25.0],
        "corrected_intensity": [16.0, 21.0, 26.0],
        "elastic_component": [1.0, 1.0, 1.0],
        "stray_background": [0.5, 0.5, 0.5],
        "valence_background": [2.0, 2.0, 2.0],
        "core_background": [3.0, 3.0, 3.0],
        "constant_background": [0.5, 0.5, 0.5],
        "total_background": [7.0, 7.0, 7.0],
        "extracted_edge": [9.0, 14.0, 19.0],
        "fit_residual": [-0.1, 0.0, 0.1],
        "statistical_uncertainty": [3.0, 4.0, 5.0],
        "model_uncertainty": [4.0, 3.0, 0.0],
        "total_uncertainty": [5.0, 5.0, 5.0],
        "background_model_name": "pearson",
        "fit_parameters": {"beta_1": 3.0, "beta_2": 4.0},
        "parameter_covariance": [[1.0, 0.0], [0.0, 2.0]],
        "fit_windows": [(1.0, 1.5), (2.5, 3.0)],
        "risk_metrics": {"R_q": 1.25},
        "warnings": ["window-sensitive"],
        "quality_grade": "warning",
        "provenance": {"source_files": [Path("input.nxs")]},
        "software_version": "0.1.0",
        "config_digest": "abc123",
        "raw_data_identifiers": ["sha256:123"],
    }
    values.update(overrides)
    return ExtractionResult(**values)  # type: ignore[arg-type]


def test_spectrum_copies_arrays_and_exposes_read_only_aliases() -> None:
    input_counts = np.array([100.0, 121.0, 144.0])
    item = spectrum(counts=input_counts)
    input_counts[0] = 999.0

    assert item.raw_counts[0] == 100.0
    assert item.energy_ev is item.energy_eV
    assert item.energy_loss_ev is item.energy_loss_eV
    assert item.i0 is item.monitor
    assert item.raw_counts.flags.writeable is False
    assert item.q_au.shape == (3,)
    assert "analyzer=analyzer-1" in item.channel_label
    assert "roi=roi-1" in item.channel_label
    with pytest.raises(ValueError):
        item.raw_counts[0] = 3.0


def test_spectrum_derives_energy_loss_from_incident_and_scattered_energy() -> None:
    item = spectrum(
        energy_loss_eV=None,
        incident_energy_ev=[100.0, 110.0, 120.0],
        scattered_energy_ev=90.0,
    )
    np.testing.assert_allclose(item.energy_loss_ev, [10.0, 20.0, 30.0])
    assert item.incident_energy_eV is item.incident_energy_ev
    assert item.scattered_energy_eV is item.scattered_energy_ev


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"counts": [1.0, 2.0]}, "length"),
        ({"counts": [1.0, -1.0, 2.0]}, "non-negative"),
        ({"energy_eV": [1.0, np.nan, 2.0]}, "NaN"),
        ({"monitor": 0.0}, "positive"),
        ({"q_au": [1.0, 2.0]}, "scalar or 3"),
        (
            {
                "incident_energy_ev": [100.0, 110.0, 120.0],
                "scattered_energy_ev": 90.0,
            },
            "must equal",
        ),
    ],
)
def test_spectrum_rejects_inconsistent_or_nonphysical_values(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(DataValidationError, match=message):
        spectrum(**overrides)


def test_metadata_is_defensively_normalized_and_json_serializable() -> None:
    metadata = {"source": Path("sample.nxs"), "values": np.array([1, 2])}
    item = spectrum(metadata=metadata)
    metadata["source"] = Path("changed.nxs")

    assert item.metadata == {"source": "sample.nxs", "values": [1, 2]}
    json.dumps(item.to_dict(), allow_nan=False)
    with pytest.raises(DataValidationError, match="non-serializable"):
        spectrum(metadata={"bad": object()})


def test_scan_analyzer_roi_sample_and_geometry_validation() -> None:
    scan = Scan(
        "scan-1",
        source_file=Path("input.nxs"),
        incident_energy_eV=[10_000.0, 10_010.0],
        acquisition_time_s=1.0,
        monitor=[100.0, 101.0],
    )
    analyzer = Analyzer(
        "analyzer-1",
        scattering_angle_deg=90.0,
        direction=[1.0, 0.0, 0.0],
        efficiency=[0.9, 0.95],
        resolution_eV=0.8,
    )
    roi = ROI("roi-1", "analyzer-1", bounds=(0, 10, 2, 12), mask=np.ones((2, 2)))
    sample = Sample("Si", composition={"Si": 1}, density_g_cm3=2.33, thickness_um=50)
    geometry = Geometry(
        scattering_angle_deg=90,
        incident_direction=[1, 0, 0],
        scattered_direction=[0, 1, 0],
        path_lengths_mm={"air": 100, "kapton": 0.025},
    )

    assert scan.source_file == "input.nxs"
    assert scan.incident_energy_ev is scan.incident_energy_eV
    assert not analyzer.direction.flags.writeable
    assert roi.mask.dtype == np.bool_
    assert sample.composition == {"Si": 1.0}
    assert geometry.path_lengths_mm["air"] == 100.0

    with pytest.raises(DataValidationError, match="length"):
        Scan("bad", incident_energy_eV=[1, 2], monitor=[1, 2, 3])
    with pytest.raises(DataValidationError, match="unit vector"):
        Analyzer("bad", direction=[2, 0, 0])
    with pytest.raises(DataValidationError, match="start values"):
        ROI("bad", bounds=(3, 2, 0, 1))
    with pytest.raises(DataValidationError, match="greater than zero"):
        Sample("bad", composition={"Si": 0})
    with pytest.raises(DataValidationError, match=r"\[0, 180\]"):
        Geometry(scattering_angle_deg=181)


def test_dataset_accepts_sequences_checks_ids_and_counts_sources() -> None:
    first = spectrum(metadata={"source_file": "first.nxs"})
    second = spectrum(
        scan_id="scan-2",
        analyzer_id="analyzer-2",
        roi_id="roi-2",
        metadata={"source_file": "second.nxs"},
    )
    dataset = XRSDataset(
        spectra=[first, second],
        scans=[Scan("scan-1"), Scan("scan-2")],
        analyzers=[Analyzer("analyzer-1"), Analyzer("analyzer-2")],
        rois=[ROI("roi-1"), ROI("roi-2")],
        provenance={"source_files": ["first.nxs", "second.nxs", "first.nxs"]},
    )

    assert isinstance(dataset.spectra, tuple)
    assert len(dataset) == 2
    assert dataset.source_count == 2
    assert dataset.spectrum(analyzer_id="analyzer-2") is second

    with pytest.raises(DataValidationError, match="duplicate"):
        XRSDataset(spectra=[first, first])
    with pytest.raises(DataValidationError, match="unique scan_id"):
        XRSDataset(spectra=[first], scans=[Scan("same"), Scan("same")])
    with pytest.raises(DataValidationError, match="unknown spectrum scan_id"):
        XRSDataset(spectra=[first], scans=[Scan("other")])


def test_dataset_source_count_falls_back_to_scan_ids() -> None:
    first = spectrum(metadata={})
    second = spectrum(
        scan_id="scan-2", analyzer_id="analyzer-2", roi_id="roi-2", metadata={}
    )
    assert XRSDataset([first, second]).source_count == 2


def test_analysis_config_normalizes_fields_and_serializes() -> None:
    config = AnalysisConfig(
        data_path=Path("数据"),
        files=[Path("a.nxs")],
        scan_ids=["1"],
        sample=Sample("Si", {"Si": 1}),
        geometry=Geometry(90, incident_direction=[1, 0, 0]),
        target_edge="Si L",
        target_edge_energy_eV=99.2,
        fit_windows=[(10, 20)],
        background_model="PEARSON",
        correction_flags={"i0": True},
        model_parameters={"seed": np.int64(1)},
        risk_thresholds={"warning": 1.0},
    )

    assert config.data_path == "数据"
    assert config.files == ("a.nxs",)
    assert config.background_model == "pearson"
    assert config.target_edge_energy_ev == 99.2
    json.dumps(config.to_dict(), allow_nan=False)

    with pytest.raises(DataValidationError, match="lower bound"):
        AnalysisConfig(fit_windows=[(2, 1)])
    with pytest.raises(DataValidationError, match="background_model"):
        AnalysisConfig(background_model="magic")
    with pytest.raises(DataValidationError, match="bool"):
        AnalysisConfig(correction_flags={"i0": 1})  # type: ignore[dict-item]


def test_correction_result_validates_pointwise_arrays() -> None:
    result = CorrectionResult(
        raw_counts=[4, 9, 16],
        normalized_intensity=[2, 3, 4],
        corrected_intensity=[2.1, 3.1, 4.1],
        correction_factors={"detector": [1, 1, 1]},
        statistical_uncertainty=[1, 1, 1],
        component_uncertainties={"i0": [0.1, 0.1, 0.1]},
    )
    assert not result.corrected_intensity.flags.writeable
    with pytest.raises(DataValidationError, match="length"):
        CorrectionResult([1, 2], [1], [1, 2])


def test_background_result_derives_total_and_checks_covariance() -> None:
    result = BackgroundResult(
        energy_loss_eV=[1, 2, 3],
        components={"core": [1, 1, 1], "valence": [2, 2, 2]},
        model_name="profile",
        fit_parameters={"scale": 1.0},
        parameter_covariance=[[0.25]],
        fit_windows=[(1, 2)],
        residual=[0, 0.1, -0.1],
    )
    np.testing.assert_allclose(result.total_background, 3.0)
    assert result.energy_loss_ev is result.energy_loss_eV

    with pytest.raises(DataValidationError, match="sum"):
        BackgroundResult([1, 2], {"core": [1, 1]}, total_background=[2, 2])
    with pytest.raises(DataValidationError, match="square"):
        BackgroundResult(
            [1, 2],
            {"core": [1, 1]},
            parameter_covariance=[[1, 2]],
        )


def test_extraction_result_keeps_all_components_and_is_serializable() -> None:
    result = extraction()

    assert result.quality_grade == "Warning"
    assert result.energy_loss_ev is result.energy_loss_eV
    assert result.q_au.shape == (3,)
    assert result.provenance["source_files"] == ["input.nxs"]
    assert not result.total_uncertainty.flags.writeable
    np.testing.assert_allclose(result.total_background, 7.0)
    json.dumps(result.to_dict(), allow_nan=False)


def test_extraction_result_derives_optional_pointwise_outputs() -> None:
    result = ExtractionResult(
        energy_loss_eV=[1, 2],
        raw_counts=[10, 20],
        q_inverse_angstrom=5.0,
    )

    np.testing.assert_allclose(result.normalized_intensity, [10, 20])
    np.testing.assert_allclose(result.corrected_intensity, [10, 20])
    np.testing.assert_allclose(result.total_background, [0, 0])
    np.testing.assert_allclose(result.extracted_edge, [10, 20])
    np.testing.assert_allclose(result.total_uncertainty, [0, 0])


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"total_background": [8, 8, 8]}, "must equal"),
        ({"total_uncertainty": [1, 1, 1]}, "quadrature"),
        ({"q_au": None, "q_inverse_angstrom": None}, "at least one"),
        ({"quality_grade": "Maybe"}, "quality_grade"),
        ({"parameter_covariance": [[1.0]]}, "dimension"),
    ],
)
def test_extraction_result_rejects_inconsistent_outputs(
    overrides: dict[str, object], message: str
) -> None:
    with pytest.raises(DataValidationError, match=message):
        extraction(**overrides)


def test_quality_report_validates_grade_metrics_and_indices() -> None:
    report = QualityReport(
        grade="pass",
        metrics={"reduced_chi_square": 1.1},
        thresholds={"max_reduced_chi_square": 2.0},
        reasons=["all checks passed"],
        anomalous_indices=[2, 4],
    )
    assert report.grade == "Pass"
    assert report.anomalous_indices == (2, 4)

    with pytest.raises(DataValidationError, match="one of"):
        QualityReport("unknown")
    with pytest.raises(DataValidationError, match="non-negative"):
        QualityReport("Reject", anomalous_indices=[-1])


def test_analysis_session_validates_results_and_invalidates_on_config_change() -> None:
    data = XRSDataset([spectrum()])
    correction = CorrectionResult([1, 2], [1, 2], [1, 2])
    session = AnalysisSession(
        dataset=data,
        correction_results={"channel": correction},
        status="READY",
    )
    session.add_log("loaded")
    assert session.status == "ready"
    assert session.logs == ("loaded",)

    session.update_config(AnalysisConfig(background_model="pearson"))
    assert session.correction_results == {}
    assert session.status == "ready"
    json.dumps(session.to_dict(), allow_nan=False)

    with pytest.raises(DataValidationError, match="CorrectionResult"):
        AnalysisSession(correction_results={"bad": object()})  # type: ignore[dict-item]

