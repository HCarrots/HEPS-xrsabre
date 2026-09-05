from __future__ import annotations

import numpy as np
import pytest

from xrs_compton_extraction.data import QualityReport
from xrs_compton_extraction.diagnostics import (
    QualityThreshold,
    adjacent_q_continuity,
    background_model_difference,
    build_quality_report,
    compute_quality_metrics,
    fit_window_sensitivity,
    nan_inf_count,
    negative_area_fraction,
    pre_edge_residual_mean,
    pre_edge_residual_std,
    reduced_chi_square,
    residual_curvature_rms,
    target_edge_integral,
)
from xrs_compton_extraction.exceptions import DataValidationError


def test_pre_edge_residual_mean_and_sample_standard_deviation() -> None:
    residual = [-1.0, 0.0, 1.0]
    assert pre_edge_residual_mean(residual) == pytest.approx(0.0)
    assert pre_edge_residual_std(residual) == pytest.approx(1.0)
    assert pre_edge_residual_std(residual, ddof=0) == pytest.approx(np.sqrt(2 / 3))


def test_reduced_chi_square_uses_degrees_of_freedom() -> None:
    value = reduced_chi_square(
        [1.0, -2.0, 1.0, 0.0],
        [1.0, 2.0, 1.0, 1.0],
        fitted_parameter_count=2,
    )
    assert value == pytest.approx(1.5)

    with pytest.raises(DataValidationError, match="positive"):
        reduced_chi_square([1, 2], [1, 0], fitted_parameter_count=0)
    with pytest.raises(DataValidationError, match="below"):
        reduced_chi_square([1, 2], [1, 1], fitted_parameter_count=2)


def test_residual_curvature_is_zero_for_a_linear_residual() -> None:
    coordinate = np.linspace(0.0, 4.0, 5)
    assert residual_curvature_rms(coordinate, 2.0 * coordinate + 1.0) == pytest.approx(
        0.0, abs=1e-14
    )
    assert residual_curvature_rms(coordinate, coordinate**2) > 0.0

    with pytest.raises(DataValidationError, match="strictly monotonic"):
        residual_curvature_rms([0, 1, 1], [0, 1, 2])


def test_negative_area_fraction_is_integral_based_and_does_not_clip_output() -> None:
    assert negative_area_fraction([0, 1, 2], [-1, -1, 1]) == pytest.approx(0.75)
    assert negative_area_fraction([0, 1], [0, 0]) == 0.0
    assert negative_area_fraction([2, 1, 0], [1, -1, -1]) == pytest.approx(0.75)


def test_window_model_and_adjacent_q_sensitivity_metrics() -> None:
    assert fit_window_sensitivity([10.0, 12.0, 14.0]) == pytest.approx(2.0)
    assert background_model_difference([[1, 2], [2, 4], [1, 1]]) == pytest.approx(
        np.sqrt(5.0)
    )
    assert adjacent_q_continuity([1, 2, 3], [2, 2, 4]) == pytest.approx(
        np.sqrt(2 / 3)
    )

    with pytest.raises(DataValidationError, match="at least two"):
        background_model_difference([[1, 2, 3]])
    with pytest.raises(DataValidationError, match="does not match"):
        adjacent_q_continuity([1, 2], [1])


def test_target_edge_integral_requires_an_explicit_valid_window() -> None:
    energy = [0.0, 1.0, 2.0, 3.0]
    edge = [0.0, 1.0, 1.0, 0.0]
    assert target_edge_integral(
        energy, edge, integration_window=(1.0, 3.0)
    ) == pytest.approx(1.5)

    with pytest.raises(DataValidationError, match="at least two"):
        target_edge_integral(energy, edge, integration_window=(1.5, 2.5))
    with pytest.raises(DataValidationError, match="strictly increasing"):
        target_edge_integral(energy, edge, integration_window=(2.0, 1.0))


def test_nan_inf_count_accepts_different_array_shapes() -> None:
    assert nan_inf_count([0, np.nan], [[np.inf, 2], [3, -np.inf]]) == 3
    with pytest.raises(DataValidationError, match="at least one"):
        nan_inf_count()


def test_compute_quality_metrics_returns_all_ten_required_metrics() -> None:
    energy = np.arange(5.0)
    metrics = compute_quality_metrics(
        pre_edge_energy_loss_ev=energy[:4],
        pre_edge_residual=[-1.0, 0.0, 1.0, 0.0],
        pre_edge_uncertainty=[1.0, 1.0, 1.0, 1.0],
        fitted_parameter_count=1,
        energy_loss_ev=energy,
        extracted_edge=[-1.0, 0.0, 2.0, 2.0, 0.0],
        target_edge_window=(1.0, 4.0),
        fit_window_integrals=[3.0, 3.2, 2.8],
        model_backgrounds=[[1, 2, 3, 2, 1], [1, 2.1, 3, 1.9, 1]],
        adjacent_q_edge=[-0.9, 0.0, 2.1, 2.0, 0.0],
        arrays_to_check=([1.0, np.nan, np.inf],),
    )

    assert set(metrics) == {
        "pre_edge_residual_mean",
        "pre_edge_residual_std",
        "reduced_chi_square",
        "residual_curvature_rms",
        "negative_area_fraction",
        "fit_window_sensitivity",
        "background_model_difference",
        "adjacent_q_continuity",
        "target_edge_integral",
        "nan_inf_count",
    }
    assert metrics["nan_inf_count"] == 2.0
    assert metrics["residual_curvature_rms"] == pytest.approx(
        residual_curvature_rms(energy[:4], [-1.0, 0.0, 1.0, 0.0])
    )


def test_quality_threshold_validates_direction_and_ordering() -> None:
    assert QualityThreshold(1.0, 2.0, "max").classify(0.9) == "Pass"
    assert QualityThreshold(1.0, 2.0, "max").classify(1.0) == "Warning"
    assert QualityThreshold(1.0, 2.0, "max").classify(2.0) == "Reject"
    assert QualityThreshold(2.0, 1.0, "min").classify(1.5) == "Warning"
    assert QualityThreshold(2.0, 1.0, "min").classify(1.0) == "Reject"
    assert QualityThreshold(0.5, 1.0, "max", absolute=True).classify(-0.75) == "Warning"

    with pytest.raises(DataValidationError, match="must not be below"):
        QualityThreshold(2.0, 1.0, "max")
    with pytest.raises(DataValidationError, match="direction"):
        QualityThreshold(1.0, 2.0, "sideways")  # type: ignore[arg-type]


def test_build_quality_report_has_no_implicit_thresholds() -> None:
    metrics = {
        "residual_mean": -0.75,
        "reduced_chi_square": 3.0,
        "target_edge_integral": 10.0,
    }
    thresholds = {
        "residual_mean": QualityThreshold(0.5, 1.0, "max", absolute=True),
        "reduced_chi_square": QualityThreshold(2.0, 2.5, "max"),
        "target_edge_integral": QualityThreshold(8.0, 4.0, "min"),
    }
    report = build_quality_report(
        metrics,
        thresholds,
        recommended_actions=["inspect background models"],
        anomalous_indices=[2],
    )

    assert isinstance(report, QualityReport)
    assert report.grade == "Reject"
    assert len(report.reasons) == 2
    assert report.thresholds["reduced_chi_square.reject"] == 2.5

    with pytest.raises(DataValidationError, match="supplied explicitly"):
        build_quality_report(metrics, {})
    with pytest.raises(DataValidationError, match="exactly match"):
        build_quality_report(metrics, {"residual_mean": thresholds["residual_mean"]})


def test_build_quality_report_returns_pass_when_every_rule_passes() -> None:
    report = build_quality_report(
        {"metric": 0.5},
        {"metric": QualityThreshold(1.0, 2.0, "max")},
    )
    assert report.grade == "Pass"
    assert report.reasons == ("all configured quality thresholds passed",)
