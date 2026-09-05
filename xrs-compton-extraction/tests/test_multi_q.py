from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest

from xrs_compton_extraction.data import ExtractionResult
from xrs_compton_extraction.exceptions import DataValidationError
from xrs_compton_extraction.multi_q import average_multi_q
from xrs_compton_extraction.plotting.multi_q import (
    plot_multi_q_average,
    plot_multi_q_evolution,
)

plt.switch_backend("Agg")


def _result(
    edge: list[float] | np.ndarray,
    *,
    energy: list[float] | np.ndarray = (0.0, 1.0, 2.0),
    q_au: float = 2.0,
    statistical: float | list[float] | np.ndarray = 1.0,
    model: float | list[float] | np.ndarray = 0.5,
    grade: str = "Pass",
) -> ExtractionResult:
    energy_array = np.asarray(energy, dtype=float)
    edge_array = np.asarray(edge, dtype=float)
    background = np.full(energy_array.size, 10.0)
    corrected = edge_array + background
    return ExtractionResult(
        energy_loss_eV=energy_array,
        raw_counts=np.maximum(corrected, 0.0),
        q_au=q_au,
        normalized_intensity=corrected,
        corrected_intensity=corrected,
        valence_background=background,
        total_background=background,
        extracted_edge=edge_array,
        statistical_uncertainty=np.broadcast_to(statistical, energy_array.shape),
        model_uncertainty=np.broadcast_to(model, energy_array.shape),
        background_model_name="synthetic",
        quality_grade=grade,
        software_version="test",
    )


def test_equal_average_preserves_negative_values_and_propagates_components() -> None:
    first = _result([-2.0, 0.0, 2.0], statistical=2.0, model=1.0)
    second = _result([0.0, 2.0, 4.0], q_au=3.0, statistical=2.0, model=3.0)

    combined = average_multi_q(
        (first, second), channel_labels=("q2", "q3"), weighting="equal"
    )

    np.testing.assert_allclose(combined.average_edge, [-1.0, 1.0, 3.0])
    np.testing.assert_allclose(combined.statistical_uncertainty, np.sqrt(2.0))
    np.testing.assert_allclose(combined.model_uncertainty, np.sqrt(2.5))
    np.testing.assert_allclose(combined.total_uncertainty, np.sqrt(4.5))
    np.testing.assert_array_equal(combined.single_channel_edges[0], first.extracted_edge)
    assert combined.average_edge[0] < 0.0
    assert combined.used_channels == ("q2", "q3")
    assert combined.rejected_channels == ()
    assert not combined.average_edge.flags.writeable
    assert combined.diagnostics.average_negative_point_fraction == pytest.approx(1 / 3)


def test_inverse_variance_uses_total_variance_and_propagates_each_term() -> None:
    first = _result([10.0, 10.0, 10.0], statistical=1.0, model=1.0)
    second = _result(
        [20.0, 20.0, 20.0], q_au=3.0, statistical=2.0, model=2.0
    )

    combined = average_multi_q((first, second), weighting="inverse_variance")

    np.testing.assert_allclose(combined.weights[0], 0.8)
    np.testing.assert_allclose(combined.weights[1], 0.2)
    np.testing.assert_allclose(combined.average_edge, 12.0)
    np.testing.assert_allclose(combined.statistical_uncertainty, np.sqrt(0.8))
    np.testing.assert_allclose(combined.model_uncertainty, np.sqrt(0.8))
    np.testing.assert_allclose(combined.total_uncertainty, np.sqrt(1.6))
    np.testing.assert_allclose(combined.diagnostics.effective_channel_count, 1 / 0.68)
    assert combined.diagnostics.reduced_chi_square is not None


def test_mismatched_grid_requires_explicit_interpolation() -> None:
    first = _result([0.0, 1.0, 2.0])
    second = _result(
        [0.0, 0.5, 1.0, 1.5, 2.0],
        energy=[0.0, 0.5, 1.0, 1.5, 2.0],
        q_au=3.0,
    )

    with pytest.raises(DataValidationError, match="interpolation='linear' explicitly"):
        average_multi_q((first, second))


def test_linear_interpolation_aligns_signal_and_propagates_variance() -> None:
    first = _result([0.0, 1.0, 2.0], statistical=1.0, model=0.0)
    second = _result(
        [0.0, 1.0, 2.0, 3.0, 4.0],
        energy=[0.0, 0.5, 1.0, 1.5, 2.0],
        q_au=3.0,
        statistical=[1.0, 2.0, 3.0, 4.0, 5.0],
        model=0.0,
    )

    combined = average_multi_q((first, second), interpolation="linear")

    np.testing.assert_allclose(combined.single_channel_edges[1], [0.0, 2.0, 4.0])
    np.testing.assert_allclose(
        combined.single_channel_statistical_uncertainties[1], [1.0, 3.0, 5.0]
    )
    assert combined.diagnostics.interpolation == "linear"
    assert combined.diagnostics.coordinate_max_abs_deviation_eV == (0.0, None)


def test_linear_interpolation_rejects_any_extrapolation() -> None:
    first = _result([0.0, 1.0, 2.0])
    second = _result(
        [0.5, 1.0, 1.5], energy=[0.5, 1.0, 1.5], q_au=3.0
    )

    with pytest.raises(DataValidationError, match="Extrapolation is not allowed"):
        average_multi_q((first, second), interpolation="linear")


def test_quality_and_explicit_rejections_are_auditable() -> None:
    first = _result([1.0, 2.0, 3.0])
    bad = _result([100.0, 100.0, 100.0], q_au=3.0, grade="Reject")
    excluded = _result([200.0, 200.0, 200.0], q_au=4.0)

    combined = average_multi_q(
        (first, bad, excluded),
        channel_labels=("good", "bad", "manual"),
        excluded_channels=("manual",),
    )

    np.testing.assert_allclose(combined.average_edge, first.extracted_edge)
    assert combined.all_channels == ("good", "bad", "manual")
    assert combined.used_channels == ("good",)
    assert combined.rejected_channels == ("bad", "manual")
    assert combined.diagnostics.rejection_reasons == (
        "quality_grade=Reject",
        "explicitly excluded",
    )
    assert combined.source_results == (first, bad, excluded)


def test_rejected_mismatched_channel_does_not_force_interpolation() -> None:
    first = _result([1.0, 2.0, 3.0])
    rejected = _result(
        [1.0, 2.0], energy=[100.0, 101.0], q_au=3.0, grade="Reject"
    )

    combined = average_multi_q((first, rejected))

    assert combined.used_channels == ("channel-0",)
    assert combined.rejected_channels == ("channel-1",)


def test_inverse_variance_rejects_zero_total_uncertainty() -> None:
    result = _result([1.0, 2.0, 3.0], statistical=0.0, model=0.0)
    with pytest.raises(DataValidationError, match="strictly positive total uncertainty"):
        average_multi_q((result,), weighting="inverse_variance")


def test_descending_grid_and_inverse_angstrom_q_are_supported() -> None:
    source = _result([3.0, 2.0, 1.0], energy=[2.0, 1.0, 0.0])
    inverse_q = ExtractionResult(
        energy_loss_eV=[2.0, 1.0, 0.0],
        raw_counts=[13.0, 12.0, 11.0],
        q_inverse_angstrom=4.0,
        corrected_intensity=[13.0, 12.0, 11.0],
        valence_background=[10.0, 10.0, 10.0],
        total_background=[10.0, 10.0, 10.0],
        extracted_edge=[3.0, 2.0, 1.0],
        statistical_uncertainty=[1.0, 1.0, 1.0],
        model_uncertainty=[0.5, 0.5, 0.5],
        software_version="test",
    )

    combined = average_multi_q((source, inverse_q))

    np.testing.assert_array_equal(combined.energy_loss_eV, [2.0, 1.0, 0.0])
    np.testing.assert_allclose(combined.q_inverse_angstrom[1], 4.0)


def test_invalid_selection_and_all_rejected_are_reported() -> None:
    first = _result([1.0, 2.0, 3.0])
    with pytest.raises(DataValidationError, match="unique"):
        average_multi_q((first, first), channel_labels=("same", "same"))
    with pytest.raises(DataValidationError, match="unknown excluded"):
        average_multi_q((first,), excluded_channels=("missing",))
    with pytest.raises(DataValidationError, match="all multi-q channels"):
        average_multi_q((first,), excluded_channels=(0,))


def test_multi_q_plots_return_figure_axes_and_preserve_negative_data() -> None:
    combined = average_multi_q(
        (
            _result([-2.0, 0.0, 2.0]),
            _result([0.0, 2.0, 4.0], q_au=3.0),
        ),
        channel_labels=("low q", "high q"),
    )

    evolution_figure, evolution_axis = plot_multi_q_evolution(combined)
    average_figure, average_axis = plot_multi_q_average(combined)

    assert evolution_axis.figure is evolution_figure
    assert average_axis.figure is average_figure
    np.testing.assert_array_equal(
        evolution_axis.lines[0].get_ydata(), combined.single_channel_edges[0]
    )
    np.testing.assert_array_equal(
        average_axis.lines[0].get_ydata(), combined.average_edge
    )
    assert evolution_axis.lines[0].get_ydata()[0] < 0.0
    assert len(average_axis.collections) == 1
    plt.close(evolution_figure)
    plt.close(average_figure)
