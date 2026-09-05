"""Plots for multi-q comparison and averaging results."""

from __future__ import annotations

from typing import Literal

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ..exceptions import DataValidationError
from ..multi_q import MultiQResult

UncertaintyKind = Literal["statistical", "model", "total"]


def _figure_and_axis(axis: Axes | None) -> tuple[Figure, Axes]:
    if axis is None:
        return plt.subplots()
    return axis.figure, axis


def plot_multi_q_evolution(
    result: MultiQResult,
    *,
    axis: Axes | None = None,
    show_average: bool = True,
) -> tuple[Figure, Axes]:
    """Plot every used q channel on its measured intensity scale."""

    if not isinstance(result, MultiQResult):
        raise TypeError("result must be a MultiQResult")
    figure, axis = _figure_and_axis(axis)
    x = result.energy_loss_eV
    for index, (label, edge) in enumerate(
        zip(result.used_channels, result.single_channel_edges, strict=True)
    ):
        q_mean = result.diagnostics.q_mean_au[index]
        axis.plot(x, edge, label=f"{label} (q={q_mean:.4g} a.u.)", alpha=0.8)
    if show_average:
        axis.plot(
            x,
            result.average_edge,
            color="black",
            linewidth=2.0,
            label="Weighted average",
        )
    axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axis.set_xlabel("Energy loss (eV)")
    axis.set_ylabel("Extracted intensity")
    axis.set_title("Multi-q evolution")
    axis.grid(alpha=0.2)
    axis.legend()
    return figure, axis


def plot_multi_q_average(
    result: MultiQResult,
    *,
    axis: Axes | None = None,
    uncertainty: UncertaintyKind = "total",
    show_channels: bool = False,
) -> tuple[Figure, Axes]:
    """Plot the average and a selected propagated uncertainty band."""

    if not isinstance(result, MultiQResult):
        raise TypeError("result must be a MultiQResult")
    if uncertainty not in ("statistical", "model", "total"):
        raise DataValidationError(
            "uncertainty must be 'statistical', 'model', or 'total'"
        )
    figure, axis = _figure_and_axis(axis)
    x = result.energy_loss_eV
    if show_channels:
        for label, edge in zip(
            result.used_channels, result.single_channel_edges, strict=True
        ):
            axis.plot(x, edge, linewidth=0.8, alpha=0.25, label=label)
    average = result.average_edge
    sigma = getattr(result, f"{uncertainty}_uncertainty")
    axis.plot(x, average, color="C0", linewidth=1.8, label="Weighted average")
    axis.fill_between(
        x,
        average - sigma,
        average + sigma,
        color="C0",
        alpha=0.25,
        linewidth=0,
        label=f"{uncertainty.capitalize()} uncertainty",
    )
    axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axis.set_xlabel("Energy loss (eV)")
    axis.set_ylabel("Extracted intensity")
    axis.set_title("Multi-q average")
    axis.grid(alpha=0.2)
    axis.legend()
    return figure, axis


__all__ = [
    "UncertaintyKind",
    "plot_multi_q_average",
    "plot_multi_q_evolution",
]
