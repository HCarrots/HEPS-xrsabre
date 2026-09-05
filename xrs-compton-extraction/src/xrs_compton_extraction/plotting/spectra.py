"""Spectrum and extracted-edge plots."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes


def _axis_or_new(axis: Axes | None) -> Axes:
    if axis is not None:
        return axis
    _, new_axis = plt.subplots()
    return new_axis


def plot_spectrum(spectrum: Any, *, axis: Axes | None = None) -> Axes:
    """Plot one spectrum's immutable raw counts against energy loss."""

    axis = _axis_or_new(axis)
    axis.plot(spectrum.energy_loss_ev, spectrum.raw_counts, label="Raw counts")
    axis.set_xlabel("Energy loss (eV)")
    axis.set_ylabel("Counts")
    axis.set_title(spectrum.channel_label)
    axis.grid(alpha=0.2)
    return axis


def plot_extraction(result: Any, *, axis: Axes | None = None) -> Axes:
    """Plot corrected signal, total background, and the un-clipped extraction."""

    axis = _axis_or_new(axis)
    x = result.energy_loss_ev
    axis.plot(x, result.corrected_intensity, label="Corrected", color="C0")
    axis.plot(x, result.total_background, label="Total background", color="C1")
    axis.plot(x, result.extracted_edge, label="Extracted edge", color="C2")
    total_uncertainty = getattr(result, "total_uncertainty", None)
    if total_uncertainty is not None:
        edge = result.extracted_edge
        axis.fill_between(
            x,
            edge - total_uncertainty,
            edge + total_uncertainty,
            color="C2",
            alpha=0.2,
            linewidth=0,
            label="Uncertainty",
        )
    axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axis.set_xlabel("Energy loss (eV)")
    axis.set_ylabel("Intensity")
    axis.legend()
    axis.grid(alpha=0.2)
    return axis

