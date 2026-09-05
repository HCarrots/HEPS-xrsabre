"""Fit-residual visualization."""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
from matplotlib.axes import Axes


def plot_residuals(result: Any, *, axis: Axes | None = None) -> Axes:
    """Plot residuals without clipping or smoothing them."""

    if axis is None:
        _, axis = plt.subplots()
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.plot(result.energy_loss_ev, result.fit_residual, color="C3", label="Residual")
    axis.set_xlabel("Energy loss (eV)")
    axis.set_ylabel("Observed - fitted")
    axis.legend()
    axis.grid(alpha=0.2)
    return axis

