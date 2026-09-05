"""Protocol shared by background-model implementations."""

from __future__ import annotations

from typing import Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray


class BackgroundModel(Protocol):
    """Minimal callable contract for a one-dimensional background model."""

    name: str

    def evaluate(self, energy_loss_ev: ArrayLike) -> NDArray[np.float64]:
        """Evaluate the background on an energy-loss coordinate."""

