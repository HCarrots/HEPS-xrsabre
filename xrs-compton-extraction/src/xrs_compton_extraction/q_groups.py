"""Explicit q-band classification for channel-level analysis."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np

QBand = Literal["low_q", "mid_high_q", "boundary"]


def classify_q_band(q_inverse_angstrom: float, *, threshold: float = 9.0) -> QBand:
    """Classify a q summary using strict ``<`` and ``>`` comparisons.

    The exact threshold is intentionally a separate ``boundary`` class; it is
    never silently assigned to either analysis regime.
    """

    q = float(q_inverse_angstrom)
    cutoff = float(threshold)
    if not np.isfinite(q) or not np.isfinite(cutoff) or cutoff <= 0:
        raise ValueError("q and threshold must be finite, with threshold > 0")
    if q < cutoff:
        return "low_q"
    if q > cutoff:
        return "mid_high_q"
    return "boundary"


def group_q_channels(
    channel_labels: Sequence[str],
    q_inverse_angstrom: Mapping[str, float],
    *,
    threshold: float = 9.0,
) -> dict[QBand, tuple[str, ...]]:
    """Return deterministic q-band channel labels, rejecting missing values."""

    labels = tuple(str(label) for label in channel_labels)
    if len(set(labels)) != len(labels):
        raise ValueError("channel_labels must be unique")
    missing = tuple(label for label in labels if label not in q_inverse_angstrom)
    if missing:
        raise ValueError(f"q is missing for channels: {', '.join(missing)}")
    grouped: dict[QBand, list[str]] = {
        "low_q": [], "mid_high_q": [], "boundary": []
    }
    for label in labels:
        grouped[classify_q_band(q_inverse_angstrom[label], threshold=threshold)].append(label)
    return {key: tuple(value) for key, value in grouped.items()}


__all__ = ["QBand", "classify_q_band", "group_q_channels"]
