"""Per-channel extraction jobs with explicit failure and exclusion records."""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from .data import ExtractionResult, XRSDataset


@dataclass(frozen=True)
class BatchResult:
    results: Mapping[str, ExtractionResult]
    failures: Mapping[str, str]
    excluded_channels: tuple[str, ...]

    def __post_init__(self):
        object.__setattr__(self, "results", MappingProxyType(dict(self.results)))
        object.__setattr__(self, "failures", MappingProxyType(dict(self.failures)))


def extract_batch(
    dataset: XRSDataset,
    *,
    extractor: Callable[..., ExtractionResult],
    channel_options: Mapping[str, Mapping[str, Any]],
    excluded_channels: Sequence[str] = (),
    on_progress: Callable[[int, int, str], None] | None = None,
) -> BatchResult:
    """Run an explicit configuration for each channel; retain every failure.

    The caller selects the extractor (Pearson or Compton, for example). Model
    selection never silently falls back to a different physical model.
    """
    labels = [s.channel_label for s in dataset.spectra]
    if len(set(labels)) != len(labels):
        raise ValueError("channel labels must be unique; merge or relabel repeated scans first")
    unknown = (set(channel_options) | set(excluded_channels)) - set(labels)
    if unknown:
        raise ValueError(f"unknown channels: {sorted(unknown)}")
    results, failures = {}, {}
    excluded = set(excluded_channels)
    for index, spectrum in enumerate(dataset.spectra):
        label = spectrum.channel_label
        if label not in excluded:
            if label not in channel_options:
                failures[label] = "Missing explicit channel configuration"
            else:
                try:
                    result = extractor(spectrum, **dict(channel_options[label]))
                    if not isinstance(result, ExtractionResult):
                        raise TypeError("extractor must return ExtractionResult")
                    results[label] = result
                except Exception as exc:  # noqa: BLE001 - batch failures are recorded, not hidden
                    failures[label] = f"{type(exc).__name__}: {exc}"
        if on_progress is not None:
            on_progress(index + 1, len(labels), label)
    return BatchResult(results, failures, tuple(label for label in labels if label in excluded))
