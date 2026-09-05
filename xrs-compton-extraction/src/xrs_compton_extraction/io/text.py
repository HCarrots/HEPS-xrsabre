"""Strict CSV/TSV input for XRS spectra.

Text files carry far less metadata than NeXus files, so this adapter never
guesses the physical meaning or unit of an energy column.  Both must be stated
in :class:`TextMapping`.
"""

from __future__ import annotations

import csv
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any, Literal, TypeAlias

import numpy as np
from numpy.typing import NDArray

from ..data import XRSDataset, XRSSpectrum
from ..exceptions import DataDiscoveryError, DataValidationError

ColumnRef: TypeAlias = str | int
EnergyKind = Literal["energy_loss", "incident_energy"]


class TextMappingError(DataValidationError):
    """Raised when a text table cannot be interpreted by its explicit mapping."""


@dataclass(frozen=True, slots=True)
class TextMapping:
    """Column and physical metadata required to read one text spectrum."""

    energy_column: ColumnRef
    counts_column: ColumnRef
    energy_kind: EnergyKind
    energy_units: str
    delimiter: str | None = None
    has_header: bool = True
    encoding: str = "utf-8-sig"
    comment_prefix: str | None = "#"
    scattered_energy_column: ColumnRef | None = None
    scattered_energy_units: str | None = None
    fixed_scattered_energy_eV: float | None = None
    monitor_column: ColumnRef | None = None
    acquisition_time_column: ColumnRef | None = None
    uncertainty_column: ColumnRef | None = None
    q_inverse_angstrom_column: ColumnRef | None = None
    q_au_column: ColumnRef | None = None
    scan_id: str | None = None
    analyzer_id: str = ""
    roi_id: str = ""
    intensity_kind: Literal["counts", "processed"] = "counts"

    def __post_init__(self) -> None:
        if self.intensity_kind not in ("counts", "processed"):
            raise ValueError("intensity_kind must be 'counts' or 'processed'")
        if self.energy_kind not in ("energy_loss", "incident_energy"):
            raise ValueError(
                "energy_kind must explicitly be 'energy_loss' or 'incident_energy'."
            )
        if not isinstance(self.energy_units, str) or not self.energy_units.strip():
            raise ValueError("energy_units must explicitly be 'eV' or 'keV'.")
        _energy_factor(self.energy_units, label="energy_units", error_type=ValueError)
        if self.delimiter is not None and (
            not isinstance(self.delimiter, str) or len(self.delimiter) != 1
        ):
            raise ValueError("delimiter must be exactly one character.")
        if not isinstance(self.has_header, bool):
            raise TypeError("has_header must be a boolean.")
        if not isinstance(self.encoding, str) or not self.encoding:
            raise ValueError("encoding must be a non-empty string.")
        if self.comment_prefix is not None and not isinstance(self.comment_prefix, str):
            raise ValueError("comment_prefix must be a string or None.")
        if not self.has_header:
            for name, reference in self.column_references().items():
                if reference is not None and not isinstance(reference, int):
                    raise ValueError(
                        f"{name} must be an integer column index when has_header=False."
                    )
        if self.scattered_energy_column is not None:
            units = self.scattered_energy_units
            if not units:
                raise ValueError(
                    "scattered_energy_units is required with scattered_energy_column."
                )
            _energy_factor(units, label="scattered_energy_units", error_type=ValueError)
        elif self.scattered_energy_units is not None:
            raise ValueError(
                "scattered_energy_units requires scattered_energy_column."
            )
        if (
            self.scattered_energy_column is not None
            and self.fixed_scattered_energy_eV is not None
        ):
            raise ValueError(
                "Specify scattered_energy_column or fixed_scattered_energy_eV, not both."
            )
        if self.fixed_scattered_energy_eV is not None:
            value = float(self.fixed_scattered_energy_eV)
            if not np.isfinite(value) or value <= 0:
                raise ValueError("fixed_scattered_energy_eV must be finite and positive.")
        if self.energy_kind == "incident_energy" and (
            self.scattered_energy_column is None
            and self.fixed_scattered_energy_eV is None
        ):
            raise ValueError(
                "incident_energy requires scattered_energy_column or "
                "fixed_scattered_energy_eV; it cannot be treated as energy loss."
            )
        for name in ("scan_id", "analyzer_id", "roi_id"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise ValueError(f"{name} must be a string or None.")

    def column_references(self) -> dict[str, ColumnRef | None]:
        """Return every mapped logical field and its source-column reference."""

        return {
            "energy_column": self.energy_column,
            "counts_column": self.counts_column,
            "scattered_energy_column": self.scattered_energy_column,
            "monitor_column": self.monitor_column,
            "acquisition_time_column": self.acquisition_time_column,
            "uncertainty_column": self.uncertainty_column,
            "q_inverse_angstrom_column": self.q_inverse_angstrom_column,
            "q_au_column": self.q_au_column,
        }

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> TextMapping:
        """Construct a mapping from YAML/JSON-compatible values."""

        try:
            return cls(**dict(values))
        except (TypeError, ValueError) as exc:
            raise TextMappingError(f"Invalid text mapping: {exc}") from exc


def load_text(
    path: str | PathLike[str],
    mapping: TextMapping | Mapping[str, Any],
) -> XRSDataset:
    """Load one explicitly mapped CSV/TSV table as an :class:`XRSDataset`."""

    return load_text_channels(path, (_coerce_mapping(mapping),))


def load_text_channels(
    path: str | PathLike[str],
    mappings: Sequence[TextMapping | Mapping[str, Any]],
) -> XRSDataset:
    """Read a wide table once using one explicit mapping per channel.

    Parsing settings must agree across channels. Processed intensities are
    retained unchanged; neither normalization nor uncertainty is invented.
    """

    selected = tuple(_coerce_mapping(mapping) for mapping in mappings)
    if not selected:
        raise TextMappingError("at least one channel mapping is required")
    settings = ("delimiter", "has_header", "encoding", "comment_prefix")
    if any(
        getattr(mapping, setting) != getattr(selected[0], setting)
        for mapping in selected for setting in settings
    ):
        raise TextMappingError("channel mappings must share parsing settings")

    source = Path(path).expanduser()
    if not source.exists():
        raise DataDiscoveryError(f"Text data file does not exist: {source}")
    if not source.is_file():
        raise DataDiscoveryError(f"Text data path is not a file: {source}")

    text_mapping = selected[0]
    delimiter = text_mapping.delimiter or _delimiter_from_suffix(source)
    header, rows, source_line_numbers = _read_rows(
        source,
        delimiter=delimiter,
        encoding=text_mapping.encoding,
        comment_prefix=text_mapping.comment_prefix,
        has_header=text_mapping.has_header,
    )
    spectra = tuple(
        _mapped_spectrum(source, mapping, delimiter, header, rows, source_line_numbers)
        for mapping in selected
    )
    labels = [spectrum.channel_label for spectrum in spectra]
    if len(set(labels)) != len(labels):
        raise TextMappingError("channel labels must be unique; set analyzer_id/roi_id")
    return XRSDataset(
        spectra=spectra,
        metadata={"source_file": str(source.resolve()), "source_format": "text"},
        provenance={"source_files": (str(source.resolve()),)},
    )


def _mapped_spectrum(
    source: Path,
    text_mapping: TextMapping,
    delimiter: str,
    header: tuple[str, ...] | None,
    rows: list[list[str]],
    source_line_numbers: list[int],
) -> XRSSpectrum:
    indices = {
        logical_name: _resolve_column(
            reference,
            header=header,
            column_count=len(rows[0]),
            logical_name=logical_name,
        )
        for logical_name, reference in text_mapping.column_references().items()
        if reference is not None
    }
    columns = {
        logical_name: _numeric_column(
            rows,
            index,
            source_line_numbers=source_line_numbers,
            source=source,
            logical_name=logical_name,
        )
        for logical_name, index in indices.items()
    }

    energy_eV = _to_eV(
        columns["energy_column"],
        text_mapping.energy_units,
        label="energy_column",
    )
    counts = columns["counts_column"]
    if text_mapping.energy_kind == "energy_loss":
        primary_energy_eV = energy_eV
        energy_loss_eV = energy_eV
        incident_energy_eV: NDArray[np.float64] | None = None
        scattered_energy_eV: NDArray[np.float64] | float | None = None
    else:
        incident_energy_eV = energy_eV
        if text_mapping.scattered_energy_column is not None:
            scattered_energy_eV = _to_eV(
                columns["scattered_energy_column"],
                text_mapping.scattered_energy_units or "",
                label="scattered_energy_column",
            )
        else:
            scattered_energy_eV = float(text_mapping.fixed_scattered_energy_eV)  # type: ignore[arg-type]
        primary_energy_eV = incident_energy_eV
        energy_loss_eV = incident_energy_eV - scattered_energy_eV

    spectrum = XRSSpectrum(
        energy_eV=primary_energy_eV,
        counts=counts,
        energy_loss_eV=energy_loss_eV,
        incident_energy_ev=incident_energy_eV,
        scattered_energy_ev=scattered_energy_eV,
        q_inverse_angstrom=columns.get("q_inverse_angstrom_column"),
        q_au=columns.get("q_au_column"),
        monitor=columns.get("monitor_column"),
        acquisition_time_s=columns.get("acquisition_time_column"),
        uncertainty=columns.get("uncertainty_column"),
        scan_id=text_mapping.scan_id or source.stem,
        analyzer_id=text_mapping.analyzer_id,
        roi_id=text_mapping.roi_id,
        metadata={
            "source_file": str(source.resolve()),
            "source_format": "text",
            "intensity_kind": text_mapping.intensity_kind,
            "delimiter": delimiter,
            "energy_coordinate_kind": text_mapping.energy_kind,
            "energy_source_unit": text_mapping.energy_units,
            "column_mapping": {
                key: value for key, value in text_mapping.column_references().items()
            },
        },
    )
    return spectrum


def _coerce_mapping(mapping: TextMapping | Mapping[str, Any]) -> TextMapping:
    if isinstance(mapping, TextMapping):
        return mapping
    if isinstance(mapping, Mapping):
        return TextMapping.from_mapping(mapping)
    raise TypeError("mapping must be TextMapping or a mapping")


def _delimiter_from_suffix(path: Path) -> str:
    suffix = path.suffix.casefold()
    if suffix == ".csv":
        return ","
    if suffix in (".tsv", ".tab"):
        return "\t"
    raise TextMappingError(
        f"Cannot infer a delimiter from {path.name!r}; set TextMapping(delimiter=...)."
    )


def _read_rows(
    path: Path,
    *,
    delimiter: str,
    encoding: str,
    comment_prefix: str | None,
    has_header: bool,
) -> tuple[tuple[str, ...] | None, list[list[str]], list[int]]:
    filtered: list[tuple[int, list[str]]] = []
    try:
        with path.open("r", encoding=encoding, newline="") as stream:
            reader = csv.reader(stream, delimiter=delimiter)
            for line_number, row in enumerate(reader, start=1):
                if not row or not any(cell.strip() for cell in row):
                    continue
                first = row[0].lstrip()
                if comment_prefix and first.startswith(comment_prefix):
                    continue
                filtered.append((line_number, row))
    except (OSError, UnicodeError, csv.Error) as exc:
        raise TextMappingError(f"Could not read text data file {path}: {exc}") from exc

    if not filtered:
        raise TextMappingError(f"Text data file is empty: {path}")

    if has_header:
        _, raw_header = filtered.pop(0)
        header = tuple(cell.strip() for cell in raw_header)
        if any(not name for name in header):
            raise TextMappingError(f"Header in {path} contains an empty column name.")
        duplicates = sorted({name for name in header if header.count(name) > 1})
        if duplicates:
            raise TextMappingError(
                f"Header in {path} contains duplicate columns: {', '.join(duplicates)}."
            )
    else:
        header = None

    if not filtered:
        raise TextMappingError(f"Text data file {path} contains no data rows.")
    width = len(filtered[0][1])
    for line_number, row in filtered:
        if len(row) != width:
            raise TextMappingError(
                f"Row {line_number} in {path} has {len(row)} columns; expected {width}."
            )
    if header is not None and len(header) != width:
        raise TextMappingError(
            f"Header in {path} has {len(header)} columns, but data rows have {width}."
        )
    return header, [row for _, row in filtered], [line for line, _ in filtered]


def _resolve_column(
    reference: ColumnRef,
    *,
    header: tuple[str, ...] | None,
    column_count: int,
    logical_name: str,
) -> int:
    if isinstance(reference, bool):
        raise TextMappingError(f"{logical_name} must be a column name or integer index.")
    if isinstance(reference, int):
        index = reference if reference >= 0 else column_count + reference
        if not 0 <= index < column_count:
            raise TextMappingError(
                f"{logical_name} index {reference} is outside a {column_count}-column table."
            )
        return index
    if not isinstance(reference, str) or not reference:
        raise TextMappingError(f"{logical_name} must be a column name or integer index.")
    if header is None:
        raise TextMappingError(
            f"{logical_name} uses name {reference!r}, but the table has no header."
        )
    try:
        return header.index(reference)
    except ValueError as exc:
        available = ", ".join(header)
        raise TextMappingError(
            f"Mapped column {reference!r} for {logical_name} is missing; "
            f"available columns: {available}."
        ) from exc


def _numeric_column(
    rows: list[list[str]],
    index: int,
    *,
    source_line_numbers: list[int],
    source: Path,
    logical_name: str,
) -> NDArray[np.float64]:
    values = np.empty(len(rows), dtype=float)
    for position, (row, line_number) in enumerate(zip(rows, source_line_numbers, strict=True)):
        text = row[index].strip()
        try:
            value = float(text)
        except ValueError as exc:
            raise TextMappingError(
                f"Non-numeric value {text!r} for {logical_name} at "
                f"{source}:{line_number}."
            ) from exc
        if not np.isfinite(value):
            raise TextMappingError(
                f"Non-finite value {text!r} for {logical_name} at "
                f"{source}:{line_number}."
            )
        values[position] = value
    return values


def _to_eV(
    values: NDArray[np.float64], units: str, *, label: str
) -> NDArray[np.float64]:
    return np.asarray(values, dtype=float) * _energy_factor(
        units, label=label, error_type=TextMappingError
    )


def _energy_factor(
    units: str,
    *,
    label: str,
    error_type: type[Exception],
) -> float:
    canonical = re.sub(r"[^a-z0-9]+", "", units.casefold())
    if canonical in ("ev", "electronvolt", "electronvolts"):
        return 1.0
    if canonical in ("kev", "kiloelectronvolt", "kiloelectronvolts"):
        return 1_000.0
    raise error_type(
        f"Unsupported {label} {units!r}; supported energy units are eV and keV."
    )


__all__ = ["TextMapping", "TextMappingError", "load_text", "load_text_channels"]
