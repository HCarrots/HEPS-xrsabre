"""Validated domain models for XRS/NRIXS analysis.

The objects in this module deliberately contain no file-format, user-interface,
or fitting logic.  They form the small, dependency-light contract shared by the
rest of :mod:`xrs_compton_extraction`.

Numerical inputs are copied on construction and exposed as read-only NumPy
arrays.  This prevents a caller from silently changing an analysis result by
mutating an array that was used to build it.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Iterator, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, ClassVar
from uuid import uuid4

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .exceptions import DataValidationError

FloatArray = NDArray[np.float64]
BoolArray = NDArray[np.bool_]

QUALITY_GRADES = frozenset({"Pass", "Warning", "Reject"})
SESSION_STATUSES = frozenset({"new", "ready", "running", "complete", "failed"})
BACKGROUND_STATUSES = frozenset({"success", "warning", "failed"})


def _require_identifier(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DataValidationError(f"{name} must be a non-empty string")
    return value.strip()


def _optional_identifier(value: object, name: str) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        raise DataValidationError(f"{name} must be a string")
    return value.strip()


def _finite_number(
    value: object,
    name: str,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise DataValidationError(f"{name} must be a finite number")
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise DataValidationError(f"{name} must be a finite number") from exc
    if not math.isfinite(number):
        raise DataValidationError(f"{name} must be finite")
    if positive and number <= 0:
        raise DataValidationError(f"{name} must be greater than zero")
    if nonnegative and number < 0:
        raise DataValidationError(f"{name} must be non-negative")
    return number


def _readonly_float_array(
    value: ArrayLike,
    name: str,
    *,
    ndim: int | tuple[int, ...] = 1,
    length: int | None = None,
    positive: bool = False,
    nonnegative: bool = False,
    allow_empty: bool = False,
) -> FloatArray:
    try:
        array = np.array(value, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise DataValidationError(f"{name} must be a numeric array") from exc

    allowed_ndim = (ndim,) if isinstance(ndim, int) else ndim
    if array.ndim not in allowed_ndim:
        allowed = " or ".join(str(item) for item in allowed_ndim)
        raise DataValidationError(f"{name} must have {allowed} dimension(s)")
    if array.ndim > 0 and not allow_empty and array.size == 0:
        raise DataValidationError(f"{name} must not be empty")
    if length is not None and array.ndim == 1 and len(array) != length:
        raise DataValidationError(
            f"{name} has length {len(array)}, expected {length}"
        )
    if not np.all(np.isfinite(array)):
        raise DataValidationError(f"{name} contains NaN or infinite values")
    if positive and np.any(array <= 0):
        raise DataValidationError(f"{name} must contain only positive values")
    if nonnegative and np.any(array < 0):
        raise DataValidationError(f"{name} must contain only non-negative values")
    array.setflags(write=False)
    return array


def _readonly_bool_array(value: ArrayLike, name: str) -> BoolArray:
    array = np.array(value, dtype=np.bool_, copy=True)
    if array.ndim != 2 or array.size == 0:
        raise DataValidationError(f"{name} must be a non-empty two-dimensional mask")
    array.setflags(write=False)
    return array


def _point_or_scalar_array(
    value: ArrayLike | None,
    name: str,
    length: int,
    *,
    positive: bool = False,
    nonnegative: bool = False,
) -> FloatArray | None:
    if value is None:
        return None
    array = _readonly_float_array(
        value,
        name,
        ndim=(0, 1),
        positive=positive,
        nonnegative=nonnegative,
    )
    if array.ndim == 1 and len(array) != length:
        raise DataValidationError(
            f"{name} has length {len(array)}, expected a scalar or {length} values"
        )
    return array


def _coordinate_array(
    value: ArrayLike | None,
    name: str,
    length: int,
    *,
    positive: bool = False,
) -> FloatArray | None:
    """Return a point-wise coordinate, broadcasting a scalar to ``length``."""

    array = _point_or_scalar_array(value, name, length, positive=positive)
    if array is None:
        return None
    if array.ndim == 0:
        broadcast = np.full(length, float(array), dtype=np.float64)
        broadcast.setflags(write=False)
        return broadcast
    return array


def _optional_series(
    value: ArrayLike | None,
    name: str,
    length: int,
    *,
    default: FloatArray | None = None,
    nonnegative: bool = False,
) -> FloatArray:
    if value is None:
        if default is None:
            result = np.zeros(length, dtype=np.float64)
        else:
            result = np.array(default, dtype=np.float64, copy=True)
        result.setflags(write=False)
        return result
    return _readonly_float_array(
        value,
        name,
        ndim=1,
        length=length,
        nonnegative=nonnegative,
    )


def _string_tuple(values: Iterable[object] | None, name: str) -> tuple[str, ...]:
    if values is None:
        return ()
    if isinstance(values, (str, bytes)):
        values = (values,)
    result: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value.strip():
            raise DataValidationError(f"{name} must contain non-empty strings")
        result.append(value.strip())
    return tuple(result)


def _fit_windows(
    values: Iterable[Sequence[object]] | None,
    name: str = "fit_windows",
) -> tuple[tuple[float, float], ...]:
    if values is None:
        return ()
    result: list[tuple[float, float]] = []
    for index, window in enumerate(values):
        if len(window) != 2:
            raise DataValidationError(f"{name}[{index}] must contain exactly two bounds")
        lower = _finite_number(window[0], f"{name}[{index}][0]")
        upper = _finite_number(window[1], f"{name}[{index}][1]")
        if lower >= upper:
            raise DataValidationError(f"{name}[{index}] lower bound must be below upper bound")
        result.append((lower, upper))
    return tuple(result)


def _numeric_mapping(
    value: Mapping[object, object] | None,
    name: str,
    *,
    nonnegative: bool = False,
) -> dict[str, float]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise DataValidationError(f"{name} must be a mapping")
    result: dict[str, float] = {}
    for raw_key, raw_value in value.items():
        key = _require_identifier(raw_key, f"{name} key")
        if key in result:
            raise DataValidationError(f"{name} contains duplicate key {key!r}")
        result[key] = _finite_number(
            raw_value,
            f"{name}[{key!r}]",
            nonnegative=nonnegative,
        )
    return result


def _array_mapping(
    value: Mapping[object, ArrayLike] | None,
    name: str,
    length: int,
    *,
    nonnegative: bool = False,
) -> dict[str, FloatArray]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise DataValidationError(f"{name} must be a mapping")
    result: dict[str, FloatArray] = {}
    for raw_key, raw_value in value.items():
        key = _require_identifier(raw_key, f"{name} key")
        if key in result:
            raise DataValidationError(f"{name} contains duplicate key {key!r}")
        result[key] = _readonly_float_array(
            raw_value,
            f"{name}[{key!r}]",
            ndim=1,
            length=length,
            nonnegative=nonnegative,
        )
    return result


def _json_safe(value: Any, path: str = "value") -> Any:
    """Copy ``value`` into a JSON-compatible representation.

    NumPy values and :class:`~pathlib.Path` objects are accepted at the public
    boundary because they commonly occur in scientific metadata, but they are
    normalized immediately so exported provenance never depends on their
    implementation-specific encoders.
    """

    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise DataValidationError(f"{path} contains NaN or an infinite value")
        return value
    if isinstance(value, np.generic):
        return _json_safe(value.item(), path)
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist(), path)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for raw_key, raw_value in value.items():
            if not isinstance(raw_key, str) or not raw_key:
                raise DataValidationError(f"{path} mapping keys must be non-empty strings")
            result[raw_key] = _json_safe(raw_value, f"{path}.{raw_key}")
        return result
    if isinstance(value, (list, tuple)):
        return [_json_safe(item, f"{path}[]") for item in value]
    raise DataValidationError(
        f"{path} contains non-serializable value of type {type(value).__name__}"
    )


def _metadata(value: Mapping[str, Any] | None, name: str = "metadata") -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise DataValidationError(f"{name} must be a mapping")
    result = _json_safe(value, name)
    # This is intentionally an eager check.  It catches implementation mistakes
    # in the normalization helper as well as unsupported values supplied by users.
    try:
        json.dumps(result, allow_nan=False)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive guard
        raise DataValidationError(f"{name} must be JSON serializable") from exc
    return result


def _serializable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if is_dataclass(value) and not isinstance(value, type):
        return {item.name: _serializable(getattr(value, item.name)) for item in fields(value)}
    if isinstance(value, Mapping):
        return {str(key): _serializable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_serializable(item) for item in value]
    if isinstance(value, list):
        return [_serializable(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return deepcopy(value)


class SerializableModel:
    """Mixin providing a JSON-compatible snapshot of a dataclass model."""

    def to_dict(self) -> dict[str, Any]:
        if not is_dataclass(self):  # pragma: no cover - protects misuse of mixin
            raise TypeError("SerializableModel must be used with a dataclass")
        return {
            item.name: _serializable(getattr(self, item.name))
            for item in fields(self)
            if not item.name.startswith("_")
        }


@dataclass(frozen=True, slots=True)
class XRSSpectrum(SerializableModel):
    """One XRS spectrum for a scan/analyzer/ROI channel."""

    energy_eV: ArrayLike
    counts: ArrayLike
    energy_loss_eV: ArrayLike | None = None
    incident_energy_ev: ArrayLike | None = None
    scattered_energy_ev: ArrayLike | None = None
    q_inverse_angstrom: ArrayLike | None = None
    q_au: ArrayLike | None = None
    monitor: ArrayLike | None = None
    acquisition_time_s: ArrayLike | None = None
    uncertainty: ArrayLike | None = None
    scan_id: str = ""
    analyzer_id: str = ""
    roi_id: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        energy = _readonly_float_array(self.energy_eV, "energy_eV")
        counts = _readonly_float_array(
            self.counts,
            "counts",
            length=len(energy),
            nonnegative=True,
        )
        object.__setattr__(self, "energy_eV", energy)
        object.__setattr__(self, "counts", counts)
        incident = _coordinate_array(
            self.incident_energy_ev,
            "incident_energy_ev",
            len(energy),
            positive=True,
        )
        scattered = _coordinate_array(
            self.scattered_energy_ev,
            "scattered_energy_ev",
            len(energy),
            positive=True,
        )
        energy_loss = _coordinate_array(
            self.energy_loss_eV, "energy_loss_eV", len(energy)
        )
        if energy_loss is None and incident is not None and scattered is not None:
            energy_loss = _readonly_float_array(
                incident - scattered, "energy_loss_eV", length=len(energy)
            )
        elif (
            energy_loss is not None
            and incident is not None
            and scattered is not None
            and not np.allclose(
                energy_loss, incident - scattered, rtol=1e-10, atol=1e-9
            )
        ):
            raise DataValidationError(
                "energy_loss_eV must equal incident_energy_ev - scattered_energy_ev"
            )
        object.__setattr__(self, "energy_loss_eV", energy_loss)
        object.__setattr__(self, "incident_energy_ev", incident)
        object.__setattr__(self, "scattered_energy_ev", scattered)
        object.__setattr__(
            self,
            "q_inverse_angstrom",
            _coordinate_array(
                self.q_inverse_angstrom,
                "q_inverse_angstrom",
                len(energy),
                positive=True,
            ),
        )
        object.__setattr__(
            self,
            "q_au",
            _coordinate_array(self.q_au, "q_au", len(energy), positive=True),
        )
        object.__setattr__(
            self,
            "monitor",
            _point_or_scalar_array(
                self.monitor,
                "monitor",
                len(energy),
                positive=True,
            ),
        )
        object.__setattr__(
            self,
            "acquisition_time_s",
            _point_or_scalar_array(
                self.acquisition_time_s,
                "acquisition_time_s",
                len(energy),
                positive=True,
            ),
        )
        object.__setattr__(
            self,
            "uncertainty",
            _point_or_scalar_array(
                self.uncertainty,
                "uncertainty",
                len(energy),
                nonnegative=True,
            ),
        )
        object.__setattr__(self, "scan_id", _optional_identifier(self.scan_id, "scan_id"))
        object.__setattr__(
            self, "analyzer_id", _optional_identifier(self.analyzer_id, "analyzer_id")
        )
        object.__setattr__(self, "roi_id", _optional_identifier(self.roi_id, "roi_id"))
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    def __len__(self) -> int:
        return len(self.energy_eV)

    @property
    def energy_ev(self) -> FloatArray:
        """PEP 8 alias for the primary energy coordinate."""

        return self.energy_eV

    @property
    def energy_loss_ev(self) -> FloatArray | None:
        """PEP 8 alias for :attr:`energy_loss_eV`."""

        return self.energy_loss_eV

    @property
    def raw_counts(self) -> FloatArray:
        """Unmodified detector counts (read-only)."""

        return self.counts

    @property
    def i0(self) -> FloatArray | None:
        """Alias for the incident-flux monitor."""

        return self.monitor

    @property
    def incident_energy_eV(self) -> FloatArray | None:
        """Compatibility alias retaining the conventional written unit."""

        return self.incident_energy_ev

    @property
    def scattered_energy_eV(self) -> FloatArray | None:
        """Compatibility alias retaining the conventional written unit."""

        return self.scattered_energy_ev

    @property
    def channel_label(self) -> str:
        """Return a stable, human-readable label for selectors and plots."""

        parts: list[str] = []
        if self.analyzer_id:
            parts.append(f"analyzer={self.analyzer_id}")
        if self.roi_id:
            parts.append(f"roi={self.roi_id}")
        if self.q_inverse_angstrom is not None and np.allclose(
            self.q_inverse_angstrom, self.q_inverse_angstrom[0]
        ):
            parts.append(f"q={self.q_inverse_angstrom[0]:g} 1/angstrom")
        if not parts and self.scan_id:
            parts.append(f"scan={self.scan_id}")
        return ", ".join(parts) if parts else "unassigned"


@dataclass(frozen=True, slots=True)
class Scan(SerializableModel):
    """Acquisition-level metadata shared by one or more spectra."""

    scan_id: str
    source_file: str | Path | None = None
    incident_energy_eV: ArrayLike | None = None
    acquisition_time_s: ArrayLike | None = None
    monitor: ArrayLike | None = None
    timestamp: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "scan_id", _require_identifier(self.scan_id, "scan_id"))
        if self.source_file is not None:
            if not isinstance(self.source_file, (str, Path)) or not str(self.source_file):
                raise DataValidationError("source_file must be a non-empty path")
            object.__setattr__(self, "source_file", str(self.source_file))

        energy = None
        if self.incident_energy_eV is not None:
            energy = _readonly_float_array(
                self.incident_energy_eV,
                "incident_energy_eV",
                ndim=(0, 1),
                positive=True,
            )
            object.__setattr__(self, "incident_energy_eV", energy)
        point_count = len(energy) if energy is not None and energy.ndim == 1 else None
        time = (
            _readonly_float_array(
                self.acquisition_time_s,
                "acquisition_time_s",
                ndim=(0, 1),
                positive=True,
            )
            if self.acquisition_time_s is not None
            else None
        )
        monitor = (
            _readonly_float_array(
                self.monitor,
                "monitor",
                ndim=(0, 1),
                positive=True,
            )
            if self.monitor is not None
            else None
        )
        for name, array in (("acquisition_time_s", time), ("monitor", monitor)):
            if (
                point_count is not None
                and array is not None
                and array.ndim == 1
                and len(array) != point_count
            ):
                raise DataValidationError(
                    f"{name} has length {len(array)}, expected {point_count}"
                )
        object.__setattr__(self, "acquisition_time_s", time)
        object.__setattr__(self, "monitor", monitor)
        if self.timestamp is not None and (
            not isinstance(self.timestamp, str) or not self.timestamp.strip()
        ):
            raise DataValidationError("timestamp must be a non-empty string when supplied")
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    @property
    def incident_energy_ev(self) -> FloatArray | None:
        """PEP 8 alias for :attr:`incident_energy_eV`."""

        return self.incident_energy_eV


@dataclass(frozen=True, slots=True)
class Analyzer(SerializableModel):
    """Analyzer crystal geometry, calibration, and efficiency metadata."""

    analyzer_id: str
    scattering_angle_deg: float | None = None
    direction: ArrayLike | None = None
    efficiency: ArrayLike | None = None
    energy_offset_eV: float = 0.0
    resolution_eV: float | None = None
    calibration: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "analyzer_id", _require_identifier(self.analyzer_id, "analyzer_id")
        )
        if self.scattering_angle_deg is not None:
            angle = _finite_number(self.scattering_angle_deg, "scattering_angle_deg")
            if not 0 <= angle <= 180:
                raise DataValidationError("scattering_angle_deg must be within [0, 180]")
            object.__setattr__(self, "scattering_angle_deg", angle)
        if self.direction is not None:
            direction = _readonly_float_array(self.direction, "direction", length=3)
            norm = float(np.linalg.norm(direction))
            if norm == 0:
                raise DataValidationError("direction must not be the zero vector")
            if not np.isclose(norm, 1.0, rtol=1e-7, atol=1e-10):
                raise DataValidationError("direction must be a unit vector")
            object.__setattr__(self, "direction", direction)
        if self.efficiency is not None:
            object.__setattr__(
                self,
                "efficiency",
                _readonly_float_array(
                    self.efficiency,
                    "efficiency",
                    ndim=(0, 1),
                    positive=True,
                ),
            )
        object.__setattr__(
            self, "energy_offset_eV", _finite_number(self.energy_offset_eV, "energy_offset_eV")
        )
        if self.resolution_eV is not None:
            object.__setattr__(
                self,
                "resolution_eV",
                _finite_number(self.resolution_eV, "resolution_eV", positive=True),
            )
        object.__setattr__(self, "calibration", _metadata(self.calibration, "calibration"))
        object.__setattr__(self, "metadata", _metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class ROI(SerializableModel):
    """Detector region of interest and its optional analyzer association."""

    roi_id: str
    analyzer_id: str = ""
    bounds: tuple[int, int, int, int] | None = None
    mask: ArrayLike | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "roi_id", _require_identifier(self.roi_id, "roi_id"))
        object.__setattr__(
            self, "analyzer_id", _optional_identifier(self.analyzer_id, "analyzer_id")
        )
        if self.bounds is not None:
            if len(self.bounds) != 4 or any(
                isinstance(value, bool) or not isinstance(value, (int, np.integer))
                for value in self.bounds
            ):
                raise DataValidationError(
                    "bounds must be four integers: (x_start, x_stop, y_start, y_stop)"
                )
            x_start, x_stop, y_start, y_stop = (int(value) for value in self.bounds)
            if min(x_start, x_stop, y_start, y_stop) < 0:
                raise DataValidationError("bounds must be non-negative")
            if x_start >= x_stop or y_start >= y_stop:
                raise DataValidationError("bounds start values must be below stop values")
            object.__setattr__(self, "bounds", (x_start, x_stop, y_start, y_stop))
        if self.mask is not None:
            object.__setattr__(self, "mask", _readonly_bool_array(self.mask, "mask"))
        object.__setattr__(self, "metadata", _metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class Sample(SerializableModel):
    """Sample composition and macroscopic absorption parameters."""

    name: str
    composition: Mapping[str, float] = field(default_factory=dict)
    density_g_cm3: float | None = None
    thickness_um: float | None = None
    environment: str | None = None
    absorption_source: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "name", _require_identifier(self.name, "name"))
        composition = _numeric_mapping(self.composition, "composition", nonnegative=True)
        if any(value == 0 for value in composition.values()):
            raise DataValidationError("composition coefficients must be greater than zero")
        object.__setattr__(self, "composition", composition)
        if self.density_g_cm3 is not None:
            object.__setattr__(
                self,
                "density_g_cm3",
                _finite_number(self.density_g_cm3, "density_g_cm3", positive=True),
            )
        if self.thickness_um is not None:
            object.__setattr__(
                self,
                "thickness_um",
                _finite_number(self.thickness_um, "thickness_um", positive=True),
            )
        for name in ("environment", "absorption_source"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise DataValidationError(f"{name} must be a non-empty string when supplied")
        object.__setattr__(self, "metadata", _metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class Geometry(SerializableModel):
    """Experiment geometry in a documented Cartesian coordinate system."""

    scattering_angle_deg: float | None = None
    incident_direction: ArrayLike | None = None
    scattered_direction: ArrayLike | None = None
    sample_normal: ArrayLike | None = None
    path_lengths_mm: Mapping[str, float] = field(default_factory=dict)
    coordinate_system: str = "laboratory"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.scattering_angle_deg is not None:
            angle = _finite_number(self.scattering_angle_deg, "scattering_angle_deg")
            if not 0 <= angle <= 180:
                raise DataValidationError("scattering_angle_deg must be within [0, 180]")
            object.__setattr__(self, "scattering_angle_deg", angle)
        for name in ("incident_direction", "scattered_direction", "sample_normal"):
            value = getattr(self, name)
            if value is None:
                continue
            vector = _readonly_float_array(value, name, length=3)
            if not np.isclose(np.linalg.norm(vector), 1.0, rtol=1e-7, atol=1e-10):
                raise DataValidationError(f"{name} must be a unit vector")
            object.__setattr__(self, name, vector)
        object.__setattr__(
            self,
            "path_lengths_mm",
            _numeric_mapping(self.path_lengths_mm, "path_lengths_mm", nonnegative=True),
        )
        object.__setattr__(
            self, "coordinate_system", _require_identifier(self.coordinate_system, "coordinate_system")
        )
        object.__setattr__(self, "metadata", _metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class XRSDataset(SerializableModel):
    """A validated collection of spectra and their experiment metadata."""

    spectra: Sequence[XRSSpectrum]
    scans: Sequence[Scan] = field(default_factory=tuple)
    analyzers: Sequence[Analyzer] = field(default_factory=tuple)
    rois: Sequence[ROI] = field(default_factory=tuple)
    sample: Sample | None = None
    geometry: Geometry | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        spectra = tuple(self.spectra)
        scans = tuple(self.scans)
        analyzers = tuple(self.analyzers)
        rois = tuple(self.rois)
        if not spectra:
            raise DataValidationError("spectra must contain at least one XRSSpectrum")
        if not all(isinstance(item, XRSSpectrum) for item in spectra):
            raise DataValidationError("spectra must contain only XRSSpectrum objects")
        if not all(isinstance(item, Scan) for item in scans):
            raise DataValidationError("scans must contain only Scan objects")
        if not all(isinstance(item, Analyzer) for item in analyzers):
            raise DataValidationError("analyzers must contain only Analyzer objects")
        if not all(isinstance(item, ROI) for item in rois):
            raise DataValidationError("rois must contain only ROI objects")
        if self.sample is not None and not isinstance(self.sample, Sample):
            raise DataValidationError("sample must be a Sample object")
        if self.geometry is not None and not isinstance(self.geometry, Geometry):
            raise DataValidationError("geometry must be a Geometry object")

        self._validate_unique_ids(scans, "scan_id", "scans")
        self._validate_unique_ids(analyzers, "analyzer_id", "analyzers")
        self._validate_unique_ids(rois, "roi_id", "rois")

        spectrum_keys = [
            (item.scan_id, item.analyzer_id, item.roi_id) for item in spectra
        ]
        if len(spectrum_keys) != len(set(spectrum_keys)):
            raise DataValidationError(
                "spectra contain duplicate (scan_id, analyzer_id, roi_id) identifiers"
            )

        known_scan_ids = {item.scan_id for item in scans}
        known_analyzer_ids = {item.analyzer_id for item in analyzers}
        known_roi_ids = {item.roi_id for item in rois}
        for spectrum in spectra:
            if known_scan_ids and spectrum.scan_id and spectrum.scan_id not in known_scan_ids:
                raise DataValidationError(f"unknown spectrum scan_id {spectrum.scan_id!r}")
            if (
                known_analyzer_ids
                and spectrum.analyzer_id
                and spectrum.analyzer_id not in known_analyzer_ids
            ):
                raise DataValidationError(
                    f"unknown spectrum analyzer_id {spectrum.analyzer_id!r}"
                )
            if known_roi_ids and spectrum.roi_id and spectrum.roi_id not in known_roi_ids:
                raise DataValidationError(f"unknown spectrum roi_id {spectrum.roi_id!r}")

        object.__setattr__(self, "spectra", spectra)
        object.__setattr__(self, "scans", scans)
        object.__setattr__(self, "analyzers", analyzers)
        object.__setattr__(self, "rois", rois)
        object.__setattr__(self, "metadata", _metadata(self.metadata))
        object.__setattr__(self, "provenance", _metadata(self.provenance, "provenance"))

    @staticmethod
    def _validate_unique_ids(items: Sequence[Any], attr: str, name: str) -> None:
        identifiers = [getattr(item, attr) for item in items]
        if len(identifiers) != len(set(identifiers)):
            raise DataValidationError(f"{name} must have unique {attr} values")

    def __len__(self) -> int:
        return len(self.spectra)

    def __iter__(self) -> Iterator[XRSSpectrum]:
        return iter(self.spectra)

    def spectrum(
        self,
        *,
        scan_id: str = "",
        analyzer_id: str = "",
        roi_id: str = "",
    ) -> XRSSpectrum:
        """Return exactly one spectrum matching the supplied identifiers."""

        matches = [
            item
            for item in self.spectra
            if (not scan_id or item.scan_id == scan_id)
            and (not analyzer_id or item.analyzer_id == analyzer_id)
            and (not roi_id or item.roi_id == roi_id)
        ]
        if len(matches) != 1:
            raise DataValidationError(
                f"spectrum selection matched {len(matches)} spectra; expected exactly one"
            )
        return matches[0]

    @property
    def source_count(self) -> int:
        """Count distinct source files, with a deterministic metadata fallback."""

        sources: list[str] = []

        def add(raw: Any) -> None:
            if isinstance(raw, (str, Path)) and str(raw):
                sources.append(str(raw))
            elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
                for item in raw:
                    if isinstance(item, (str, Path)) and str(item):
                        sources.append(str(item))

        for mapping in (self.provenance, self.metadata):
            add(mapping.get("source_files"))
            add(mapping.get("source_file"))
        for scan in self.scans:
            add(scan.source_file)
        for spectrum in self.spectra:
            add(spectrum.metadata.get("source_file"))
            add(spectrum.metadata.get("source_path"))
        if sources:
            return len(dict.fromkeys(sources))
        scan_ids = tuple(dict.fromkeys(item.scan_id for item in self.spectra if item.scan_id))
        return len(scan_ids) if scan_ids else len(self.spectra)


@dataclass(frozen=True, slots=True)
class AnalysisConfig(SerializableModel):
    """Versioned, serializable analysis configuration."""

    schema_version: str = "1.0"
    data_path: str | Path | None = None
    files: Sequence[str | Path] = field(default_factory=tuple)
    scan_ids: Sequence[str] = field(default_factory=tuple)
    roi_ids: Sequence[str] = field(default_factory=tuple)
    analyzer_ids: Sequence[str] = field(default_factory=tuple)
    sample: Sample | None = None
    geometry: Geometry | None = None
    target_edge: str | None = None
    target_edge_energy_eV: float | None = None
    fit_windows: Iterable[Sequence[float]] = field(default_factory=tuple)
    core_normalization_windows: Iterable[Sequence[float]] = field(default_factory=tuple)
    background_model: str = "auto"
    correction_flags: Mapping[str, bool] = field(default_factory=dict)
    model_parameters: Mapping[str, Any] = field(default_factory=dict)
    parameter_bounds: Mapping[str, Any] = field(default_factory=dict)
    smoothing_sigma: float | None = None
    risk_thresholds: Mapping[str, float] = field(default_factory=dict)
    excluded_channels: Sequence[str] = field(default_factory=tuple)
    software_version: str | None = None
    analysis_time: str | None = None
    raw_data_identifiers: Sequence[str] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    SUPPORTED_BACKGROUND_MODELS: ClassVar[frozenset[str]] = frozenset(
        {"auto", "pearson", "compton_profile", "polynomial"}
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "schema_version", _require_identifier(self.schema_version, "schema_version")
        )
        if self.data_path is not None:
            if not isinstance(self.data_path, (str, Path)) or not str(self.data_path):
                raise DataValidationError("data_path must be a non-empty path when supplied")
            object.__setattr__(self, "data_path", str(self.data_path))
        files = tuple(str(item) for item in self.files)
        if any(not item for item in files):
            raise DataValidationError("files must contain non-empty paths")
        object.__setattr__(self, "files", files)
        for name in ("scan_ids", "roi_ids", "analyzer_ids", "excluded_channels"):
            values = _string_tuple(getattr(self, name), name)
            if len(values) != len(set(values)):
                raise DataValidationError(f"{name} must not contain duplicates")
            object.__setattr__(self, name, values)
        if self.sample is not None and not isinstance(self.sample, Sample):
            raise DataValidationError("sample must be a Sample object")
        if self.geometry is not None and not isinstance(self.geometry, Geometry):
            raise DataValidationError("geometry must be a Geometry object")
        if self.target_edge is not None:
            object.__setattr__(
                self, "target_edge", _require_identifier(self.target_edge, "target_edge")
            )
        if self.target_edge_energy_eV is not None:
            object.__setattr__(
                self,
                "target_edge_energy_eV",
                _finite_number(
                    self.target_edge_energy_eV,
                    "target_edge_energy_eV",
                    positive=True,
                ),
            )
        object.__setattr__(self, "fit_windows", _fit_windows(self.fit_windows))
        object.__setattr__(
            self,
            "core_normalization_windows",
            _fit_windows(self.core_normalization_windows, "core_normalization_windows"),
        )
        model = _require_identifier(self.background_model, "background_model").lower()
        if model not in self.SUPPORTED_BACKGROUND_MODELS:
            choices = ", ".join(sorted(self.SUPPORTED_BACKGROUND_MODELS))
            raise DataValidationError(f"background_model must be one of: {choices}")
        object.__setattr__(self, "background_model", model)
        if not isinstance(self.correction_flags, Mapping) or any(
            not isinstance(key, str) or not key or not isinstance(value, bool)
            for key, value in self.correction_flags.items()
        ):
            raise DataValidationError("correction_flags must map non-empty strings to bool values")
        object.__setattr__(self, "correction_flags", dict(self.correction_flags))
        object.__setattr__(
            self,
            "model_parameters",
            _metadata(self.model_parameters, "model_parameters"),
        )
        object.__setattr__(
            self,
            "parameter_bounds",
            _metadata(self.parameter_bounds, "parameter_bounds"),
        )
        if self.smoothing_sigma is not None:
            object.__setattr__(
                self,
                "smoothing_sigma",
                _finite_number(self.smoothing_sigma, "smoothing_sigma", nonnegative=True),
            )
        object.__setattr__(
            self,
            "risk_thresholds",
            _numeric_mapping(self.risk_thresholds, "risk_thresholds", nonnegative=True),
        )
        for name in ("software_version", "analysis_time"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise DataValidationError(f"{name} must be a non-empty string when supplied")
        object.__setattr__(
            self,
            "raw_data_identifiers",
            _string_tuple(self.raw_data_identifiers, "raw_data_identifiers"),
        )
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    @property
    def target_edge_energy_ev(self) -> float | None:
        """PEP 8 alias for :attr:`target_edge_energy_eV`."""

        return self.target_edge_energy_eV


@dataclass(frozen=True, slots=True)
class CorrectionResult(SerializableModel):
    """The auditable output of the correction pipeline."""

    raw_counts: ArrayLike
    normalized_intensity: ArrayLike
    corrected_intensity: ArrayLike
    correction_factors: Mapping[str, ArrayLike] = field(default_factory=dict)
    statistical_uncertainty: ArrayLike | None = None
    component_uncertainties: Mapping[str, ArrayLike] = field(default_factory=dict)
    warnings: Sequence[str] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        raw = _readonly_float_array(self.raw_counts, "raw_counts", nonnegative=True)
        length = len(raw)
        object.__setattr__(self, "raw_counts", raw)
        object.__setattr__(
            self,
            "normalized_intensity",
            _readonly_float_array(
                self.normalized_intensity, "normalized_intensity", length=length
            ),
        )
        object.__setattr__(
            self,
            "corrected_intensity",
            _readonly_float_array(self.corrected_intensity, "corrected_intensity", length=length),
        )
        object.__setattr__(
            self,
            "correction_factors",
            _array_mapping(
                self.correction_factors,
                "correction_factors",
                length,
                nonnegative=True,
            ),
        )
        object.__setattr__(
            self,
            "statistical_uncertainty",
            None
            if self.statistical_uncertainty is None
            else _readonly_float_array(
                self.statistical_uncertainty,
                "statistical_uncertainty",
                length=length,
                nonnegative=True,
            ),
        )
        object.__setattr__(
            self,
            "component_uncertainties",
            _array_mapping(
                self.component_uncertainties,
                "component_uncertainties",
                length,
                nonnegative=True,
            ),
        )
        object.__setattr__(self, "warnings", _string_tuple(self.warnings, "warnings"))
        object.__setattr__(self, "metadata", _metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class BackgroundResult(SerializableModel):
    """Background components and diagnostics from one fitted model."""

    energy_loss_eV: ArrayLike
    components: Mapping[str, ArrayLike]
    total_background: ArrayLike | None = None
    model_name: str = "unknown"
    fit_parameters: Mapping[str, float] = field(default_factory=dict)
    parameter_covariance: ArrayLike | None = None
    fit_windows: Iterable[Sequence[float]] = field(default_factory=tuple)
    residual: ArrayLike | None = None
    status: str = "success"
    warnings: Sequence[str] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        energy = _readonly_float_array(self.energy_loss_eV, "energy_loss_eV")
        length = len(energy)
        components = _array_mapping(self.components, "components", length)
        if not components:
            raise DataValidationError("components must contain at least one background component")
        component_total = np.sum(np.stack(tuple(components.values())), axis=0)
        if self.total_background is None:
            total = _readonly_float_array(component_total, "total_background")
        else:
            total = _readonly_float_array(
                self.total_background, "total_background", length=length
            )
            if not np.allclose(total, component_total, rtol=1e-8, atol=1e-10):
                raise DataValidationError(
                    "total_background must equal the sum of all background components"
                )
        parameters = _numeric_mapping(self.fit_parameters, "fit_parameters")
        covariance = None
        if self.parameter_covariance is not None:
            covariance = _readonly_float_array(
                self.parameter_covariance,
                "parameter_covariance",
                ndim=2,
            )
            if covariance.shape[0] != covariance.shape[1]:
                raise DataValidationError("parameter_covariance must be square")
            if parameters and covariance.shape[0] != len(parameters):
                raise DataValidationError(
                    "parameter_covariance dimension must match fit_parameters"
                )
        residual = None
        if self.residual is not None:
            residual = _readonly_float_array(self.residual, "residual", length=length)
        status = _require_identifier(self.status, "status").lower()
        if status not in BACKGROUND_STATUSES:
            raise DataValidationError(
                f"status must be one of: {', '.join(sorted(BACKGROUND_STATUSES))}"
            )
        object.__setattr__(self, "energy_loss_eV", energy)
        object.__setattr__(self, "components", components)
        object.__setattr__(self, "total_background", total)
        object.__setattr__(self, "model_name", _require_identifier(self.model_name, "model_name"))
        object.__setattr__(self, "fit_parameters", parameters)
        object.__setattr__(self, "parameter_covariance", covariance)
        object.__setattr__(self, "fit_windows", _fit_windows(self.fit_windows))
        object.__setattr__(self, "residual", residual)
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "warnings", _string_tuple(self.warnings, "warnings"))
        object.__setattr__(self, "metadata", _metadata(self.metadata))

    @property
    def energy_loss_ev(self) -> FloatArray:
        """PEP 8 alias for :attr:`energy_loss_eV`."""

        return self.energy_loss_eV


@dataclass(frozen=True, slots=True)
class ExtractionResult(SerializableModel):
    """Complete, point-wise result for one extracted target edge."""

    energy_loss_eV: ArrayLike
    raw_counts: ArrayLike
    q_au: ArrayLike | None = None
    q_inverse_angstrom: ArrayLike | None = None
    normalized_intensity: ArrayLike | None = None
    corrected_intensity: ArrayLike | None = None
    elastic_component: ArrayLike | None = None
    stray_background: ArrayLike | None = None
    valence_background: ArrayLike | None = None
    core_background: ArrayLike | None = None
    constant_background: ArrayLike | None = None
    total_background: ArrayLike | None = None
    extracted_edge: ArrayLike | None = None
    fit_residual: ArrayLike | None = None
    statistical_uncertainty: ArrayLike | None = None
    model_uncertainty: ArrayLike | None = None
    total_uncertainty: ArrayLike | None = None
    background_model_name: str = "unknown"
    fit_parameters: Mapping[str, float] = field(default_factory=dict)
    parameter_covariance: ArrayLike | None = None
    fit_windows: Iterable[Sequence[float]] = field(default_factory=tuple)
    risk_metrics: Mapping[str, float] = field(default_factory=dict)
    warnings: Sequence[str] = field(default_factory=tuple)
    quality_grade: str = "Warning"
    provenance: Mapping[str, Any] = field(default_factory=dict)
    software_version: str = "unknown"
    config_digest: str = ""
    raw_data_identifiers: Sequence[str] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        energy = _readonly_float_array(self.energy_loss_eV, "energy_loss_eV")
        raw = _readonly_float_array(
            self.raw_counts, "raw_counts", length=len(energy), nonnegative=True
        )
        length = len(energy)
        normalized = _optional_series(
            self.normalized_intensity,
            "normalized_intensity",
            length,
            default=raw,
        )
        corrected = _optional_series(
            self.corrected_intensity,
            "corrected_intensity",
            length,
            default=normalized,
        )
        elastic = _optional_series(self.elastic_component, "elastic_component", length)
        stray = _optional_series(self.stray_background, "stray_background", length)
        valence = _optional_series(self.valence_background, "valence_background", length)
        core = _optional_series(self.core_background, "core_background", length)
        constant = _optional_series(self.constant_background, "constant_background", length)
        calculated_total = elastic + stray + valence + core + constant
        if self.total_background is None:
            total = _readonly_float_array(calculated_total, "total_background")
        else:
            total = _readonly_float_array(
                self.total_background, "total_background", length=length
            )
            if not np.allclose(total, calculated_total, rtol=1e-8, atol=1e-10):
                raise DataValidationError(
                    "total_background must equal elastic_component + stray_background + "
                    "valence_background + core_background + constant_background"
                )
        if self.extracted_edge is None:
            extracted = _readonly_float_array(corrected - total, "extracted_edge")
        else:
            extracted = _readonly_float_array(
                self.extracted_edge, "extracted_edge", length=length
            )
        residual = _optional_series(self.fit_residual, "fit_residual", length)
        statistical = _optional_series(
            self.statistical_uncertainty,
            "statistical_uncertainty",
            length,
            nonnegative=True,
        )
        model = _optional_series(
            self.model_uncertainty,
            "model_uncertainty",
            length,
            nonnegative=True,
        )
        calculated_uncertainty = np.hypot(statistical, model)
        if self.total_uncertainty is None:
            total_uncertainty = _readonly_float_array(
                calculated_uncertainty, "total_uncertainty"
            )
        else:
            total_uncertainty = _readonly_float_array(
                self.total_uncertainty,
                "total_uncertainty",
                length=length,
                nonnegative=True,
            )
            if not np.allclose(
                total_uncertainty,
                calculated_uncertainty,
                rtol=1e-8,
                atol=1e-10,
            ):
                raise DataValidationError(
                    "total_uncertainty must be the quadrature sum of statistical_uncertainty "
                    "and model_uncertainty"
                )
        q_au = _coordinate_array(self.q_au, "q_au", length, positive=True)
        q_angstrom = _coordinate_array(
            self.q_inverse_angstrom,
            "q_inverse_angstrom",
            length,
            positive=True,
        )
        if q_au is None and q_angstrom is None:
            raise DataValidationError("at least one of q_au or q_inverse_angstrom is required")
        parameters = _numeric_mapping(self.fit_parameters, "fit_parameters")
        covariance = None
        if self.parameter_covariance is not None:
            covariance = _readonly_float_array(
                self.parameter_covariance,
                "parameter_covariance",
                ndim=2,
            )
            if covariance.shape[0] != covariance.shape[1]:
                raise DataValidationError("parameter_covariance must be square")
            if parameters and covariance.shape[0] != len(parameters):
                raise DataValidationError(
                    "parameter_covariance dimension must match fit_parameters"
                )

        grade = _require_identifier(self.quality_grade, "quality_grade").capitalize()
        if grade not in QUALITY_GRADES:
            raise DataValidationError(
                f"quality_grade must be one of: {', '.join(sorted(QUALITY_GRADES))}"
            )
        object.__setattr__(self, "energy_loss_eV", energy)
        object.__setattr__(self, "raw_counts", raw)
        object.__setattr__(self, "q_au", q_au)
        object.__setattr__(self, "q_inverse_angstrom", q_angstrom)
        object.__setattr__(self, "normalized_intensity", normalized)
        object.__setattr__(self, "corrected_intensity", corrected)
        object.__setattr__(self, "elastic_component", elastic)
        object.__setattr__(self, "stray_background", stray)
        object.__setattr__(self, "valence_background", valence)
        object.__setattr__(self, "core_background", core)
        object.__setattr__(self, "constant_background", constant)
        object.__setattr__(self, "total_background", total)
        object.__setattr__(self, "extracted_edge", extracted)
        object.__setattr__(self, "fit_residual", residual)
        object.__setattr__(self, "statistical_uncertainty", statistical)
        object.__setattr__(self, "model_uncertainty", model)
        object.__setattr__(self, "total_uncertainty", total_uncertainty)
        object.__setattr__(
            self,
            "background_model_name",
            _require_identifier(self.background_model_name, "background_model_name"),
        )
        object.__setattr__(self, "fit_parameters", parameters)
        object.__setattr__(self, "parameter_covariance", covariance)
        object.__setattr__(self, "fit_windows", _fit_windows(self.fit_windows))
        object.__setattr__(
            self,
            "risk_metrics",
            _numeric_mapping(self.risk_metrics, "risk_metrics", nonnegative=True),
        )
        object.__setattr__(self, "warnings", _string_tuple(self.warnings, "warnings"))
        object.__setattr__(self, "quality_grade", grade)
        object.__setattr__(self, "provenance", _metadata(self.provenance, "provenance"))
        object.__setattr__(
            self, "software_version", _require_identifier(self.software_version, "software_version")
        )
        if not isinstance(self.config_digest, str):
            raise DataValidationError("config_digest must be a string")
        object.__setattr__(
            self,
            "raw_data_identifiers",
            _string_tuple(self.raw_data_identifiers, "raw_data_identifiers"),
        )

    @property
    def energy_loss_ev(self) -> FloatArray:
        """PEP 8 alias for :attr:`energy_loss_eV`."""

        return self.energy_loss_eV


@dataclass(frozen=True, slots=True)
class QualityReport(SerializableModel):
    """Machine-readable quality grade with evidence and suggested actions."""

    grade: str
    metrics: Mapping[str, float] = field(default_factory=dict)
    thresholds: Mapping[str, float] = field(default_factory=dict)
    reasons: Sequence[str] = field(default_factory=tuple)
    anomalous_indices: Sequence[int] = field(default_factory=tuple)
    recommended_actions: Sequence[str] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        grade = _require_identifier(self.grade, "grade").capitalize()
        if grade not in QUALITY_GRADES:
            raise DataValidationError(
                f"grade must be one of: {', '.join(sorted(QUALITY_GRADES))}"
            )
        indices: list[int] = []
        for value in self.anomalous_indices:
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise DataValidationError("anomalous_indices must contain integers")
            index = int(value)
            if index < 0:
                raise DataValidationError("anomalous_indices must be non-negative")
            indices.append(index)
        if len(indices) != len(set(indices)):
            raise DataValidationError("anomalous_indices must not contain duplicates")
        object.__setattr__(self, "grade", grade)
        object.__setattr__(self, "metrics", _numeric_mapping(self.metrics, "metrics"))
        object.__setattr__(
            self,
            "thresholds",
            _numeric_mapping(self.thresholds, "thresholds", nonnegative=True),
        )
        object.__setattr__(self, "reasons", _string_tuple(self.reasons, "reasons"))
        object.__setattr__(self, "anomalous_indices", tuple(indices))
        object.__setattr__(
            self,
            "recommended_actions",
            _string_tuple(self.recommended_actions, "recommended_actions"),
        )
        object.__setattr__(self, "metadata", _metadata(self.metadata))


@dataclass(slots=True)
class AnalysisSession(SerializableModel):
    """Mutable orchestration state; numerical result objects remain immutable."""

    config: AnalysisConfig = field(default_factory=AnalysisConfig)
    dataset: XRSDataset | None = None
    correction_results: Mapping[str, CorrectionResult] = field(default_factory=dict)
    background_results: Mapping[str, BackgroundResult] = field(default_factory=dict)
    extraction_results: Mapping[str, ExtractionResult] = field(default_factory=dict)
    quality_reports: Mapping[str, QualityReport] = field(default_factory=dict)
    status: str = "new"
    run_id: str = field(default_factory=lambda: str(uuid4()))
    created_at: str = field(default_factory=lambda: datetime.now().astimezone().isoformat())
    logs: Sequence[str] = field(default_factory=tuple)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    _RESULT_TYPES: ClassVar[dict[str, type[Any]]] = {
        "correction_results": CorrectionResult,
        "background_results": BackgroundResult,
        "extraction_results": ExtractionResult,
        "quality_reports": QualityReport,
    }

    def __post_init__(self) -> None:
        if not isinstance(self.config, AnalysisConfig):
            raise DataValidationError("config must be an AnalysisConfig object")
        if self.dataset is not None and not isinstance(self.dataset, XRSDataset):
            raise DataValidationError("dataset must be an XRSDataset object")
        for name, expected_type in self._RESULT_TYPES.items():
            value = getattr(self, name)
            if not isinstance(value, Mapping):
                raise DataValidationError(f"{name} must be a mapping")
            normalized: dict[str, Any] = {}
            for raw_key, result in value.items():
                key = _require_identifier(raw_key, f"{name} key")
                if not isinstance(result, expected_type):
                    raise DataValidationError(
                        f"{name}[{key!r}] must be a {expected_type.__name__} object"
                    )
                normalized[key] = result
            setattr(self, name, normalized)
        self.set_status(self.status)
        self.run_id = _require_identifier(self.run_id, "run_id")
        self.created_at = _require_identifier(self.created_at, "created_at")
        self.logs = _string_tuple(self.logs, "logs")
        self.metadata = _metadata(self.metadata)

    def set_status(self, status: str) -> None:
        normalized = _require_identifier(status, "status").lower()
        if normalized not in SESSION_STATUSES:
            raise DataValidationError(
                f"status must be one of: {', '.join(sorted(SESSION_STATUSES))}"
            )
        self.status = normalized

    def update_config(self, config: AnalysisConfig) -> None:
        """Replace the configuration and invalidate all derived results."""

        if not isinstance(config, AnalysisConfig):
            raise DataValidationError("config must be an AnalysisConfig object")
        # Dataclass equality is unsuitable when nested geometry objects contain
        # NumPy arrays (their element-wise comparison has no scalar truth value).
        if config.to_dict() != self.config.to_dict():
            self.config = config
            self.clear_results()

    def clear_results(self) -> None:
        self.correction_results.clear()
        self.background_results.clear()
        self.extraction_results.clear()
        self.quality_reports.clear()
        self.status = "ready" if self.dataset is not None else "new"

    def add_log(self, message: str) -> None:
        self.logs = (*self.logs, _require_identifier(message, "message"))


__all__ = [
    "QUALITY_GRADES",
    "ROI",
    "AnalysisConfig",
    "AnalysisSession",
    "Analyzer",
    "BackgroundResult",
    "CorrectionResult",
    "ExtractionResult",
    "Geometry",
    "QualityReport",
    "Sample",
    "Scan",
    "XRSDataset",
    "XRSSpectrum",
]
