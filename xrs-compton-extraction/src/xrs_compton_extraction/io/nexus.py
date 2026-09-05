"""Read XRS spectra from NeXus/HDF5 files.

The reader intentionally implements a small, strict NeXus mapping layer.  It
uses the standard ``NXentry``/``NXdata`` ``signal`` and ``axes`` attributes
when possible and lets a beamline provide explicit dataset paths when the
standard metadata is incomplete.

Energy semantics are deliberately conservative: a generic ``energy`` axis is
not assumed to be energy loss.  An energy-loss coordinate is accepted only
when it is labelled as such, or is calculated from an incident-energy axis and
a fixed scattered-photon energy.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from os import PathLike
from pathlib import Path
from typing import Any, Literal

import h5py
import numpy as np
from numpy.typing import NDArray

from ..data import XRSDataset, XRSSpectrum
from ..exceptions import DataDiscoveryError, NexusMappingError

EnergyKind = Literal["energy_loss", "incident_energy"]


@dataclass(frozen=True, slots=True)
class NexusMapping:
    """Optional beamline-specific HDF5 dataset mapping.

    Paths beginning with ``/`` are resolved from the HDF5 root.  Relative
    signal and energy paths are first resolved against ``NXdata``; other
    relative paths are also allowed against ``NXentry``.

    ``energy_loss_path`` and ``incident_energy_path`` encode the coordinate's
    physical meaning directly.  ``energy_path`` is useful for generic mapping
    files, but then ``energy_kind`` should be supplied unless the dataset name
    or metadata is unambiguous.  A numeric fixed scattered energy is expressed
    in ``fixed_scattered_energy_units`` (eV by default).
    """

    entry_path: str | None = None
    nxdata_path: str | None = None
    signal_path: str | None = None
    energy_path: str | None = None
    energy_loss_path: str | None = None
    incident_energy_path: str | None = None
    energy_kind: EnergyKind | None = None
    energy_units: str | None = None
    scattered_energy_path: str | None = None
    fixed_scattered_energy_eV: float | None = None
    fixed_scattered_energy_units: str = "eV"
    monitor_path: str | None = None
    acquisition_time_path: str | None = None
    channel_ids_path: str | None = None
    energy_axis: int | None = None

    def __post_init__(self) -> None:
        coordinate_paths = (
            self.energy_path,
            self.energy_loss_path,
            self.incident_energy_path,
        )
        if sum(value is not None for value in coordinate_paths) > 1:
            raise ValueError(
                "Specify only one of energy_path, energy_loss_path, and "
                "incident_energy_path."
            )
        if self.energy_kind not in (None, "energy_loss", "incident_energy"):
            raise ValueError(
                "energy_kind must be 'energy_loss' or 'incident_energy'."
            )
        if self.energy_loss_path is not None and self.energy_kind == "incident_energy":
            raise ValueError("energy_loss_path conflicts with energy_kind='incident_energy'.")
        if self.incident_energy_path is not None and self.energy_kind == "energy_loss":
            raise ValueError("incident_energy_path conflicts with energy_kind='energy_loss'.")
        if self.energy_axis is not None and self.energy_axis not in (0, 1, -1, -2):
            raise ValueError("energy_axis must identify dimension 0 or 1.")
        if self.fixed_scattered_energy_eV is not None:
            value = float(self.fixed_scattered_energy_eV)
            if not np.isfinite(value) or value <= 0:
                raise ValueError("fixed_scattered_energy_eV must be finite and positive.")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> NexusMapping:
        """Build a mapping from a YAML/JSON-compatible dictionary."""

        try:
            return cls(**dict(values))
        except TypeError as exc:
            raise NexusMappingError(f"Invalid NeXus mapping fields: {exc}") from exc
        except ValueError as exc:
            raise NexusMappingError(f"Invalid NeXus mapping: {exc}") from exc


def discover_nexus_files(
    path: str | PathLike[str], scan_id: str | int | None = None
) -> tuple[Path, ...]:
    """Resolve a NeXus file or deterministically discover files in a directory.

    A full ``.nxs`` filename always resolves to that single file.  For a
    directory, ``<scan_id>.nxs`` is preferred when ``scan_id`` is supplied;
    otherwise ``<directory-name>.nxs`` is preferred.  If that file does not
    exist, all direct child ``.nxs`` files are returned in a deterministic,
    case-independent order so a caller can present an explicit choice.

    Case-insensitive duplicate exact names are treated as ambiguous.  This
    matters on case-sensitive filesystems and prevents different behaviour on
    Windows and Linux.
    """

    candidate = Path(path).expanduser()
    if not candidate.exists():
        raise DataDiscoveryError(f"NeXus path does not exist: {candidate}")

    if candidate.is_file():
        if candidate.suffix.casefold() != ".nxs":
            raise DataDiscoveryError(
                f"Expected a .nxs file, got: {candidate}"
            )
        return (candidate.resolve(),)

    if not candidate.is_dir():
        raise DataDiscoveryError(f"NeXus path is neither a file nor directory: {candidate}")

    files = tuple(
        sorted(
            (
                child.resolve()
                for child in candidate.iterdir()
                if child.is_file() and child.suffix.casefold() == ".nxs"
            ),
            key=lambda item: (item.name.casefold(), item.name, str(item)),
        )
    )
    if not files:
        raise DataDiscoveryError(f"No .nxs files found in directory: {candidate}")

    preferred_stem = str(scan_id) if scan_id is not None else candidate.name
    preferred_name = f"{preferred_stem}.nxs".casefold()
    exact = tuple(item for item in files if item.name.casefold() == preferred_name)
    if len(exact) > 1:
        names = ", ".join(str(item) for item in exact)
        raise DataDiscoveryError(
            f"Ambiguous case-insensitive match for {preferred_stem!r}: {names}"
        )
    if exact:
        return exact
    return files


def load_nexus(
    path: str | PathLike[str],
    mapping: NexusMapping | Mapping[str, Any] | None = None,
    *,
    scan_id: str | int | None = None,
) -> XRSDataset:
    """Load one NeXus file into an :class:`XRSDataset`.

    Directory input is accepted only when discovery resolves to one file.  If
    fallback discovery finds several files, callers must select one explicitly
    rather than relying on an implicit first-file choice.
    """

    file_paths = discover_nexus_files(path, scan_id=scan_id)
    if len(file_paths) != 1:
        choices = ", ".join(item.name for item in file_paths)
        raise DataDiscoveryError(
            "Multiple .nxs files matched the directory; pass one complete file "
            f"path (deterministic choices: {choices})."
        )

    nexus_mapping = _coerce_mapping(mapping)
    source_path = file_paths[0]
    try:
        with h5py.File(source_path, "r") as handle:
            return _load_open_file(
                handle,
                source_path=source_path,
                mapping=nexus_mapping,
                scan_id=scan_id,
            )
    except NexusMappingError:
        raise
    except OSError as exc:
        raise NexusMappingError(
            f"Could not open NeXus/HDF5 file {source_path}: {exc}"
        ) from exc


def _coerce_mapping(
    mapping: NexusMapping | Mapping[str, Any] | None,
) -> NexusMapping:
    if mapping is None:
        return NexusMapping()
    if isinstance(mapping, NexusMapping):
        return mapping
    if isinstance(mapping, Mapping):
        return NexusMapping.from_mapping(mapping)
    raise TypeError("mapping must be NexusMapping, a mapping, or None")


def _load_open_file(
    handle: h5py.File,
    *,
    source_path: Path,
    mapping: NexusMapping,
    scan_id: str | int | None,
) -> XRSDataset:
    entry = _select_entry(handle, mapping.entry_path, scan_id=scan_id)
    nxdata = _select_nxdata(handle, entry, mapping.nxdata_path)
    signal = _select_signal(handle, entry, nxdata, mapping.signal_path)
    signal_values = _numeric_array(signal, label="signal")
    if signal_values.ndim not in (1, 2):
        raise NexusMappingError(
            f"Signal dataset {signal.name} must be one- or two-dimensional; "
            f"got shape {signal_values.shape}."
        )

    energy_dataset, declared_kind = _select_energy_axis(
        handle, entry, nxdata, signal, mapping
    )
    energy_values = _energy_values_eV(
        energy_dataset, units_override=mapping.energy_units
    )
    if energy_values.ndim != 1:
        raise NexusMappingError(
            f"Energy coordinate {energy_dataset.name} must be one-dimensional; "
            f"got shape {energy_values.shape}."
        )

    energy_dimension = _resolve_energy_dimension(
        signal,
        signal_values,
        energy_dataset,
        energy_values,
        nxdata,
        explicit=mapping.energy_axis,
    )

    scattered_energy_eV = _resolve_fixed_scattered_energy(
        handle, entry, nxdata, mapping
    )
    coordinate_kind = declared_kind or _infer_energy_kind(energy_dataset)
    if coordinate_kind == "energy_loss":
        energy_loss_eV = energy_values
        primary_energy_eV = energy_values
    elif scattered_energy_eV is not None:
        # A generic energy axis is treated as incident energy only when an
        # independently identified fixed scattered energy makes the conversion
        # physically defined.
        primary_energy_eV = energy_values
        energy_loss_eV = energy_values - scattered_energy_eV
        coordinate_kind = "incident_energy"
    else:
        semantic_name = energy_dataset.name
        if coordinate_kind == "incident_energy":
            detail = "is incident energy"
        else:
            detail = "does not explicitly identify energy loss"
        raise NexusMappingError(
            f"Energy coordinate {semantic_name} {detail}, and no fixed scattered "
            "energy is available. Supply energy_loss_path, or provide "
            "scattered_energy_path/fixed_scattered_energy_eV."
        )

    channel_count = 1 if signal_values.ndim == 1 else signal_values.shape[1 - energy_dimension]
    channel_ids = _resolve_channel_ids(
        handle,
        entry,
        nxdata,
        signal,
        energy_dataset,
        energy_dimension,
        channel_count,
        mapping.channel_ids_path,
    )
    monitor_dataset = _select_optional_dataset(
        handle,
        entry,
        nxdata,
        explicit_path=mapping.monitor_path,
        canonical_names=("monitor", "i0", "incidentmonitor", "incidentintensity"),
        label="monitor",
    )
    time_dataset = _select_optional_dataset(
        handle,
        entry,
        nxdata,
        explicit_path=mapping.acquisition_time_path,
        canonical_names=(
            "acquisitiontime",
            "counttime",
            "countingtime",
            "exposuretime",
            "integrationtime",
        ),
        label="acquisition time",
    )

    spectra: list[XRSSpectrum] = []
    for channel_index in range(channel_count):
        counts = _channel_slice(signal_values, energy_dimension, channel_index)
        monitor = _read_auxiliary_channel(
            monitor_dataset,
            signal_shape=signal_values.shape,
            energy_dimension=energy_dimension,
            channel_index=channel_index,
            channel_count=channel_count,
            energy_count=energy_values.size,
            label="monitor",
        )
        acquisition_time = _read_auxiliary_channel(
            time_dataset,
            signal_shape=signal_values.shape,
            energy_dimension=energy_dimension,
            channel_index=channel_index,
            channel_count=channel_count,
            energy_count=energy_values.size,
            label="acquisition time",
        )
        analyzer_id = channel_ids[channel_index]
        spectrum_metadata = {
            "source_file": str(source_path),
            "entry_path": entry.name,
            "nxdata_path": nxdata.name,
            "signal_path": signal.name,
            "energy_path": energy_dataset.name,
            "energy_coordinate_kind": coordinate_kind,
            "channel_index": channel_index,
        }
        if scattered_energy_eV is not None:
            spectrum_metadata["fixed_scattered_energy_eV"] = float(scattered_energy_eV)

        spectra.append(
            XRSSpectrum(
                energy_eV=np.array(primary_energy_eV, dtype=float, copy=True),
                counts=np.array(counts, copy=True),
                energy_loss_eV=np.array(energy_loss_eV, dtype=float, copy=True),
                incident_energy_ev=(
                    np.array(primary_energy_eV, dtype=float, copy=True)
                    if coordinate_kind == "incident_energy"
                    else None
                ),
                scattered_energy_ev=(
                    float(scattered_energy_eV)
                    if coordinate_kind == "incident_energy"
                    else None
                ),
                monitor=monitor,
                acquisition_time_s=acquisition_time,
                scan_id=str(scan_id) if scan_id is not None else source_path.stem,
                analyzer_id=analyzer_id,
                metadata=spectrum_metadata,
            )
        )

    return XRSDataset(
        spectra=tuple(spectra),
        metadata={
            "source_file": str(source_path),
            "entry_path": entry.name,
            "nxdata_path": nxdata.name,
            "signal_path": signal.name,
        },
        provenance={"source_files": (str(source_path),)},
    )


def _select_entry(
    handle: h5py.File, explicit_path: str | None, *, scan_id: str | int | None
) -> h5py.Group:
    if explicit_path is not None:
        return _require_group(
            _resolve_node(handle, None, None, explicit_path),
            label="NXentry",
            expected_class="NXentry",
        )

    default_name = _single_attr_text(handle.attrs.get("default"))
    if default_name:
        node = _try_resolve_node(handle, handle, None, default_name)
        if isinstance(node, h5py.Group) and _nx_class(node) == "NXentry":
            return node

    entries = [
        child
        for child in handle.values()
        if isinstance(child, h5py.Group) and _nx_class(child) == "NXentry"
    ]
    if scan_id is not None:
        target = str(scan_id).casefold()
        matching = [item for item in entries if item.name.rsplit("/", 1)[-1].casefold() == target]
        if len(matching) == 1:
            return matching[0]
        if len(matching) > 1:
            raise NexusMappingError(
                f"More than one NXentry matches scan_id {scan_id!r}."
            )
    if len(entries) == 1:
        return entries[0]
    if not entries:
        raise NexusMappingError(
            "No NXentry group found. Supply NexusMapping(entry_path=...)."
        )
    choices = ", ".join(item.name for item in entries)
    raise NexusMappingError(
        f"Multiple NXentry groups found ({choices}); supply entry_path."
    )


def _select_nxdata(
    handle: h5py.File, entry: h5py.Group, explicit_path: str | None
) -> h5py.Group:
    if explicit_path is not None:
        return _require_group(
            _resolve_node(handle, entry, None, explicit_path),
            label="NXdata",
            expected_class="NXdata",
        )

    default_name = _single_attr_text(entry.attrs.get("default"))
    if default_name:
        node = _try_resolve_node(handle, entry, None, default_name)
        if isinstance(node, h5py.Group) and _nx_class(node) == "NXdata":
            return node

    groups = [
        child
        for child in entry.values()
        if isinstance(child, h5py.Group) and _nx_class(child) == "NXdata"
    ]
    if len(groups) == 1:
        return groups[0]
    if not groups:
        raise NexusMappingError(
            f"No NXdata group found below {entry.name}; supply nxdata_path."
        )
    choices = ", ".join(item.name for item in groups)
    raise NexusMappingError(
        f"Multiple NXdata groups found ({choices}); supply nxdata_path."
    )


def _select_signal(
    handle: h5py.File,
    entry: h5py.Group,
    nxdata: h5py.Group,
    explicit_path: str | None,
) -> h5py.Dataset:
    if explicit_path is not None:
        return _require_dataset(
            _resolve_node(handle, entry, nxdata, explicit_path), label="signal"
        )

    signal_name = _single_attr_text(nxdata.attrs.get("signal"))
    if signal_name:
        return _require_dataset(
            _resolve_node(handle, entry, nxdata, signal_name), label="NXdata signal"
        )

    legacy = [
        item
        for item in nxdata.values()
        if isinstance(item, h5py.Dataset) and _is_legacy_signal(item)
    ]
    if len(legacy) == 1:
        return legacy[0]
    if len(legacy) > 1:
        choices = ", ".join(item.name for item in legacy)
        raise NexusMappingError(f"Multiple datasets are marked as signal: {choices}")
    raise NexusMappingError(
        f"NXdata group {nxdata.name} has no usable signal attribute; supply signal_path."
    )


def _select_energy_axis(
    handle: h5py.File,
    entry: h5py.Group,
    nxdata: h5py.Group,
    signal: h5py.Dataset,
    mapping: NexusMapping,
) -> tuple[h5py.Dataset, EnergyKind | None]:
    explicit_path: str | None = None
    declared_kind = mapping.energy_kind
    if mapping.energy_loss_path is not None:
        explicit_path = mapping.energy_loss_path
        declared_kind = "energy_loss"
    elif mapping.incident_energy_path is not None:
        explicit_path = mapping.incident_energy_path
        declared_kind = "incident_energy"
    elif mapping.energy_path is not None:
        explicit_path = mapping.energy_path

    if explicit_path is not None:
        return (
            _require_dataset(
                _resolve_node(handle, entry, nxdata, explicit_path), label="energy coordinate"
            ),
            declared_kind,
        )

    axis_names = _attr_texts(nxdata.attrs.get("axes"))
    axis_datasets: list[h5py.Dataset] = []
    for name in axis_names:
        if name in ("", "."):
            continue
        node = _try_resolve_node(handle, entry, nxdata, name)
        if isinstance(node, h5py.Dataset):
            axis_datasets.append(node)

    if not axis_datasets:
        # A limited fallback for otherwise conventional NXdata groups.  The
        # name is still interpreted below; a generic ``energy`` remains unsafe.
        common = {
            "energyloss",
            "energytransfer",
            "transferredenergy",
            "incidentenergy",
            "initialenergy",
            "energy",
        }
        axis_datasets = [
            item
            for item in nxdata.values()
            if isinstance(item, h5py.Dataset)
            and _canonical(item.name.rsplit("/", 1)[-1]) in common
        ]

    compatible = [
        item
        for item in axis_datasets
        if item.ndim == 1 and item.shape[0] in signal.shape
    ]
    energy_like = [item for item in compatible if _infer_energy_kind(item) is not None]
    if len(energy_like) == 1:
        return energy_like[0], _infer_energy_kind(energy_like[0])
    if len(energy_like) > 1:
        loss_axes = [item for item in energy_like if _infer_energy_kind(item) == "energy_loss"]
        if len(loss_axes) == 1:
            return loss_axes[0], "energy_loss"
        choices = ", ".join(item.name for item in energy_like)
        raise NexusMappingError(
            f"Multiple energy-like NXdata axes found ({choices}); supply an explicit energy path."
        )
    if len(compatible) == 1:
        return compatible[0], None
    if not compatible:
        raise NexusMappingError(
            f"Could not identify an energy axis for signal {signal.name}; "
            "supply energy_loss_path or incident_energy_path."
        )
    choices = ", ".join(item.name for item in compatible)
    raise NexusMappingError(
        f"NXdata axes are ambiguous ({choices}); supply an explicit energy path."
    )


def _resolve_energy_dimension(
    signal: h5py.Dataset,
    signal_values: NDArray[Any],
    energy_dataset: h5py.Dataset,
    energy_values: NDArray[np.float64],
    nxdata: h5py.Group,
    *,
    explicit: int | None,
) -> int:
    if signal_values.ndim == 1:
        if signal_values.shape[0] != energy_values.size:
            raise NexusMappingError(
                f"Signal length {signal_values.shape[0]} does not match energy-axis "
                f"length {energy_values.size}."
            )
        return 0

    if explicit is not None:
        dimension = explicit % signal_values.ndim
        if signal_values.shape[dimension] != energy_values.size:
            raise NexusMappingError(
                f"Configured energy_axis={explicit} has length "
                f"{signal_values.shape[dimension]}, but {energy_dataset.name} has "
                f"length {energy_values.size}."
            )
        return dimension

    axis_name = energy_dataset.name.rsplit("/", 1)[-1]
    index_attr = nxdata.attrs.get(f"{axis_name}_indices")
    if index_attr is not None:
        indices = np.asarray(index_attr).reshape(-1)
        if indices.size == 1:
            dimension = int(indices[0])
            if 0 <= dimension < signal_values.ndim and signal_values.shape[dimension] == energy_values.size:
                return dimension

    # In a standard NXdata group, a full ``axes`` list is ordered by signal
    # dimension.  This also disambiguates square two-dimensional signals.
    axis_names = _attr_texts(nxdata.attrs.get("axes"))
    if len(axis_names) == signal_values.ndim:
        for dimension, name in enumerate(axis_names):
            node = _try_resolve_node(signal.file, nxdata.parent, nxdata, name)
            if (
                isinstance(node, h5py.Dataset)
                and node.name == energy_dataset.name
                and signal_values.shape[dimension] == energy_values.size
            ):
                return dimension

    legacy_axis = energy_dataset.attrs.get("axis")
    if legacy_axis is not None:
        values = np.asarray(legacy_axis).reshape(-1)
        if values.size == 1:
            # Legacy NeXus ``axis`` values are one-based.
            dimension = int(values[0]) - 1
            if 0 <= dimension < signal_values.ndim and signal_values.shape[dimension] == energy_values.size:
                return dimension

    matching = [
        index for index, length in enumerate(signal_values.shape) if length == energy_values.size
    ]
    if len(matching) == 1:
        return matching[0]
    if not matching:
        raise NexusMappingError(
            f"No dimension of signal {signal.name} shape {signal_values.shape} matches "
            f"energy-axis length {energy_values.size}."
        )
    raise NexusMappingError(
        f"Signal shape {signal_values.shape} makes the energy dimension ambiguous; "
        "set NexusMapping(energy_axis=0 or 1)."
    )


def _resolve_fixed_scattered_energy(
    handle: h5py.File,
    entry: h5py.Group,
    nxdata: h5py.Group,
    mapping: NexusMapping,
) -> float | None:
    if mapping.fixed_scattered_energy_eV is not None:
        converted = _convert_energy_to_eV(
            np.asarray(float(mapping.fixed_scattered_energy_eV)),
            mapping.fixed_scattered_energy_units,
            label="fixed scattered energy",
        )
        return float(converted)

    if mapping.scattered_energy_path is not None:
        dataset = _require_dataset(
            _resolve_node(handle, entry, nxdata, mapping.scattered_energy_path),
            label="scattered energy",
        )
        values = _energy_values_eV(dataset)
        return _constant_value(values, dataset.name)

    matches = _find_named_datasets(
        entry,
        {
            "scatteredenergy",
            "finalenergy",
            "analyzerenergy",
            "analyserenergy",
            "fixedscatteredenergy",
        },
    )
    if len(matches) > 1:
        choices = ", ".join(item.name for item in matches)
        raise NexusMappingError(
            f"Multiple possible fixed scattered-energy datasets found ({choices}); "
            "supply scattered_energy_path."
        )
    if not matches:
        return None
    values = _energy_values_eV(matches[0])
    return _constant_value(values, matches[0].name)


def _resolve_channel_ids(
    handle: h5py.File,
    entry: h5py.Group,
    nxdata: h5py.Group,
    signal: h5py.Dataset,
    energy_dataset: h5py.Dataset,
    energy_dimension: int,
    channel_count: int,
    explicit_path: str | None,
) -> tuple[str, ...]:
    dataset: h5py.Dataset | None = None
    if explicit_path is not None:
        dataset = _require_dataset(
            _resolve_node(handle, entry, nxdata, explicit_path), label="channel IDs"
        )
    elif signal.ndim == 2:
        for axis_name in _attr_texts(nxdata.attrs.get("axes")):
            if axis_name in ("", "."):
                continue
            node = _try_resolve_node(handle, entry, nxdata, axis_name)
            if not isinstance(node, h5py.Dataset) or node.name == energy_dataset.name:
                continue
            if node.ndim == 1 and node.shape[0] == channel_count:
                dataset = node
                break

    if dataset is None:
        return tuple(f"channel-{index}" for index in range(channel_count))

    values = np.asarray(dataset[()]).reshape(-1)
    if values.size != channel_count:
        raise NexusMappingError(
            f"Channel-ID dataset {dataset.name} has {values.size} values; "
            f"expected {channel_count}."
        )
    return tuple(_display_value(item) for item in values)


def _select_optional_dataset(
    handle: h5py.File,
    entry: h5py.Group,
    nxdata: h5py.Group,
    *,
    explicit_path: str | None,
    canonical_names: Sequence[str],
    label: str,
) -> h5py.Dataset | None:
    if explicit_path is not None:
        return _require_dataset(
            _resolve_node(handle, entry, nxdata, explicit_path), label=label
        )
    matches = _find_named_datasets(entry, set(canonical_names))
    if len(matches) == 1:
        return matches[0]
    # Optional metadata is not guessed when several plausible sources exist.
    return None


def _read_auxiliary_channel(
    dataset: h5py.Dataset | None,
    *,
    signal_shape: tuple[int, ...],
    energy_dimension: int,
    channel_index: int,
    channel_count: int,
    energy_count: int,
    label: str,
) -> NDArray[np.float64] | None:
    if dataset is None:
        return None
    values = _numeric_array(dataset, label=label).astype(float, copy=False)
    if values.ndim == 0 or values.size == 1:
        return np.full(energy_count, float(values.reshape(-1)[0]), dtype=float)
    if values.shape == signal_shape:
        return np.asarray(
            _channel_slice(values, energy_dimension, channel_index), dtype=float
        )
    if values.ndim == 1 and values.size == energy_count:
        return np.array(values, dtype=float, copy=True)
    if values.ndim == 1 and values.size == channel_count:
        return np.full(energy_count, float(values[channel_index]), dtype=float)
    raise NexusMappingError(
        f"{label.capitalize()} dataset {dataset.name} shape {values.shape} is not "
        f"compatible with signal shape {signal_shape}."
    )


def _channel_slice(
    values: NDArray[Any], energy_dimension: int, channel_index: int
) -> NDArray[Any]:
    if values.ndim == 1:
        return np.asarray(values)
    channel_dimension = 1 - energy_dimension
    return np.asarray(np.take(values, channel_index, axis=channel_dimension))


def _energy_values_eV(
    dataset: h5py.Dataset, *, units_override: str | None = None
) -> NDArray[np.float64]:
    values = _numeric_array(dataset, label="energy coordinate").astype(float, copy=False)
    units = units_override or _single_attr_text(dataset.attrs.get("units"))
    if not units:
        raise NexusMappingError(
            f"Energy dataset {dataset.name} has no units. Add a NeXus units "
            "attribute or set NexusMapping(energy_units='eV' or 'keV')."
        )
    return np.asarray(
        _convert_energy_to_eV(values, units, label=dataset.name), dtype=float
    )


def _convert_energy_to_eV(
    values: NDArray[Any], units: str, *, label: str
) -> NDArray[np.float64]:
    canonical = _canonical_unit(units)
    factors = {
        "ev": 1.0,
        "electronvolt": 1.0,
        "electronvolts": 1.0,
        "kev": 1_000.0,
        "kiloelectronvolt": 1_000.0,
        "kiloelectronvolts": 1_000.0,
    }
    try:
        factor = factors[canonical]
    except KeyError as exc:
        raise NexusMappingError(
            f"Unsupported energy unit {units!r} for {label}; supported units are eV and keV."
        ) from exc
    converted = np.asarray(values, dtype=float) * factor
    if not np.all(np.isfinite(converted)):
        raise NexusMappingError(f"Energy values in {label} contain NaN or infinity.")
    return converted


def _constant_value(values: NDArray[np.float64], label: str) -> float:
    flattened = np.asarray(values, dtype=float).reshape(-1)
    if flattened.size == 0:
        raise NexusMappingError(f"Scattered-energy dataset {label} is empty.")
    if not np.allclose(flattened, flattened[0], rtol=1e-12, atol=1e-9):
        raise NexusMappingError(
            f"Scattered-energy dataset {label} is not fixed; pointwise final-energy "
            "conversion is not supported by this first mapping implementation."
        )
    value = float(flattened[0])
    if not np.isfinite(value) or value <= 0:
        raise NexusMappingError(f"Fixed scattered energy in {label} must be positive and finite.")
    return value


def _numeric_array(dataset: h5py.Dataset, *, label: str) -> NDArray[Any]:
    values = np.asarray(dataset[()])
    if values.dtype.kind not in "uif":
        raise NexusMappingError(
            f"{label.capitalize()} dataset {dataset.name} must be numeric; got dtype {values.dtype}."
        )
    if not np.all(np.isfinite(values)):
        raise NexusMappingError(
            f"{label.capitalize()} dataset {dataset.name} contains NaN or infinity."
        )
    return values


def _infer_energy_kind(dataset: h5py.Dataset) -> EnergyKind | None:
    terms = [dataset.name.rsplit("/", 1)[-1]]
    for attribute in ("long_name", "standard_name", "interpretation"):
        terms.extend(_attr_texts(dataset.attrs.get(attribute)))
    canonical_terms = tuple(_canonical(term) for term in terms)
    if any(
        marker in term
        for term in canonical_terms
        for marker in ("energyloss", "energytransfer", "transferredenergy")
    ):
        return "energy_loss"
    if any(
        marker in term
        for term in canonical_terms
        for marker in ("incidentenergy", "initialenergy", "excitationenergy")
    ):
        return "incident_energy"
    return None


def _find_named_datasets(
    group: h5py.Group, canonical_names: set[str]
) -> list[h5py.Dataset]:
    matches: list[h5py.Dataset] = []

    def visitor(name: str, node: h5py.Group | h5py.Dataset) -> None:
        if isinstance(node, h5py.Dataset) and _canonical(name.rsplit("/", 1)[-1]) in canonical_names:
            matches.append(node)

    group.visititems(visitor)
    return sorted(matches, key=lambda item: item.name)


def _resolve_node(
    handle: h5py.File,
    entry: h5py.Group | None,
    nxdata: h5py.Group | None,
    path: str,
) -> h5py.Group | h5py.Dataset:
    node = _try_resolve_node(handle, entry, nxdata, path)
    if node is None:
        bases = [base.name for base in (nxdata, entry) if base is not None]
        detail = f" relative to {', '.join(bases)}" if bases else ""
        raise NexusMappingError(f"HDF5 path {path!r} was not found{detail}.")
    return node


def _try_resolve_node(
    handle: h5py.File,
    entry: h5py.Group | None,
    nxdata: h5py.Group | None,
    path: str,
) -> h5py.Group | h5py.Dataset | None:
    clean_path = str(path).strip()
    if not clean_path:
        return None
    if clean_path.startswith("/"):
        return handle.get(clean_path)
    for base in (nxdata, entry, handle):
        if base is None:
            continue
        node = base.get(clean_path)
        if node is not None:
            return node
    return None


def _require_dataset(
    node: h5py.Group | h5py.Dataset, *, label: str
) -> h5py.Dataset:
    if not isinstance(node, h5py.Dataset):
        raise NexusMappingError(f"Mapped {label} path {node.name} is not a dataset.")
    return node


def _require_group(
    node: h5py.Group | h5py.Dataset,
    *,
    label: str,
    expected_class: str,
) -> h5py.Group:
    if not isinstance(node, h5py.Group):
        raise NexusMappingError(f"Mapped {label} path {node.name} is not a group.")
    actual_class = _nx_class(node)
    if actual_class not in (None, expected_class):
        raise NexusMappingError(
            f"Mapped {label} group {node.name} has NX_class={actual_class!r}, "
            f"expected {expected_class!r}."
        )
    return node


def _nx_class(group: h5py.Group) -> str | None:
    return _single_attr_text(group.attrs.get("NX_class"))


def _is_legacy_signal(dataset: h5py.Dataset) -> bool:
    value = dataset.attrs.get("signal")
    if value is None:
        return False
    flattened = np.asarray(value).reshape(-1)
    return flattened.size == 1 and _display_value(flattened[0]) == "1"


def _attr_texts(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        # Some producers serialize multiple axes as comma/colon-separated text.
        return tuple(part.strip() for part in re.split(r"[:,]", value) if part.strip())
    if isinstance(value, bytes):
        return (_decode_bytes(value),)
    flattened = np.asarray(value).reshape(-1)
    return tuple(_display_value(item).strip() for item in flattened if _display_value(item).strip())


def _single_attr_text(value: Any) -> str | None:
    values = _attr_texts(value)
    return values[0] if values else None


def _display_value(value: Any) -> str:
    if isinstance(value, bytes):
        return _decode_bytes(value)
    if isinstance(value, np.bytes_):
        return _decode_bytes(bytes(value))
    if isinstance(value, np.generic):
        value = value.item()
    return str(value)


def _decode_bytes(value: bytes) -> str:
    try:
        return value.decode("utf-8")
    except UnicodeDecodeError:
        return value.decode("latin-1")


def _canonical(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", value.casefold())


def _canonical_unit(value: str) -> str:
    return _canonical(value).replace("kiloelectronvolt", "kiloelectronvolt")


__all__ = ["NexusMapping", "discover_nexus_files", "load_nexus"]
