from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from xrs_compton_extraction.exceptions import DataDiscoveryError, NexusMappingError
from xrs_compton_extraction.io import NexusMapping, discover_nexus_files, load_nexus


def _write_standard_nxdata(
    path: Path,
    *,
    energy: np.ndarray | None = None,
    energy_name: str = "energy_loss",
    energy_units: str = "eV",
    counts: np.ndarray | None = None,
) -> Path:
    energy_values = np.asarray(
        energy if energy is not None else [0.0, 10.0, 20.0, 30.0]
    )
    count_values = np.asarray(
        counts if counts is not None else [10, 20, 30, 40]
    )
    with h5py.File(path, "w") as handle:
        handle.attrs["default"] = "entry"
        entry = handle.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        entry.attrs["default"] = "data"
        nxdata = entry.create_group("data")
        nxdata.attrs["NX_class"] = "NXdata"
        nxdata.attrs["signal"] = "counts"
        nxdata.attrs["axes"] = energy_name
        nxdata.create_dataset("counts", data=count_values)
        axis = nxdata.create_dataset(energy_name, data=energy_values)
        axis.attrs["units"] = energy_units
    return path


def test_discover_accepts_full_file_with_unicode_and_spaces(tmp_path: Path) -> None:
    folder = tmp_path / "中文 数据 (scan)"
    folder.mkdir()
    source = folder / "扫描 42.nxs"
    source.touch()

    assert discover_nexus_files(source) == (source.resolve(),)


def test_discover_prefers_scan_id_name_and_falls_back_sorted(tmp_path: Path) -> None:
    scan_dir = tmp_path / "scan directory"
    scan_dir.mkdir()
    (scan_dir / "z-last.nxs").touch()
    preferred = scan_dir / "0042.nxs"
    preferred.touch()
    (scan_dir / "A-first.NXS").touch()

    assert discover_nexus_files(scan_dir, scan_id="0042") == (preferred.resolve(),)

    fallback = discover_nexus_files(scan_dir, scan_id="missing")
    assert [item.name for item in fallback] == ["0042.nxs", "A-first.NXS", "z-last.nxs"]


def test_discover_prefers_file_named_after_directory(tmp_path: Path) -> None:
    scan_dir = tmp_path / "1087"
    scan_dir.mkdir()
    preferred = scan_dir / "1087.nxs"
    preferred.touch()
    (scan_dir / "other.nxs").touch()

    assert discover_nexus_files(scan_dir) == (preferred.resolve(),)


def test_discovery_reports_missing_and_load_reports_multiple(tmp_path: Path) -> None:
    with pytest.raises(DataDiscoveryError, match="does not exist"):
        discover_nexus_files(tmp_path / "absent")

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(DataDiscoveryError, match="No \\.nxs files"):
        discover_nexus_files(empty)

    several = tmp_path / "several"
    several.mkdir()
    _write_standard_nxdata(several / "b.nxs")
    _write_standard_nxdata(several / "a.nxs")
    with pytest.raises(DataDiscoveryError, match=r"a\.nxs, b\.nxs"):
        load_nexus(several)


def test_loads_standard_nxentry_nxdata_energy_loss(tmp_path: Path) -> None:
    source = _write_standard_nxdata(tmp_path / "standard.nxs")

    dataset = load_nexus(source)

    assert len(dataset.spectra) == 1
    spectrum = dataset.spectra[0]
    np.testing.assert_allclose(spectrum.energy_eV, [0.0, 10.0, 20.0, 30.0])
    np.testing.assert_allclose(spectrum.energy_loss_eV, spectrum.energy_eV)
    np.testing.assert_array_equal(spectrum.counts, [10, 20, 30, 40])
    assert spectrum.scan_id == "standard"
    assert spectrum.analyzer_id == "channel-0"
    assert spectrum.metadata["energy_coordinate_kind"] == "energy_loss"
    assert dataset.metadata["source_file"] == str(source.resolve())


def test_explicit_mapping_converts_kev_incident_energy_to_loss(tmp_path: Path) -> None:
    source = tmp_path / "beamline-layout.nxs"
    with h5py.File(source, "w") as handle:
        raw = handle.create_group("raw")
        raw.create_dataset("detector", data=[100, 110, 120])
        incident = raw.create_dataset("mono", data=[10.000, 10.010, 10.020])
        incident.attrs["units"] = "keV"
        raw.create_dataset("ring_current", data=250.0)
        raw.create_dataset("seconds", data=2.0)

    mapping = NexusMapping(
        entry_path="/raw",
        nxdata_path="/raw",
        signal_path="/raw/detector",
        incident_energy_path="/raw/mono",
        fixed_scattered_energy_eV=9_950.0,
        monitor_path="/raw/ring_current",
        acquisition_time_path="/raw/seconds",
    )
    dataset = load_nexus(source, mapping)
    spectrum = dataset.spectra[0]

    np.testing.assert_allclose(spectrum.energy_eV, [10_000.0, 10_010.0, 10_020.0])
    np.testing.assert_allclose(spectrum.energy_loss_eV, [50.0, 60.0, 70.0])
    np.testing.assert_allclose(spectrum.incident_energy_ev, spectrum.energy_eV)
    np.testing.assert_allclose(spectrum.scattered_energy_ev, [9_950.0] * 3)
    np.testing.assert_allclose(spectrum.monitor, [250.0, 250.0, 250.0])
    np.testing.assert_allclose(spectrum.acquisition_time_s, [2.0, 2.0, 2.0])


def test_explicit_energy_loss_path_and_units_override(tmp_path: Path) -> None:
    source = tmp_path / "no-units.nxs"
    with h5py.File(source, "w") as handle:
        entry = handle.create_group("entry")
        entry.create_dataset("signal", data=[2, 3])
        entry.create_dataset("loss", data=[0.01, 0.02])

    dataset = load_nexus(
        source,
        NexusMapping(
            entry_path="/entry",
            nxdata_path="/entry",
            signal_path="signal",
            energy_loss_path="loss",
            energy_units="keV",
        ),
    )

    np.testing.assert_allclose(dataset.spectra[0].energy_loss_eV, [10.0, 20.0])


def test_two_dimensional_signal_splits_into_channels(tmp_path: Path) -> None:
    source = tmp_path / "multi-channel.nxs"
    with h5py.File(source, "w") as handle:
        entry = handle.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        data = entry.create_group("data")
        data.attrs["NX_class"] = "NXdata"
        data.attrs["signal"] = "counts"
        data.attrs["axes"] = np.asarray(["analyzer", "energy_loss"], dtype="S")
        data.attrs["energy_loss_indices"] = 1
        data.create_dataset("counts", data=[[1, 2, 3, 4], [11, 12, 13, 14]])
        analyzer = data.create_dataset(
            "analyzer", data=np.asarray(["A-01", "A-02"], dtype="S")
        )
        analyzer.attrs["long_name"] = "analyzer ID"
        loss = data.create_dataset("energy_loss", data=[5.0, 10.0, 15.0, 20.0])
        loss.attrs["units"] = "eV"

    dataset = load_nexus(source)

    assert len(dataset.spectra) == 2
    assert [item.analyzer_id for item in dataset.spectra] == ["A-01", "A-02"]
    np.testing.assert_array_equal(dataset.spectra[0].counts, [1, 2, 3, 4])
    np.testing.assert_array_equal(dataset.spectra[1].counts, [11, 12, 13, 14])
    np.testing.assert_allclose(dataset.spectra[1].energy_loss_eV, [5, 10, 15, 20])


def test_generic_or_incident_energy_is_not_silently_treated_as_loss(
    tmp_path: Path,
) -> None:
    generic = _write_standard_nxdata(
        tmp_path / "generic.nxs", energy_name="energy", energy_units="eV"
    )
    with pytest.raises(NexusMappingError, match="does not explicitly identify energy loss"):
        load_nexus(generic)

    incident = _write_standard_nxdata(
        tmp_path / "incident.nxs", energy_name="incident_energy", energy_units="keV"
    )
    with pytest.raises(NexusMappingError, match="is incident energy"):
        load_nexus(incident)


def test_fixed_scattered_energy_dataset_enables_loss_calculation(tmp_path: Path) -> None:
    source = _write_standard_nxdata(
        tmp_path / "with-final-energy.nxs",
        energy=np.asarray([9.98, 9.99, 10.00]),
        energy_name="incident_energy",
        energy_units="keV",
        counts=np.asarray([5, 6, 7]),
    )
    with h5py.File(source, "a") as handle:
        final = handle["entry"].create_dataset("scattered_energy", data=9.95)
        final.attrs["units"] = "keV"

    spectrum = load_nexus(source).spectra[0]
    np.testing.assert_allclose(spectrum.energy_loss_eV, [30.0, 40.0, 50.0])


def test_missing_signal_and_energy_units_have_actionable_errors(tmp_path: Path) -> None:
    no_signal = tmp_path / "no-signal.nxs"
    with h5py.File(no_signal, "w") as handle:
        entry = handle.create_group("entry")
        entry.attrs["NX_class"] = "NXentry"
        data = entry.create_group("data")
        data.attrs["NX_class"] = "NXdata"
        data.attrs["axes"] = "energy_loss"
        data.create_dataset("energy_loss", data=[1.0, 2.0])
    with pytest.raises(NexusMappingError, match="no usable signal"):
        load_nexus(no_signal)

    missing_units = _write_standard_nxdata(tmp_path / "missing-units.nxs")
    with h5py.File(missing_units, "a") as handle:
        del handle["entry/data/energy_loss"].attrs["units"]
    with pytest.raises(NexusMappingError, match="has no units"):
        load_nexus(missing_units)


def test_mapping_rejects_conflicting_energy_paths() -> None:
    with pytest.raises(ValueError, match="only one"):
        NexusMapping(energy_loss_path="loss", incident_energy_path="incident")
