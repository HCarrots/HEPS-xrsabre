from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from xrs_compton_extraction.exceptions import DataDiscoveryError
from xrs_compton_extraction.io.text import (
    TextMapping,
    TextMappingError,
    load_text,
    load_text_channels,
)


def test_load_csv_energy_loss_from_unicode_path(tmp_path: Path) -> None:
    folder = tmp_path / "中文 数据"
    folder.mkdir()
    source = folder / "边 扫描.csv"
    source.write_text(
        "loss,counts,i0,time\n0,100,250,2\n10,121,251,2\n20,144,252,2\n",
        encoding="utf-8",
    )

    dataset = load_text(
        source,
        TextMapping(
            energy_column="loss",
            counts_column="counts",
            energy_kind="energy_loss",
            energy_units="eV",
            monitor_column="i0",
            acquisition_time_column="time",
            analyzer_id="A-1",
        ),
    )

    spectrum = dataset.spectra[0]
    np.testing.assert_allclose(spectrum.energy_loss_eV, [0.0, 10.0, 20.0])
    np.testing.assert_array_equal(spectrum.counts, [100, 121, 144])
    np.testing.assert_allclose(spectrum.monitor, [250, 251, 252])
    np.testing.assert_allclose(spectrum.acquisition_time_s, [2, 2, 2])
    assert spectrum.scan_id == "边 扫描"
    assert spectrum.analyzer_id == "A-1"
    assert dataset.metadata["source_file"] == str(source.resolve())


def test_load_tsv_incident_energy_with_fixed_scattered_energy(tmp_path: Path) -> None:
    source = tmp_path / "incident.tsv"
    source.write_text(
        "incident\tcounts\tsigma\n9.980\t10\t1\n9.990\t20\t2\n10.000\t30\t3\n",
        encoding="utf-8",
    )

    spectrum = load_text(
        source,
        TextMapping(
            energy_column="incident",
            counts_column="counts",
            uncertainty_column="sigma",
            energy_kind="incident_energy",
            energy_units="keV",
            fixed_scattered_energy_eV=9_950.0,
        ),
    ).spectra[0]

    np.testing.assert_allclose(spectrum.incident_energy_ev, [9980, 9990, 10000])
    np.testing.assert_allclose(spectrum.scattered_energy_ev, [9950, 9950, 9950])
    np.testing.assert_allclose(spectrum.energy_loss_eV, [30, 40, 50])
    np.testing.assert_allclose(spectrum.uncertainty, [1, 2, 3])


def test_scattered_energy_column_has_its_own_explicit_unit(tmp_path: Path) -> None:
    source = tmp_path / "pointwise.csv"
    source.write_text(
        "ei,ef,n,q\n10.00,9.95,5,2.0\n10.02,9.96,6,2.1\n",
        encoding="utf-8",
    )

    spectrum = load_text(
        source,
        TextMapping(
            energy_column="ei",
            counts_column="n",
            energy_kind="incident_energy",
            energy_units="keV",
            scattered_energy_column="ef",
            scattered_energy_units="keV",
            q_inverse_angstrom_column="q",
        ),
    ).spectra[0]

    np.testing.assert_allclose(spectrum.energy_loss_eV, [50, 60])
    np.testing.assert_allclose(spectrum.q_inverse_angstrom, [2.0, 2.1])


def test_headerless_table_uses_integer_columns_and_comments(tmp_path: Path) -> None:
    source = tmp_path / "headerless.dat"
    source.write_text("# loss counts\n0 10\n5 20\n", encoding="utf-8")

    spectrum = load_text(
        source,
        TextMapping(
            energy_column=0,
            counts_column=1,
            energy_kind="energy_loss",
            energy_units="eV",
            delimiter=" ",
            has_header=False,
        ),
    ).spectra[0]

    np.testing.assert_allclose(spectrum.energy_loss_eV, [0, 5])
    np.testing.assert_allclose(spectrum.counts, [10, 20])


def test_incident_energy_requires_scattered_energy() -> None:
    with pytest.raises(ValueError, match="cannot be treated as energy loss"):
        TextMapping(
            energy_column="energy",
            counts_column="counts",
            energy_kind="incident_energy",
            energy_units="eV",
        )


def test_mapping_requires_explicit_valid_energy_semantics_and_units() -> None:
    with pytest.raises(ValueError, match="energy_kind"):
        TextMapping(
            energy_column="energy",
            counts_column="counts",
            energy_kind="unknown",  # type: ignore[arg-type]
            energy_units="eV",
        )
    with pytest.raises(ValueError, match="supported energy units"):
        TextMapping(
            energy_column="loss",
            counts_column="counts",
            energy_kind="energy_loss",
            energy_units="joule",
        )


def test_missing_column_and_bad_numeric_value_are_actionable(tmp_path: Path) -> None:
    missing = tmp_path / "missing.csv"
    missing.write_text("loss,signal\n0,1\n", encoding="utf-8")
    mapping = TextMapping(
        energy_column="loss",
        counts_column="counts",
        energy_kind="energy_loss",
        energy_units="eV",
    )
    with pytest.raises(TextMappingError, match="available columns: loss, signal"):
        load_text(missing, mapping)

    malformed = tmp_path / "malformed.csv"
    malformed.write_text("loss,counts\n0,one\n", encoding="utf-8")
    with pytest.raises(TextMappingError, match=r"counts_column.*:2"):
        load_text(malformed, mapping)


def test_text_path_and_delimiter_errors_are_clear(tmp_path: Path) -> None:
    with pytest.raises(DataDiscoveryError, match="does not exist"):
        load_text(
            tmp_path / "missing.csv",
            TextMapping("loss", "counts", "energy_loss", "eV"),
        )

    source = tmp_path / "spectrum.txt"
    source.write_text("loss counts\n0 1\n", encoding="utf-8")
    with pytest.raises(TextMappingError, match="Cannot infer a delimiter"):
        load_text(
            source,
            TextMapping("loss", "counts", "energy_loss", "eV"),
        )


def test_mapping_dictionary_is_supported(tmp_path: Path) -> None:
    source = tmp_path / "mapped.csv"
    source.write_text("x,y\n1,3\n2,4\n", encoding="utf-8")

    spectrum = load_text(
        source,
        {
            "energy_column": "x",
            "counts_column": "y",
            "energy_kind": "energy_loss",
            "energy_units": "keV",
        },
    ).spectra[0]

    np.testing.assert_allclose(spectrum.energy_loss_eV, [1000, 2000])


def test_wide_processed_table_preserves_channels_and_missing_uncertainty(tmp_path: Path) -> None:
    source = tmp_path / "wide.tsv"
    source.write_text("loss\tA\tB\n0\t1.2\t0\n1\t2.5\t0\n", encoding="utf-8")
    dataset = load_text_channels(source, [
        TextMapping("loss", name, "energy_loss", "eV", analyzer_id=name, intensity_kind="processed")
        for name in ("A", "B")
    ])
    assert [s.analyzer_id for s in dataset.spectra] == ["A", "B"]
    np.testing.assert_array_equal(dataset.spectra[0].counts, [1.2, 2.5])
    np.testing.assert_array_equal(dataset.spectra[1].counts, [0, 0])
    assert dataset.spectra[0].metadata["intensity_kind"] == "processed"
    assert dataset.spectra[0].uncertainty is None
    assert dataset.source_count == 1


def test_wide_mapping_requires_unique_labels_and_shared_settings(tmp_path: Path) -> None:
    from dataclasses import replace

    source = tmp_path / "wide.tsv"
    source.write_text("x\ty\n0\t1\n", encoding="utf-8")
    mapping = TextMapping("x", "y", "energy_loss", "eV")
    with pytest.raises(TextMappingError, match="unique"):
        load_text_channels(source, [mapping, mapping])
    with pytest.raises(TextMappingError, match="parsing settings"):
        load_text_channels(source, [mapping, replace(mapping, delimiter=",")])
    with pytest.raises(TextMappingError, match="at least one"):
        load_text_channels(source, [])


@pytest.mark.parametrize("engine", ["correction", "pearson"])
def test_processed_intensity_never_gets_an_implicit_poisson_error(tmp_path: Path, engine: str) -> None:
    from xrs_compton_extraction.corrections import correct_spectrum
    from xrs_compton_extraction.exceptions import DataValidationError
    from xrs_compton_extraction.pipeline import extract_pearson

    source = tmp_path / "processed.tsv"
    source.write_text("x\ty\n0\t1.2\n1\t2.5\n", encoding="utf-8")
    spectrum = load_text(source, TextMapping("x", "y", "energy_loss", "eV", intensity_kind="processed")).spectra[0]
    with pytest.raises(DataValidationError, match="explicit uncertainty"):
        if engine == "correction":
            correct_spectrum(spectrum, normalize_i0=False, normalize_acquisition_time=False)
        else:
            extract_pearson(spectrum, fit_windows_ev=((0, 1),), q_au=2,
                            normalize_i0=False, normalize_acquisition_time=False)


def test_processed_intensity_uses_explicit_error_unchanged(tmp_path: Path) -> None:
    from xrs_compton_extraction.corrections import correct_spectrum

    source = tmp_path / "errors.tsv"
    source.write_text("e\ty\tsigma\n0\t100\t2\n1\t400\t3\n", encoding="utf-8")
    spectrum = load_text(source, TextMapping(
        "e", "y", "energy_loss", "eV", intensity_kind="processed", uncertainty_column="sigma"
    )).spectra[0]
    result = correct_spectrum(spectrum, normalize_i0=False, normalize_acquisition_time=False)
    np.testing.assert_array_equal(result.corrected_intensity, [100, 400])
    np.testing.assert_array_equal(result.statistical_uncertainty, [2, 3])
