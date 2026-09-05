"""Opt-in local regression: beamline data is not bundled or required by CI."""

import hashlib
import os
from pathlib import Path

import numpy as np
import pytest

from xrs_compton_extraction.io import TextMapping, load_text_channels


def test_real_ho_processed_scan_import() -> None:
    configured = os.environ.get("XRS_HO_TEST_DATA")
    if not configured:
        pytest.skip("set XRS_HO_TEST_DATA to the Ho standard all_data.txt")
    source = Path(configured)
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    with source.open(encoding="utf-8-sig") as stream:
        columns = stream.readline().strip().split("\t")
    assert columns[0] == "Energy Transfer (eV)"
    dataset = load_text_channels(source, [
        TextMapping(columns[0], column, "energy_loss", "eV", delimiter="\t",
                    analyzer_id=column, intensity_kind="processed")
        for column in columns[1:]
    ])
    assert len(dataset.spectra) == 52
    for spectrum in dataset.spectra:
        assert len(spectrum) == 4021
        assert spectrum.uncertainty is None
        assert spectrum.monitor is None
        np.testing.assert_allclose(spectrum.energy_loss_ev[[0, -1]], [-1.6, 802.4])
        np.testing.assert_allclose(np.diff(spectrum.energy_loss_ev), 0.2)
        assert np.all(np.isfinite(spectrum.counts))
    assert [s.analyzer_id for s in dataset.spectra if np.all(s.counts == 0)] == ["HB-E1", "HB-E2"]
    assert hashlib.sha256(source.read_bytes()).hexdigest() == before
