import json

import numpy as np
import pytest

from xrs_compton_extraction import (
    XRSDataset,
    XRSSpectrum,
    XRSWorkbench,
    extract_batch,
    extract_compton_profile,
    save_results,
)
from xrs_compton_extraction.backgrounds import (
    CoreProfileResult,
    ValenceReferenceCandidate,
    build_valence_profile,
)
from xrs_compton_extraction.constants import HARTREE_ENERGY_EV
from xrs_compton_extraction.geometry import pz_to_energy_loss


def profile_fixture():
    pz = np.linspace(-3, 3, 601)
    q = 8.0
    energy = pz_to_energy_loss(pz, q)
    jv = np.exp(-pz**2)
    jv *= 4 / np.trapezoid(jv, pz)
    jc = 0.1 * np.exp(-pz**2 / 8)
    core = CoreProfileResult(pz, {"O:K": jc}, jc, component_metadata={"O:K": {"electron_occupancy": 2}}, source_provenance={"provider": "synthetic"})
    reference = ValenceReferenceCandidate("ref", energy, jv / (q * HARTREE_ENERGY_EV), q, np.zeros(len(pz), dtype=bool))
    valence = build_valence_profile([reference], score_weights={"uncontaminated_fraction": 1}, valence_electron_count=4, normalization_convention="full_symmetric")
    edge = np.where(np.abs(pz) < 0.4, 3.0, 0.0)
    counts = 500 * (jc + jv) / (q * HARTREE_ENERGY_EV) + 2 + edge
    spectrum = XRSSpectrum(energy, counts, energy_loss_eV=energy, q_au=q, monitor=1, acquisition_time_s=1, analyzer_id="A1")
    options = {"q_au": q, "core_profile": core, "valence_profile": valence, "fit_windows_ev": ((energy[0], energy[220]), (energy[380], energy[-1]))}
    return spectrum, options, edge


def test_profile_pipeline_recovers_core_valence_scale_and_edge(tmp_path):
    spectrum, options, edge = profile_fixture()
    result = extract_compton_profile(spectrum, **options)
    assert result.fit_parameters["profile_scale"] == pytest.approx(500)
    assert result.fit_parameters["constant"] == pytest.approx(2)
    np.testing.assert_allclose(result.extracted_edge, edge, atol=1e-10)
    assert result.quality_grade == "Warning"
    assert np.all(result.model_uncertainty > 0)
    output = save_results(result, tmp_path / "profile")
    metadata = json.loads((output / "extraction.metadata.json").read_text())
    assert metadata["provenance"]["core_source"]["provider"] == "synthetic"


def test_wrong_q_grid_is_rejected():
    spectrum, options, _ = profile_fixture()
    options["q_au"] = 7
    with pytest.raises(ValueError, match="grid"):
        extract_compton_profile(spectrum, **options)


def test_batch_records_failure_and_progress():
    spectrum, options, _ = profile_fixture()
    dataset = XRSDataset(spectra=[spectrum])
    events = []
    good = extract_batch(dataset, extractor=extract_compton_profile, channel_options={spectrum.channel_label: options}, on_progress=lambda *args: events.append(args))
    assert len(good.results) == 1 and not good.failures
    assert events == [(1, 1, spectrum.channel_label)]
    bad = extract_batch(dataset, extractor=extract_compton_profile, channel_options={})
    assert "Missing" in bad.failures[spectrum.channel_label]


def test_workbench_profile_and_batch_are_connected():
    spectrum, options, _ = profile_fixture()
    app = XRSWorkbench()
    app.session.dataset = XRSDataset(spectra=[spectrum])
    result = app.run_compton_profile(**options)
    assert result.background_model_name == "compton_profile"
    assert app.session.status == "complete"
    batch = app.run_batch(extractor=extract_compton_profile, channel_options={spectrum.channel_label: options})
    assert len(batch.results) == len(app.results) == 1
