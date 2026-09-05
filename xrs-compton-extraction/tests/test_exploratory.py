"""Regression for notebook semantics, including deliberately asymmetric data."""

import hashlib
import json

import numpy as np
import pytest

from xrs_compton_extraction.backgrounds.valence_profile import ValenceReferenceCandidate
from xrs_compton_extraction.constants import HARTREE_ENERGY_EV
from xrs_compton_extraction.exploratory import (
    fit_unweighted_pearson,
    fit_unweighted_templates,
    hf_missing,
    map_available,
    prompt_coordinates,
    prompt_energy,
    prompt_pz,
    read_crystals,
    reference_scores,
    save_result,
    valence_stages,
)
from xrs_compton_extraction.geometry import pz_to_energy_loss


def test_prompt_sign_preserves_asymmetric_pairing():
    pz = np.linspace(-3, 3, 301)
    asymmetric = np.exp(-pz**2) * (1 + .1*pz)
    reflected, values = prompt_coordinates(pz, asymmetric)
    np.testing.assert_allclose(values, np.exp(-reflected**2)*(1-.1*reflected))
    energy = pz_to_energy_loss(pz, 8)
    np.testing.assert_allclose(prompt_pz(energy, 8), -pz, atol=1e-14)
    np.testing.assert_allclose(prompt_energy(prompt_pz(energy, 8), 8), energy)


def test_stages_mask_normalization_and_available_mapping():
    pz = np.linspace(-3, 3, 601)
    energy = pz_to_energy_loss(pz, 8)
    raw = np.exp(-pz**2) * (1+.1*pz)
    mask = (pz > .5) & (pz < .7)
    raw[mask] += 10
    candidate = ValenceReferenceCandidate("test", energy, raw/(8*HARTREE_ENERGY_EV), 8, mask)
    profile, stages = valence_stages(candidate, 4)
    assert np.trapezoid(profile.profile, profile.pz_au) == pytest.approx(4)
    np.testing.assert_allclose(stages["corrected"], stages["corrected"][::-1])
    assert stages["raw"].max() > 10
    assert stages["corrected"].max() < 3
    assert stages["mask"].sum() == mask.sum()
    query = np.linspace(-4, 4, 801)
    target_energy = pz_to_energy_loss(query, 5)
    mapped, available = map_available(profile, target_energy, 5)
    assert np.all(np.isnan(mapped[~available]))
    assert not available[0] and not available[-1]
    assert np.trapezoid(mapped[available], target_energy[available]) == pytest.approx(4, rel=1e-3)


def test_unweighted_templates_protect_target_and_retain_negative(tmp_path):
    energy = np.linspace(0, 100, 101)
    core = .01*energy
    valence = np.exp(-((energy-40)/30)**2)
    y = 3*(core+valence)+2
    y[50] -= 4
    y[52] += 20
    valence[:3] = np.nan
    result = fit_unweighted_templates(energy, y, core, valence, ((0, 100),), (45, 55))
    assert result["parameters"]["scale"] == pytest.approx(3)
    assert result["parameters"]["constant"] == pytest.approx(2)
    assert result["residual"][50] == pytest.approx(-4)
    assert not result["fit_mask"][52]
    assert np.isnan(result["background"][:3]).all()
    assert result["statistical_uncertainty"] is None
    assert result["parameter_covariance"] is None
    assert "reduced_chi_square" not in result
    save_result(tmp_path, "test", energy, y, result, edge=(45, 55), provenance={})
    saved = json.loads((tmp_path / "test.json").read_text())
    assert saved["parameter_covariance"] is None
    assert "reduced_chi_square" not in saved
    with pytest.raises(ValueError, match="support"):
        fit_unweighted_templates(energy, y, core, np.full(101, np.nan), ((0, 100),), (45, 55))


def test_pearson_unweighted_does_not_report_statistical_covariance():
    x = np.linspace(20, 700, 201)
    y = 12*(1+(.02*(x-150))**2)**-1.5
    result = fit_unweighted_pearson(x, y, ((20, 80), (230, 700)), (111, 211))
    assert result["statistical_uncertainty"] is None
    assert result["parameter_covariance"] is None
    assert result["fit_rmse"] < 1e-5
    with pytest.raises(ValueError, match="insufficient"):
        fit_unweighted_pearson(x, y, ((20, 21),), (111, 211))


def test_crystal_join_validation_and_zero_retention(tmp_path):
    data, fit, info = (tmp_path/n for n in ("data.txt", "fit.txt", "run.json"))
    data.write_text("Energy Transfer (eV)\tB\tA\n20\t0\t1\n30\t0\t2\n40\t0\t3\n")
    header = "crystal\tcenter_eV\tfwhm_eV\tq_ave\tdq_ave\tq_range\tdq_range\n"
    a = "A\t9680\t1\t9.5\t.1\t.2\t.1\n"
    b = "B\t9681\t1\t9.6\t.1\t.2\t.1\n"
    fit.write_text(header+a+b)
    info.write_text("{}")
    before = hashlib.sha256(data.read_bytes()).hexdigest()
    dataset, records, provenance = read_crystals(data, fit, info)
    assert [r["crystal"] for r in records] == ["B", "A"]
    assert records[0]["all_zero"] and not records[1]["all_zero"]
    assert all(s.uncertainty is None for s in dataset.spectra)
    assert provenance["input_sha256"][str(data.resolve())] == before
    assert hashlib.sha256(data.read_bytes()).hexdigest() == before
    assert [r["crystal"] for r in reference_scores(dataset, records)] == ["A"]
    fit.write_text(header+a)
    with pytest.raises(ValueError, match="missing"):
        read_crystals(data, fit, info)
    fit.write_text(header+a+a+b)
    with pytest.raises(ValueError, match="duplicate"):
        read_crystals(data, fit, info)


def test_hf_gate_never_guesses_physics():
    missing = hf_missing({})
    assert "shells_by_element" in missing
    assert "atomic_review_passed" in missing
    assert "reference_scale" in missing


def test_real_notebook_join(monkeypatch):
    import os
    from pathlib import Path
    configured = os.environ.get("XRS_HO_TEST_DATA")
    if not configured:
        pytest.skip("set XRS_HO_TEST_DATA for private input regression")
    data = Path(configured)
    prefix = str(data).removesuffix("_all_data.txt")
    dataset, records, _ = read_crystals(data, prefix+"_fit_results.txt", prefix+"_run_info.json")
    assert len(dataset.spectra) == len(records) == 52
    assert {r["crystal"] for r in records if r["all_zero"]} == {"HB-E1", "HB-E2"}
