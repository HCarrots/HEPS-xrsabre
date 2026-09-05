"""Check moment units, fit scaling, and retention of the target continuum."""

import numpy as np
import pytest

from xrs_compton_extraction.backgrounds import DabaxProfileSource
from xrs_compton_extraction.constants import HARTREE_ENERGY_EV as EH
from xrs_compton_extraction.hf_workflow import (
    build_hf_energy,
    extract_hf_target,
    match_hf_scale,
)


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "test.dat"
    lines = ["#S 3 Li", "#N 5", "#UOCCUP 1 1 1", "#UBIND 50 120 0",
             "#L pz total Shell_1 Shell_2 Shell_3"]
    for p in np.linspace(0, 100, 2001):
        values = np.exp(-np.array([.2, .4, 1])*p*p)
        lines.append(" ".join(map(str, [p, values.sum(), *values])))
    path.write_text("\n".join(lines))
    return DabaxProfileSource(path)


def build(energy, source, **kwargs):
    return build_hf_energy(energy, 5, source, {"Li": 2}, target=("Li", "Shell_1"),
                           valence_cutoff_ev=20, **kwargs)


def test_threshold_first_moment_and_scan_independent_scale(source):
    energy = np.linspace(50, 2500, 40001)
    hf = build(energy, source)
    target = hf.components["Li:Shell_1"]
    # Two formula units, one electron in this shell; S is per eV.
    assert np.trapezoid(energy*target, energy) == pytest.approx(2*EH*25/2, rel=2e-4)
    short = build(np.linspace(0, 500, 501), source)
    assert np.all(short.target[short.energy_ev < 50] == 0)
    assert np.all(short.components["Li:Shell_2"][short.energy_ev < 120] == 0)
    assert short.metadata["shells"]["Li:Shell_1"]["fsum_scale"] == hf.metadata["shells"]["Li:Shell_1"]["fsum_scale"]
    np.testing.assert_allclose(short.core, short.target+short.other_core)
    np.testing.assert_allclose(short.total, short.core+short.valence)


def test_match_recovers_scale_linear_and_ignores_target(source):
    energy = np.linspace(0, 900, 1801)
    hf = build(energy, source)
    baseline = .04 + .00003*energy
    raw = 1700*(hf.total+baseline)
    protected = (energy >= 50) & (energy <= 90)
    raw[protected] += 12345
    result = match_hf_scale(energy, raw, hf.total, windows=((20, 800),),
        edge=(50, 90), prenorm_windows=((20, 800),))
    assert result["metadata"]["raw_to_hf_scale"] == pytest.approx(1/1700, rel=1e-9)
    np.testing.assert_allclose(result["calibrated"][~protected], hf.total[~protected], atol=1e-10)
    assert not result["pre_mask"][protected].any()
    assert not result["fit_mask"][protected].any()
    assert result["metadata"]["parameter_covariance"] is None


def test_target_is_retained_instead_of_subtracted(source):
    energy = np.linspace(0, 900, 1801)
    hf = build(energy, source)
    feature = np.zeros_like(energy)
    feature[(energy >= 60) & (energy <= 80)] = -.003  # negative fine-structure retained
    raw = 1200*(hf.total+.005+.00002*energy+feature)
    mapped = hf.valence.copy()
    mapped[energy > 850] = np.nan
    result = extract_hf_target(energy, raw, hf, mapped, windows=((20, 800),),
                              edge=(50, 90), prenorm_windows=((20, 800),))
    valid = result["available"]
    np.testing.assert_allclose(result["residual"][valid], (hf.target+feature)[valid], atol=1e-10)
    np.testing.assert_allclose(result["model_fit_residual"][valid], feature[valid], atol=1e-10)
    assert np.isnan(result["residual"][~valid]).all()
    assert result["parameter_covariance"] is None


def test_bad_scales_and_empty_support_fail():
    energy = np.arange(101.)
    for y, template in ((np.zeros(101), np.ones(101)),
                        (np.ones(101), np.full(101, np.nan)),
                        (np.ones(101), np.ones(101))):
        with pytest.raises(ValueError):
            match_hf_scale(energy, y, template, windows=((0, 100),),
                           edge=(40, 60), prenorm_windows=((0, 100),))


def test_disjoint_area_does_not_bridge_gap():
    energy = np.arange(101.)
    model = 1+np.sin(energy/10)**2
    raw = 2*model
    raw[40:61] = 1e9
    result = match_hf_scale(energy, raw, model, windows=((0, 100),),
        edge=(40, 60), prenorm_windows=((0, 100),), fit_linear=False)
    assert result["metadata"]["area_pre_scale"] == pytest.approx(.5)


def test_target_cannot_be_valence(source):
    with pytest.raises(ValueError, match="target cannot"):
        build_hf_energy(np.arange(301.), 5, source, {"Li": 1},
                        target=("Li", "Shell_3"), valence_cutoff_ev=20)
