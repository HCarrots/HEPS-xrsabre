"""DABAX parsing and occupancy semantics require no installed xraylib."""

import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from xrs_compton_extraction.backgrounds import DabaxProfileSource, build_core_profile


@pytest.fixture
def table(tmp_path):
    path = tmp_path / "ComptonProfiles.dat"
    path.write_text(
        "#D test date\n#S 2 He\n#N 3\n#UOCCUP 2\n#UBIND 24.6\n"
        "#L pz total Shell_1\n0 2 1\n1 1 .5\n100 0 0\n")
    return path


def test_native_occupancy_interpolation_and_weight_once(table):
    source = DabaxProfileSource(table)
    assert source.electron_occupancy("He", "Shell_1") == 2
    assert source.binding_energy_ev(2, "Shell_1") == 24.6
    np.testing.assert_allclose(source.partial_profile("He", "Shell_1", [-.5, 0, .5]), [.75, 1, .75])
    core = build_core_profile([-1, 0, 1], {"He": 3}, source,
                              shells_by_element={"He": ["Shell_1"]})
    np.testing.assert_allclose(core.total_profile, [3, 6, 3])
    assert source.provenance["source_sha256"] == hashlib.sha256(table.read_bytes()).hexdigest()
    excluded = build_core_profile([-1, 0, 1], {"He": 3}, source,
        shells_by_element={"He": ["Shell_1"]}, exclude_target=("He", "Shell_1"))
    np.testing.assert_array_equal(excluded.total_profile, 0)


def test_invalid_shell_and_extrapolation_rejected(table):
    source = DabaxProfileSource(table)
    with pytest.raises(ValueError, match="shell"):
        source.partial_profile("He", "K", [0])
    with pytest.raises(ValueError, match="shell"):
        source.partial_profile("He", "Shell_2", [0])
    with pytest.raises(ValueError, match="extrapolation"):
        source.total_profile("He", [101])
    with pytest.raises(ValueError, match="finite"):
        source.total_profile("He", [np.nan])


def test_malformed_table_rejected(table):
    original = table.read_text()
    for modified in (original.replace("#UOCCUP 2", "#UOCCUP 1"),
                     original.replace("#N 3", "#N 4"),
                     original.replace("1 1 .5", "0 1 .5"),
                     original + original,
                     original.replace("#UBIND 24.6", "")):
        table.write_text(modified)
        with pytest.raises(ValueError):
            DabaxProfileSource(table)


def test_source_runs_when_xraylib_unavailable(table):
    code = ("import sys; sys.modules['xraylib']=None; "
            "from xrs_compton_extraction.backgrounds import DabaxProfileSource; "
            "s=DabaxProfileSource(sys.argv[1]); "
            "assert s.electron_occupancy('He','Shell_1')==2")
    subprocess.run([sys.executable, "-c", code, str(table)], check=True, capture_output=True)


def test_local_ho_b_table():
    path = Path(__file__).resolve().parents[1] / "resources/compton_profiles/ComptonProfiles.dat"
    if not path.exists():
        pytest.skip("external DABAX table not bundled")
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    source = DabaxProfileSource(path)
    for element, z in (("Ho", 67), ("B", 5)):
        shells = source.available_shells(element)
        assert sum(source.electron_occupancy(element, shell) for shell in shells) == z
        pz = np.linspace(0, 100, 401)
        total = source.total_profile(element, pz)
        summed = sum(source.electron_occupancy(element, shell)*source.partial_profile(element, shell, pz)
                     for shell in shells)
        assert np.linalg.norm(summed-total)/np.linalg.norm(total) < .01
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
