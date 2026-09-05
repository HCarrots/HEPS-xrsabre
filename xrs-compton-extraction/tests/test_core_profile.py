from types import SimpleNamespace

import numpy as np
import pytest

from xrs_compton_extraction.backgrounds.core_profile import (
    CoreAsymmetry,
    ProfileDependencyError,
    XraylibProfileSource,
    build_core_profile,
    gaussian_resolution_convolution,
    interpolate_profile,
)


def fake_backend():
    return SimpleNamespace(
        __version__="test", K_SHELL=0, L1_SHELL=1,
        AtomicNumberToSymbol=lambda z: {6: "C", 8: "O"}[z],
        SymbolToAtomicNumber=lambda s: {"C": 6, "O": 8}[s],
        ElectronConfig_Biggs=lambda z, s: 2.0,
        ComptonProfile=lambda z, p: 4 * np.exp(-p * p),
        ComptonProfile_Partial=lambda z, s, p: np.exp(-p * p),
    )


def test_core_weights_exclusion_and_readonly():
    source = XraylibProfileSource(fake_backend())
    pz = np.linspace(-3, 3, 61)
    result = build_core_profile(
        pz, {"C": 1, "O": 2}, source,
        shells_by_element={"C": ["K"], "O": ["K"]}, exclude_target=("C", "K"),
    )
    assert set(result.components) == {"O:K"}
    np.testing.assert_allclose(result.total_profile, 4 * np.exp(-pz * pz))
    assert result.excluded_target == "C:K"
    assert not result.total_profile.flags.writeable
    assert result.source_provenance["occupancy_source"] == "xraylib.ElectronConfig_Biggs"


def test_core_requires_explicit_shell_selection():
    with pytest.raises(ValueError, match="explicitly select"):
        build_core_profile([-1, 0, 1], {"C": 1}, XraylibProfileSource(fake_backend()))


def test_missing_biggs_api_requires_explicit_alternative():
    backend = fake_backend()
    backend.ElectronConfig = backend.ElectronConfig_Biggs
    del backend.ElectronConfig_Biggs
    with pytest.raises(ProfileDependencyError, match="occupancy_source"):
        XraylibProfileSource(backend)
    source = XraylibProfileSource(backend, occupancy_source="electron_config")
    assert source.electron_occupancy("C", "K") == 2
    assert source.provenance["occupancy_source"] == "xraylib.ElectronConfig"


def test_even_interpolation_and_no_extrapolation():
    np.testing.assert_allclose(interpolate_profile([0, 1, 2], [3, 2, 1], [-1.5, 0, 1.5]), [1.5, 3, 1.5])
    with pytest.raises(ValueError, match="extrapolation"):
        interpolate_profile([0, 1], [1, 0.5], [2])


def test_convolution_preserves_well_supported_area_and_rejects_nonuniform_grid():
    pz = np.linspace(-10, 10, 2001)
    y = np.exp(-pz**2)
    convolved = gaussian_resolution_convolution(y, pz, 0.3)
    assert np.trapezoid(convolved, pz) == pytest.approx(np.trapezoid(y, pz), rel=1e-6)
    assert convolved.max() < y.max()
    with pytest.raises(ValueError, match="uniform"):
        gaussian_resolution_convolution([1, 2, 1], [-1, 0, 2], 0.3)
    np.testing.assert_allclose(CoreAsymmetry(0.2, 1).factor([-1, 1]).sum(), 2)


def test_real_xraylib_one_electron_normalization_and_helium_sum():
    backend = pytest.importorskip("xraylib")
    source = XraylibProfileSource(backend, occupancy_source="electron_config")
    pz = np.linspace(-30, 30, 6001)
    hydrogen = source.partial_profile(1, "K", pz)
    assert np.trapezoid(hydrogen, pz) == pytest.approx(1.0, rel=0.02)
    np.testing.assert_allclose(
        source.total_profile(2, [-1, 0, 1]),
        2 * source.partial_profile(2, "K", [-1, 0, 1]), rtol=0.01,
    )
    assert source.provenance["provider_version"] != "unknown"
