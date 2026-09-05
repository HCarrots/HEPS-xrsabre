from __future__ import annotations

import numpy as np
import pytest

from xrs_compton_extraction.geometry import au_to_inverse_angstrom
from xrs_compton_extraction.synthetic import (
    SyntheticData,
    generate_synthetic_data,
    generate_synthetic_spectrum,
)


def _components():
    energy = np.linspace(0.0, 4.0, 5)
    edge = np.array([0.0, 0.0, 2.0, 3.0, 4.0])
    valence = np.array([4.0, 3.0, 2.0, 1.0, 0.0])
    core = np.full(5, 5.0)
    background = 1.0
    return energy, edge, valence, core, background


def test_noiseless_sum_is_exact() -> None:
    energy, edge, valence, core, background = _components()
    generated = generate_synthetic_data(
        energy,
        edge_intensity=edge,
        valence_intensity=valence,
        core_intensity=core,
        background_intensity=background,
        add_poisson_noise=False,
    )

    expected = edge + valence + core + background
    np.testing.assert_array_equal(generated.expected_counts, expected)
    np.testing.assert_array_equal(generated.counts, expected)
    np.testing.assert_array_equal(generated.poisson_noise, 0.0)
    np.testing.assert_array_equal(
        sum(generated.components.values(), start=np.zeros_like(energy)), expected
    )


def test_seed_makes_poisson_sampling_reproducible() -> None:
    energy = np.linspace(0.0, 10.0, 101)
    kwargs = {
        "edge_intensity": lambda x: np.where(x >= 5.0, 30.0, 0.0),
        "valence_intensity": 20.0,
        "core_intensity": np.linspace(1.0, 5.0, energy.size),
        "background_intensity": 3.0,
        "seed": 8675309,
    }
    first = generate_synthetic_data(energy, **kwargs)
    second = generate_synthetic_data(energy, **kwargs)
    third = generate_synthetic_data(energy, **{**kwargs, "seed": 8675310})

    np.testing.assert_array_equal(first.counts, second.counts)
    assert not np.array_equal(first.counts, third.counts)
    np.testing.assert_array_equal(first.poisson_noise, first.counts - first.expected_counts)
    assert first.metadata["synthetic"] is True
    assert first.metadata["synthetic_seed"] == 8675309


def test_component_scalars_arrays_and_callables_are_supported() -> None:
    energy = np.arange(4.0)
    generated = generate_synthetic_data(
        energy,
        edge_intensity=lambda x: x,
        valence_intensity=2.0,
        core_intensity=np.arange(4.0),
        background_intensity=np.array([1.0]),
        add_poisson_noise=False,
    )
    np.testing.assert_array_equal(generated.expected_counts, 2.0 * energy + 3.0)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"edge_intensity": [1.0, 2.0]}, "cannot be broadcast"),
        ({"valence_intensity": -1.0}, "non-negative"),
        ({"core_intensity": np.nan}, "finite"),
        ({"q_au": 0.0}, "strictly positive"),
        ({"seed": 1.5}, "integer or None"),
        ({"add_poisson_noise": "yes"}, "must be a boolean"),
    ],
)
def test_invalid_generator_inputs_raise(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        generate_synthetic_data(np.arange(3.0), **kwargs)


def test_energy_grid_must_be_one_dimensional_finite_and_increasing() -> None:
    with pytest.raises(ValueError, match="one-dimensional"):
        generate_synthetic_data(np.ones((2, 2)))
    with pytest.raises(ValueError, match="finite"):
        generate_synthetic_data([0.0, np.inf])
    with pytest.raises(ValueError, match="strictly increasing"):
        generate_synthetic_data([0.0, 2.0, 1.0])


def test_generated_truth_arrays_are_read_only() -> None:
    generated = generate_synthetic_data([0.0, 1.0], background_intensity=1.0)
    with pytest.raises(ValueError):
        generated.counts[0] = 123.0
    with pytest.raises(TypeError):
        generated.metadata["new"] = "value"  # type: ignore[index]


def test_return_components_switch() -> None:
    generated = generate_synthetic_spectrum(
        [0.0, 1.0],
        background_intensity=5.0,
        return_components=True,
    )
    assert isinstance(generated, SyntheticData)


def test_generate_standard_spectrum_with_synthetic_provenance() -> None:
    from xrs_compton_extraction.data import XRSSpectrum

    energy_loss = np.linspace(10.0, 20.0, 6)
    spectrum = generate_synthetic_spectrum(
        energy_loss,
        edge_intensity=10.0,
        valence_intensity=20.0,
        core_intensity=30.0,
        background_intensity=5.0,
        q_au=2.0,
        seed=42,
    )

    assert isinstance(spectrum, XRSSpectrum)
    np.testing.assert_array_equal(spectrum.energy_loss_eV, energy_loss)
    np.testing.assert_allclose(spectrum.q_au, 2.0)
    np.testing.assert_allclose(
        spectrum.q_inverse_angstrom, au_to_inverse_angstrom(2.0)
    )
    np.testing.assert_allclose(spectrum.uncertainty, np.sqrt(65.0))
    assert spectrum.metadata["synthetic"] is True
    assert spectrum.metadata["synthetic_seed"] == 42
