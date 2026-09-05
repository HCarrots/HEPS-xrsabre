"""Deterministic synthetic spectra for numerical and pipeline tests.

The generator deliberately has no default experimental geometry or material
parameters.  Callers supply an energy-loss grid and the four expected-count
components explicitly.  This prevents convenient test defaults from being
mistaken for measured or literature-backed values.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, TypeAlias, overload

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .geometry import au_to_inverse_angstrom

if TYPE_CHECKING:
    from .data import XRSSpectrum

Component: TypeAlias = ArrayLike | Callable[[NDArray[np.float64]], ArrayLike]


def _one_dimensional_finite(value: ArrayLike, name: str) -> NDArray[np.float64]:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a real numeric array") from exc
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if array.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _component_values(
    component: Component,
    energy_loss_eV: NDArray[np.float64],
    name: str,
) -> NDArray[np.float64]:
    raw = component(energy_loss_eV.copy()) if callable(component) else component
    try:
        values = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"{name} must be numeric or a callable returning numeric values"
        ) from exc
    try:
        values = np.broadcast_to(values, energy_loss_eV.shape)
    except ValueError as exc:
        raise ValueError(
            f"{name} with shape {values.shape} cannot be broadcast to "
            f"energy_loss_eV shape {energy_loss_eV.shape}"
        ) from exc
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any(values < 0.0):
        raise ValueError(f"{name} must be non-negative expected counts")
    return np.array(values, dtype=np.float64, copy=True)


def _readonly(array: ArrayLike, *, dtype: np.dtype | type = np.float64) -> NDArray:
    result = np.array(array, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


@dataclass(frozen=True, slots=True)
class SyntheticData:
    """A generated spectrum together with its exact noiseless decomposition."""

    energy_loss_eV: NDArray[np.float64]
    edge: NDArray[np.float64]
    valence: NDArray[np.float64]
    core: NDArray[np.float64]
    background: NDArray[np.float64]
    expected_counts: NDArray[np.float64]
    counts: NDArray[np.float64]
    poisson_noise: NDArray[np.float64]
    seed: int | None
    energy_eV: NDArray[np.float64]
    q_au: float | NDArray[np.float64] | None = None
    metadata: Mapping[str, object] = MappingProxyType({})

    def __post_init__(self) -> None:
        array_fields = (
            "energy_loss_eV",
            "edge",
            "valence",
            "core",
            "background",
            "expected_counts",
            "counts",
            "poisson_noise",
            "energy_eV",
        )
        for field_name in array_fields:
            object.__setattr__(self, field_name, _readonly(getattr(self, field_name)))
        if isinstance(self.q_au, np.ndarray):
            object.__setattr__(self, "q_au", _readonly(self.q_au))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    @property
    def components(self) -> Mapping[str, NDArray[np.float64]]:
        """Return a read-only mapping of the four noiseless components."""

        return MappingProxyType(
            {
                "edge": self.edge,
                "valence": self.valence,
                "core": self.core,
                "background": self.background,
            }
        )

    def to_xrs_spectrum(self) -> XRSSpectrum:
        """Create the package's standard :class:`XRSSpectrum` representation."""

        # Import lazily: the truth/decomposition object remains usable in small
        # numerical tests even when only this module and NumPy are available.
        from .data import XRSSpectrum

        q_inverse_angstrom = (
            None if self.q_au is None else au_to_inverse_angstrom(self.q_au)
        )
        spectrum_metadata = dict(self.metadata)
        spectrum_metadata.update(
            {
                "synthetic": True,
                "synthetic_seed": self.seed,
                "synthetic_generator": "xrs_compton_extraction.synthetic",
            }
        )
        return XRSSpectrum(
            energy_eV=self.energy_eV,
            counts=self.counts,
            energy_loss_eV=self.energy_loss_eV,
            q_inverse_angstrom=q_inverse_angstrom,
            q_au=self.q_au,
            uncertainty=np.sqrt(self.expected_counts),
            scan_id="synthetic",
            metadata=spectrum_metadata,
        )


def generate_synthetic_data(
    energy_loss_eV: ArrayLike,
    *,
    edge_intensity: Component = 0.0,
    valence_intensity: Component = 0.0,
    core_intensity: Component = 0.0,
    background_intensity: Component = 0.0,
    energy_eV: ArrayLike | None = None,
    q_au: ArrayLike | None = None,
    seed: int | None = 0,
    add_poisson_noise: bool = True,
    metadata: Mapping[str, object] | None = None,
) -> SyntheticData:
    """Generate counts from four explicitly supplied expected-count components.

    The noiseless expectation is

    ``edge + valence + core + background``.

    With ``add_poisson_noise=True`` (the default), observed counts are sampled
    from a Poisson distribution with that expectation.  Consequently the
    reported additive ``poisson_noise`` is ``counts - expected_counts``.  The
    default ``seed=0`` is a deterministic test convention, not an experimental
    parameter; pass ``None`` for nondeterministic sampling.

    A component may be a scalar, an array matching the energy grid, or a
    callable receiving a copy of that grid.  All components are interpreted as
    expected counts per bin and therefore must be finite and non-negative.
    """

    loss = _one_dimensional_finite(energy_loss_eV, "energy_loss_eV")
    if loss.size > 1 and np.any(np.diff(loss) <= 0.0):
        raise ValueError("energy_loss_eV must be strictly increasing")

    components = {
        "edge": _component_values(edge_intensity, loss, "edge_intensity"),
        "valence": _component_values(valence_intensity, loss, "valence_intensity"),
        "core": _component_values(core_intensity, loss, "core_intensity"),
        "background": _component_values(
            background_intensity, loss, "background_intensity"
        ),
    }
    expected = sum(components.values(), start=np.zeros_like(loss))
    if not np.all(np.isfinite(expected)):
        raise ValueError("the summed expected counts overflowed to a non-finite value")

    if not isinstance(add_poisson_noise, (bool, np.bool_)):
        raise TypeError("add_poisson_noise must be a boolean")
    if seed is not None and (
        isinstance(seed, bool) or not isinstance(seed, (int, np.integer))
    ):
        raise TypeError("seed must be an integer or None")
    normalized_seed = None if seed is None else int(seed)

    if add_poisson_noise:
        try:
            counts = (
                np.random.default_rng(normalized_seed)
                .poisson(expected)
                .astype(np.float64)
            )
        except ValueError as exc:
            raise ValueError("expected counts are outside NumPy's supported Poisson range") from exc
    else:
        counts = expected.copy()

    if energy_eV is None:
        # This is merely a coordinate-preserving test representation.  It does
        # not assert that energy loss is an incident or scattered photon energy.
        energy = loss.copy()
        energy_axis_kind = "energy_loss_proxy"
    else:
        energy = _one_dimensional_finite(energy_eV, "energy_eV")
        if energy.shape != loss.shape:
            raise ValueError("energy_eV must have the same shape as energy_loss_eV")
        energy_axis_kind = "caller_supplied"

    if q_au is None:
        normalized_q: float | NDArray[np.float64] | None = None
    else:
        try:
            q_array = np.asarray(q_au, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise TypeError("q_au must be a real numeric scalar or array") from exc
        if not np.all(np.isfinite(q_array)):
            raise ValueError("q_au must contain only finite values")
        if np.any(q_array <= 0.0):
            raise ValueError("q_au must be strictly positive")
        if q_array.ndim == 0:
            normalized_q = float(q_array)
        elif q_array.shape == loss.shape:
            normalized_q = np.array(q_array, copy=True)
        else:
            raise ValueError("q_au must be scalar or have the same shape as energy_loss_eV")

    result_metadata = dict(metadata or {})
    result_metadata.update(
        {
            "synthetic": True,
            "synthetic_seed": normalized_seed,
            "synthetic_energy_axis": energy_axis_kind,
        }
    )

    return SyntheticData(
        energy_loss_eV=loss,
        edge=components["edge"],
        valence=components["valence"],
        core=components["core"],
        background=components["background"],
        expected_counts=expected,
        counts=counts,
        poisson_noise=counts - expected,
        seed=normalized_seed,
        energy_eV=energy,
        q_au=normalized_q,
        metadata=result_metadata,
    )


@overload
def generate_synthetic_spectrum(
    energy_loss_eV: ArrayLike,
    *,
    return_components: bool = False,
    **kwargs: object,
) -> XRSSpectrum: ...


@overload
def generate_synthetic_spectrum(
    energy_loss_eV: ArrayLike,
    *,
    return_components: bool,
    **kwargs: object,
) -> XRSSpectrum | SyntheticData: ...


def generate_synthetic_spectrum(
    energy_loss_eV: ArrayLike,
    *,
    return_components: bool = False,
    **kwargs: object,
) -> XRSSpectrum | SyntheticData:
    """Return an ``XRSSpectrum``, or its truth data when requested.

    All keyword arguments apart from ``return_components`` are forwarded to
    :func:`generate_synthetic_data`.
    """

    generated = generate_synthetic_data(energy_loss_eV, **kwargs)
    return generated if return_components else generated.to_xrs_spectrum()


__all__ = [
    "Component",
    "SyntheticData",
    "generate_synthetic_data",
    "generate_synthetic_spectrum",
]
