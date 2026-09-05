"""Hartree--Fock atomic core-background profiles.

The module intentionally keeps the profile provider behind a small protocol.
This makes the scientific operation (stoichiometric and shell-occupation
weighting) testable without downloading or redistributing the Biggs tables.

``partial_profile`` values are defined *per electron*.  The builder therefore
multiplies every shell profile by ``electron_occupancy`` exactly once.  This is
the convention used by the Biggs/DABAX tables exposed by xraylib.
"""

from __future__ import annotations

import importlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Protocol, TypeAlias, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import gaussian_filter1d

FloatArray = NDArray[np.float64]
Element: TypeAlias = int | str
Shell: TypeAlias = int | str

BIGGS_REFERENCE_DOI = "10.1016/0092-640X(75)90030-3"
BIGGS_DATA_SOURCE = (
    "Biggs, Mendelsohn and Mann Hartree-Fock/Dirac-Hartree-Fock "
    "Compton profiles via xraylib"
)


class ProfileDependencyError(RuntimeError):
    """Raised when an optional atomic-profile provider is unavailable."""


@dataclass(frozen=True, slots=True)
class ElementIdentity:
    """Canonical identity returned by a :class:`ProfileSource`."""

    atomic_number: int
    symbol: str

    def __post_init__(self) -> None:
        if not isinstance(self.atomic_number, int):
            raise TypeError("atomic_number must be a positive integer")
        if isinstance(self.atomic_number, bool) or self.atomic_number <= 0:
            raise ValueError("atomic_number must be a positive integer")
        if not isinstance(self.symbol, str) or not self.symbol.strip():
            raise ValueError("symbol must be a non-empty string")
        object.__setattr__(self, "symbol", self.symbol.strip())


@runtime_checkable
class ProfileSource(Protocol):
    """Provider contract for atom- and shell-resolved Compton profiles.

    A partial profile must be the profile of one electron in ``shell``.  The
    separate occupation method is part of the contract so that composition and
    orbital occupation are never conflated.  Providers may evaluate analytic
    functions or interpolate their own tables; all momenta are in atomic units.
    """

    @property
    def provenance(self) -> Mapping[str, str]:
        """Return provider/version/reference metadata."""

    def resolve_element(self, element: Element) -> ElementIdentity:
        """Resolve a symbol or atomic number to a canonical identity."""

    def available_shells(self, element: Element) -> Sequence[Shell]:
        """Return occupied shells available for ``element``."""

    def shell_label(self, shell: Shell) -> str:
        """Return a stable, human-readable shell label."""

    def total_profile(self, element: Element, pz_au: ArrayLike) -> FloatArray:
        """Evaluate the occupation-weighted total atomic profile."""

    def partial_profile(
        self,
        element: Element,
        shell: Shell,
        pz_au: ArrayLike,
    ) -> FloatArray:
        """Evaluate a one-electron partial shell profile."""

    def electron_occupancy(self, element: Element, shell: Shell) -> float:
        """Return the Biggs-compatible shell occupation."""


@dataclass(frozen=True, slots=True)
class TargetShell:
    """Element and shell to omit from the calculated core background."""

    element: Element
    shell: Shell


@dataclass(frozen=True, slots=True)
class CoreAsymmetry:
    """Explicit phenomenological odd correction to an even atomic profile.

    The applied factor is ``1 + amplitude * tanh(pz / scale_au)``.  It is not
    part of the Hartree--Fock table and is disabled unless an instance is passed
    to :func:`build_core_profile`.  ``abs(amplitude) < 1`` keeps the factor
    positive without clipping the calculated profile.
    """

    amplitude: float
    scale_au: float

    def __post_init__(self) -> None:
        amplitude = _finite_scalar(self.amplitude, "amplitude")
        scale = _finite_scalar(self.scale_au, "scale_au")
        if abs(amplitude) >= 1.0:
            raise ValueError("asymmetry amplitude must satisfy abs(amplitude) < 1")
        if scale <= 0.0:
            raise ValueError("asymmetry scale_au must be greater than zero")
        object.__setattr__(self, "amplitude", amplitude)
        object.__setattr__(self, "scale_au", scale)

    def factor(self, pz_au: ArrayLike) -> FloatArray:
        """Evaluate the multiplicative correction on a signed momentum grid."""

        pz = _float_array(pz_au, "pz_au")
        factor = 1.0 + self.amplitude * np.tanh(pz / self.scale_au)
        return _readonly(factor)


@dataclass(frozen=True, slots=True)
class CoreProfileResult:
    """Immutable shell components and their summed core profile."""

    pz_au: ArrayLike
    components: Mapping[str, ArrayLike]
    total_profile: ArrayLike
    component_metadata: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    stoichiometry: Mapping[str, float] = field(default_factory=dict)
    source_provenance: Mapping[str, str] = field(default_factory=dict)
    excluded_target: str | None = None
    resolution_sigma_au: float | None = None
    asymmetry: CoreAsymmetry | None = None

    def __post_init__(self) -> None:
        pz = _strict_grid(self.pz_au, "pz_au")
        components: dict[str, FloatArray] = {}
        for raw_name, raw_values in self.components.items():
            name = _nonempty_string(raw_name, "component name")
            if name in components:
                raise ValueError(f"duplicate component name {name!r}")
            values = _profile_array(raw_values, f"components[{name!r}]", len(pz))
            components[name] = values

        expected = (
            np.sum(np.stack(tuple(components.values())), axis=0)
            if components
            else np.zeros_like(pz)
        )
        total = _profile_array(self.total_profile, "total_profile", len(pz))
        if not np.allclose(total, expected, rtol=1e-12, atol=1e-14):
            raise ValueError("total_profile must equal the sum of components")

        metadata: dict[str, Mapping[str, Any]] = {}
        if set(self.component_metadata) != set(components):
            raise ValueError("component_metadata keys must match components")
        for name, item in self.component_metadata.items():
            if not isinstance(item, Mapping):
                raise TypeError("each component_metadata value must be a mapping")
            metadata[name] = MappingProxyType(dict(item))

        composition: dict[str, float] = {}
        for raw_symbol, raw_amount in self.stoichiometry.items():
            symbol = _nonempty_string(raw_symbol, "stoichiometry symbol")
            amount = _finite_scalar(raw_amount, f"stoichiometry[{symbol!r}]")
            if amount <= 0:
                raise ValueError("stoichiometric coefficients must be greater than zero")
            composition[symbol] = amount

        provenance: dict[str, str] = {}
        for raw_key, raw_value in self.source_provenance.items():
            key = _nonempty_string(raw_key, "source provenance key")
            provenance[key] = _nonempty_string(raw_value, f"source_provenance[{key!r}]")

        sigma = self.resolution_sigma_au
        if sigma is not None:
            sigma = _finite_scalar(sigma, "resolution_sigma_au")
            if sigma <= 0:
                raise ValueError("resolution_sigma_au must be greater than zero")
        if self.asymmetry is not None and not isinstance(self.asymmetry, CoreAsymmetry):
            raise ValueError("asymmetry must be a CoreAsymmetry or None")

        pz.setflags(write=False)
        total.setflags(write=False)
        object.__setattr__(self, "pz_au", pz)
        object.__setattr__(self, "components", MappingProxyType(components))
        object.__setattr__(self, "total_profile", total)
        object.__setattr__(self, "component_metadata", MappingProxyType(metadata))
        object.__setattr__(self, "stoichiometry", MappingProxyType(composition))
        object.__setattr__(self, "source_provenance", MappingProxyType(provenance))
        object.__setattr__(self, "resolution_sigma_au", sigma)


class XraylibProfileSource:
    """Optional xraylib adapter for the Biggs atomic Compton-profile tables."""

    def __init__(
        self, module: Any | None = None, *, occupancy_source: str = "biggs"
    ) -> None:
        if module is None:
            try:
                module = importlib.import_module("xraylib")
            except ImportError as exc:
                raise ProfileDependencyError(
                    "xraylib is required for XraylibProfileSource; install the "
                    "optional dependency with `pip install xraylib`"
                ) from exc
        self._module = module
        required = (
            "AtomicNumberToSymbol",
            "SymbolToAtomicNumber",
            "ComptonProfile",
            "ComptonProfile_Partial",
        )
        missing = [name for name in required if not callable(getattr(module, name, None))]
        if missing:
            raise ProfileDependencyError(
                "the imported xraylib module lacks required APIs: " + ", ".join(missing)
            )
        occupancy_apis = {"biggs": "ElectronConfig_Biggs", "electron_config": "ElectronConfig"}
        if occupancy_source not in occupancy_apis:
            raise ValueError("occupancy_source must be 'biggs' or 'electron_config'")
        self._occupancy_api = occupancy_apis[occupancy_source]
        self._occupancy = getattr(module, self._occupancy_api, None)
        if not callable(self._occupancy):
            raise ProfileDependencyError(
                f"xraylib does not expose {self._occupancy_api}. "
                "To explicitly use its alternative electron configuration, pass "
                "occupancy_source='electron_config'; verify occupations for the selected shells."
            )

        shell_constants: dict[str, int] = {}
        for name in dir(module):
            value = getattr(module, name)
            if name.endswith("_SHELL") and isinstance(value, int) and not isinstance(value, bool):
                shell_constants[name.removesuffix("_SHELL").upper()] = value
        if not shell_constants:
            raise ProfileDependencyError("xraylib exposes no *_SHELL constants")
        self._shell_by_label = shell_constants
        self._label_by_shell = {
            value: name
            for name, value in sorted(shell_constants.items(), key=lambda item: item[1])
        }

        version = getattr(module, "__version__", None)
        if version is None:
            version = getattr(module, "XRL_VERSION", "unknown")
        self._provenance = MappingProxyType(
            {
                "provider": "xraylib",
                "provider_version": str(version),
                "profile_data_source": BIGGS_DATA_SOURCE,
                "reference_doi": BIGGS_REFERENCE_DOI,
                "occupancy_source": f"xraylib.{self._occupancy_api}",
            }
        )

    @property
    def provenance(self) -> Mapping[str, str]:
        return self._provenance

    def resolve_element(self, element: Element) -> ElementIdentity:
        if isinstance(element, bool):
            raise TypeError("element must be an atomic number or symbol")
        if isinstance(element, int):
            atomic_number = element
        elif isinstance(element, str) and element.strip():
            symbol = element.strip()
            try:
                atomic_number = int(self._module.SymbolToAtomicNumber(symbol))
            except Exception as exc:
                raise ValueError(f"unknown element symbol {symbol!r}") from exc
        else:
            raise ValueError("element must be an atomic number or symbol")
        if not 1 <= atomic_number <= 102:
            raise ValueError("Biggs Compton profiles are available only for Z=1..102")
        try:
            symbol = str(self._module.AtomicNumberToSymbol(atomic_number))
        except Exception as exc:
            raise ValueError(f"invalid atomic number {atomic_number}") from exc
        return ElementIdentity(atomic_number=atomic_number, symbol=symbol)

    def _resolve_shell(self, shell: Shell) -> int:
        if isinstance(shell, bool):
            raise TypeError("shell must be an integer constant or shell label")
        if isinstance(shell, int):
            shell_id = shell
        elif isinstance(shell, str) and shell.strip():
            label = shell.strip().upper().removesuffix("_SHELL")
            try:
                shell_id = self._shell_by_label[label]
            except KeyError as exc:
                raise ValueError(f"unknown xraylib shell label {shell!r}") from exc
        else:
            raise ValueError("shell must be an integer constant or shell label")
        if shell_id not in self._label_by_shell:
            raise ValueError(f"unknown xraylib shell constant {shell_id}")
        return shell_id

    def shell_label(self, shell: Shell) -> str:
        return self._label_by_shell[self._resolve_shell(shell)]

    def available_shells(self, element: Element) -> tuple[int, ...]:
        identity = self.resolve_element(element)
        occupied: list[int] = []
        for shell in sorted(self._label_by_shell):
            try:
                occupancy = float(
                    self._occupancy(identity.atomic_number, shell)
                )
                self._module.ComptonProfile_Partial(identity.atomic_number, shell, 0.0)
            except ValueError:
                # xraylib rejects shells outside the table for an element.
                continue
            if math.isfinite(occupancy) and occupancy > 0.0:
                occupied.append(shell)
        return tuple(occupied)

    def total_profile(self, element: Element, pz_au: ArrayLike) -> FloatArray:
        identity = self.resolve_element(element)
        pz = _float_array(pz_au, "pz_au")
        values = np.asarray(
            [
                self._module.ComptonProfile(identity.atomic_number, abs(float(value)))
                for value in pz
            ],
            dtype=np.float64,
        )
        return _profile_array(values, "xraylib total profile", len(pz))

    def partial_profile(
        self,
        element: Element,
        shell: Shell,
        pz_au: ArrayLike,
    ) -> FloatArray:
        identity = self.resolve_element(element)
        shell_id = self._resolve_shell(shell)
        pz = _float_array(pz_au, "pz_au")
        values = np.asarray(
            [
                self._module.ComptonProfile_Partial(
                    identity.atomic_number,
                    shell_id,
                    abs(float(value)),
                )
                for value in pz
            ],
            dtype=np.float64,
        )
        return _profile_array(values, "xraylib partial profile", len(pz))

    def electron_occupancy(self, element: Element, shell: Shell) -> float:
        identity = self.resolve_element(element)
        shell_id = self._resolve_shell(shell)
        occupancy = _finite_scalar(
            self._occupancy(identity.atomic_number, shell_id),
            "xraylib electron occupancy",
        )
        if occupancy < 0:
            raise ValueError("xraylib returned a negative electron occupancy")
        return occupancy


def interpolate_profile(
    source_pz_au: ArrayLike,
    source_profile: ArrayLike,
    target_pz_au: ArrayLike,
) -> FloatArray:
    """Linearly interpolate an even profile without extrapolation.

    ``source_pz_au`` is a strictly increasing non-negative grid.  Signed target
    momenta are mapped through their absolute value.  A target outside the
    supplied source range is rejected rather than silently extrapolated.
    """

    source_pz = _strict_grid(source_pz_au, "source_pz_au", nonnegative=True)
    source_values = _profile_array(source_profile, "source_profile", len(source_pz))
    target = _float_array(target_pz_au, "target_pz_au")
    query = np.abs(target)
    tolerance = np.finfo(np.float64).eps * max(1.0, float(source_pz[-1])) * 8
    if np.any(query < source_pz[0] - tolerance) or np.any(query > source_pz[-1] + tolerance):
        raise ValueError("target_pz_au lies outside source_pz_au; extrapolation is forbidden")
    interpolated = np.interp(query, source_pz, source_values)
    return _readonly(interpolated)


def gaussian_resolution_convolution(
    profile: ArrayLike,
    pz_au: ArrayLike,
    sigma_au: float,
) -> FloatArray:
    """Convolve a profile with a Gaussian on an explicitly uniform grid."""

    grid = _strict_grid(pz_au, "pz_au")
    values = _profile_array(profile, "profile", len(grid))
    sigma = _finite_scalar(sigma_au, "sigma_au")
    if sigma <= 0:
        raise ValueError("sigma_au must be greater than zero")
    spacing = np.diff(grid)
    tolerance = max(np.finfo(np.float64).eps * 32, abs(float(spacing[0])) * 1e-10)
    if not np.allclose(spacing, spacing[0], rtol=1e-10, atol=tolerance):
        raise ValueError(
            "Gaussian convolution requires a uniform pz_au grid; explicitly "
            "resample non-uniform data first"
        )
    broadened = gaussian_filter1d(
        values,
        sigma=sigma / float(spacing[0]),
        mode="constant",
        cval=0.0,
        truncate=6.0,
    )
    return _profile_array(broadened, "broadened profile", len(grid))


def build_core_profile(
    pz_au: ArrayLike,
    stoichiometry: Mapping[Element, float],
    source: ProfileSource,
    *,
    exclude_target: TargetShell | tuple[Element, Shell] | None = None,
    shells_by_element: Mapping[Element, Sequence[Shell]] | None = None,
    source_grid_pz_au: ArrayLike | None = None,
    resolution_sigma_au: float | None = None,
    asymmetry: CoreAsymmetry | None = None,
) -> CoreProfileResult:
    """Build a stoichiometric, shell-resolved Hartree--Fock core profile.

    If ``source_grid_pz_au`` is given, every source profile is evaluated there
    and explicitly interpolated to ``abs(pz_au)``.  The grid must span every
    requested momentum; extrapolation is never performed.  Resolution
    convolution is optional and requires the experimental grid to be uniform.
    """

    experimental_pz = _strict_grid(pz_au, "pz_au")
    if not isinstance(source, ProfileSource):
        raise TypeError("source does not implement the ProfileSource protocol")
    if not isinstance(stoichiometry, Mapping) or not stoichiometry:
        raise ValueError("stoichiometry must be a non-empty mapping")

    composition: dict[int, tuple[ElementIdentity, float]] = {}
    original_elements: dict[int, Element] = {}
    for raw_element, raw_amount in stoichiometry.items():
        identity = source.resolve_element(raw_element)
        amount = _finite_scalar(raw_amount, f"stoichiometry[{raw_element!r}]")
        if amount <= 0:
            raise ValueError("stoichiometric coefficients must be greater than zero")
        if identity.atomic_number in composition:
            raise ValueError(f"duplicate element aliases for {identity.symbol}")
        composition[identity.atomic_number] = (identity, amount)
        original_elements[identity.atomic_number] = raw_element

    shell_overrides: dict[int, tuple[Shell, ...]] = {}
    if shells_by_element is not None:
        if not isinstance(shells_by_element, Mapping):
            raise ValueError("shells_by_element must be a mapping")
        for raw_element, raw_shells in shells_by_element.items():
            identity = source.resolve_element(raw_element)
            if identity.atomic_number not in composition:
                raise ValueError("shells_by_element contains an element absent from stoichiometry")
            if identity.atomic_number in shell_overrides:
                raise ValueError(f"duplicate shell override for {identity.symbol}")
            if isinstance(raw_shells, (str, bytes)):
                raise TypeError("shell selections must be a sequence, not a string")
            selected = tuple(raw_shells)
            if len({source.shell_label(shell) for shell in selected}) != len(selected):
                raise ValueError(f"duplicate shell selection for {identity.symbol}")
            shell_overrides[identity.atomic_number] = selected

    if set(shell_overrides) != set(composition):
        raise ValueError(
            "shells_by_element must explicitly select core shells for every element; "
            "use an empty sequence for an element with no included core shells"
        )

    excluded_identity: ElementIdentity | None = None
    excluded_label: str | None = None
    if exclude_target is not None:
        if isinstance(exclude_target, TargetShell):
            target = exclude_target
        else:
            try:
                target = TargetShell(*exclude_target)
            except (TypeError, ValueError) as exc:
                raise ValueError("exclude_target must contain exactly (element, shell)") from exc
        excluded_identity = source.resolve_element(target.element)
        if excluded_identity.atomic_number not in composition:
            raise ValueError("excluded target element is absent from stoichiometry")
        excluded_label = source.shell_label(target.shell)

    sampling_grid: FloatArray | None = None
    if source_grid_pz_au is not None:
        sampling_grid = _strict_grid(
            source_grid_pz_au,
            "source_grid_pz_au",
            nonnegative=True,
        )

    if resolution_sigma_au is not None:
        sigma = _finite_scalar(resolution_sigma_au, "resolution_sigma_au")
        if sigma <= 0:
            raise ValueError("resolution_sigma_au must be greater than zero")
        # Validate before spending time evaluating every source component.
        _require_uniform_grid(experimental_pz)
    else:
        sigma = None
    if asymmetry is not None and not isinstance(asymmetry, CoreAsymmetry):
        raise ValueError("asymmetry must be a CoreAsymmetry or None")
    asymmetry_factor = None if asymmetry is None else asymmetry.factor(experimental_pz)

    components: dict[str, FloatArray] = {}
    component_metadata: dict[str, Mapping[str, Any]] = {}
    output_composition: dict[str, float] = {}

    for atomic_number in sorted(composition):
        identity, amount = composition[atomic_number]
        output_composition[identity.symbol] = amount
        shells = shell_overrides.get(atomic_number)
        if shells is None:
            shells = tuple(source.available_shells(original_elements[atomic_number]))
        for shell in shells:
            label = source.shell_label(shell)
            if (
                excluded_identity is not None
                and atomic_number == excluded_identity.atomic_number
                and label == excluded_label
            ):
                continue
            occupancy = _finite_scalar(
                source.electron_occupancy(original_elements[atomic_number], shell),
                f"occupancy for {identity.symbol}:{label}",
            )
            if occupancy < 0:
                raise ValueError("electron occupancies must be non-negative")
            if sampling_grid is None:
                per_electron = source.partial_profile(
                    original_elements[atomic_number],
                    shell,
                    np.abs(experimental_pz),
                )
                per_electron = _profile_array(
                    per_electron,
                    f"partial profile for {identity.symbol}:{label}",
                    len(experimental_pz),
                )
            else:
                sampled = source.partial_profile(
                    original_elements[atomic_number], shell, sampling_grid
                )
                per_electron = interpolate_profile(
                    sampling_grid,
                    sampled,
                    experimental_pz,
                )

            values = np.asarray(per_electron * occupancy * amount, dtype=np.float64)
            if asymmetry_factor is not None:
                values = values * asymmetry_factor
            if sigma is not None:
                values = gaussian_resolution_convolution(values, experimental_pz, sigma)
            name = f"{identity.symbol}:{label}"
            if name in components:
                raise ValueError(f"duplicate component name {name!r}")
            components[name] = _profile_array(values, name, len(experimental_pz))
            component_metadata[name] = {
                "atomic_number": atomic_number,
                "element": identity.symbol,
                "shell": label,
                "stoichiometric_coefficient": amount,
                "electron_occupancy": occupancy,
                "combined_weight": amount * occupancy,
            }

    total = (
        np.sum(np.stack(tuple(components.values())), axis=0)
        if components
        else np.zeros_like(experimental_pz)
    )
    excluded = (
        None
        if excluded_identity is None
        else f"{excluded_identity.symbol}:{excluded_label}"
    )
    return CoreProfileResult(
        pz_au=experimental_pz,
        components=components,
        total_profile=total,
        component_metadata=component_metadata,
        stoichiometry=output_composition,
        source_provenance=dict(source.provenance),
        excluded_target=excluded,
        resolution_sigma_au=sigma,
        asymmetry=asymmetry,
    )


def _finite_scalar(value: object, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise TypeError(f"{name} must be a finite number")
    try:
        result = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _nonempty_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def _float_array(value: ArrayLike, name: str) -> FloatArray:
    try:
        array = np.array(value, dtype=np.float64, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric array") from exc
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values")
    return array


def _strict_grid(
    value: ArrayLike,
    name: str,
    *,
    nonnegative: bool = False,
) -> FloatArray:
    grid = _float_array(value, name)
    if grid.size < 2:
        raise ValueError(f"{name} must contain at least two values")
    if np.any(np.diff(grid) <= 0):
        raise ValueError(f"{name} must be strictly increasing")
    if nonnegative and np.any(grid < 0):
        raise ValueError(f"{name} must be non-negative")
    grid.setflags(write=False)
    return grid


def _profile_array(value: ArrayLike, name: str, length: int) -> FloatArray:
    array = _float_array(value, name)
    if len(array) != length:
        raise ValueError(f"{name} has length {len(array)}, expected {length}")
    if np.any(array < 0):
        raise ValueError(f"{name} contains negative profile values")
    array.setflags(write=False)
    return array


def _readonly(value: ArrayLike) -> FloatArray:
    result = np.array(value, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def _require_uniform_grid(grid: FloatArray) -> None:
    spacing = np.diff(grid)
    tolerance = max(np.finfo(np.float64).eps * 32, abs(float(spacing[0])) * 1e-10)
    if not np.allclose(spacing, spacing[0], rtol=1e-10, atol=tolerance):
        raise ValueError(
            "Gaussian convolution requires a uniform pz_au grid; explicitly "
            "resample non-uniform data first"
        )


__all__ = [
    "BIGGS_DATA_SOURCE",
    "BIGGS_REFERENCE_DOI",
    "CoreAsymmetry",
    "CoreProfileResult",
    "ElementIdentity",
    "ProfileDependencyError",
    "ProfileSource",
    "TargetShell",
    "XraylibProfileSource",
    "build_core_profile",
    "gaussian_resolution_convolution",
    "interpolate_profile",
]
