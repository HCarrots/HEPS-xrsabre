"""Empirical high-q valence Compton-profile construction and transfer.

The routines in this module deliberately do not define a universal ``high q``
cutoff.  Callers provide the candidate channels and explicit, dimensionless
quality scores that are appropriate for their experiment.  Every score and
weight used to select the reference channel is retained in the result.

The extracted profile follows the impulse-approximation convention

``p_z = omega / q - q / 2``

with energy in Hartree and momenta in atomic units.  A mapped profile is
returned as a density per eV, ``J(p_z) / (q * E_h)``, so its energy integral
has the same electron-count convention as the momentum-space profile.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.ndimage import gaussian_filter1d

from ..constants import HARTREE_ENERGY_EV
from ..geometry import energy_loss_to_pz

FloatArray: TypeAlias = NDArray[np.float64]
BoolArray: TypeAlias = NDArray[np.bool_]
MaskPolicy: TypeAlias = Literal["exclude", "linear"]
NormalizationConvention: TypeAlias = Literal[
    "full_symmetric", "doubled_positive_half"
]
AsymmetryMode: TypeAlias = Literal["off", "fit_first_order"]


def _readonly_array(
    value: ArrayLike,
    name: str,
    *,
    dtype: Any = np.float64,
    length: int | None = None,
) -> NDArray[Any]:
    try:
        result = np.array(value, dtype=dtype, copy=True)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric array") from exc
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if length is not None and result.size != length:
        raise ValueError(f"{name} must contain {length} values")
    if np.issubdtype(result.dtype, np.floating) and not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    result.setflags(write=False)
    return result


def _readonly_series(value: ArrayLike, name: str, length: int) -> FloatArray:
    try:
        broadcast = np.broadcast_to(np.asarray(value, dtype=np.float64), (length,))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be scalar or contain {length} values") from exc
    if not np.all(np.isfinite(broadcast)):
        raise ValueError(f"{name} must contain only finite values")
    result = np.array(broadcast, copy=True)
    result.setflags(write=False)
    return result


def _freeze(value: Any) -> Any:
    """Recursively make provenance-like values immutable."""

    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, np.ndarray):
        result = np.array(value, copy=True)
        result.setflags(write=False)
        return result
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_freeze(item) for item in value)
    return value


def _validated_score_mapping(
    values: Mapping[str, float], name: str
) -> Mapping[str, float]:
    if not isinstance(values, Mapping):
        raise TypeError(f"{name} must be a mapping")
    normalized: dict[str, float] = {}
    for raw_key, raw_value in values.items():
        key = str(raw_key).strip()
        if not key:
            raise ValueError(f"{name} keys must not be empty")
        if key == "uncontaminated_fraction":
            raise ValueError(
                "uncontaminated_fraction is derived from contamination_mask and "
                "must not be supplied"
            )
        value = float(raw_value)
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"{name}[{key!r}] must lie within [0, 1]")
        normalized[key] = value
    return MappingProxyType(normalized)


@dataclass(frozen=True, slots=True)
class ValenceReferenceCandidate:
    """One caller-approved high-q channel considered as a valence reference.

    ``quality_scores`` are caller-defined dimensionless values in ``[0, 1]``;
    larger always means better.  The reserved score
    ``uncontaminated_fraction`` is calculated from ``contamination_mask``.
    ``True`` mask entries identify samples that must not be trusted.
    """

    channel_id: str
    energy_loss_ev: ArrayLike
    corrected_intensity: ArrayLike
    q_au: float
    contamination_mask: ArrayLike
    core_background: ArrayLike = 0.0
    constant_background: ArrayLike = 0.0
    quality_scores: Mapping[str, float] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)
    intensity_convention: Literal["density_per_ev", "density_per_hartree"] = "density_per_ev"

    def __post_init__(self) -> None:
        channel_id = str(self.channel_id).strip()
        if not channel_id:
            raise ValueError("channel_id must not be empty")
        energy = _readonly_array(self.energy_loss_ev, "energy_loss_ev")
        if energy.size < 3:
            raise ValueError("energy_loss_ev must contain at least three samples")
        if np.any(np.diff(energy) <= 0.0):
            raise ValueError("energy_loss_ev must be strictly increasing")
        intensity = _readonly_array(
            self.corrected_intensity,
            "corrected_intensity",
            length=energy.size,
        )
        mask = _readonly_array(
            self.contamination_mask,
            "contamination_mask",
            dtype=np.bool_,
            length=energy.size,
        )
        q_au = float(self.q_au)
        if not np.isfinite(q_au) or q_au <= 0.0:
            raise ValueError("q_au must be finite and strictly positive")
        if self.intensity_convention not in ("density_per_ev", "density_per_hartree"):
            raise ValueError("intensity_convention must specify density_per_ev or density_per_hartree")
        object.__setattr__(self, "channel_id", channel_id)
        object.__setattr__(self, "energy_loss_ev", energy)
        object.__setattr__(self, "corrected_intensity", intensity)
        object.__setattr__(self, "q_au", q_au)
        object.__setattr__(self, "contamination_mask", mask)
        object.__setattr__(
            self,
            "core_background",
            _readonly_series(self.core_background, "core_background", energy.size),
        )
        object.__setattr__(
            self,
            "constant_background",
            _readonly_series(
                self.constant_background, "constant_background", energy.size
            ),
        )
        object.__setattr__(
            self,
            "quality_scores",
            _validated_score_mapping(self.quality_scores, "quality_scores"),
        )
        object.__setattr__(self, "provenance", _freeze(self.provenance))

    @property
    def uncontaminated_fraction(self) -> float:
        return float(np.mean(~self.contamination_mask))


@dataclass(frozen=True, slots=True)
class ReferenceSelectionResult:
    """Auditable result of weighted reference-channel selection."""

    selected_channel_id: str
    selected_index: int
    normalized_weights: Mapping[str, float]
    candidate_scores: Mapping[str, float]
    score_components: Mapping[str, Mapping[str, float]]
    tied_channel_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "normalized_weights", _freeze(self.normalized_weights))
        object.__setattr__(self, "candidate_scores", _freeze(self.candidate_scores))
        object.__setattr__(self, "score_components", _freeze(self.score_components))
        object.__setattr__(self, "tied_channel_ids", tuple(self.tied_channel_ids))


@dataclass(frozen=True, slots=True)
class ValenceProfileResult:
    """Read-only, normalized and symmetric empirical valence profile."""

    reference_channel_id: str
    pz_au: FloatArray
    profile: FloatArray
    source_energy_loss_ev: FloatArray
    source_pz_au: FloatArray
    source_residual: FloatArray
    contamination_mask: BoolArray
    support_intervals_pz_au: tuple[tuple[float, float], ...]
    selection: ReferenceSelectionResult
    diagnostics: Mapping[str, Any]
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        pz = _readonly_array(self.pz_au, "pz_au")
        profile = _readonly_array(self.profile, "profile", length=pz.size)
        if pz.size < 3 or np.any(np.diff(pz) <= 0.0):
            raise ValueError("pz_au must contain at least three increasing samples")
        source_energy = _readonly_array(
            self.source_energy_loss_ev, "source_energy_loss_ev"
        )
        source_pz = _readonly_array(
            self.source_pz_au, "source_pz_au", length=source_energy.size
        )
        source_residual = _readonly_array(
            self.source_residual, "source_residual", length=source_energy.size
        )
        source_mask = _readonly_array(
            self.contamination_mask,
            "contamination_mask",
            dtype=np.bool_,
            length=source_energy.size,
        )
        intervals = tuple(
            (float(start), float(stop))
            for start, stop in self.support_intervals_pz_au
        )
        if not intervals or any(start > stop for start, stop in intervals):
            raise ValueError("support intervals must contain start <= stop")
        object.__setattr__(self, "pz_au", pz)
        object.__setattr__(self, "profile", profile)
        object.__setattr__(self, "source_energy_loss_ev", source_energy)
        object.__setattr__(self, "source_pz_au", source_pz)
        object.__setattr__(self, "source_residual", source_residual)
        object.__setattr__(self, "contamination_mask", source_mask)
        object.__setattr__(self, "support_intervals_pz_au", intervals)
        object.__setattr__(self, "diagnostics", _freeze(self.diagnostics))
        object.__setattr__(self, "provenance", _freeze(self.provenance))


@dataclass(frozen=True, slots=True)
class ValenceMappingResult:
    """Read-only valence profile mapped onto one target energy-loss grid."""

    energy_loss_ev: FloatArray
    pz_au: FloatArray
    intensity_per_ev: FloatArray
    target_q_au: float
    diagnostics: Mapping[str, Any]
    provenance: Mapping[str, Any]

    def __post_init__(self) -> None:
        energy = _readonly_array(self.energy_loss_ev, "energy_loss_ev")
        pz = _readonly_array(self.pz_au, "pz_au", length=energy.size)
        intensity = _readonly_array(
            self.intensity_per_ev, "intensity_per_ev", length=energy.size
        )
        object.__setattr__(self, "energy_loss_ev", energy)
        object.__setattr__(self, "pz_au", pz)
        object.__setattr__(self, "intensity_per_ev", intensity)
        object.__setattr__(self, "diagnostics", _freeze(self.diagnostics))
        object.__setattr__(self, "provenance", _freeze(self.provenance))

    @property
    def valence_background(self) -> FloatArray:
        """Alias suitable for assembling an extraction background."""

        return self.intensity_per_ev


def _normalized_weights(score_weights: Mapping[str, float]) -> dict[str, float]:
    if not isinstance(score_weights, Mapping) or not score_weights:
        raise ValueError("score_weights must be a non-empty mapping")
    weights: dict[str, float] = {}
    for raw_key, raw_value in score_weights.items():
        key = str(raw_key).strip()
        value = float(raw_value)
        if not key or not np.isfinite(value) or value < 0.0:
            raise ValueError("score weights must have non-empty keys and finite values >= 0")
        weights[key] = value
    total = float(sum(weights.values()))
    if total <= 0.0:
        raise ValueError("at least one score weight must be greater than zero")
    return {key: value / total for key, value in weights.items()}


def select_reference_candidate(
    candidates: Sequence[ValenceReferenceCandidate],
    *,
    score_weights: Mapping[str, float],
) -> ReferenceSelectionResult:
    """Select one explicit candidate using caller-supplied weighted scores.

    No criterion depends on an internal q threshold.  Exact ties are resolved
    by the caller's candidate order and reported in ``tied_channel_ids``.
    """

    candidate_tuple = tuple(candidates)
    if not candidate_tuple:
        raise ValueError("at least one explicit reference candidate is required")
    if not all(isinstance(item, ValenceReferenceCandidate) for item in candidate_tuple):
        raise TypeError("all candidates must be ValenceReferenceCandidate objects")
    identifiers = [item.channel_id for item in candidate_tuple]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("candidate channel_id values must be unique")
    weights = _normalized_weights(score_weights)
    scores: dict[str, float] = {}
    components: dict[str, dict[str, float]] = {}
    for candidate in candidate_tuple:
        available = dict(candidate.quality_scores)
        available["uncontaminated_fraction"] = candidate.uncontaminated_fraction
        missing = set(weights).difference(available)
        if missing:
            formatted = ", ".join(sorted(missing))
            raise ValueError(
                f"candidate {candidate.channel_id!r} lacks weighted scores: {formatted}"
            )
        weighted = {key: weights[key] * available[key] for key in weights}
        components[candidate.channel_id] = weighted
        scores[candidate.channel_id] = float(sum(weighted.values()))
    maximum = max(scores.values())
    tied = tuple(
        identifier
        for identifier in identifiers
        if np.isclose(scores[identifier], maximum, rtol=0.0, atol=1.0e-15)
    )
    selected_id = tied[0]
    return ReferenceSelectionResult(
        selected_channel_id=selected_id,
        selected_index=identifiers.index(selected_id),
        normalized_weights=weights,
        candidate_scores=scores,
        score_components=components,
        tied_channel_ids=tied,
    )


def _is_uniform_grid(coordinate: FloatArray) -> tuple[bool, float]:
    differences = np.diff(coordinate)
    step = float(np.median(differences))
    tolerance = max(abs(step) * 1.0e-6, 1.0e-12)
    return bool(np.all(np.abs(differences - step) <= tolerance)), step


def _deduplicate_sorted(values: FloatArray) -> FloatArray:
    if values.size == 0:
        return values
    scale = max(1.0, float(np.max(np.abs(values))))
    tolerance = 1.0e-10 * scale
    kept = [float(values[0])]
    for value in values[1:]:
        if float(value) - kept[-1] > tolerance:
            kept.append(float(value))
    return np.asarray(kept, dtype=np.float64)


def _candidate_magnitudes(pz: FloatArray, usable: BoolArray) -> tuple[FloatArray, bool]:
    common_limit = min(-float(pz[0]), float(pz[-1]))
    if common_limit <= 0.0:
        raise ValueError("reference pz coverage must extend to both sides of pz=0")
    uniform, step = _is_uniform_grid(pz)
    if uniform:
        count = int(np.floor(common_limit / step + 1.0e-9))
        if count < 1:
            raise ValueError("common pz support must span at least one grid interval")
        return np.arange(count + 1, dtype=np.float64) * step, True
    magnitudes = np.abs(pz[usable & (np.abs(pz) <= common_limit)])
    magnitudes = np.sort(np.concatenate((magnitudes, np.array([0.0, common_limit]))))
    return _deduplicate_sorted(magnitudes), False


def _masked_interpolate(
    x: FloatArray,
    y: FloatArray,
    usable: BoolArray,
    queries: FloatArray,
) -> FloatArray:
    """Interpolate only between adjacent usable source points."""

    output = np.full(queries.shape, np.nan, dtype=np.float64)
    scale = max(1.0, float(np.max(np.abs(x))))
    tolerance = 1.0e-10 * scale
    for index, query in enumerate(queries):
        insertion = int(np.searchsorted(x, query))
        exact_indices: list[int] = []
        if insertion < x.size and abs(float(x[insertion] - query)) <= tolerance:
            exact_indices.append(insertion)
        if insertion > 0 and abs(float(x[insertion - 1] - query)) <= tolerance:
            exact_indices.append(insertion - 1)
        exact = next((item for item in exact_indices if usable[item]), None)
        if exact is not None:
            output[index] = y[exact]
            continue
        left, right = insertion - 1, insertion
        if left >= 0 and right < x.size and usable[left] and usable[right]:
            fraction = (query - x[left]) / (x[right] - x[left])
            output[index] = y[left] + fraction * (y[right] - y[left])
    return output


def _fill_contamination_linearly(
    pz: FloatArray, residual: FloatArray, mask: BoolArray
) -> FloatArray:
    if not np.any(mask):
        return np.array(residual, copy=True)
    clean = ~mask
    if np.count_nonzero(clean) < 2:
        raise ValueError("linear mask filling requires at least two clean samples")
    contaminated_pz = pz[mask]
    clean_pz = pz[clean]
    if contaminated_pz.min() < clean_pz.min() or contaminated_pz.max() > clean_pz.max():
        raise ValueError("linear mask filling would require extrapolation")
    filled = np.array(residual, copy=True)
    filled[mask] = np.interp(contaminated_pz, clean_pz, residual[clean])
    return filled


def _fit_asymmetry_coefficient(
    magnitudes: FloatArray,
    positive: FloatArray,
    negative: FloatArray,
    max_fractional_asymmetry: float,
) -> tuple[float, float, bool, int]:
    paired = np.isfinite(positive) & np.isfinite(negative) & (magnitudes > 0.0)
    paired_count = int(np.count_nonzero(paired))
    if paired_count < 2:
        raise ValueError("first-order asymmetry fitting requires two paired nonzero samples")
    m = magnitudes[paired]
    summed = positive[paired] + negative[paired]
    difference = positive[paired] - negative[paired]
    design = m * summed
    denominator = float(np.dot(design, design))
    if denominator <= np.finfo(float).eps:
        raise ValueError("first-order asymmetry coefficient is not identifiable")
    unconstrained = float(np.dot(design, difference) / denominator)
    bound = max_fractional_asymmetry / float(np.max(magnitudes))
    fitted = float(np.clip(unconstrained, -bound, bound))
    hit_bound = bool(not np.isclose(fitted, unconstrained, rtol=1.0e-12, atol=1.0e-15))
    return fitted, bound, hit_bound, paired_count


def _support_intervals(
    magnitudes: FloatArray, valid: BoolArray
) -> tuple[tuple[float, float], ...]:
    groups: list[tuple[float, float]] = []
    starts = np.flatnonzero(valid & np.concatenate(([True], ~valid[:-1])))
    stops = np.flatnonzero(valid & np.concatenate((~valid[1:], [True])))
    for start_index, stop_index in zip(starts, stops, strict=True):
        groups.append((float(magnitudes[start_index]), float(magnitudes[stop_index])))
    intervals: list[tuple[float, float]] = []
    for start, stop in reversed(groups):
        if start == 0.0:
            intervals.append((-stop, stop))
        else:
            intervals.append((-stop, -start))
    intervals.extend((start, stop) for start, stop in groups if start != 0.0)
    return tuple(sorted(intervals))


def _integral_over_intervals(
    pz: FloatArray,
    profile: FloatArray,
    intervals: tuple[tuple[float, float], ...],
) -> float:
    integral = 0.0
    tolerance = 1.0e-12 * max(1.0, float(np.max(np.abs(pz))))
    for start, stop in intervals:
        selected = (pz >= start - tolerance) & (pz <= stop + tolerance)
        if np.count_nonzero(selected) >= 2:
            integral += float(np.trapezoid(profile[selected], pz[selected]))
    return integral


def build_valence_profile(
    candidates: Sequence[ValenceReferenceCandidate],
    *,
    score_weights: Mapping[str, float],
    valence_electron_count: float,
    normalization_convention: NormalizationConvention,
    masked_region_policy: MaskPolicy = "exclude",
    gaussian_sigma_pz_au: float | None = None,
    asymmetry_mode: AsymmetryMode = "off",
    max_fractional_asymmetry: float | None = None,
) -> ValenceProfileResult:
    """Build a normalized empirical ``J_v(p_z)`` from a selected high-q channel.

    Negative samples are retained.  Normalization is possible only when the
    signed integral under the chosen convention is positive.
    """

    electron_count = float(valence_electron_count)
    if not np.isfinite(electron_count) or electron_count <= 0.0:
        raise ValueError("valence_electron_count must be finite and positive")
    if normalization_convention not in (
        "full_symmetric",
        "doubled_positive_half",
    ):
        raise ValueError("unsupported normalization_convention")
    if masked_region_policy not in ("exclude", "linear"):
        raise ValueError("masked_region_policy must be 'exclude' or 'linear'")
    if asymmetry_mode not in ("off", "fit_first_order"):
        raise ValueError("asymmetry_mode must be 'off' or 'fit_first_order'")
    if asymmetry_mode == "fit_first_order":
        if max_fractional_asymmetry is None:
            raise ValueError(
                "max_fractional_asymmetry is required for fit_first_order"
            )
        maximum_asymmetry = float(max_fractional_asymmetry)
        if not np.isfinite(maximum_asymmetry) or not 0.0 < maximum_asymmetry < 1.0:
            raise ValueError("max_fractional_asymmetry must lie strictly within (0, 1)")
    else:
        if max_fractional_asymmetry is not None:
            raise ValueError("max_fractional_asymmetry must be omitted when asymmetry is off")
        maximum_asymmetry = 0.0
    if gaussian_sigma_pz_au is not None:
        smoothing_sigma = float(gaussian_sigma_pz_au)
        if not np.isfinite(smoothing_sigma) or smoothing_sigma <= 0.0:
            raise ValueError("gaussian_sigma_pz_au must be finite and positive")
    else:
        smoothing_sigma = 0.0

    candidate_tuple = tuple(candidates)
    selection = select_reference_candidate(candidate_tuple, score_weights=score_weights)
    reference = candidate_tuple[selection.selected_index]
    source_pz = np.asarray(
        energy_loss_to_pz(reference.energy_loss_ev, reference.q_au), dtype=np.float64
    )
    source_residual = np.asarray(
        reference.corrected_intensity
        - reference.core_background
        - reference.constant_background,
        dtype=np.float64,
    )
    jacobian = reference.q_au * (
        HARTREE_ENERGY_EV if reference.intensity_convention == "density_per_ev" else 1.0
    )
    source_profile = source_residual * jacobian
    mask = np.asarray(reference.contamination_mask, dtype=np.bool_)
    if masked_region_policy == "linear":
        prepared = _fill_contamination_linearly(source_pz, source_profile, mask)
        usable = np.ones(mask.shape, dtype=np.bool_)
        interpolated_count = int(np.count_nonzero(mask))
    else:
        prepared = np.array(source_profile, copy=True)
        usable = ~mask
        interpolated_count = 0
    if np.count_nonzero(usable) < 3:
        raise ValueError("fewer than three usable reference samples remain")

    magnitudes, source_grid_uniform = _candidate_magnitudes(source_pz, usable)
    positive = _masked_interpolate(source_pz, prepared, usable, magnitudes)
    negative = _masked_interpolate(source_pz, prepared, usable, -magnitudes)
    available = np.isfinite(positive) | np.isfinite(negative)
    if np.count_nonzero(available) < 2:
        raise ValueError("insufficient paired pz support after applying contamination mask")

    if asymmetry_mode == "fit_first_order":
        coefficient, coefficient_bound, hit_bound, paired_count = (
            _fit_asymmetry_coefficient(
                magnitudes,
                positive,
                negative,
                maximum_asymmetry,
            )
        )
    else:
        coefficient, coefficient_bound, hit_bound = 0.0, 0.0, False
        paired_count = int(
            np.count_nonzero(
                np.isfinite(positive) & np.isfinite(negative) & (magnitudes > 0.0)
            )
        )
    corrected_positive = positive / (1.0 + coefficient * magnitudes)
    corrected_negative = negative / (1.0 - coefficient * magnitudes)
    count = np.isfinite(corrected_positive).astype(int) + np.isfinite(
        corrected_negative
    ).astype(int)
    summed = np.nan_to_num(corrected_positive, nan=0.0) + np.nan_to_num(
        corrected_negative, nan=0.0
    )
    even_profile = np.divide(
        summed,
        count,
        out=np.full(summed.shape, np.nan, dtype=np.float64),
        where=count > 0,
    )
    valid = np.isfinite(even_profile)
    valid_magnitudes = magnitudes[valid]
    valid_profile = even_profile[valid]
    positive_nonzero = valid_magnitudes > 0.0
    pz = np.concatenate(
        (-valid_magnitudes[positive_nonzero][::-1], valid_magnitudes)
    )
    profile = np.concatenate(
        (valid_profile[positive_nonzero][::-1], valid_profile)
    )
    intervals = _support_intervals(magnitudes, valid)
    if len(intervals) != 1 or not intervals[0][0] < 0.0 < intervals[0][1]:
        raise ValueError(
            "profile has unresolved contamination gaps; electron-count normalization "
            "requires continuous support. Choose masked_region_policy='linear' explicitly"
        )

    if smoothing_sigma > 0.0:
        uniform, step = _is_uniform_grid(pz)
        continuous = len(intervals) == 1 and intervals[0][0] < 0.0 < intervals[0][1]
        if not uniform or not continuous:
            raise ValueError(
                "Gaussian smoothing requires a continuous, uniformly spaced pz grid"
            )
        profile = gaussian_filter1d(
            profile,
            sigma=smoothing_sigma / step,
            mode="reflect",
        )
        profile = 0.5 * (profile + profile[::-1])

    if normalization_convention == "full_symmetric":
        unnormalized_area = _integral_over_intervals(pz, profile, intervals)
    else:
        positive_intervals = tuple(
            (max(0.0, start), stop)
            for start, stop in intervals
            if stop > 0.0
        )
        positive_area = _integral_over_intervals(pz, profile, positive_intervals)
        unnormalized_area = 2.0 * positive_area
    if not np.isfinite(unnormalized_area) or unnormalized_area <= 0.0:
        raise ValueError("profile has a non-positive signed normalization integral")
    normalization_scale = electron_count / unnormalized_area
    profile = profile * normalization_scale

    diagnostics = {
        "selected_score": selection.candidate_scores[reference.channel_id],
        "source_sample_count": int(source_pz.size),
        "contaminated_sample_count": int(np.count_nonzero(mask)),
        "interpolated_sample_count": interpolated_count,
        "source_uncontaminated_fraction": reference.uncontaminated_fraction,
        "source_grid_uniform": source_grid_uniform,
        "profile_grid_uniform": _is_uniform_grid(pz)[0],
        "paired_sample_count": paired_count,
        "asymmetry_coefficient_per_au": coefficient,
        "asymmetry_coefficient_bound_per_au": coefficient_bound,
        "asymmetry_hit_bound": hit_bound,
        "unnormalized_signed_integral": unnormalized_area,
        "normalization_scale": normalization_scale,
        "negative_source_sample_count": int(np.count_nonzero(source_residual < 0.0)),
        "negative_profile_sample_count": int(np.count_nonzero(profile < 0.0)),
    }
    provenance = {
        "method": "empirical_high_q_valence_profile",
        "reference_channel_id": reference.channel_id,
        "reference_q_au": reference.q_au,
        "reference_provenance": reference.provenance,
        "selection": {
            "normalized_weights": selection.normalized_weights,
            "candidate_scores": selection.candidate_scores,
            "score_components": selection.score_components,
            "tie_resolution": "first candidate in caller order",
        },
        "subtractions": ("core_background", "constant_background"),
        "pz_transform": "p_z = (energy_loss_eV / Hartree_eV) / q_au - q_au / 2",
        "input_intensity_convention": reference.intensity_convention,
        "input_to_profile_jacobian": jacobian,
        "finite_support_normalization": True,
        "contamination_mask_policy": masked_region_policy,
        "symmetrization": "mean of available +/- p_z samples after asymmetry correction",
        "gaussian_sigma_pz_au": None if smoothing_sigma == 0.0 else smoothing_sigma,
        "asymmetry_mode": asymmetry_mode,
        "max_fractional_asymmetry": max_fractional_asymmetry,
        "normalization_convention": normalization_convention,
        "valence_electron_count": electron_count,
        "negative_values_clipped": False,
    }
    return ValenceProfileResult(
        reference_channel_id=reference.channel_id,
        pz_au=pz,
        profile=profile,
        source_energy_loss_ev=reference.energy_loss_ev,
        source_pz_au=source_pz,
        source_residual=source_residual,
        contamination_mask=mask,
        support_intervals_pz_au=intervals,
        selection=selection,
        diagnostics=diagnostics,
        provenance=provenance,
    )


def _inside_support(
    values: FloatArray, intervals: tuple[tuple[float, float], ...]
) -> BoolArray:
    tolerance = 1.0e-10 * max(
        1.0,
        max(abs(bound) for interval in intervals for bound in interval),
    )
    inside = np.zeros(values.shape, dtype=np.bool_)
    for start, stop in intervals:
        inside |= (values >= start - tolerance) & (values <= stop + tolerance)
    return inside


def map_valence_profile(
    result: ValenceProfileResult,
    target_energy_loss_ev: ArrayLike,
    *,
    target_q_au: float,
) -> ValenceMappingResult:
    """Map ``J_v(p_z)`` to a target energy grid without extrapolation.

    The returned intensity is a density per eV.  No amplitude fitting or
    negative-value clipping is performed here.
    """

    if not isinstance(result, ValenceProfileResult):
        raise TypeError("result must be a ValenceProfileResult")
    energy = _readonly_array(target_energy_loss_ev, "target_energy_loss_ev")
    if energy.size < 2 or np.any(np.diff(energy) <= 0.0):
        raise ValueError("target_energy_loss_ev must contain increasing samples")
    q_au = float(target_q_au)
    if not np.isfinite(q_au) or q_au <= 0.0:
        raise ValueError("target_q_au must be finite and strictly positive")
    target_pz = np.asarray(energy_loss_to_pz(energy, q_au), dtype=np.float64)
    supported = _inside_support(target_pz, result.support_intervals_pz_au)
    if not np.all(supported):
        rejected = int(np.count_nonzero(~supported))
        raise ValueError(
            f"target grid contains {rejected} pz samples outside continuous "
            "profile support; extrapolation is forbidden"
        )
    interpolated_profile = np.interp(target_pz, result.pz_au, result.profile)
    intensity_per_ev = interpolated_profile / (q_au * HARTREE_ENERGY_EV)
    diagnostics = {
        "target_sample_count": int(energy.size),
        "target_pz_min_au": float(np.min(target_pz)),
        "target_pz_max_au": float(np.max(target_pz)),
        "negative_mapped_sample_count": int(np.count_nonzero(intensity_per_ev < 0.0)),
    }
    provenance = {
        "method": "impulse_approximation_profile_transfer",
        "reference_channel_id": result.reference_channel_id,
        "target_q_au": q_au,
        "mapping_convention": "density_per_eV = J_v(p_z) / (q_au * Hartree_eV)",
        "extrapolation": "forbidden",
        "negative_values_clipped": False,
        "profile_provenance": result.provenance,
    }
    return ValenceMappingResult(
        energy_loss_ev=energy,
        pz_au=target_pz,
        intensity_per_ev=intensity_per_ev,
        target_q_au=q_au,
        diagnostics=diagnostics,
        provenance=provenance,
    )


__all__ = [
    "AsymmetryMode",
    "MaskPolicy",
    "NormalizationConvention",
    "ReferenceSelectionResult",
    "ValenceMappingResult",
    "ValenceProfileResult",
    "ValenceReferenceCandidate",
    "build_valence_profile",
    "map_valence_profile",
    "select_reference_candidate",
]
