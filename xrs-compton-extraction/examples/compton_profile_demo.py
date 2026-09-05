"""Synthetic two-q extraction using real xraylib oxygen K-shell profiles.

Run from the project directory:
    pixi run python examples/compton_profile_demo.py --output output/profile-demo

All geometry, intensities and fit windows below are synthetic test parameters.
"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")

from xrs_compton_extraction import (
    XRSSpectrum,
    average_multi_q,
    extract_compton_profile,
    save_results,
)
from xrs_compton_extraction.backgrounds import (
    ValenceReferenceCandidate,
    XraylibProfileSource,
    build_core_profile,
    build_valence_profile,
)
from xrs_compton_extraction.constants import HARTREE_ENERGY_EV
from xrs_compton_extraction.geometry import (
    energy_loss_to_pz,
    pz_to_energy_loss,
)
from xrs_compton_extraction.plotting import (
    plot_extraction,
    plot_multi_q_evolution,
)


def run_demo(output: Path) -> Path:
    source = XraylibProfileSource(occupancy_source="electron_config")
    ref_pz = np.linspace(-3.5, 3.5, 1401)
    ref_q = 8.0
    ref_energy = pz_to_energy_loss(ref_pz, ref_q)
    jv = np.exp(-ref_pz**2)
    jv *= 6 / np.trapezoid(jv, ref_pz)
    candidate = ValenceReferenceCandidate(
        "synthetic-reference", ref_energy, jv / (ref_q * HARTREE_ENERGY_EV), ref_q,
        np.zeros(ref_pz.size, dtype=bool),
        provenance={"synthetic": True, "valence_model": "Gaussian with 6 electrons"},
    )
    valence = build_valence_profile(
        [candidate], score_weights={"uncontaminated_fraction": 1},
        valence_electron_count=6, normalization_convention="full_symmetric",
    )
    energy = np.linspace(250, 800, 551)
    injected_edge = np.where((energy >= 480) & (energy <= 540), 5.0, 0.0)
    results, figures = {}, {}
    for q in (6.0, 7.0):
        pz = np.asarray(energy_loss_to_pz(energy, q))
        core = build_core_profile(pz, {"O": 1}, source, shells_by_element={"O": ["K"]})
        mapped_jv = np.interp(pz, valence.pz_au, valence.profile)
        expected = 1500 * (core.total_profile + mapped_jv) / (q * HARTREE_ENERGY_EV) + 2 + injected_edge
        spectrum = XRSSpectrum(
            energy, expected, energy_loss_eV=energy, q_au=q, monitor=1, acquisition_time_s=1,
            scan_id="synthetic", analyzer_id=f"q{q:g}", metadata={"synthetic": True},
        )
        result = extract_compton_profile(
            spectrum, q_au=q, core_profile=core, valence_profile=valence,
            fit_windows_ev=((250, 450), (570, 800)),
        )
        np.testing.assert_allclose(result.extracted_edge, injected_edge, atol=1e-8)
        results[f"q{q:g}"] = result
        figures[f"q{q:g}"] = plot_extraction(result).figure
    combined = average_multi_q(tuple(results.values()), channel_labels=tuple(results))
    figures["multi-q"], _ = plot_multi_q_evolution(combined)
    return save_results(results, output, figures=figures)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("output") / "profile-demo")
    args = parser.parse_args()
    print(run_demo(args.output))
