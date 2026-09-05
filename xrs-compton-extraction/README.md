# xrs-compton-extraction

`xrs_compton_extraction` is a standalone Python package for extracting core-level
X-ray Raman scattering (XRS/NRIXS) spectra from momentum-transfer-dependent
backgrounds.

The package is developed inside the `xrsabre` workspace, but it is architecturally
independent from `xrs_processing`: it does not import that package, does not reuse
its internal data structures, and does not promise compatibility with its APIs.

## Current implementation milestone

The first executable MVP slice now includes:

- installable `src`-layout package;
- validated domain and result models;
- deterministic NeXus/HDF5 discovery and loading;
- energy, momentum-transfer, and longitudinal-momentum conversions;
- deterministic synthetic spectra;
- acquisition-time and I0 normalization with Poisson propagation;
- bounded, weighted, robust Pearson background fitting;
- a single-channel extraction pipeline that never clips negative results;
- YAML/JSON configuration and CSV/JSON result export;
- a minimal Jupyter/ipywidgets workbench.

The second implementation slice adds:

- strict CSV/TSV mappings and repeated-scan merging with explicit interpolation;
- elastic, stray, absorption and kinematic correction primitives;
- explicit time, monitor, detector-efficiency and attenuation error propagation;
- polynomial fitting, configurable model-selection criteria and QC metrics;
- optional xraylib Hartree–Fock core profiles with explicit core-shell selection;
- empirical high-q valence profiles, contamination masks, symmetry, smoothing,
  finite-support electron-count normalization and q transfer;
- Compton-template extraction with a fitted common scale and constant;
- per-channel batch execution, multi-q comparison/averaging and Markdown reports;
- five workbench pages, with interactive Pearson extraction, batch progress,
  configuration and export controls. Advanced Compton inputs use the Python API.

## Quick start

For step-by-step analysis without the widget workbench, open
[`notebooks/Compton_Analysis.ipynb`](notebooks/Compton_Analysis.ipynb).
From this directory, use `pixi run jupyter lab notebooks/Compton_Analysis.ipynb`.
For another environment, install `python -m pip install -e ".[notebook]"`.
The notebook reads `resources/compton_profiles/ComptonProfiles.dat` directly;
it does not use xraylib. `DabaxProfileSource(path)` reads native `Shell_N` columns,
occupations from `#UOCCUP` and binding energies from `#UBIND`, and records SHA256.
The external table is intentionally not committed. Place the reviewed local file
at that path before running the HF sections; the expected SHA256 is documented in
`docs/compton-profile-data.md`.
Run cells in order. The top cell contains the explicit physical parameters.
The notebook joins crystal metadata, audits Ho/B atomic profiles, runs unweighted
Pearson diagnostics and high-q HF matching, and verifies target preservation on synthetic data.
Ho N4 is explicitly `Ho:Shell_13` (164 eV in this table); the original experimental
axis is retained. HF scale and a linear baseline are fitted automatically after
same-window area pre-matching. The configurable exploratory valence cutoff is
binding energy < 20 eV. High-q extraction preserves N4 and subtracts other core
shells, valence and the baseline. See [HF matching workflow](docs/hf-matching.md).
Outputs under `output/ho-b4-hf-matching/real` and `synthetic` remain separate.
Previous output and the pre-migration executed notebook are preserved.
Section 6.1 displays each module's extracted target-region mean and the overall
equal-module mean, with an equal-crystal comparison. HF densities are first
converted back to input intensity units using their saved scale; no peak/area
normalization is added. PNGs, CSVs and contributing-channel counts are saved in
`output/ho-b4-hf-matching/real/module-averages`.
The checked-in notebook has no execution outputs; it does not import UI modules.

The `exploratory` helpers use explicit missing-support masks and null statistical
uncertainties/covariance. They do not change the weighted extraction interfaces.
Notebook pz uses `q/2 - omega/q`; existing package APIs retain the opposite sign.
Paired arrays are reflected and reordered explicitly at the notebook boundary.

For the supplied Ho scan, see [real-data input validation](docs/ho-data-validation.md).
Wide processed tables are supported without inventing Poisson uncertainties.

```python
from xrs_compton_extraction import XRSWorkbench

app = XRSWorkbench()
app.display()
```

The programmatic low-q path requires explicit background-only fit windows and a
known momentum transfer:

```python
from xrs_compton_extraction.io import load_nexus
from xrs_compton_extraction.pipeline import extract_pearson

dataset = load_nexus("scan.nxs")
result = extract_pearson(
    dataset.spectra[0],
    q_au=2.0,                       # use measured/calculated q; example only
    fit_windows_ev=((0, 35), (65, 100)),  # choose for the actual target edge
)
```

The numbers above illustrate API shape only and are not default experimental
parameters. The pipeline raises an error rather than guessing missing q, energy
semantics, acquisition time, or I0.

## Development

From this directory in the shared Pixi workspace:

```text
pixi run python -m pytest
pixi run ruff check src tests
```

For an editable installation with the workbench dependencies:

```text
python -m pip install -e ".[workbench,test]"
```

## Hartree–Fock / Compton workflow

The current HoB4 notebook uses `DabaxProfileSource` and the user-supplied local
table. Native shell labels must be selected explicitly: e.g. `Shell_1`, not `K`.
It runs both real-data diagnostics and the synthetic example without xraylib.
The API example below documents the older optional xraylib backend, retained for
compatibility only; it is not used by the HoB4 notebook.

Install the optional backend and run the complete synthetic example:

```text
pixi run python -m pip install -e ".[profiles]" --no-build-isolation
pixi run python examples/compton_profile_demo.py --output output/profile-demo
```

The example uses actual xraylib oxygen K-shell profiles and a synthetic valence
profile/target edge. Its output includes two channel CSV files, complete source
metadata, three PNG plots, a manifest and `report.md`.

The profile builder deliberately requires explicit core shells for each element:

```python
from xrs_compton_extraction.backgrounds import XraylibProfileSource, build_core_profile
from xrs_compton_extraction.geometry import energy_loss_to_pz

# xraylib 4.2.1 does not expose ElectronConfig_Biggs through its Python API.
# Explicitly opt in to ElectronConfig and verify the selected occupations.
source = XraylibProfileSource(occupancy_source="electron_config")
core = build_core_profile(
    energy_loss_to_pz(energy_loss_ev, q_au),
    {"Si": 1, "O": 2},
    source,
    shells_by_element={"Si": ["K", "L1", "L2", "L3"], "O": ["K"]},
    exclude_target=("O", "K"),
)
```

`energy_loss_ev` and `q_au` above must come from the experiment. Shell selections
are an example for an O K-edge analysis and require scientific review for the
sample. Use `build_valence_profile` to construct the empirical reference, then
`extract_compton_profile` or `app.run_compton_profile` to extract a channel.
See `examples/compton_profile_demo.py` for an executable call sequence.

Mapped profiles are densities per eV: `J(pz) / (q_au * Hartree_eV)`.
Core and valence templates share a fitted experimental amplitude. Atomic tables
are not included in this package. See [data policy](docs/compton-profile-data.md).

## Scientific status

This is a tested development version. Model-selection criteria recommend
candidates; they do not automatically run a scientifically validated ensemble.
Multi-q averaging assumes independent channel uncertainties and retains q-specific
spectra for comparison. Shared reference-profile errors and real q-dependent
spectral changes require separate treatment. Empirical electron-count
normalization applies to the available symmetric support; omitted tails are not
estimated. Full experimental regression and advanced GUI controls remain pending.

## License

The project license is **待确认**. Do not redistribute the package or bundled
scientific data until the project owner has selected a license and confirmed the
licenses of all profile data resources.
