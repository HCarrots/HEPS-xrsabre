# Ho processed scan: input validation

## Current HF matching implementation

The former missing-scale pause is superseded by automatic HF area pre-matching
and positive scale/linear-baseline fitting. See [HF matching](hf-matching.md).
Target is Ho N4 = Shell_13, 164 eV in the local table (161 eV is N5).
With an explicit exploratory binding < 20 eV valence partition, the notebook now
runs 14 high-q target-preserving extractions and 36 Pearson diagnostics; two zero
channels remain excluded. All original axes/data are retained. Reference VB-A2
has significant negative valence tails, so execution success is not scientific
validation. New output is in `output/ho-b4-hf-matching`; the notes below describe
earlier implementation stages where indicated.

## Step-by-step notebook (2026-09-05)

**Current source update:** the notebook now reads the local
`resources/compton_profiles/ComptonProfiles.dat` via `DabaxProfileSource`.
It no longer calls xraylib, including in the synthetic section. Table occupations
sum to B=5 and Ho=67; the earlier B=4.333333 result below is historical and does
not apply to the local DABAX source. Target shell mapping uses explicit `Shell_N`
identifiers; review and real intensity parameters remain required.

`notebooks/HoB4_Compton_Analysis.ipynb` now executes the input audit, original-axis
diagnostics, atomic occupation checks and unweighted channel fits without the UI.
The sample stoichiometry is Ho:B = 1:4. With the current explicit missing HF inputs,
the executed notebook records 36 exploratory Pearson channels, 14 paused high-q
channels and two retained/excluded zero channels. These are execution statuses,
not scientific acceptance grades. Results are in `output/ho-b4-notebook`.

The historical xraylib 4.2.1 `ElectronConfig` diagnostic gave Ho occupation sum 67 and B
available-shell sum 4.333333. The occupation-weighted partial-profile relative L2
differences against the total are approximately 0.0030 and 0.1405 respectively.
Integrals use the finite interval |pz| <= 100 a.u.; no tails are inferred.
These discrepancies are recorded, not patched with guessed occupations. The
external DABAX table and installed backend byte identity remain unverified.
Shell selection, electron count and reference intensity scale still require review.

The synthetic O example uses local DABAX Shell_1 (not HoB4) and exercises HF subtraction, a contaminated and
asymmetric reference, symmetry, Gaussian smoothing and mapping to q=5,6,8 a.u.
It saves target truth separately from recovered residuals. Real-data errors and
covariance remain absent; RMSE is reported in arbitrary intensity units.

The user-supplied local file is
`workspace/Ho/processed/Ho_Comptonscan_standard/Ho_Comptonscan_standard_all_data.txt`
relative to the xrsabre workspace. Input files are never modified or bundled.

## Observed input

- A tab-separated header with energy transfer in eV and 52 crystal channels.
- 4021 points, from -1.6 to 802.4 eV, in 0.2 eV steps.
- No nonfinite or negative intensities in this file.
- HB-E1 and HB-E2 are entirely zero; retained and flagged in the audit.
- The adjacent run-info JSON records I0 normalization, filtering and interpolation.
- No propagated uncertainties or sample covariance are present in the intensity table.
- The adjacent fit-results table contains elastic centers and q summaries.
  The producer's documented q convention is inverse angstrom. The mean q range
  is 0.987744–9.859149 inverse angstrom (0.522692–5.217237 a.u.).
  These are channel summaries, not constant-q or energy-resolved calibrations.

## Reproduce the input audit

From the package directory:

```powershell
pixi run python examples/inspect_processed_scan.py `
  ../workspace/Ho/processed/Ho_Comptonscan_standard/Ho_Comptonscan_standard_all_data.txt `
  --fit-results ../workspace/Ho/processed/Ho_Comptonscan_standard/Ho_Comptonscan_standard_fit_results.txt `
  --run-info ../workspace/Ho/processed/Ho_Comptonscan_standard/Ho_Comptonscan_standard_run_info.json `
  --output output/ho-input-check

$env:XRS_HO_TEST_DATA = '../workspace/Ho/processed/Ho_Comptonscan_standard/Ho_Comptonscan_standard_all_data.txt'
pixi run python -m pytest tests/test_real_ho_scan.py
```

The audit writes `inspection.json` (per-channel statistics and SHA-256 hashes),
`inspection.md`, and `overview.png`. It performs no background fit. The opt-in
regression is skipped when `XRS_HO_TEST_DATA` is absent, so public CI does not
depend on private beamline files. The package does not import `xrs_processing`.

## Wide-table API

Use `load_text_channels(path, mappings)` or
`XRSWorkbench().load_text(path, mappings=mappings)`. Each mapping is a
`TextMapping` with explicit energy semantics, a signal column, and a unique
analyzer/ROI label. Set `intensity_kind="processed"` for this dataset.
The table is parsed once and every channel is retained unchanged.

Processed intensities without explicit uncertainties are rejected by weighted
extraction: taking their square root would not reproduce the original counting
uncertainty. Do not invent unit monitor/time arrays or normalize I0 again.
When propagated uncertainties become available, map their columns and explicitly
disable any normalization already applied upstream.

## Still required for scientific extraction

The q split is explicit: q < 9 Å⁻¹ is `low_q`, q > 9 Å⁻¹ is `mid_high_q`, and
exactly 9 Å⁻¹ is left as `boundary`. The window diagnostic can be reproduced with:

```powershell
pixi run python examples/debug_ho_n4_windows.py `
  ../workspace/Ho/processed/Ho_Comptonscan_standard/Ho_Comptonscan_standard_all_data.txt `
  --fit-results ../workspace/Ho/processed/Ho_Comptonscan_standard/Ho_Comptonscan_standard_fit_results.txt `
  --output output/ho-n4-window-scan
```

The first diagnostic scan excludes the elastic region below 20 eV, uses a
provisional N4 reference of 161 eV, and excludes 111–211 eV while scoring
candidate windows. It does not assert that 161 eV is the calibrated position of
this energy-transfer axis.

- Confirm the target edge, sample composition and intended fit/exclusion windows.
- Supply propagated uncertainties, or explicitly choose an exploratory fit whose
  outputs do not claim statistical confidence intervals.
- Confirm q calibration/convention across the energy scan and any correction
  factors not already applied.

This is real-data **input** regression, not validation of recovered edge spectra.
