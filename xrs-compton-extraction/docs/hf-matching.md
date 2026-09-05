# HF matching and target-preserving extraction

## Module target-region display

The active notebook is now `notebooks/Compton_Analysis.ipynb` (user rename).
Section 6.1 averages each module after extraction in `TARGET_VIEW`, initially
111–211 eV. `module_averages.average_target_exports` reads current successful
exports and divides HF-channel residuals by saved raw_to_hf_scale before averaging,
bringing them back to the input arbitrary intensity units of Pearson channels.
It does not redo subtraction or normalize peaks. This reverses the scale change,
not detector efficiencies or q-dependent physical response.

Module means weight valid crystals equally at each energy; the main overall curve
weights valid module means equally. A separate overall curve weights every crystal
equally. Missing samples are omitted with per-point counts, never replaced by zero.
No covariance/error bars are inferred. Current counts: VB 14, VU 13, HB 9, HL 7,
VD 7; HB-E1 and HB-E2 remain excluded as all-zero inputs (50 contributors total).
Outputs are in `real/module-averages`: per-module CSVs, overall CSV, two PNGs and
JSON with scales, included/excluded crystals and exact weighting definitions.
The current rerun reveals large negative Pearson residuals in HL (notably HL-E1
and HL-E2 near 139–140 eV), dominating the combined mean. These are retained;
a residual/input peak ratio > 10 triggers a visible diagnostic warning without
automatic exclusion. An additional symmetric-log figure exposes smaller module
signals while preserving signs. The linear figure and exported values are unchanged.

The HoB4 notebook now uses `hf_workflow.py`, independently implemented after
reading the local XRStools workflow. There is no import or runtime dependency on
XRStools, xraylib, or xrs_processing. The original weighted extraction API is unchanged.

## Model and normalization

`build_hf_energy(energy_ev, q_au, source, composition, target=...,
valence_cutoff_ev=...)` returns total, core, valence, target and other-core
densities, along with per-shell arrays and provenance. It reads the supplied
DABAX file and applies each shell's binding-energy threshold before enforcing
the constant-q first moment:

`integral E*S(E) dE = N * Hartree_eV * q_au**2 / 2`.

Normalization is computed on a separate 16001-point momentum grid from the
binding threshold to the table's positive support endpoint. It is independent
of the observed scan endpoint. Occupation and stoichiometry enter exactly once
through N. This is a finite-table f-sum convention, not a claim to know missing
atomic tails. A threshold beyond table support is explicitly recorded; the
reader rejects queries requiring extrapolation. Raw DABAX arrays are not changed.

`match_hf_scale` first compares HF and experimental areas on exactly the same
observed support, excluding the protected interval and without bridging masked
gaps. It then fits `I_pre = a*HF + b0 + b1*(E-Ec)` in background windows with
positive a. The matched density is `(I_pre-baseline)/a`. It records the area
factor, fitted HF factor, net raw-to-HF factor, linear terms, masks and RMSE.
No artificial sigma, covariance or statistical chi-square is generated.

The reference valence is matched density minus **all** HF core contributions.
Its protected near-edge region is interpolated between observed endpoints,
then symmetrized and smoothed in pz. Finite-support electron normalization and
all negative samples are retained and reported.

`extract_hf_target` fits total core plus mapped empirical valence, but subtracts
only **other** core plus valence and the fitted linear baseline. `residual`
therefore contains the target HF continuum and its fine structure.
`model_fit_residual = residual - target_hf` is the background-window diagnostic.
Exports retain raw intensity, scaled intensity, each subtracted component,
target HF, availability and fit masks. High-q output units are HF-model density
per eV; low-q Pearson output remains arbitrary units and must not be mixed into
the same quantitative average.

## Explicit HoB4 settings

- Composition Ho:B = 1:4; target `('Ho', 'Shell_13')`, N4, four electrons, 164 eV.
  N5 is Shell_14 at 161 eV. The former report of N4=161 eV was incorrect for this table.
- Exploratory valence partition: binding < 20 eV. Binding = 20 eV is core.
  This gives 25 valence electrons per formula unit under this model, not a
  validated solid-state electron partition. Ho 4f response makes this assumption consequential.
- Original energy axis unchanged; protect 111–211 eV. Fit 20–80 and 230–700 eV;
  pre-match 20–700 eV, also excluding protection. The atomic 164 eV value does
  not establish experimental calibration.
- q_ave converted from inverse angstrom to atomic units and held fixed per channel.
  High-q route uses q > 9 inverse angstrom; lower-q Pearson remains diagnostic.
- Gaussian width 0.05 a.u.; no asymmetry fit. Empirical tails outside support
  remain unavailable, never replaced by zero.

## Local execution and limitations

The notebook completed with xraylib, XRStools and xrs_processing blocked in the
kernel. It processed 36 Pearson and 14 HF channels, retaining/excluding two zero
channels. The selected reference is VB-A2. Its fitted raw-to-HF scale is about
4.51731e-6, and finite-support valence normalization adds a factor of about 1.48972.
Its smoothed profile contains 1288 negative samples: retained as a clear model/
window warning, not silently clipped or presented as scientific acceptance.

Outputs: `output/ho-b4-hf-matching/{real,synthetic}`; executed copy:
`output/ho-b4-hf-matching/executed.ipynb`; pre-migration backup:
`output/ho-b4-hf-matching/before-migration.ipynb`.

Synthetic validation recovers known scale/baseline and the target response,
including the HF continuum. The multi-q notebook demonstration uses the mapped
reference to generate its synthetic input and verifies bookkeeping; it does
not independently validate the physical model. Tests additionally check moment
units, thresholding, scan independence, negative fine structure, masks, missing
support and unidentifiable fits. Current full suite with real Ho input: 227 passed.

No final experimental N4 edge is claimed. Calibration, the valence partition,
hard threshold approximation, fit-window sensitivity, and shared reference
uncertainty still require scientific assessment.

## XRStools reading notes

Local reference locations (not imported): `XRStools/bin/run_diamond_example.py`
for the calling sequence; `XRStools/extraction.py` lines 54, 1524 and 1637 for
area pre-matching, linear/scale fitting and valence removal;
`XRStools/xrs_ComptonProfiles.py` lines 563 and 694 for threshold/f-sum processing
and N4 column mapping. The new implementation deliberately uses same-support
area ratios, explicit target masks and no tail extrapolation. It does not copy
the questionable EXPnorm*HFnorm pre-scale found in the other legacy extraction file.
