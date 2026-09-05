# Atomic Compton-profile data policy

## Current HoB4 source: local DABAX file

`DabaxProfileSource(path)` reads the supplied
`resources/compton_profiles/ComptonProfiles.dat` without xraylib or network access.
It preserves `Shell_N` labels, reads per-shell occupations from `#UOCCUP` and
binding energies in eV from `#UBIND`, and records the source path, table date and
SHA256. The supplied file date is 2003-01-29; its SHA256 is
`db42c6cefad0916886e96acf03b0186b538d01dbde2988d7bdec4c8675d814ba`.

Partial columns are per electron; the core builder applies occupation and
stoichiometry exactly once. The reader checks dimensions, finite/nonnegative
values, monotonic momentum and occupation sums. It linearly interpolates in
absolute pz, rejects extrapolation, and never renormalizes the tabulated total.
Native column names are not automatically mapped to spectroscopic N4/N5 labels.
The notebook displays binding energies for review before choosing the target.

The new source fixes the earlier occupation-source mismatch: B occupations are
2,2,1 (sum 5); Ho occupations sum to 67. Finite-grid partial/total relative L2
differences are about 0.00086 for B and 0.00271 for Ho. Integration of the coarse
table is a numerical diagnostic, not an exact electron-count renormalization.
The table remains an external file and is not added to package-data patterns.

The following sections describe the legacy optional backend, not the active notebook.

## Selected source

The first supported Hartree–Fock source is the Biggs–Mendelsohn–Mann atomic
profile set:

- F. Biggs, L. B. Mendelsohn, and J. B. Mann, *Hartree-Fock Compton profiles
  for the elements*, Atomic Data and Nuclear Data Tables 16 (1975), 201–309;
- DOI: <https://doi.org/10.1016/0092-640X(75)90030-3>;
- machine-readable implementation: xraylib `ComptonProfile`,
  `ComptonProfile_Partial`, and an explicitly identified occupancy API;
- upstream project: <https://github.com/tschoonj/xraylib>.

The tabulation covers Z=1–102. It uses non-relativistic Hartree–Fock profiles
for the lighter elements and Dirac–Hartree–Fock profiles for the heavier
elements. Values are tabulated against non-negative longitudinal momentum in
atomic units. Free-atom profiles are even in momentum, so the adapter evaluates
the upstream source at `abs(p_z)` while retaining the signed experimental grid.

## Integration policy

`xraylib` is an optional runtime dependency, not a dependency on
`xrs_processing`. The adapter records:

- backend and backend version;
- element, shell, occupancy, and stoichiometric weight;
- excluded target shell;
- Biggs DOI;
- resolution convolution and asymmetry settings;
- the signed experimental `p_z` grid.

No profile table is copied into the wheel. The xraylib software is distributed
under a BSD-style license, but redistribution of the underlying DABAX table must
be reviewed independently before vendoring. A future vendored dataset must add
its exact version, source URL, license, and SHA-256 checksum to `manifest.yaml`.

The Python interface in the tested xraylib 4.2.1 build does not expose
`ElectronConfig_Biggs`. The default adapter reports this incompatibility.
`XraylibProfileSource(occupancy_source="electron_config")` explicitly selects
the alternative `ElectronConfig` API and records it in provenance. Occupancy
conventions may differ, especially for open shells; confirm the selected
occupations before using the result scientifically. The core builder requires
`shells_by_element` for every element so valence electrons cannot silently be
counted again when an empirical valence profile is added.

## Scientific limitations

These profiles describe isolated atoms in the impulse approximation. They do
not encode chemical bonding or solid-state valence redistribution. They are
therefore suitable as an auditable starting point for non-target core-shell
backgrounds; they are not a substitute for experimental validation or for the
high-q valence-profile extraction described by Sternemann et al.
