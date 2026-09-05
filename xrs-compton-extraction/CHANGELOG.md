# Changelog

All notable changes to this project will be documented in this file.

## Unreleased

- Added finite-table shell-threshold/f-sum energy profiles, same-support HF area
  matching and automatic scale/linear-baseline fits without legacy dependencies.
- HoB4 high-q extraction now preserves N4 (Shell_13, 164 eV); exports distinguish
  target response from target-minus-HF residual and retain original intensity.

- Switched the HoB4 notebook (including synthetic data) to a local DABAX reader,
  with native shell labels, table occupations/binding energies and SHA256 provenance.
- Verified the local source works with xraylib unavailable; preserved the legacy
  optional backend only for existing callers.

- Added a Chinese step-by-step HoB4 notebook independent of the widget UI.
- Added explicit unweighted diagnostics, coordinate reflection, atomic review gates,
  missing-support mapping and a separate synthetic profile-transfer demonstration.
- Kept real HF subtraction paused pending shell, occupation and scale inputs.

- Added single-pass wide-table loading and a reproducible Ho processed-data audit.
- Blocked implicit Poisson error estimates for explicitly marked processed intensities.

- Added an optional xraylib profile adapter with explicit occupancy provenance.
- Added core-shell selection, target-shell exclusion, resolution and asymmetry hooks.
- Added empirical valence-reference selection, masking, symmetry and q transfer.
- Added Compton-template extraction, batch jobs and multi-q averaging/plots.
- Added a real-xraylib/synthetic two-q example and Markdown result reports.
- Added five workbench pages with Pearson batch, configuration and export controls.
- Preserved supplied count uncertainties and propagated explicit correction-factor errors.
- Raised the NumPy minimum to 2.0 to match the integration API used by the package.

- Created the independent `xrs_compton_extraction` package skeleton.
- Implemented all required domain/result objects with immutable numerical arrays.
- Implemented strict NeXus discovery and standard/configurable NXdata loading.
- Implemented photon kinematics, q/pz conversion, and deterministic synthetic data.
- Implemented time/I0 normalization and Poisson/I0 uncertainty propagation.
- Implemented bounded, weighted Pearson fitting and a single-channel extraction path.
- Added YAML/JSON configuration, CSV/JSON export, plotting, and a minimal workbench.
- Verified source tests, lint, source/wheel builds, and wheel import smoke tests.
