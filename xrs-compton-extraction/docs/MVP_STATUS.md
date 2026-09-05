# MVP implementation status

## Completed and automatically verified

- M0: source and wheel package import successfully.
- M1: a standard or explicitly mapped one-dimensional/multi-channel NeXus signal
  can be loaded and displayed.
- M2 computation core: one channel can be normalized and fitted with a bounded,
  weighted Pearson model; all background/result arrays and residuals are retained.
- Configuration can be round-tripped through YAML or JSON.
- Results can be exported to CSV plus JSON metadata and a manifest.
- The Jupyter entry point is `XRSWorkbench().display()` and contains no notebook
  algorithm definitions.
- Package-boundary tests reject any import from `xrs_processing`.
- M3 computation path: real xraylib profiles can form selected core backgrounds;
  a synthetic high-q reference can be normalized and transferred to another q;
  template fitting recovers an injected edge in end-to-end tests.
- M4 computation path: batch jobs retain successful channels, failures and
  exclusions; multi-q averaging and plots are available through public APIs.
- Workbench has five pages and interactive Pearson batch/configuration/export
  controls. The Compton workflow is callable through the controller's Python API.
- An executable two-q example uses real oxygen K-shell profiles with synthetic
  experimental intensities. This is a numerical regression, not real-data validation.

## Scientific release blockers

- The Ho processed wide table has passed input regression; target-edge settings,
  propagated uncertainties and scientific extraction tolerances are not yet confirmed.
- The project license is not selected.
- Hartree–Fock profiles use the optional xraylib backend, with Biggs provenance.
  Data redistribution review is needed only if tables will be vendored.
- Python xraylib 4.2.1 lacks ElectronConfig_Biggs. Using ElectronConfig is an
  explicit opt-in and selected shell occupations must be checked.
- Absorption/self-absorption and cross-section conventions await domain review.
- Experimental regression tolerances are not yet approved.

## Remaining engineering work

- Full advanced correction, profile and ensemble controls in the GUI.
- Automated multi-model/window/smoothing uncertainty experiments and shared
  reference covariance across q channels.
- Experimental energy calibration and regression with beamline datasets.
- Linux execution in CI (the local verification environment is Windows).

The current version is therefore `0.1.0.dev0`, not a scientifically validated
production release.
