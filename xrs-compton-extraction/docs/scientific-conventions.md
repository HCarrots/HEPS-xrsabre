# Scientific conventions

Status: initial engineering baseline; domain-owner review is required before a
formal scientific release.

## Coordinates and signs

- Incident and scattered photon energies are positive and stored in eV.
- Energy loss is defined as `incident_energy_ev - scattered_energy_ev`.
- Photon wavenumber is stored in inverse ångström when used with laboratory
  geometry.
- Momentum transfer is the magnitude of the difference between incident and
  scattered photon wavevectors and is therefore non-negative.
- Longitudinal electron momentum uses atomic units. Both energy loss and momentum
  transfer must be converted to atomic units before evaluating
  `p_z = omega / q - q / 2`.
- Scattering angles exposed by the public API are in degrees unless the argument
  name explicitly says otherwise.

## Data integrity

- Raw detector counts are never overwritten by a correction or extraction step.
- Negative values produced by background subtraction are retained and diagnosed;
  they are not clipped to zero.
- Missing geometry, units, or fixed-analyzer energy must not be replaced by a
  physically meaningful-looking default.
- Every interpolation, mask, and excluded fit window must be present in result
  metadata or provenance.

## Pending domain decisions

- Exact scattering-angle orientation for the first beamline mapping.
- Authoritative physical-constant release used for formal validation.
- Fixed-energy metadata path and analyzer-specific calibration model.
- Acceptance tolerances for experimental regression data.

