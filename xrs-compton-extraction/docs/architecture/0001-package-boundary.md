# ADR 0001: Package boundary

## Decision

`xrs_compton_extraction` is developed in the `xrsabre` workspace as a separate
installable package. It may share the workspace's development environment, but it
must not import `xrs_processing`, inspect its private state, or accept its internal
objects as a required public interface.

The Jupyter workbench depends on the public computation and I/O APIs. Computation
modules never depend on the workbench.

## Automated enforcement

Tests scan package imports for `xrs_processing`. Integration fixtures construct
only `xrs_compton_extraction` domain objects.

## Consequences

- Scientific behavior can be tested and released independently.
- Any future data interchange with `xrs_processing` must be an explicit file or
  public adapter boundary and requires a separate design decision.

