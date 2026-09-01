"""Helpers used by the Jupyter workbench.

The notebook remains the only user-facing entry point.  These helpers keep
input preflight and the QC visualisation reusable while allowing a freshly
cloned workspace (which intentionally has no raw NXS files) to open and run
without an opaque ``FileNotFoundError``.
"""

from __future__ import annotations

from .paths import WorkspacePaths


def missing_scan_ids(config, workspace: WorkspacePaths) -> tuple[int, ...]:
    """Return configured scan IDs whose raw scan directory is unavailable."""
    raw_directory = workspace.raw / config.element
    if not raw_directory.is_dir():
        return tuple(dict.fromkeys((*config.elastic_scan_ids, *config.xrs_scan_ids)))

    missing: list[int] = []
    for scan_id in (*config.elastic_scan_ids, *config.xrs_scan_ids):
        prefix = f"{int(scan_id)}_"
        if not any(
            candidate.is_dir() and candidate.name.startswith(prefix)
            for candidate in raw_directory.iterdir()
        ):
            missing.append(int(scan_id))
    return tuple(dict.fromkeys(missing))


def run_qc_review(config, workspace: WorkspacePaths):
    """Prepare an analysis, display its QC tables and render diagnostic plots."""
    import matplotlib.pyplot as plt
    import numpy as np
    from IPython.display import display
    from matplotlib.colors import LogNorm
    from matplotlib.patches import Rectangle

    from xrslab.workflow import (
        QcApproval,
        build_qc_report,
        finalize_analysis,
        prepare_analysis,
    )

    prepared = prepare_analysis(config, workspace)
    qc = build_qc_report(prepared)
    display(qc.summary)
    display(qc.scan_table)
    display(qc.roi_table)

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    for axis, originals, corrected, title in (
        (axes[0], prepared.elastic_i0_original, prepared.elastic_i0_corrected, "Elastic I0"),
        (axes[1], prepared.xrs_i0_original, prepared.xrs_i0_corrected, "XRS I0"),
    ):
        for index, (before, after) in enumerate(zip(originals, corrected)):
            axis.plot(before, alpha=0.4, label=f"{index} raw")
            axis.plot(after, linewidth=1, label=f"{index} corrected")
        axis.set_title(title)
        axis.set_xlabel("Point")
        axis.legend(fontsize=7)
    plt.tight_layout()

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    for axis, stack, items, title in (
        (axes[0], prepared.elastic_batch.lambda_data[0], prepared.roi_collection.rois, "lambda"),
        (axes[1], prepared.elastic_batch.minipix_data[0], prepared.roi_collection.minipix_rois, "minipix"),
    ):
        image = np.nansum(stack, axis=0)
        positive = image[image > 0]
        norm = LogNorm(vmin=max(float(positive.min()), 1e-12), vmax=float(positive.max())) if positive.size else None
        axis.imshow(image, norm=norm)
        axis.set_title(title)
        for item in items:
            axis.add_patch(Rectangle(
                (item.x1, item.y1), item.x_width, item.y_width,
                edgecolor="red", facecolor="none", linewidth=0.5,
            ))
    plt.tight_layout()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(qc.roi_table["r-square"].dropna(), bins=20)
    axes[0].axvline(config.min_r_squared, color="red")
    axes[0].set_title("R-squared")
    axes[1].hist(qc.roi_table["width"].dropna(), bins=20)
    axes[1].axvline(config.max_fwhm_ev, color="red")
    axes[1].set_title("FWHM (eV)")
    plt.tight_layout()
    display(qc.roi_table.loc[
        ~qc.roi_table["automatic_accepted"],
        ["roi_id", "automatic_exclusion_reasons"],
    ])

    preview = finalize_analysis(prepared, QcApproval(approved=False))
    plt.figure(figsize=(9, 4))
    plt.plot(preview.energy_transfer, preview.intensity_sum)
    plt.xlabel("Energy Transfer (eV)")
    plt.ylabel("Intensity sum")
    plt.title("Provisional spectrum")
    plt.tight_layout()
    return prepared, qc


__all__ = ["missing_scan_ids", "run_qc_review"]
