"""XRSlab: X-ray Raman scattering (XRS) data processing toolkit."""

from . import analysis, math_func, roi_editor, workflow, xrs_roi
from xrsabre.paths import WorkspacePaths, load_workspace
from .xrs_roi import XRSRoi, data_fit, q_calc, read_h5, roi_path, save_data
from .roi_editor import RoiBox, RoiEditResult, RoiEditor
from .analysis import (
    DEFAULT_MODULE_ANGLES,
    FitSummary,
    InterpolationResult,
    RoiCollection,
    RoiMasks,
    RoiSums,
    ScanBatch,
    add_q_columns,
    build_roi_masks,
    configure_roi_selection,
    fit_elastic_rois,
    interpolate_and_sum,
    load_roi_collection,
    load_scan_batch,
    normalize_detector_array,
    normalize_detector_data,
    normalize_scan_batch,
    remove_i0_glitches,
    resolve_motor_pv,
    save_interpolation,
    sum_roi_spectra,
    sum_roi_spectra_batch,
)
from .workflow import (
    AnalysisConfig,
    AnalysisResult,
    PreparedAnalysis,
    QcApproval,
    QCReport,
    build_qc_report,
    export_analysis,
    finalize_analysis,
    prepare_analysis,
)

__version__ = "0.1.0"

__all__ = [
    "math_func",
    "xrs_roi",
    "analysis",
    "roi_editor",
    "workflow",
    # workspace paths
    "WorkspacePaths",
    "load_workspace",
    # roi
    "XRSRoi",
    "data_fit",
    "q_calc",
    "read_h5",
    "roi_path",
    "save_data",
    # interactive ROI editing
    "RoiBox",
    "RoiEditResult",
    "RoiEditor",
    # reusable analysis pipeline
    "DEFAULT_MODULE_ANGLES",
    "ScanBatch",
    "RoiCollection",
    "RoiMasks",
    "RoiSums",
    "FitSummary",
    "InterpolationResult",
    "normalize_detector_array",
    "normalize_detector_data",
    "load_scan_batch",
    "normalize_scan_batch",
    "remove_i0_glitches",
    "resolve_motor_pv",
    "load_roi_collection",
    "build_roi_masks",
    "sum_roi_spectra",
    "sum_roi_spectra_batch",
    "fit_elastic_rois",
    "add_q_columns",
    "configure_roi_selection",
    "interpolate_and_sum",
    "save_interpolation",
    # quality-gated workflow
    "AnalysisConfig",
    "QcApproval",
    "PreparedAnalysis",
    "QCReport",
    "AnalysisResult",
    "prepare_analysis",
    "build_qc_report",
    "finalize_analysis",
    "export_analysis",
]
