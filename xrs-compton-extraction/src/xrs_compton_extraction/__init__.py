"""Independent tools for multi-q XRS Compton-background extraction."""

from .batch import BatchResult, extract_batch
from .config import load_config, save_config
from .corrections import correct_spectrum
from .data import (
    ROI,
    AnalysisConfig,
    AnalysisSession,
    Analyzer,
    BackgroundResult,
    CorrectionResult,
    ExtractionResult,
    Geometry,
    QualityReport,
    Sample,
    Scan,
    XRSDataset,
    XRSSpectrum,
)
from .diagnostics import QualityThreshold, build_quality_report
from .io import NexusMapping, TextMapping, load_nexus, load_text, save_results
from .merge import MergeDiagnostics, MergeResult, merge_dataset, merge_spectra
from .multi_q import MultiQResult, average_multi_q
from .pipeline import extract_pearson
from .profile_pipeline import extract_compton_profile
from .q_groups import QBand, classify_q_band, group_q_channels
from .uncertainty import (
    BootstrapResult,
    NormalizationUncertainty,
    ScanRepeatability,
    bootstrap_statistic,
    combine_independent_uncertainties,
    multi_scan_repeatability,
    poisson_uncertainty,
    propagate_normalization_uncertainty,
)
from .workbench import XRSWorkbench

__version__ = "0.1.0.dev0"

__all__ = [
    "ROI",
    "AnalysisConfig",
    "AnalysisSession",
    "Analyzer",
    "BackgroundResult",
    "BatchResult",
    "BootstrapResult",
    "CorrectionResult",
    "ExtractionResult",
    "Geometry",
    "MergeDiagnostics",
    "MergeResult",
    "MultiQResult",
    "NexusMapping",
    "NormalizationUncertainty",
    "QBand",
    "QualityReport",
    "QualityThreshold",
    "Sample",
    "Scan",
    "ScanRepeatability",
    "TextMapping",
    "XRSDataset",
    "XRSSpectrum",
    "XRSWorkbench",
    "average_multi_q",
    "bootstrap_statistic",
    "build_quality_report",
    "classify_q_band",
    "combine_independent_uncertainties",
    "correct_spectrum",
    "extract_batch",
    "extract_compton_profile",
    "extract_pearson",
    "group_q_channels",
    "load_config",
    "load_nexus",
    "load_text",
    "merge_dataset",
    "merge_spectra",
    "multi_scan_repeatability",
    "poisson_uncertainty",
    "propagate_normalization_uncertainty",
    "save_config",
    "save_results",
]
