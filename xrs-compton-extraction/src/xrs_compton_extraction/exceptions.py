"""Domain-specific exceptions for :mod:`xrs_compton_extraction`."""

from __future__ import annotations


class XRSComptonExtractionError(Exception):
    """Base class for recoverable package errors."""


class DataValidationError(XRSComptonExtractionError, ValueError):
    """Raised when a domain object's values or dimensions are inconsistent."""


class DataDiscoveryError(XRSComptonExtractionError):
    """Raised when an input path cannot be resolved unambiguously."""


class NexusMappingError(XRSComptonExtractionError):
    """Raised when a NeXus file cannot be mapped to the required XRS fields."""


class MissingOptionalDependencyError(XRSComptonExtractionError, ImportError):
    """Raised when an optional feature is requested without its dependency."""

