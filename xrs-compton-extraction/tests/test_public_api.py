from __future__ import annotations


def test_top_level_workflow_api_is_importable() -> None:
    from xrs_compton_extraction import (
        NexusMapping,
        QualityThreshold,
        TextMapping,
        XRSWorkbench,
        bootstrap_statistic,
        build_quality_report,
        correct_spectrum,
        extract_pearson,
        load_nexus,
        load_text,
        merge_spectra,
        poisson_uncertainty,
        save_results,
    )

    assert all(
        value is not None
        for value in (
            NexusMapping,
            QualityThreshold,
            TextMapping,
            XRSWorkbench,
            bootstrap_statistic,
            build_quality_report,
            correct_spectrum,
            extract_pearson,
            load_nexus,
            load_text,
            merge_spectra,
            poisson_uncertainty,
            save_results,
        )
    )


def test_subpackage_background_api_is_importable() -> None:
    from xrs_compton_extraction.backgrounds import (
        ModelSelectionFeatures,
        ModelSelectionPolicy,
        fit_pearson,
        fit_polynomial,
        select_background_models,
    )

    assert all(
        value is not None
        for value in (
            ModelSelectionFeatures,
            ModelSelectionPolicy,
            fit_pearson,
            fit_polynomial,
            select_background_models,
        )
    )


def test_io_text_api_is_importable() -> None:
    from xrs_compton_extraction.io import TextMapping, TextMappingError, load_text

    assert TextMapping is not None
    assert TextMappingError is not None
    assert load_text is not None
