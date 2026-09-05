"""Minimal Jupyter workbench shell.

The module deliberately imports ``ipywidgets`` only when :meth:`display` is
called, so importing the scientific package does not require UI dependencies.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from ..config import load_config as load_analysis_config
from ..config import save_config as save_analysis_config
from ..data import AnalysisConfig, AnalysisSession, ExtractionResult, XRSDataset
from ..exceptions import MissingOptionalDependencyError


@dataclass(slots=True)
class XRSWorkbench:
    """Stateful controller for the minimal single-channel extraction workbench."""

    data_path: Path | None = None
    config: AnalysisConfig = field(
        default_factory=lambda: AnalysisConfig(
            background_model="pearson",
            correction_flags={
                "normalize_acquisition_time": True,
                "normalize_i0": True,
            },
        )
    )
    session: AnalysisSession = field(init=False, repr=False)
    _widget: Any | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.data_path is not None:
            self.data_path = Path(self.data_path).expanduser()
            self.config = replace(self.config, data_path=str(self.data_path))
        self.session = AnalysisSession(config=self.config)

    @property
    def dataset(self) -> XRSDataset | None:
        return self.session.dataset

    @property
    def results(self) -> Mapping[str, ExtractionResult]:
        return self.session.extraction_results

    def load(self, path: str | Path | None = None, *, mapping: Any | None = None) -> XRSDataset:
        """Load one unambiguous NeXus file into the analysis session."""

        from ..io import load_nexus

        selected = Path(path).expanduser() if path is not None else self.data_path
        if selected is None:
            raise ValueError("a NeXus file or directory path is required")
        dataset = load_nexus(selected, mapping=mapping)
        self.session.extraction_results.clear()
        self.data_path = selected
        self.config = replace(self.config, data_path=str(selected))
        self.session.update_config(self.config)
        self.session.dataset = dataset
        self.session.set_status("ready")
        self.session.add_log(f"Loaded {dataset.source_count} source file(s) from {selected}")
        return dataset

    def load_text(
        self, path: str | Path, *, mapping: Any | None = None,
        mappings: Sequence[Any] | None = None,
    ) -> XRSDataset:
        """Load CSV/TSV with one mapping, or mappings for a wide channel table."""
        from ..io import load_text, load_text_channels

        if (mapping is None) == (mappings is None):
            raise ValueError("supply exactly one of mapping or mappings")
        dataset = load_text(path, mapping=mapping) if mapping is not None else load_text_channels(path, mappings)
        self.data_path = Path(path).expanduser()
        self.config = replace(self.config, data_path=str(self.data_path))
        self.session.update_config(self.config)
        self.session.extraction_results.clear()
        self.session.dataset = dataset
        self.session.set_status("ready")
        return dataset

    def run_compton_profile(self, *, spectrum_index: int = 0, **options: Any) -> ExtractionResult:
        """Run profile extraction with explicit core/valence inputs and windows."""
        from ..profile_pipeline import extract_compton_profile

        if self.dataset is None:
            raise RuntimeError("load a dataset before running extraction")
        if not 0 <= spectrum_index < len(self.dataset.spectra):
            raise IndexError("spectrum_index is out of range")
        spectrum = self.dataset.spectra[spectrum_index]
        self.session.set_status("running")
        try:
            result = extract_compton_profile(spectrum, **options)
        except Exception:
            self.session.set_status("failed")
            raise
        self.session.extraction_results[spectrum.channel_label] = result
        self.config = replace(self.config, background_model="compton_profile", fit_windows=result.fit_windows)
        self.session.config = self.config
        self.session.set_status("complete")
        return result

    def run_batch(self, *, extractor: Any, channel_options: Mapping[str, Mapping[str, Any]], **options: Any) -> Any:
        """Run a batch and retain its per-channel failures in the session log."""
        from ..batch import extract_batch

        if self.dataset is None:
            raise RuntimeError("load a dataset before running extraction")
        self.session.set_status("running")
        try:
            batch = extract_batch(self.dataset, extractor=extractor, channel_options=channel_options, **options)
        except Exception:
            self.session.set_status("failed")
            raise
        self.session.extraction_results.clear()
        self.session.extraction_results.update(batch.results)
        for channel, error in batch.failures.items():
            self.session.add_log(f"{channel}: {error}")
        self.session.set_status("failed" if batch.failures else "complete")
        return batch

    def run_pearson(
        self,
        *,
        spectrum_index: int = 0,
        fit_windows_ev: Sequence[tuple[float, float]] | None = None,
        q_au: float | None = None,
        normalize_acquisition_time: bool | None = None,
        normalize_i0: bool | None = None,
        initial: Sequence[float] | None = None,
        loss: str = "soft_l1",
    ) -> ExtractionResult:
        """Run the UI-independent Pearson pipeline for one selected channel."""

        from ..pipeline import extract_pearson

        if self.dataset is None:
            raise RuntimeError("load a dataset before running extraction")
        if not 0 <= spectrum_index < len(self.dataset.spectra):
            raise IndexError("spectrum_index is out of range")
        try:
            spectrum = self.dataset.spectra[spectrum_index]
        except IndexError as exc:
            raise IndexError(f"spectrum_index {spectrum_index} is out of range") from exc
        windows = tuple(fit_windows_ev or self.config.fit_windows)
        time_flag = (
            self.config.correction_flags.get("normalize_acquisition_time", True)
            if normalize_acquisition_time is None
            else normalize_acquisition_time
        )
        i0_flag = (
            self.config.correction_flags.get("normalize_i0", True)
            if normalize_i0 is None
            else normalize_i0
        )
        self.session.set_status("running")
        try:
            result = extract_pearson(
                spectrum,
                fit_windows_ev=windows,
                q_au=q_au,
                normalize_acquisition_time=time_flag,
                normalize_i0=i0_flag,
                initial=initial,
                loss=loss,
            )
        except Exception:
            self.session.set_status("failed")
            raise
        key = spectrum.channel_label
        self.session.extraction_results[key] = result
        self.config = replace(
            self.config,
            fit_windows=windows,
            background_model="pearson",
            correction_flags={
                **self.config.correction_flags,
                "normalize_acquisition_time": bool(time_flag),
                "normalize_i0": bool(i0_flag),
            },
        )
        self.session.config = self.config
        self.session.set_status("complete")
        self.session.add_log(f"Completed Pearson extraction for {key}")
        return result

    def save_config(self, path: str | Path) -> Path:
        """Save the current workbench configuration."""

        return save_analysis_config(self.config, path)

    def load_config(self, path: str | Path) -> AnalysisConfig:
        """Load a configuration and invalidate prior derived results."""

        config = load_analysis_config(path)
        self.config = config
        self.session.update_config(config)
        self.data_path = Path(config.data_path) if config.data_path else None
        return config

    def save_results(self, output_directory: str | Path) -> Path:
        """Save all extraction results, configuration, and manifest."""

        from ..io import save_results

        if not self.results:
            raise RuntimeError("no extraction results are available to save")
        return save_results(self.results, output_directory, config=self.config)

    def build(self) -> Any:
        """Build and return the workbench widget without displaying it."""

        if self._widget is not None:
            return self._widget
        try:
            import ipywidgets as widgets
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise MissingOptionalDependencyError(
                "The Jupyter workbench requires the 'workbench' extra: "
                "python -m pip install -e '.[workbench]'"
            ) from exc

        path_value = "" if self.data_path is None else str(self.data_path)
        path_input = widgets.Text(
            value=path_value,
            description="Data path",
            layout=widgets.Layout(width="80%"),
        )
        load_button = widgets.Button(description="Load", button_style="primary")
        q_selector = widgets.Dropdown(options=[], description="q channel", disabled=True)
        q_input = widgets.FloatText(value=0.0, description="q (a.u.)")
        windows_input = widgets.Text(
            value=self._format_windows(self.config.fit_windows),
            placeholder="e.g. 0:35, 65:100",
            description="Fit windows",
            layout=widgets.Layout(width="80%"),
        )
        normalize_time = widgets.Checkbox(
            value=self.config.correction_flags.get("normalize_acquisition_time", True),
            description="Normalize time",
        )
        normalize_i0 = widgets.Checkbox(
            value=self.config.correction_flags.get("normalize_i0", True),
            description="Normalize I0",
        )
        fit_button = widgets.Button(description="Fit Pearson", button_style="success", disabled=True)
        batch_button = widgets.Button(description="Run all Pearson", disabled=True)
        progress = widgets.IntProgress(value=0, min=0, max=1, description="Progress")
        output_path = widgets.Text(value="output", description="Output dir")
        config_path = widgets.Text(value="analysis.yaml", description="Config file")
        save_button = widgets.Button(description="Export results")
        save_config_button = widgets.Button(description="Save config")
        load_config_button = widgets.Button(description="Load config")
        status = widgets.HTML(value="<em>No dataset loaded.</em>")
        output = widgets.Output()

        def on_load(_: Any) -> None:
            output.clear_output(wait=True)
            try:
                self.data_path = Path(path_input.value).expanduser()
                dataset = self.load(self.data_path)
                options = [
                    (spectrum.channel_label, index)
                    for index, spectrum in enumerate(dataset.spectra)
                ]
                q_selector.options = options
                q_selector.value = options[0][1] if options else None
                q_selector.disabled = not options
                fit_button.disabled = not options
                batch_button.disabled = not options
                status.value = (
                    f"<b>Loaded:</b> {len(options)} spectrum channel(s) from "
                    f"{dataset.source_count} source file(s)."
                )
                if options:
                    channel_q = dataset.spectra[0].q_au
                    if channel_q is not None and np.allclose(channel_q, channel_q[0]):
                        q_input.value = float(channel_q[0])
                    with output:
                        self._plot_spectrum(0)
            except Exception as exc:  # noqa: BLE001 - UI callback renders domain errors
                self.session.dataset = None
                q_selector.options = []
                q_selector.disabled = True
                fit_button.disabled = True
                batch_button.disabled = True
                status.value = f"<b>Load failed:</b> {type(exc).__name__}: {exc}"

        def on_channel(change: dict[str, Any]) -> None:
            if change.get("name") != "value" or self.dataset is None:
                return
            output.clear_output(wait=True)
            if change["new"] is None:
                return
            channel_q = self.dataset.spectra[int(change["new"])].q_au
            q_input.value = (
                float(channel_q[0])
                if channel_q is not None and np.allclose(channel_q, channel_q[0])
                else 0.0
            )
            with output:
                self._plot_spectrum(int(change["new"]))

        def on_fit(_: Any) -> None:
            output.clear_output(wait=True)
            try:
                windows = self._parse_windows(windows_input.value)
                selected_q = q_input.value if q_input.value > 0.0 else None
                result = self.run_pearson(
                    spectrum_index=int(q_selector.value),
                    fit_windows_ev=windows,
                    q_au=selected_q,
                    normalize_acquisition_time=normalize_time.value,
                    normalize_i0=normalize_i0.value,
                )
                status.value = (
                    f"<b>{result.quality_grade}:</b> Pearson extraction complete; "
                    f"reduced chi-square={result.risk_metrics['reduced_chi_square']:.4g}."
                )
                with output:
                    self._plot_result(result)
            except Exception as exc:  # noqa: BLE001 - UI callback renders domain errors
                status.value = f"<b>Fit failed:</b> {type(exc).__name__}: {exc}"

        load_button.on_click(on_load)
        fit_button.on_click(on_fit)
        def on_batch(_: Any) -> None:
            if self.dataset is None:
                return
            from ..pipeline import extract_pearson

            try:
                windows = self._parse_windows(windows_input.value)
                channel_options = {
                    spectrum.channel_label: {
                        "fit_windows_ev": windows,
                        "normalize_acquisition_time": normalize_time.value,
                        "normalize_i0": normalize_i0.value,
                    }
                    for spectrum in self.dataset.spectra
                }
                progress.max = len(channel_options)
                progress.value = 0
                batch = self.run_batch(
                    extractor=extract_pearson,
                    channel_options=channel_options,
                    on_progress=lambda done, total, label: setattr(progress, "value", done),
                )
                status.value = f"Completed: {len(batch.results)}; failed: {len(batch.failures)}"
                output.clear_output(wait=True)
                with output:
                    for channel, error in batch.failures.items():
                        print(f"{channel}: {error}")
                    if batch.results:
                        self._plot_result(next(iter(batch.results.values())))
            except Exception as exc:  # noqa: BLE001 - show actionable batch errors
                status.value = f"Batch failed: {type(exc).__name__}: {exc}"

        def on_save(_: Any) -> None:
            try:
                destination = self.save_results(output_path.value)
                status.value = f"Exported: {destination}"
            except Exception as exc:  # noqa: BLE001 - render export errors
                status.value = f"Export failed: {type(exc).__name__}: {exc}"

        def on_save_config(_: Any) -> None:
            try:
                self.config = replace(
                    self.config, fit_windows=self._parse_windows(windows_input.value),
                    correction_flags={"normalize_acquisition_time": normalize_time.value, "normalize_i0": normalize_i0.value},
                )
                self.session.update_config(self.config)
                status.value = f"Saved: {self.save_config(config_path.value)}"
            except Exception as exc:  # noqa: BLE001 - render configuration errors
                status.value = f"Save config failed: {type(exc).__name__}: {exc}"

        def on_load_config(_: Any) -> None:
            try:
                config = self.load_config(config_path.value)
                path_input.value = str(config.data_path or "")
                windows_input.value = self._format_windows(config.fit_windows)
                normalize_time.value = config.correction_flags.get("normalize_acquisition_time", True)
                normalize_i0.value = config.correction_flags.get("normalize_i0", True)
                status.value = "Configuration loaded; load the selected data before extraction."
                self.session.dataset = None
                q_selector.options = []
                q_selector.disabled = fit_button.disabled = batch_button.disabled = True
            except Exception as exc:  # noqa: BLE001 - render configuration errors
                status.value = f"Load config failed: {type(exc).__name__}: {exc}"

        batch_button.on_click(on_batch)
        save_button.on_click(on_save)
        save_config_button.on_click(on_save_config)
        load_config_button.on_click(on_load_config)
        q_selector.observe(on_channel, names="value")
        pages = widgets.Tab(children=[
            widgets.VBox([widgets.HBox([path_input, load_button]), q_selector]),
            widgets.VBox([normalize_time, normalize_i0, widgets.HTML("Advanced corrections are available through correct_spectrum().")]),
            widgets.VBox([q_input, windows_input, widgets.HTML("Pearson interactive fitting. Profile extraction: app.run_compton_profile(...).")]),
            widgets.VBox([widgets.HBox([fit_button, batch_button]), progress, widgets.HTML("Batch uses each channel's measured q and the displayed fit windows.")]),
            widgets.VBox([widgets.HBox([output_path, save_button]), config_path, widgets.HBox([save_config_button, load_config_button])]),
        ])
        for index, title in enumerate(("Data", "Correction", "Background", "Extraction", "Results")):
            pages.set_title(index, title)
        self._widget = widgets.VBox(
            [
                widgets.HTML("<h3>XRS Compton Extraction</h3>"),
                pages,
                status,
                output,
            ]
        )
        return self._widget

    def display(self) -> Any:
        """Display the workbench and return its root widget."""

        try:
            from IPython.display import display
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise MissingOptionalDependencyError(
                "Displaying the workbench requires IPython."
            ) from exc
        widget = self.build()
        display(widget)
        return widget

    def _plot_spectrum(self, index: int) -> None:
        import matplotlib.pyplot as plt

        if self.dataset is None:
            return
        spectrum = self.dataset.spectra[index]
        _, axis = plt.subplots()
        x = (
            spectrum.energy_loss_ev
            if spectrum.energy_loss_ev is not None
            else spectrum.energy_ev
        )
        axis.plot(x, spectrum.raw_counts)
        axis.set_xlabel(
            "Energy loss (eV)"
            if spectrum.energy_loss_ev is not None
            else "Energy (eV)"
        )
        axis.set_ylabel("Raw counts")
        axis.set_title(spectrum.channel_label)
        axis.grid(alpha=0.2)
        plt.show()

    @staticmethod
    def _parse_windows(value: str) -> tuple[tuple[float, float], ...]:
        windows: list[tuple[float, float]] = []
        for item in value.split(","):
            item = item.strip()
            if not item:
                continue
            try:
                start_text, stop_text = item.split(":", maxsplit=1)
                start, stop = float(start_text), float(stop_text)
            except ValueError as exc:
                raise ValueError(
                    "fit windows must use 'start:stop, start:stop' syntax"
                ) from exc
            if not np.isfinite(start) or not np.isfinite(stop) or start >= stop:
                raise ValueError("each fit window must have finite start < stop")
            windows.append((start, stop))
        if not windows:
            raise ValueError("at least one fit window is required")
        return tuple(windows)

    @staticmethod
    def _format_windows(windows: Sequence[tuple[float, float]]) -> str:
        return ", ".join(f"{start:g}:{stop:g}" for start, stop in windows)

    @staticmethod
    def _plot_result(result: ExtractionResult) -> None:
        import matplotlib.pyplot as plt

        figure, (spectrum_axis, residual_axis) = plt.subplots(2, 1, sharex=True)
        spectrum_axis.plot(
            result.energy_loss_ev, result.corrected_intensity, label="Corrected"
        )
        spectrum_axis.plot(
            result.energy_loss_ev, result.total_background, label="Background"
        )
        spectrum_axis.plot(
            result.energy_loss_ev, result.extracted_edge, label="Extracted"
        )
        spectrum_axis.legend()
        spectrum_axis.set_ylabel("Intensity")
        residual_axis.axhline(0.0, color="black", linewidth=0.8)
        residual_axis.plot(result.energy_loss_ev, result.fit_residual, color="C3")
        residual_axis.set_xlabel("Energy loss (eV)")
        residual_axis.set_ylabel("Residual")
        figure.tight_layout()
        plt.show()
