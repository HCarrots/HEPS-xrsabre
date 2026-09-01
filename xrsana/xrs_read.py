import ast
import glob
import os
import re

import numpy as np
import pandas as pd


def print_citation_message():
    """Prints plea for citing the XRStools article when using this software."""
    print("                                                                                ")
    print(" ############################# Welcome to XRStools #############################")
    print(" # If you are using this software, please cite the following work:             #")
    print(" # Ch.J. Sahle, A. Mirone, J. Niskanen, J. Inkinen, M. Krisch, and S. Huotari: #")
    print(" # \"Planning, performing, and analyzing X-ray Raman scattering experiments.\"   #")
    print(" # Journal of Synchrotron Radiation 22, No. 2 (2015): 400-409.                 #")
    print(" ###############################################################################")
    print("                                                                                ")


class read_heps_id33:
    """
    Reader for the one-dimensional HEPS ID33 Ho reduced data export.

    The Ho data directory contains one tab-separated spectrum matrix
    (``*_all_data.txt``), one ROI metadata table (``*_rois.txt``), and one
    key/value metadata file (``*_info.txt``).  The reader exposes the loaded
    data directly as arrays:

    ``eloss``
        Energy transfer axis in eV.
    ``signals``
        Matrix with shape ``(n_energy, n_crystals)``.
    ``errors``
        Poisson-style uncertainty estimate, ``sqrt(abs(signals))``.
    ``energy``
        Incident-energy axis in keV, computed from the selected ROI centers.
    """

    def __init__(
        self,
        #data_path
        exp_dir=None,
        data_file=None,
        roi_file=None,
        info_file=None,
        #read intensity name
        intensity_columns_index=None,
        energy_column_index="Energy Transfer (eV)",
        use_roi_selection=True,
        q_column_index="q_ave",
        e0_column_index="center",
        q_range=None,
        #select analyzer name (is not intensity_columns_index)
        analyzer_names=None,
        remove_bad_fits=True,
        auto_read=True,
    ):
        self.path = os.path.abspath(exp_dir) if exp_dir is not None else None
        self.user_data_file = data_file
        self.user_roi_file = roi_file
        self.user_info_file = info_file

        self.user_intensity_columns = intensity_columns_index
        self.energy_column_index = energy_column_index
        self.use_roi_selection = use_roi_selection
        self.q_column_index = q_column_index
        self.e0_column_index = e0_column_index
        self.q_range = q_range
        self.analyzer_names = analyzer_names
        self.remove_bad_fits = remove_bad_fits

        self.info_file = None
        self.roi_file = None
        self.data_file = None
        self.source_name = None

        self.header_attrs = {}
        self.rois = pd.DataFrame()
        self.selected_rois = pd.DataFrame()
        self.dataframe = pd.DataFrame()
        self.intensity_columns_index = []

        self.data = {}
        self.key = {}
        self.cenom_dict = {}
        self.eloss = np.array([])
        self.signals = np.array([])
        self.errors = np.array([])
        self.E0 = np.nan
        self.cenom = []
        self.energy = np.array([])
        self.q = np.array([])
        self.q_average = np.nan
        self.tth = []

        self._validate_inputs()
        self._resolve_input_files()
        if auto_read:
            self.read_data()

    def _validate_inputs(self):
        has_exp_dir = self.path is not None
        given_files = {
            "data_file": self.user_data_file,
            "roi_file": self.user_roi_file,
            "info_file": self.user_info_file,
        }
        provided_files = [name for name, value in given_files.items() if value is not None]

        if has_exp_dir and provided_files:
            raise ValueError("Pass either exp_dir or data_file/roi_file/info_file, not both.")

        if has_exp_dir:
            if not os.path.isdir(self.path):
                raise ValueError("Invalid exp_dir: %s" % self.path)
            return

        missing_files = [name for name, value in given_files.items() if value is None]
        if missing_files:
            raise ValueError(
                "If exp_dir is not provided, data_file, roi_file and info_file "
                "must all be provided. Missing: " + ", ".join(missing_files)
            )

    def _resolve_input_files(self):
        if self.path is not None:
            self.data_file = self._find_single_file(self.path, "*_all_data.txt", "data")
            self.roi_file = self._find_single_file(self.path, "*_rois.txt", "ROI")
            self.info_file = self._find_single_file(self.path, "*_info.txt", "info")
        else:
            self.data_file = os.path.abspath(self.user_data_file)
            self.roi_file = os.path.abspath(self.user_roi_file)
            self.info_file = os.path.abspath(self.user_info_file)

        for file_path, label in [
            (self.data_file, "data_file"),
            (self.roi_file, "roi_file"),
            (self.info_file, "info_file"),
        ]:
            if not os.path.isfile(file_path):
                raise FileNotFoundError("%s does not exist: %s" % (label, file_path))

        self.source_name = re.sub(r"_all_data\.txt$", "", os.path.basename(self.data_file))

    def _find_single_file(self, directory, pattern, label):
        candidates = sorted(glob.glob(os.path.join(directory, pattern)))
        if len(candidates) == 1:
            return os.path.abspath(candidates[0])
        if len(candidates) > 1:
            raise FileExistsError(
                "Multiple %s files found: %s"
                % (label, ", ".join(os.path.basename(path) for path in candidates))
            )
        raise FileNotFoundError("Could not find %s file matching %s in %s" % (label, pattern, directory))

    def _read_table(self, filename):
        dataframe = pd.read_csv(filename, sep="\t")
        unnamed = [col for col in dataframe.columns if str(col).startswith("Unnamed:")]
        if unnamed:
            dataframe = dataframe.drop(columns=unnamed)
        return dataframe

    def _read_info(self, filename):
        attrs = {}
        with open(filename, "r") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or "=" not in line:
                    continue
                key, value = [part.strip() for part in line.split("=", 1)]
                try:
                    attrs[key] = ast.literal_eval(value)
                except Exception:
                    try:
                        attrs[key] = float(value)
                    except ValueError:
                        attrs[key] = value
        return attrs

    def _resolve_energy_column(self, dataframe):
        if self.energy_column_index in dataframe.columns:
            return self.energy_column_index
        if isinstance(self.energy_column_index, int):
            return dataframe.columns[self.energy_column_index]
        columns = [col for col in dataframe.columns if not str(col).startswith("Unnamed:")]
        if not columns:
            raise ValueError("Could not find an energy column in %s" % self.data_file)
        return columns[0]

    def _select_rois(self, rois):
        selected = rois.copy()
        if not self.use_roi_selection:
            return selected

        if self.analyzer_names and "crystal" in selected:
            analyzer_names = [self.analyzer_names] if isinstance(self.analyzer_names, str) else list(self.analyzer_names)
            prefixes = tuple(analyzer_name + "-" for analyzer_name in analyzer_names)
            selected = selected[selected["crystal"].astype(str).str.startswith(prefixes)]

        remove_list = set(self.header_attrs.get("ROIremoveList", []))
        if self.remove_bad_fits:
            remove_list.update(self.header_attrs.get("BadFitList", []))
        if remove_list and "crystal" in selected:
            selected = selected[~selected["crystal"].isin(remove_list)]

        q_range = self.q_range if self.q_range is not None else self.header_attrs.get("q_range")
        if q_range and self.q_column_index in selected:
            q_min, q_max = q_range
            selected = selected[
                (selected[self.q_column_index] >= q_min)
                & (selected[self.q_column_index] <= q_max)
            ]

        return selected

    def _resolve_intensity_columns(self, dataframe, selected_rois, energy_column):
        if self.user_intensity_columns is not None:
            columns = [
                dataframe.columns[col] if isinstance(col, int) else col
                for col in self.user_intensity_columns
            ]
            missing = [col for col in columns if col not in dataframe.columns]
            if missing:
                raise KeyError("Intensity columns not found: %s" % ", ".join(missing))
            selected_rois = self._match_rois_to_columns(selected_rois, columns)
            return columns, selected_rois

        candidate_columns = [
            col for col in dataframe.columns
            if col != energy_column and not str(col).startswith("Unnamed:")
        ]
        if "crystal" not in selected_rois:
            return candidate_columns, selected_rois

        roi_crystals = selected_rois["crystal"].astype(str).tolist()
        columns = [crystal for crystal in roi_crystals if crystal in dataframe.columns]
        if not columns:
            columns = candidate_columns
        selected_rois = self._match_rois_to_columns(selected_rois, columns)
        return columns, selected_rois

    def _match_rois_to_columns(self, rois, columns):
        if "crystal" not in rois:
            return rois
        matched = rois.set_index("crystal").reindex(columns).dropna(how="all").reset_index()
        return matched

    def _selected_values(self, selected_rois, column):
        if column not in selected_rois:
            return np.array([], dtype="float64")
        return selected_rois[column].to_numpy(dtype="float64")

    def _q_values_for_columns(self, selected_rois, intensity_columns):
        if self.q_column_index not in selected_rois:
            return np.full(len(intensity_columns), np.nan, dtype="float64")
        q_values = selected_rois[self.q_column_index].to_numpy(dtype="float64")
        if len(q_values) == len(intensity_columns):
            return q_values
        q_average = float(np.nanmean(q_values)) if q_values.size else np.nan
        return np.full(len(intensity_columns), q_average, dtype="float64")

    def _make_tth(self, q_values, number_of_columns, E0):
        q_values = np.asarray(q_values, dtype="float64")
        if q_values.size != number_of_columns:
            q_values = np.full(number_of_columns, np.nan, dtype="float64")
        if not np.isfinite(E0) or E0 <= 0.0:
            return [np.nan] * number_of_columns

        k = 0.5067731 * E0
        cos_tth = 1.0 - (q_values ** 2) / (2.0 * k ** 2)
        return [float(value) for value in np.degrees(np.arccos(np.clip(cos_tth, -1.0, 1.0)))]

    def _update_analyzer_metadata(self, selected_rois, intensity_columns):
        self.key = {"Analyzer%02d" % (idx + 1): idx for idx in range(len(intensity_columns))}
        self.data = {key: {} for key in self.key}
        self.cenom_dict = {}
        for key, idx in self.key.items():
            row = selected_rois.iloc[idx] if idx < len(selected_rois) else None
            fwhm = float(row["width"]) if row is not None and "width" in selected_rois else np.nan
            e0 = float(row[self.e0_column_index]) if row is not None and self.e0_column_index in selected_rois else np.nan
            self.cenom_dict[key] = {"average": {"fwhm": fwhm, "e0": e0}}
    
    def read_data(self):
        self.header_attrs = self._read_info(self.info_file)
        self.rois = self._read_table(self.roi_file)
        self.dataframe = self._read_table(self.data_file)

        energy_column = self._resolve_energy_column(self.dataframe)
        selected_rois = self._select_rois(self.rois)
        intensity_columns, selected_rois = self._resolve_intensity_columns(
            self.dataframe,
            selected_rois,
            energy_column,
        )

        if not intensity_columns:
            raise ValueError("No intensity columns were found in %s" % self.data_file)

        self.intensity_columns_index = intensity_columns
        self.selected_rois = selected_rois
        self.eloss = self.dataframe[energy_column].to_numpy(dtype="float64")
        self.signals = self.dataframe[intensity_columns].to_numpy(dtype="float64")
        if self.signals.ndim == 1:
            self.signals = self.signals[:, np.newaxis]
        self.errors = np.sqrt(np.absolute(self.signals))

        e0_values = self._selected_values(selected_rois, self.e0_column_index)
        self.E0 = float(np.nanmean(e0_values) / 1e3) if e0_values.size else np.nan
        self.cenom = (e0_values / 1e3).tolist() if e0_values.size else []
        self.energy = self.E0 + self.eloss / 1e3 if np.isfinite(self.E0) else self.eloss / 1e3

        self.q = self._q_values_for_columns(selected_rois, intensity_columns)
        self.q_average = float(np.nanmean(self.q)) if self.q.size else np.nan
        self.tth = self._make_tth(self.q, len(intensity_columns), self.E0)
        self._update_analyzer_metadata(selected_rois, intensity_columns)
        return self