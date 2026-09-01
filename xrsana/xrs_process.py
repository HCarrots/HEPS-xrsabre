import numpy as np
import copy
from xrsana.xrs_public import e2pz,mpr_compds,abscorr2,element,convg,pz2e1
from xrsana import xrs_ComptonProfiles
import os
from importlib.resources import files
from pathlib import Path
from scipy import interpolate , optimize
import matplotlib.pyplot as plt
from xrsana.math_functions import pearson7
from xrsabre.paths import WorkspacePaths, load_workspace
data_installation_dir = str(files("xrsana").joinpath("resources", "data"))
HFCP_PATH = os.path.join(data_installation_dir,'ComptonProfiles.dat')

class HF_dataset:
    """
    **dataset**
    A class to hold all information from HF Compton profiles necessary to subtract background from the experiment.
    """
    def __init__(self, data, formulas, stoich_weights, edges):
        self.formulas       = formulas
        self.stoich_weights = stoich_weights
        self.edges          = edges #e.g. {'Li':['K','L23'], 'O':'K'}
        self.E0             = data.E0
        self.cenom          = data.cenom
        self.tth            = data.tth
        self.eloss          = data.eloss
        self.HFProfile      = xrs_ComptonProfiles.HFProfile(formulas, stoich_weights, HFCP_PATH)
        self.HFProfile.get_elossProfiles(self.E0,self.tth)

        # interpolate total HF profiles onto experimental eloss scale
        self.J_total   = np.zeros((len(self.eloss),len(self.tth)))
        self.C_total   = np.zeros((len(self.eloss),len(self.tth)))
        self.V_total   = np.zeros((len(self.eloss),len(self.tth)))
        self.q_vals    = np.zeros((len(self.eloss),len(self.tth)))
        for ii in range(len(self.tth)):
            self.J_total[:,ii] = np.interp(self.eloss, self.HFProfile.eloss,self.HFProfile.J_total[:,ii])
            self.C_total[:,ii] = np.interp(self.eloss, self.HFProfile.eloss,self.HFProfile.C_total[:,ii])
            self.V_total[:,ii] = np.interp(self.eloss, self.HFProfile.eloss,self.HFProfile.V_total[:,ii])
            self.q_vals[:,ii]  = np.interp(self.eloss, self.HFProfile.eloss,self.HFProfile.q_vals[:,ii])

        # initialize double dicts for {'element1':{'edge1','edge2',...}, 'element2':{'edge1','edge2',...} }
        self.C_edges = {}
        for key in self.edges:
            self.C_edges[key] = {}
            for edge in self.edges[key]:
                self.C_edges[key][edge] = np.zeros((len(self.eloss),len(self.tth)))

        # interpolate the core profiles for the desired elements
        for key in self.edges:
            for edge in self.edges[key]:
                edge_keyword = xrs_ComptonProfiles.mapShellNames(edge,element(key))
                for formula in self.formulas:
                    if key in formula:
                        # cp core-edge profile
                        atom_profile = self.HFProfile.FormulaProfiles[formula].AtomProfiles[key]
                        for jj in range(len(self.tth)):
                            self.C_edges[key][edge][:,jj] = np.interp(self.eloss,atom_profile.eloss,atom_profile.CperShell[edge_keyword][:,jj])
                    else:
                        print('Could not find ' + key + ' in any of the provided formulas.')

    def get_J_total_av(self,columns):
        return np.mean(self.J_total[:,columns])

    def get_C_total(self,columns):
        return np.mean(self.J_total[:,columns])

    def get_C_edges_av(self,element,edge,columns):
        return np.mean(self.C_edges[element][edge][:,columns])

class XRSProcess:
    def __init__(
        self,
        exp_data,
        formulas,
        stoich_weights,
        edges,
        prenormrange=[5, np.inf],
        *,
        workspace: WorkspacePaths | None = None,
    ):
        self.workspace = load_workspace() if workspace is None else workspace
        self.eloss          = copy.deepcopy( exp_data.eloss )
        self.signals        = copy.deepcopy( exp_data.signals )
        self.errors         = copy.deepcopy( exp_data.errors )
        self.tth            = exp_data.tth
        self.formulas       = formulas
        self.E0             = exp_data.E0
        self.qvals          = exp_data.q
        self.prenormrange   = prenormrange
        self.HF_dataset     = HF_dataset(exp_data, formulas, stoich_weights, edges)
        self.concentrations = stoich_weights
        self.background     = np.zeros(np.shape(exp_data.signals))
        self.sqw            = np.zeros(np.shape(exp_data.signals))
        pass

    def xrs_remove_elastic(self,effective_energy_range=(5, np.inf)):
        """
        Remove elastic peak from XRS data.
        """
        mask = (self.eloss >= effective_energy_range[0]) & (self.eloss <= effective_energy_range[1])

        if not np.any(mask):
            raise ValueError(f"IN [{effective_energy_range[0]}, {effective_energy_range[1]}] has no data points, please check the range settings")
        self.eloss = self.eloss[mask]
        self.signals = self.signals[mask, :]
        self.errors = self.errors[mask, :]
        self.background = self.background[mask, :]
        self.sqw = self.sqw[mask, :]
        self.HF_dataset.eloss = self.HF_dataset.eloss[mask]
        self.HF_dataset.J_total = self.HF_dataset.J_total[mask, :]
        self.HF_dataset.C_total = self.HF_dataset.C_total[mask, :]
        self.HF_dataset.V_total = self.HF_dataset.V_total[mask, :]
        self.HF_dataset.q_vals = self.HF_dataset.q_vals[mask, :]
        for key in self.HF_dataset.C_edges:
            for edge in self.HF_dataset.C_edges[key]:
                self.HF_dataset.C_edges[key][edge] = self.HF_dataset.C_edges[key][edge][mask, :]
    
    def xrs_remove_stray_background(self, method='linear', fit_range=(10, 110)):
        """
        Remove stray background from XRS data using the specified method.

        Parameters
        ----------
        method : {'linear', 'constant'}
            Background model to subtract.
        fit_range : tuple[float, float]
            Energy-loss range used to estimate the background.
        """
        mask = (self.eloss >= fit_range[0]) & (self.eloss <= fit_range[1])

        if not np.any(mask):
            raise ValueError("fit_range does not overlap with self.eloss.")

        if method == 'linear':
            k, b = np.polyfit(self.eloss[mask], self.signals[mask], deg=1)
            background = self.eloss[:, None] * k[None, :] + b[None, :]

        elif method == 'constant':
            c = np.mean(self.signals[mask])
            background = np.full_like(self.signals, c[None, :], dtype=float)

        else:
            raise ValueError("Unsupported method for background removal. Use 'linear' or 'constant'.")

        self.signals = self.signals - background

    def xrs_energy_correction(self, alpha, densities, samthickness, whichq=None):
        """
        Apply energy dependent corrections to selected measured data columns based on
        scattering angles, sample material, and sample thickness.

        Parameters
        ----------
        alpha : float
            Incident beam angle relative to sample surface normal.
            Negative for transmission geometry.

        densities : float or list
            Single value for one compound, or list of values for mixture.

        samthickness : float
            Sample thickness in cm.

        whichq : int, list of int, or None
            q-channel / detector column index or indices to correct.
            If None, all columns are corrected.

        Returns
        -------
        pz : ndarray
            Momentum scale for the selected columns.
            Shape is (len(self.eloss), len(whichq)).
        """

        import numpy as np

        # --------------------------------------------------
        # Select q columns
        # --------------------------------------------------
        ncols_total = self.signals.shape[1]

        if whichq is None:
            cols = np.arange(ncols_total, dtype=int)
        elif np.isscalar(whichq):
            cols = np.array([int(whichq)], dtype=int)
        else:
            cols = np.array([int(col) for col in whichq], dtype=int)

        if cols.size == 0:
            raise ValueError("whichq contains no columns.")

        if np.any(cols < 0) or np.any(cols >= ncols_total):
            raise IndexError(
                f"whichq contains invalid column indices. "
                f"Valid range is 0 to {ncols_total - 1}."
            )

        # --------------------------------------------------
        # Make densities iterable
        # --------------------------------------------------
        if isinstance(densities, (list, tuple, np.ndarray)):
            denses = list(densities)
        else:
            denses = [densities]

        # --------------------------------------------------
        # Safety checks
        # --------------------------------------------------
        if self.errors.shape != self.signals.shape:
            raise ValueError("self.errors must have the same shape as self.signals.")

        if self.HF_dataset.J_total.shape != self.signals.shape:
            raise ValueError(
                "self.HF_dataset.J_total must have the same shape as self.signals."
            )

        if len(self.tth) < ncols_total:
            raise ValueError(
                "self.tth must contain at least one scattering angle per signal column."
            )

        # --------------------------------------------------
        # Calculate beta for selected columns
        # --------------------------------------------------
        beta = np.zeros(len(cols))

        if alpha >= 0:
            # Reflection geometry
            for n, col in enumerate(cols):
                beta[n] = np.abs(180.0 - alpha - self.tth[col])
        else:
            # Transmission geometry
            for n, col in enumerate(cols):
                beta[n] = -np.abs(np.abs(alpha) - self.tth[col])

        # --------------------------------------------------
        # Absorption and self-absorption correction
        # --------------------------------------------------
        mu_in, mu_out = mpr_compds(
            self.eloss / 1.0e3 + self.E0,
            self.formulas,
            self.concentrations,
            self.E0,
            denses,
        )

        ac = np.zeros((len(self.eloss), len(cols)))

        for n, col in enumerate(cols):
            ac[:, n] = abscorr2(
                mu_in,
                mu_out,
                alpha,
                beta[n],
                samthickness,
            )

        # --------------------------------------------------
        # Cross section correction
        # --------------------------------------------------
        pz = np.zeros((len(self.eloss), len(cols)))
        cf = np.zeros((len(self.eloss), len(cols)))

        for n, col in enumerate(cols):
            pz[:, n], cf[:, n] = e2pz(
                self.E0 + self.eloss / 1.0e3,
                self.E0,
                self.tth[col],
            )

        # --------------------------------------------------
        # Normalization range
        # --------------------------------------------------
        inds = np.where(
            (self.eloss >= self.prenormrange[0])
            & (self.eloss <= self.prenormrange[1])
        )[0]

        if len(inds) < 2:
            raise ValueError("Not enough points in prenormrange for normalization.")

        # --------------------------------------------------
        # Apply corrections only to selected columns
        # --------------------------------------------------
        scales = {}

        for n, col in enumerate(cols):
            correction = ac[:, n] * cf[:, n]

            self.signals[:, col] = self.signals[:, col] * correction
            self.errors[:, col] = self.errors[:, col] * correction

            HFnorm = np.trapezoid(
                self.HF_dataset.J_total[inds, col],
                self.eloss[inds],
            )

            EXPnorm = np.trapezoid(
                self.signals[inds, col],
                self.eloss[inds],
            )

            if np.isclose(EXPnorm, 0.0):
                raise ValueError(
                    f"EXPnorm is zero or too close to zero for column {col}."
                )

            scale = HFnorm / EXPnorm

            self.signals[:, col] = self.signals[:, col] * scale
            self.errors[:, col] = self.errors[:, col] * scale

            scales[int(col)] = scale
            print(f"column {col}: scale = {scale}")

        # Optional: save diagnostic information
        self.energy_correction_info = {
            "whichq": cols.tolist(),
            "alpha": alpha,
            "beta": beta,
            "densities": denses,
            "samthickness": samthickness,
            "scales": scales,
        }

        return pz

    def xrs_remove_poly_core(
        self,
        whichq,
        polyregion,
        coreregion,
        weights=(1.0, 1.0),
        polyorder=2,
        scale=1.0,
        hfcoreshift=0.0,
        plot=True,
        ewindow=100.0,
        save_result=True,
        ):
        """
        Fit and subtract a polynomial background from one or more q channels,
        while using the corresponding HF core profile as a constraint.

        This version does NOT use averaged signals.

        Model for each q channel:

            scaled_signal(E) ≈ polynomial(E) + shifted_core_profile(E)

        After fitting:

            sqw(E, q) = scaled_signal(E, q) - polynomial(E)

        Parameters
        ----------
        whichq : int or list of int
            q-channel index or list of q-channel indices.

        polyregion : list or tuple
            Energy range [emin, emax] mainly used to constrain the smooth
            polynomial background.

        coreregion : list or tuple
            Energy range [emin, emax] mainly used to constrain the known core
            profile.

        weights : tuple of float
            Relative fitting weights for polyregion and coreregion.
            Example: weights=(5.0, 1.0) gives the polynomial region more influence.

        polyorder : int
            Polynomial order. Default is 2.

        scale : float
            Multiplicative scale applied to each experimental signal before fitting.

        hfcoreshift : float
            Energy shift applied to the HF core profile.

        plot : bool
            If True, show diagnostic plots.

        ewindow : float
            Extra plotting window around selected fitting regions.

        save_result : bool
            If True, save background and corrected spectra into:

                self.background[:, col]
                self.sqw[:, col]

        Returns
        -------
        results : dict
            Dictionary keyed by q-channel index. Each entry contains fit coefficients,
            fitted background, shifted core profile, corrected spectrum, and fit cost.
        """

        import numpy as np
        import matplotlib.pyplot as plt
        from scipy import optimize

        # -----------------------------
        # 1. Basic validation
        # -----------------------------
        if not hasattr(self, "eloss"):
            raise AttributeError("self.eloss is missing.")

        if not hasattr(self, "signals"):
            raise AttributeError("self.signals is missing.")

        if not hasattr(self, "HF_dataset") or not hasattr(self.HF_dataset, "C_edges"):
            raise AttributeError("self.HF_dataset.C_edges is missing. Core profile self.HF_dataset.C_edges is required.")

        if not hasattr(self, "sqw"):
            raise AttributeError("self.sqw is missing. Please initialize self.sqw first.")

        if save_result and not hasattr(self, "background"):
            raise AttributeError(
                "self.background is missing. Please initialize self.background first."
            )

        eloss = np.asarray(self.eloss)
        signals = np.asarray(self.signals)
        core_profiles = np.zeros_like(signals)
        for element_edges in self.HF_dataset.C_edges.values():
            for edge_profile in element_edges.values():
                core_profiles += np.asarray(edge_profile)

        if eloss.ndim != 1:
            raise ValueError("self.eloss must be a 1D array.")

        if signals.ndim != 2:
            raise ValueError("self.signals must be a 2D array with shape (energy, q).")

        if core_profiles.shape != signals.shape:
            raise ValueError("self.HF_dataset.C_edges profiles must have the same shape as self.signals.")

        if signals.shape[0] != eloss.size:
            raise ValueError("self.signals.shape[0] must match len(self.eloss).")

        if len(polyregion) != 2:
            raise ValueError("polyregion must be [emin, emax].")

        if len(coreregion) != 2:
            raise ValueError("coreregion must be [emin, emax].")

        polyorder = int(polyorder)

        if polyorder < 0:
            raise ValueError("polyorder must be non-negative.")

        if np.isscalar(whichq):
            columns = [int(whichq)]
        else:
            columns = [int(col) for col in whichq]

        n_energy, n_q = signals.shape

        for col in columns:
            if col < 0 or col >= n_q:
                raise IndexError(f"q-channel index {col} is out of range.")

        # -----------------------------
        # 2. Select fitting regions
        # -----------------------------
        poly_inds = np.flatnonzero(
            (eloss >= polyregion[0]) & (eloss <= polyregion[1])
        )

        core_inds = np.flatnonzero(
            (eloss >= coreregion[0]) & (eloss <= coreregion[1])
        )

        if poly_inds.size == 0:
            raise ValueError("polyregion contains no data points.")

        if core_inds.size == 0:
            raise ValueError("coreregion contains no data points.")

        if poly_inds.size + core_inds.size <= polyorder:
            raise ValueError(
                "Not enough fitting points for the requested polynomial order."
            )

        w_poly = float(weights[0])
        w_core = float(weights[1])

        if w_poly <= 0 or w_core <= 0:
            raise ValueError("weights must be positive.")

        fit_inds = np.concatenate([poly_inds, core_inds])

        region_weights = np.concatenate([
            np.full(poly_inds.size, np.sqrt(w_poly)),
            np.full(core_inds.size, np.sqrt(w_core)),
        ])

        results = {}

        # -----------------------------
        # 3. Fit each q channel
        # -----------------------------
        for col in columns:
            signal = signals[:, col]
            core = core_profiles[:, col]

            scaled_signal = signal * scale

            shifted_core = np.interp(
                eloss,
                eloss + hfcoreshift,
                core,
                left=0.0,
                right=0.0,
            )

            # Initial polynomial guess:
            # scaled_signal - shifted_core ≈ polynomial
            initial_coeffs = np.polyfit(
                eloss[fit_inds],
                scaled_signal[fit_inds] - shifted_core[fit_inds],
                deg=polyorder,
            )

            def residual(coeffs):
                polynomial = np.polyval(coeffs, eloss[fit_inds])
                raw_residual = (
                    scaled_signal[fit_inds]
                    - shifted_core[fit_inds]
                    - polynomial
                )
                return region_weights * raw_residual

            fit_result = optimize.least_squares(residual, initial_coeffs)

            if not fit_result.success:
                raise RuntimeError(
                    f"Polynomial-core fit failed for q-channel {col}: "
                    f"{fit_result.message}"
                )

            coeffs = fit_result.x

            polynomial_background = np.polyval(coeffs, eloss)

            corrected_spectrum = scaled_signal - polynomial_background

            fit_cost = np.sum(residual(coeffs) ** 2)

            # -----------------------------
            # 4. Save results
            # -----------------------------
            if save_result:
                self.background[:, col] = polynomial_background
                self.sqw[:, col] = corrected_spectrum

            results[col] = {
                "coeffs": coeffs,
                "polyorder": polyorder,
                "scale": scale,
                "hfcoreshift": hfcoreshift,
                "polyregion": polyregion,
                "coreregion": coreregion,
                "weights": weights,
                "polynomial_background": polynomial_background,
                "shifted_core": shifted_core,
                "corrected_spectrum": corrected_spectrum,
                "fit_cost": fit_cost,
                "fit_result": fit_result,
            }

            # -----------------------------
            # 5. Optional diagnostic plot
            # -----------------------------
            if plot:
            # 全局学术样式（和你之前完全一致）
                plt.rcParams.update({
                    'font.family': 'serif',
                    'font.serif': ['Times New Roman'],
                    'font.size': 12,
                    'axes.labelsize': 14,
                    'axes.titlesize': 12,
                    'xtick.labelsize': 12,
                    'ytick.labelsize': 12,
                    'legend.fontsize': 12,
                    'figure.dpi': 300,
                    'savefig.dpi': 300,
                    'savefig.bbox': 'tight',
                })

                # 创建画布
                fig, ax = plt.subplots(figsize=(8, 5))

                # 绘制多条曲线
                ax.plot(eloss, scaled_signal, label="scaled signal", linewidth=1.2)
                ax.plot(eloss, polynomial_background + shifted_core, label="polynomial + shifted core",
                        linestyle='--', linewidth=1.2)
                ax.plot(eloss, polynomial_background, label="polynomial background", linewidth=1.2)
                ax.plot(eloss, corrected_spectrum, label="signal - polynomial", linewidth=1.2)
                ax.plot(eloss, shifted_core, label="shifted core profile", linewidth=1.2)

                # 坐标轴范围（完全保留原逻辑）
                xmin = min(polyregion[0], coreregion[0]) - ewindow
                xmax = polyregion[1] if np.isfinite(polyregion[1]) else eloss[-1]
                xmax = max(xmax, coreregion[1] if np.isfinite(coreregion[1]) else eloss[-1]) + ewindow
                ax.set_xlim(xmin, xmax)

                # 标题、坐标轴
                ax.set_xlabel("energy loss [eV]")
                ax.set_ylabel("DDSCS [arb. units]")
                ax.set_title(f"q-channel {col}", pad=15)

                # 图例 + 网格（底层显示，不遮挡曲线）
                ax.legend(frameon=False)
                ax.grid(True, linestyle='--', alpha=0.5, color='lightgray', zorder=0)
                ax.set_axisbelow(True)

                # 显示 + 关闭画布
                plt.tight_layout()
                plot_dir = self.workspace.diagnostics / "xrs_process"
                plot_dir.mkdir(parents=True, exist_ok=True)
                fig.savefig(plot_dir / f"q_channel_{col}.pdf")
                fig.savefig(plot_dir / f"q_channel_{col}.png")
                plt.close(fig)
        return results    



    def xrs_remove_poly_core_2(
        self,
        whichq,
        polyregion,
        coreregion,
        weights=(1.0, 1.0),
        polyorder=2,
        scale=1.0,
        hfcoreshift=0.0,
        plot=True,
        ewindow=100.0,
        save_result=True,
    ):
        """
        Fit and subtract a polynomial background from one or more q channels,
        while using the corresponding HF core profile as a constraint.

        This version uses two polynomial fitting regions:
            1. pre-peak background region
            2. post-peak background region

        Model for each q channel:

            scaled_signal(E) ≈ polynomial_background(E) + shifted_core_profile(E)

        After fitting:

            sqw(E, q) = scaled_signal(E, q) - polynomial_background(E)

        Parameters
        ----------
        whichq : int or list of int
            q-channel index or list of q-channel indices.

        polyregion : list or tuple
            Two background fitting regions, before and after the peak.

            Recommended format:
                [[pre_min, pre_max], [post_min, post_max]]

            Also accepted:
                [pre_min, pre_max, post_min, post_max]

            Example:
                polyregion = [[-100, -20], [80, 150]]

        coreregion : list or tuple
            Energy range [emin, emax] mainly used to constrain the known core
            profile.

            Example:
                coreregion = [0, 60]

        weights : tuple of float
            Relative fitting weights for polyregion and coreregion.

            weights = (w_poly, w_core)

            Example:
                weights=(5.0, 1.0)
            gives the polynomial background regions more influence.

        polyorder : int
            Polynomial order. Default is 2.

        scale : float
            Multiplicative scale applied to each experimental signal before fitting.

        hfcoreshift : float
            Energy shift applied to the HF core profile.

        plot : bool
            If True, show diagnostic plots.

        ewindow : float
            Extra plotting window around selected fitting regions.

        save_result : bool
            If True, save background and corrected spectra into:

                self.background[:, col]
                self.sqw[:, col]

        Returns
        -------
        results : dict
            Dictionary keyed by q-channel index. Each entry contains fit coefficients,
            fitted background, shifted core profile, corrected spectrum, and fit cost.
        """

        import numpy as np
        import matplotlib.pyplot as plt
        from scipy import optimize

        # -----------------------------
        # 1. Basic validation
        # -----------------------------
        if not hasattr(self, "eloss"):
            raise AttributeError("self.eloss is missing.")

        if not hasattr(self, "signals"):
            raise AttributeError("self.signals is missing.")

        if not hasattr(self, "HF_dataset") or not hasattr(self.HF_dataset, "C_edges"):
            raise AttributeError(
                "self.HF_dataset.C_edges is missing. "
                "Core profile self.HF_dataset.C_edges is required."
            )

        if not hasattr(self, "sqw"):
            raise AttributeError("self.sqw is missing. Please initialize self.sqw first.")

        if save_result and not hasattr(self, "background"):
            raise AttributeError(
                "self.background is missing. Please initialize self.background first."
            )

        eloss = np.asarray(self.eloss)
        signals = np.asarray(self.signals)
        core_profiles = np.zeros_like(signals)
        for element_edges in self.HF_dataset.C_edges.values():
            for edge_profile in element_edges.values():
                core_profiles += np.asarray(edge_profile)

        if eloss.ndim != 1:
            raise ValueError("self.eloss must be a 1D array.")

        if signals.ndim != 2:
            raise ValueError("self.signals must be a 2D array with shape (energy, q).")

        if core_profiles.shape != signals.shape:
            raise ValueError(
                "self.HF_dataset.C_edges must have the same shape as self.signals."
            )

        if signals.shape[0] != eloss.size:
            raise ValueError("self.signals.shape[0] must match len(self.eloss).")

        polyorder = int(polyorder)

        if polyorder < 0:
            raise ValueError("polyorder must be non-negative.")

        if len(weights) != 2:
            raise ValueError("weights must be a tuple/list of length 2: (w_poly, w_core).")

        w_poly = float(weights[0])
        w_core = float(weights[1])

        if w_poly <= 0 or w_core <= 0:
            raise ValueError("weights must be positive.")

        if np.isscalar(whichq):
            columns = [int(whichq)]
        else:
            columns = [int(col) for col in whichq]

        n_energy, n_q = signals.shape

        for col in columns:
            if col < 0 or col >= n_q:
                raise IndexError(f"q-channel index {col} is out of range.")

        # -----------------------------
        # 2. Normalize fitting regions
        # -----------------------------
        def _normalize_polyregion(region):
            """
            Normalize polyregion to:
                [(pre_min, pre_max), (post_min, post_max)]

            Accepted formats:
                [[pre_min, pre_max], [post_min, post_max]]
                [pre_min, pre_max, post_min, post_max]
            """
            arr = np.asarray(region, dtype=float)

            if arr.ndim == 1:
                if arr.size == 4:
                    arr = arr.reshape(2, 2)
                else:
                    raise ValueError(
                        "polyregion must contain two intervals, e.g. "
                        "[[pre_min, pre_max], [post_min, post_max]] "
                        "or [pre_min, pre_max, post_min, post_max]."
                    )

            elif arr.ndim == 2:
                if arr.shape != (2, 2):
                    raise ValueError(
                        "polyregion must have shape (2, 2), e.g. "
                        "[[pre_min, pre_max], [post_min, post_max]]."
                    )

            else:
                raise ValueError(
                    "polyregion must be [[pre_min, pre_max], [post_min, post_max]]."
                )

            regions = []

            for i, pair in enumerate(arr):
                emin, emax = pair

                if not np.isfinite(emin) or not np.isfinite(emax):
                    raise ValueError("polyregion values must be finite.")

                if emax <= emin:
                    raise ValueError(
                        f"polyregion interval {i} has invalid range: "
                        f"[{emin}, {emax}]. emax must be larger than emin."
                    )

                regions.append((float(emin), float(emax)))

            return regions

        def _normalize_coreregion(region):
            """
            Normalize coreregion to:
                (core_min, core_max)
            """
            arr = np.asarray(region, dtype=float)

            if arr.ndim != 1 or arr.size != 2:
                raise ValueError("coreregion must be [emin, emax].")

            emin, emax = arr

            if not np.isfinite(emin) or not np.isfinite(emax):
                raise ValueError("coreregion values must be finite.")

            if emax <= emin:
                raise ValueError(
                    f"coreregion has invalid range: [{emin}, {emax}]. "
                    "emax must be larger than emin."
                )

            return float(emin), float(emax)

        poly_regions = _normalize_polyregion(polyregion)
        core_emin, core_emax = _normalize_coreregion(coreregion)

        pre_emin, pre_emax = poly_regions[0]
        post_emin, post_emax = poly_regions[1]

        # Optional but recommended logical check:
        # The two polynomial regions should normally be outside the core region.
        if pre_emax > post_emin:
            raise ValueError(
                "polyregion intervals appear to overlap or are in the wrong order. "
                "Expected: pre-peak region first, post-peak region second."
            )

        # Build masks
        poly_mask = np.zeros_like(eloss, dtype=bool)

        for emin, emax in poly_regions:
            poly_mask |= (eloss >= emin) & (eloss <= emax)

        core_mask = (eloss >= core_emin) & (eloss <= core_emax)

        if not np.any(poly_mask):
            raise ValueError("polyregion contains no data points.")

        if not np.any(core_mask):
            raise ValueError("coreregion contains no data points.")

        # Avoid double-use of the same data points as both polynomial and core constraints.
        if np.any(poly_mask & core_mask):
            raise ValueError(
                "polyregion and coreregion overlap. "
                "Please choose non-overlapping background and core fitting regions."
            )

        poly_inds = np.flatnonzero(poly_mask)
        core_inds = np.flatnonzero(core_mask)

        if poly_inds.size + core_inds.size <= polyorder:
            raise ValueError(
                "Not enough fitting points for the requested polynomial order."
            )

        fit_inds = np.concatenate([poly_inds, core_inds])

        region_weights = np.concatenate(
            [
                np.full(poly_inds.size, np.sqrt(w_poly)),
                np.full(core_inds.size, np.sqrt(w_core)),
            ]
        )

        results = {}

        # -----------------------------
        # 3. Fit each q channel
        # -----------------------------
        for col in columns:
            signal = signals[:, col]
            core = core_profiles[:, col]

            scaled_signal = signal * scale

            shifted_core = np.interp(
                eloss,
                eloss + hfcoreshift,
                core,
                left=0.0,
                right=0.0,
            )

            # Initial polynomial guess:
            # scaled_signal - shifted_core ≈ polynomial background
            initial_coeffs = np.polyfit(
                eloss[fit_inds],
                scaled_signal[fit_inds] - shifted_core[fit_inds],
                deg=polyorder,
            )

            def residual(coeffs):
                polynomial = np.polyval(coeffs, eloss[fit_inds])

                raw_residual = (
                    scaled_signal[fit_inds]
                    - shifted_core[fit_inds]
                    - polynomial
                )

                return region_weights * raw_residual

            fit_result = optimize.least_squares(residual, initial_coeffs)

            if not fit_result.success:
                raise RuntimeError(
                    f"Polynomial-core fit failed for q-channel {col}: "
                    f"{fit_result.message}"
                )

            coeffs = fit_result.x

            polynomial_background = np.polyval(coeffs, eloss)
            polynomial_background[polynomial_background < 0] = 0

            corrected_spectrum = scaled_signal - polynomial_background

            fit_cost = np.sum(residual(coeffs) ** 2)

            # -----------------------------
            # 4. Save results
            # -----------------------------
            if save_result:
                self.background[:, col] = polynomial_background
                self.sqw[:, col] = corrected_spectrum

            results[col] = {
                "coeffs": coeffs,
                "polyorder": polyorder,
                "scale": scale,
                "hfcoreshift": hfcoreshift,
                "polyregion": poly_regions,
                "coreregion": (core_emin, core_emax),
                "weights": weights,
                "polynomial_background": polynomial_background,
                "shifted_core": shifted_core,
                "corrected_spectrum": corrected_spectrum,
                "fit_cost": fit_cost,
                "fit_result": fit_result,
            }

            # -----------------------------
            # 5. Optional diagnostic plot
            # -----------------------------
            if plot:
                plt.rcParams.update({
                    'font.family': 'serif',
                    'font.serif': ['Times New Roman'],
                    'font.size': 12,
                    'axes.labelsize': 14,
                    'axes.titlesize': 16,
                    'xtick.labelsize': 12,
                    'ytick.labelsize': 12,
                    'legend.fontsize': 12,
                    'figure.dpi': 300,
                    'savefig.dpi': 300,
                    'savefig.bbox': 'tight',
                })

                # 创建画布（固定尺寸，绘图更稳定）
                fig, ax = plt.subplots(figsize=(8, 5))

                # 1. 绘制所有曲线（完全保留原线条宽度、线型）
                ax.plot(eloss, scaled_signal, label="scaled signal", linewidth=1.2)

                ax.plot(
                    eloss,
                    polynomial_background + shifted_core,
                    label="polynomial + shifted core",
                    linestyle="--",
                    linewidth=1.2
                )

                ax.plot(
                    eloss,
                    polynomial_background,
                    label="polynomial background",
                    linewidth=0.5
                )

                ax.plot(
                    eloss,
                    corrected_spectrum,
                    label="signal - polynomial",
                    linewidth=1.2
                )

                ax.plot(
                    eloss,
                    shifted_core,
                    label="shifted core profile",
                    linewidth=0.5
                )

                # 2. 绘制多项式拟合区域阴影（完全保留原逻辑）
                for i, (emin, emax) in enumerate(poly_regions):
                    label = "polyregion" if i == 0 else None
                    ax.axvspan(
                        emin,
                        emax,
                        alpha=0.15,
                        label=label,
                        color='gray'  # 统一阴影颜色，更美观
                    )

                # 3. 绘制核心拟合区域阴影
                ax.axvspan(
                    core_emin,
                    core_emax,
                    alpha=0.15,
                    label="coreregion",
                    color='orange'  # 区分区域，可视化更清晰
                )

                # 4. 网格设置（置于底层，不遮挡曲线）
                ax.grid(True, alpha=0.3, linestyle='--', color='lightgray')
                ax.set_axisbelow(True)

                # 5. 计算坐标轴范围（原逻辑完全保留）
                all_region_edges = [
                    pre_emin,
                    pre_emax,
                    post_emin,
                    post_emax,
                    core_emin,
                    core_emax,
                ]
                xmin = min(all_region_edges) - ewindow
                xmax = max(all_region_edges) + ewindow
                ax.set_xlim(xmin, xmax)

                # 6. 坐标轴与标题（学术规范：标题间距+统一字体）
                ax.set_xlabel("energy loss [eV]")
                ax.set_ylabel("DDSCS [arb. units]")
                ax.set_title(f"q-channel {col} Valence electron profile correction", pad=15)

                # 7. 图例（无边框，期刊标准）
                ax.legend(frameon=False)

                # 8. 紧凑布局 + 显示 + 释放内存
                plt.tight_layout()
                #plt.show()
                plot_dir = self.workspace.diagnostics / "xrs_process"
                plot_dir.mkdir(parents=True, exist_ok=True)
                fig.savefig(plot_dir / f"q_channel_2{col}.pdf")
                fig.savefig(plot_dir / f"q_channel_2{col}.png")
                plt.close(fig)
        return results

    def extractval(
        self,
        whichq,
        mirror=False,
        linrange1=None,
        linrange2=None,
        edge_pz=None,
        fit_core_scale=True,
        asymmetry_sign=-1.0,
        final_interp_kind="linear",
        make_plots=True,
        wait_for_input=False,
    ):
        """
        Extract valence profiles for one or multiple q-values.

        This is a cleaned-up version based on extractval() and extractval_test().
        It preserves the original workflow:

            1. Fit a linear background in selected energy-loss regions.
            2. Fit/subtract HF core profile.
            3. Extract raw valence profile.
            4. Either mirror the negative-pz side or replace near-edge region
            by a Pearson VII function.
            5. Fit valence asymmetry.
            6. Interpolate final results onto self.pzscale.

        Parameters
        ----------
        whichq : int or list[int]
            Column index or list of column indices to process.

        mirror : bool
            If True, use the negative-pz side to mirror the profile.
            If False, use Pearson VII replacement near edge.

        linrange1, linrange2 : tuple/list or None
            Energy-loss fitting ranges, e.g. (10, 20). If both are None,
            the original default criterion is used:
                0.1 * HF_dataset.C_edges[:, col] > self.V[:, col]

        edge_pz : float or None
            Boundary pz value for Pearson replacement.
            If None and mirror=False, the function asks the user to click a point.

        fit_core_scale : bool
            If True, fit a scale factor for the HF/core profile, as in extractval_test().
            If False, subtract the core profile with fixed scale = 1, closer to extractval().

        asymmetry_sign : float
            Sign convention for asymmetry.
            -1.0 matches the behavior in extractval_test().
            +1.0 is closer to extractval().

        final_interp_kind : str
            Interpolation kind for mapping each q grid back to self.pzscale.
            Recommended: "linear". Use "cubic" only if the data are smooth enough.

        make_plots : bool
            Whether to show diagnostic plots.

        wait_for_input : bool
            Whether to pause after plotting each profile.

        Notes
        -----
        Assumes the following already exist:
            np, plt, pylab, optimize, interpolate, e2pz, pearson7
        """

        if not isinstance(whichq, (list, tuple, np.ndarray)):
            columns = [whichq]
        else:
            columns = list(whichq)

        if len(columns) == 0:
            raise ValueError("whichq is empty.")

        n_points = len(self.eloss)

        # ------------------------------------------------------------------
        # Ensure output arrays exist.
        # ------------------------------------------------------------------
        
        n_cols = self.sqw.shape[1]

        if not hasattr(self, "background") or self.background.shape != self.sqw.shape:
            self.background = np.zeros_like(self.sqw)

        if not hasattr(self, "valencepz") or self.valencepz.shape != self.sqw.shape:
            self.valencepz = np.zeros_like(self.sqw)

        if not hasattr(self, "valasymmetrypz") or self.valasymmetrypz.shape != self.sqw.shape:
            self.valasymmetrypz = np.zeros_like(self.sqw)

        # Backward-compatible arrays, if older code expects them.
        if not hasattr(self, "valence") or self.valence.shape != self.sqw.shape:
            self.valence = np.zeros_like(self.sqw)

        if not hasattr(self, "valasymmetry") or self.valasymmetry.shape != self.sqw.shape:
            self.valasymmetry = np.zeros_like(self.sqw)

        # ------------------------------------------------------------------
        # Set master pz grid.
        # If self.pzscale already exists and has the right length, keep it.
        # Otherwise use the first requested q column as master grid.
        # ------------------------------------------------------------------
        if (
            not hasattr(self, "pzscale")
            or self.pzscale is None
            or len(self.pzscale) != n_points
        ):
            first_col = columns[0]
            self.pzscale = e2pz(
                self.eloss / 1e3 + self.E0,
                self.E0,
                self.tth[first_col]
            )[0]

        master_pz = np.asarray(self.pzscale)

        # ------------------------------------------------------------------
        # Small interpolation helpers.
        # ------------------------------------------------------------------
        def _unique_sorted_xy(x, y):
            """Return sorted x,y with duplicate x values removed."""
            x = np.asarray(x)
            y = np.asarray(y)

            mask = np.isfinite(x) & np.isfinite(y)
            x = x[mask]
            y = y[mask]

            order = np.argsort(x)
            x = x[order]
            y = y[order]

            x_unique, unique_idx = np.unique(x, return_index=True)
            y_unique = y[unique_idx]

            return x_unique, y_unique

        def _interp_to_master(pz_src, y_src, kind="linear", fill_value=0.0):
            """Interpolate y_src(pz_src) onto master_pz."""
            x, y = _unique_sorted_xy(pz_src, y_src)

            if len(x) < 2:
                return np.zeros_like(master_pz)

            use_kind = kind
            if kind == "cubic" and len(x) < 4:
                use_kind = "linear"

            f = interpolate.interp1d(
                x,
                y,
                kind=use_kind,
                bounds_error=False,
                fill_value=fill_value,
            )
            return f(master_pz)

        def _interp_on_grid(x_src, y_src, x_new, kind="linear", fill_value=0.0):
            """Safe interpolation onto arbitrary x_new."""
            x, y = _unique_sorted_xy(x_src, y_src)

            if len(x) < 2:
                return np.zeros_like(x_new)

            use_kind = kind
            if kind == "cubic" and len(x) < 4:
                use_kind = "linear"

            f = interpolate.interp1d(
                x,
                y,
                kind=use_kind,
                bounds_error=False,
                fill_value=fill_value,
            )
            return f(x_new)

        # ------------------------------------------------------------------
        # Prepare storage for Pearson VII info (so user can draw later).
        # ------------------------------------------------------------------
        if not hasattr(self, "pearson_info"):
            self.pearson_info = {}

        # ------------------------------------------------------------------
        # Main loop over q columns.
        # ------------------------------------------------------------------
        plt.ion()

        for col in columns:
            if col < 0 or col >= n_cols:
                raise IndexError(f"Column index {col} is out of range.")

            # Current q-specific pz grid.
            pz_col = e2pz(
                self.eloss / 1e3 + self.E0,
                self.E0,
                self.tth[col]
            )[0]

            # Sum all core-edge profiles for this column.
            core_total_col = np.zeros(n_points)
            for element_edges in self.HF_dataset.C_edges.values():
                for edge_profile in element_edges.values():
                    core_total_col += np.asarray(edge_profile[:, col])

            # --------------------------------------------------------------
            # Select linear-background fitting region.
            # --------------------------------------------------------------
            if linrange1 is not None and linrange2 is not None:
                range1 = np.where(
                    (self.eloss >= linrange1[0]) & (self.eloss <= linrange1[1])
                )[0]
                range2 = np.where(
                    (self.eloss >= linrange2[0]) & (self.eloss <= linrange2[1])
                )[0]
                linrange = np.append(range1, range2)

            elif linrange1 is not None:
                linrange = np.where(
                    (self.eloss >= linrange1[0]) & (self.eloss <= linrange1[1])
                )[0]

            else:
                linrange = np.where(0.1 * core_total_col > self.HF_dataset.V_total[:, col])[0]

            linrange = np.asarray(linrange, dtype=int)

            if len(linrange) < 3:
                raise ValueError(
                    f"Not enough points in linear fitting range for column {col}. "
                    f"Got {len(linrange)} points."
                )

            # --------------------------------------------------------------
            # Fit linear background and optional core scale.
            # Original versions fit against self.J but subtract HF_dataset.C_edges.
            # This keeps that behavior, but applies the fitted scale to HF_dataset.C_edges.
            # --------------------------------------------------------------
            core_fit = self.HF_dataset.J_total[:, col]
            core_subtract = core_total_col

            if fit_core_scale:
                def fitfct(a):
                    background = np.polyval([a[0], a[1]], self.eloss[linrange])
                    return (
                        self.sqw[linrange, col]
                        - background
                        - a[2] * core_fit[linrange]
                    )

                res = optimize.leastsq(fitfct, [0.0, 0.0, 1.0])[0]
                slope = res[0]
                intercept = res[1]
                core_scale = res[2]

            else:
                def fitfct(a):
                    background = np.polyval([a[0], a[1]], self.eloss[linrange])
                    return (
                        self.sqw[linrange, col]
                        - background
                        - core_fit[linrange]
                    )

                res = optimize.leastsq(fitfct, [0.0, 0.0])[0]
                slope = res[0]
                intercept = res[1]
                core_scale = 1.0

            background = np.polyval([slope, intercept], self.eloss)
            self.background[:, col] = background

            # Raw valence profile on current q-specific pz grid.
            val_raw = (
                self.sqw[:, col]
                - background
                - core_scale * core_subtract
            )

            # --------------------------------------------------------------
            # Branch 1: mirror negative-pz side.
            # --------------------------------------------------------------
            if mirror:
                neg_mask = pz_col <= 0.0

                if np.count_nonzero(neg_mask) < 2:
                    raise ValueError(
                        f"Not enough negative-pz points for mirror extraction "
                        f"in column {col}."
                    )

                mirror_pz = np.append(
                    pz_col[neg_mask],
                    -np.flipud(pz_col[neg_mask])
                )
                mirror_val = np.append(
                    val_raw[neg_mask],
                    np.flipud(val_raw[neg_mask])
                )

                extracted = _interp_on_grid(
                    mirror_pz,
                    mirror_val,
                    pz_col,
                    kind="linear",
                    fill_value=0.0,
                )

                asym = np.zeros_like(extracted)
                corrected = extracted.copy()

                if make_plots:
                    plt.figure(figsize=(10, 6))
                    plt.plot(pz_col, val_raw, label="exp. S(q,w) — HF core profile")
                    plt.plot(pz_col, extracted, label="mirrored extracted valence profile")
                    plt.legend()
                    plt.xlabel("pz [a.u.]")
                    plt.ylabel("S(q,w) [1/eV]")
                    plt.title(f"q column {col}: mirrored valence extraction")
                    plt.grid(True, linestyle='--', alpha=0.5, color='gray')
                    plt.tight_layout()
                    plt.draw()

                    if wait_for_input:
                        _ = input("Press [enter] to continue.")

                    plt.close()

            # --------------------------------------------------------------
            # Branch 2: Pearson near-edge replacement plus asymmetry fitting.
            # --------------------------------------------------------------
            else:
                if edge_pz is None:
                    if not make_plots:
                        raise ValueError(
                            "edge_pz must be provided when make_plots=False "
                            "and mirror=False."
                        )

                    print(
                        "Select a point above which the valence profile "
                        "should be replaced by a Pearson function."
                    )

                    plt.figure(figsize=(10, 6))
                    plt.plot(pz_col, val_raw, label="raw valence profile")

                    mask_for_ylim = pz_col < 2.0
                    if np.any(mask_for_ylim):
                        ymin = np.amin(val_raw[mask_for_ylim]) * 0.9
                        ymax = np.amax(val_raw[mask_for_ylim]) * 1.1
                        if ymin < ymax:
                            plt.ylim((ymin, ymax))

                    plt.xlabel("pz [a.u.]")
                    plt.ylabel("S(q,w) [1/eV]")
                    plt.title(f"q column {col}: select Pearson edge boundary")
                    plt.legend()
                    plt.grid(True, linestyle='--', alpha=0.5, color='gray')
                    plt.tight_layout()
                    plt.draw()

                    xyval = plt.ginput(1)[0]
                    edge = xyval[0]
                    plt.close()

                else:
                    edge = edge_pz

                edge_region = np.where(pz_col < edge)[0]

                if len(edge_region) < 5:
                    raise ValueError(
                        f"Not enough points below edge_pz={edge} for Pearson fit "
                        f"in column {col}. Got {len(edge_region)} points."
                    )

                max_idx = edge_region[np.argmax(val_raw[edge_region])]

                start_param = [
                    pz_col[max_idx],
                    4.0,
                    1.0,
                    val_raw[max_idx],
                    0.0,
                ]

                def pearson_fitfct(a):
                    return val_raw[edge_region] - pearson7(
                        pz_col[edge_region],
                        a
                    )

                pearson_param = optimize.leastsq(
                    pearson_fitfct,
                    start_param
                )[0]

                self.pearson_info[col] = {
                    "params": pearson_param,
                    "edge": edge,
                }

                pearson = pearson7(pz_col, pearson_param)

                extracted = val_raw.copy()
                extracted[pz_col > edge] = pearson[pz_col > edge]

                # ----------------------------------------------------------
                # Fit asymmetry.
                # ----------------------------------------------------------
                print("Trying to extract valence asymmetry.")

                pos_mask = pz_col >= 0.0
                neg_mask = pz_col < 0.0

                pzp = pz_col[pos_mask]
                pzm = pz_col[neg_mask]

                jvalp = extracted[pos_mask]

                if len(pzp) < 2 or len(pzm) < 2:
                    print(
                        f"Warning: not enough positive/negative pz points for "
                        f"asymmetry fitting in column {col}. Using zero asymmetry."
                    )
                    asym = np.zeros_like(extracted)

                else:
                    # Mirror negative side onto positive pz.
                    jvalm = _interp_on_grid(
                        -pzm,
                        extracted[neg_mask],
                        pzp,
                        kind="linear",
                        fill_value=0.0,
                    )

                    def asym_model(a, x):
                        eps = 1e-12

                        width_1 = a[1]
                        if abs(width_1) < eps:
                            width_1 = eps if width_1 >= 0 else -eps

                        width_2 = abs(a[2])
                        if width_2 < eps:
                            width_2 = eps

                        return a[0] * (
                            np.tanh(x / width_1)
                            * np.exp(-((x / width_2) ** 4.0))
                        )

                    def asym_fitfct(a):
                        return jvalp - jvalm - asym_model(a, pzp)

                    asym_param = optimize.leastsq(
                        asym_fitfct,
                        [0.0, 1.0, 1.0]
                    )[0]

                    asym = asymmetry_sign * asym_model(asym_param, pz_col) / 2.0

                corrected = extracted - asym

                if make_plots:
                    plt.figure(figsize=(10, 6))
                    plt.plot(pz_col, extracted, label="extracted valence profile")
                    plt.plot(pz_col, asym, label="fitted valence asymmetry")
                    plt.plot(pz_col, corrected, label="asymmetry corrected valence profile")
                    plt.legend()
                    plt.xlabel("pz [a.u.]")
                    plt.ylabel("S(q,w) [1/eV]")
                    plt.title(f"q column {col}: asymmetry correction")
                    plt.grid(True, linestyle='--', alpha=0.5, color='gray')
                    plt.tight_layout()
                    plt.draw()

                    if wait_for_input:
                        _ = input("Press [enter] to continue.")

                    plt.close()

            # --------------------------------------------------------------
            # Interpolate all q-specific results onto master pz grid.
            # This fixes the main inconsistency between the two original methods.
            # --------------------------------------------------------------
            corrected_master = _interp_to_master(
                pz_col,
                corrected,
                kind=final_interp_kind,
                fill_value=0.0,
            )

            asym_master = _interp_to_master(
                pz_col,
                asym,
                kind=final_interp_kind,
                fill_value=0.0,
            )

            self.valencepz[:, col] = corrected_master
            self.valasymmetrypz[:, col] = asym_master

            # Backward-compatible assignment.
            self.valence[:, col] = corrected_master
            self.valasymmetry[:, col] = asym_master

        plt.ioff()


    @staticmethod
    def _as_columns(whichq, n_cols: int) -> list[int]:
        """Normalize one or many column indices to a validated list."""
        if isinstance(whichq, (list, tuple, np.ndarray)):
            columns = [int(c) for c in np.asarray(whichq).ravel()]
        else:
            columns = [int(whichq)]

        if not columns:
            raise ValueError("whichq is empty.")

        bad = [c for c in columns if c < 0 or c >= n_cols]
        if bad:
            raise IndexError(f"Column index out of range: {bad}; valid range is [0, {n_cols - 1}].")

        return columns

    @staticmethod
    def _interp_to_grid(
        x_src,
        y_src,
        x_new,
        *,
        kind: str = "linear",
        fill_value: float = 0.0,
    ):
        """
        Safe 1D interpolation:
        - removes NaN/inf
        - sorts x
        - removes duplicate x
        - downgrades cubic to linear if there are too few points
        """
        x_src = np.asarray(x_src, dtype=float).ravel()
        y_src = np.asarray(y_src, dtype=float).ravel()
        x_new = np.asarray(x_new, dtype=float)

        if x_src.shape != y_src.shape:
            raise ValueError(
                f"x_src and y_src must have the same shape. "
                f"Got {x_src.shape} and {y_src.shape}."
            )

        mask = np.isfinite(x_src) & np.isfinite(y_src)
        x = x_src[mask]
        y = y_src[mask]

        if len(x) < 2:
            return np.full_like(x_new, fill_value, dtype=float)

        order = np.argsort(x)
        x = x[order]
        y = y[order]

        x_unique, unique_idx = np.unique(x, return_index=True)
        y_unique = y[unique_idx]

        if len(x_unique) < 2:
            return np.full_like(x_new, fill_value, dtype=float)

        use_kind = kind
        if kind == "cubic" and len(x_unique) < 4:
            use_kind = "linear"

        f = interpolate.interp1d(
            x_unique,
            y_unique,
            kind=use_kind,
            bounds_error=False,
            fill_value=fill_value,
            assume_sorted=True,
        )
        return f(x_new)

    def get_all_valprof(
        self,
        whichq: int,
        smoothgval: float = 0.0,
        *,
        make_plots: bool = False,
        wait_for_input: bool = False,
        interp_kind: str = "linear",
        q_epsilon: float = 1e-12,
        return_components: bool = False,
    ):
        """
        Transform one extracted valence profile from pz-space onto all q-values.

        Parameters
        ----------
        whichq:
            Column index whose self.valencepz[:, whichq] and
            self.valasymmetrypz[:, whichq] are used as source profiles.

        smoothgval:
            FWHM for Gaussian smoothing. If <= 0, no smoothing is applied.

        make_plots:
            If True, plot transformed valence and asymmetry for each q column.

        wait_for_input:
            If True and make_plots=True, pause after each plot.

        interp_kind:
            Interpolation kind passed to scipy.interpolate.interp1d.
            Usually "linear" is safest.

        q_epsilon:
            Values of abs(qvals) <= q_epsilon are treated as invalid to avoid
            division by zero.

        return_components:
            If True, return intermediate arrays for inspection.

        Updates
        -------
        self.valence:
            Transformed total valence profile on self.eloss grid.

        self.valasymmetry:
            Transformed asymmetry profile on self.eloss grid.
        """
        n_energy = len(self.eloss)
        n_q = len(self.tth)

        whichq = int(whichq)
        if whichq < 0 or whichq >= self.valencepz.shape[1]:
            raise IndexError(
                f"whichq={whichq} is out of range for valencepz with "
                f"{self.valencepz.shape[1]} columns."
            )

        if len(self.pzscale) != self.valencepz.shape[0]:
            raise ValueError(
                "len(self.pzscale) must match self.valencepz.shape[0]."
            )

        newenergy = np.zeros((len(self.pzscale), n_q), dtype=float)
        newvalence = np.zeros((n_energy, n_q), dtype=float)
        newasym = np.zeros((n_energy, n_q), dtype=float)

        was_interactive = plt.isinteractive()
        if make_plots:
            plt.ion()

        try:
            source_valence = np.asarray(self.valencepz[:, whichq], dtype=float)
            source_asym = np.asarray(self.valasymmetrypz[:, whichq], dtype=float)

            for n in range(n_q):
                # Convert common pz-grid to this q-column's energy-loss grid in eV.
                energy_grid = (
                    pz2e1(self.E0, self.pzscale, self.tth[n]) - self.E0
                ) * 1e3

                newenergy[:, n] = energy_grid

                valence_interp = self._interp_to_grid(
                    energy_grid,
                    source_valence,
                    self.eloss,
                    kind=interp_kind,
                    fill_value=0.0,
                )

                asym_interp = self._interp_to_grid(
                    energy_grid,
                    source_asym,
                    self.eloss,
                    kind=interp_kind,
                    fill_value=0.0,
                )

                q = np.asarray(self.HF_dataset.q_vals[:, n], dtype=float)
                valid_q = np.isfinite(q) & (np.abs(q) > q_epsilon)

                if q.shape[0] != n_energy:
                    raise ValueError(
                        f"self.HF_dataset.q_vals[:, {n}] has length {q.shape[0]}, "
                        f"but self.eloss has length {n_energy}."
                    )

                newvalence[valid_q, n] = valence_interp[valid_q] / q[valid_q]
                newasym[valid_q, n] = asym_interp[valid_q] / q[valid_q]

                if make_plots:
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.plot(self.eloss, newvalence[:, n], label="valence")
                    ax.plot(self.eloss, newasym[:, n], label="asymmetry")
                    ax.set_xlabel("Energy loss [eV]")
                    ax.set_ylabel("Intensity")
                    ax.set_title(f"Transformed valence profile, q column {n}")
                    ax.legend()
                    ax.grid(True, linestyle='--', alpha=0.4, color='gray')
                    fig.tight_layout()
                    fig.canvas.draw_idle()

                    if wait_for_input:
                        input("Press [enter] to continue.")

                    plt.close(fig)

            if smoothgval > 0.0:
                smoothed_valence = np.zeros_like(newvalence)
                for n in range(n_q):
                    smoothed_valence[:, n] = convg(
                        self.eloss,
                        newvalence[:, n],
                        smoothgval,
                    )
                self.valence = smoothed_valence + newasym
            else:
                self.valence = newvalence + newasym

            self.valasymmetry = newasym

        finally:
            if make_plots and not was_interactive:
                plt.ioff()

        if return_components:
            return {
                "energy": newenergy,
                "valence_without_asymmetry": newvalence,
                "asymmetry": newasym,
                "total_valence": self.valence,
            }

        return None

    def remv_alence_prof(
        self,
        whichq,
        *,
        eoffset: float = 0.0,
        fit_shift: bool = False,
        initial_scale: float = 1.0,
        initial_shift: float = 0.0,
        fit_start: float | None = None,
        make_plots: bool = False,
        wait_for_input: bool = False,
        update_valence: bool = True,
        interp_kind: str = "linear",
        return_fit: bool = True,
    ):
        """
        Remove fitted valence contribution from selected q columns.

        Model without shift
        -------------------
        signal(E, q) ~= C(E, q) + scale * valence(E, q) + linear_background(E)

        Model with shift
        ----------------
        signal(E, q) ~= C(E, q) + scale * valence(E + shift, q)
                        + linear_background(E)

        Parameters
        ----------
        whichq:
            Column index or list/tuple/array of column indices.

        eoffset:
            Fixed pre-shift applied before fitting.

        fit_shift:
            If False, fit only scale + linear background.
            If True, also fit an additional energy shift.

        initial_scale:
            Initial guess for valence scale.

        initial_shift:
            Initial guess for fitted shift. Only used when fit_shift=True.

        fit_start:
            Lower bound of fitting region in energy loss.
            If None, uses self.prenormrange[0].

        make_plots:
            If True, plot signal, fitted model, background, and residual/core.

        wait_for_input:
            If True and make_plots=True, pause after each plot.

        update_valence:
            If True, replace self.valence[:, col] by the fitted scaled/shifted
            valence profile.

        interp_kind:
            Interpolation kind for applying fixed eoffset.

        return_fit:
            If True, return a dict of fitted parameters for each column.

        Updates
        -------
        self.sqw[:, col]:
            signal - fitted_valence - fitted_background

        self.background[:, col]:
            fitted linear background

        self.valence[:, col]:
            fitted valence profile, if update_valence=True
        """
        n_energy, n_cols = self.signals.shape
        columns = self._as_columns(whichq, n_cols)

        if not hasattr(self, "sqw") or self.sqw.shape != self.signals.shape:
            self.sqw = np.zeros_like(self.signals)

        if not hasattr(self, "background") or self.background.shape != self.signals.shape:
            self.background = np.zeros_like(self.signals)

        if fit_start is None:
            fit_start = self.prenormrange[0]

        inds = np.where(self.eloss >= fit_start)[0]

        n_params = 4 if fit_shift else 3
        if len(inds) < n_params:
            raise ValueError(
                f"Not enough fitting points. Need at least {n_params}, "
                f"got {len(inds)}."
            )

        fit_results = {}

        was_interactive = plt.isinteractive()
        if make_plots:
            plt.ion()

        try:
            for col in columns:
                # Apply fixed external offset first.
                base_valence = self._interp_to_grid(
                    self.eloss + eoffset,
                    self.valence[:, col],
                    self.eloss,
                    kind=interp_kind,
                    fill_value=0.0,
                )

                def shifted_valence(shift: float):
                    return np.interp(
                        self.eloss,
                        self.eloss + shift,
                        base_valence,
                        left=0.0,
                        right=0.0,
                    )

                if fit_shift:
                    def residual(params):
                        scale, shift, slope, intercept = params
                        val = shifted_valence(shift)
                        background = np.polyval([slope, intercept], self.eloss)
                        return (
                            self.signals[inds, col]
                            - self.HF_dataset.C_total[inds, col]
                            - scale * val[inds]
                            - background[inds]
                        )

                    x0 = [initial_scale, initial_shift, 0.0, 0.0]
                    result = optimize.leastsq(residual, x0, full_output=True)
                    params, cov_x, info, message, ier = result

                    scale, shift, slope, intercept = params
                    fitted_valence = scale * shifted_valence(shift)

                else:
                    def residual(params):
                        scale, slope, intercept = params
                        background = np.polyval([slope, intercept], self.eloss)
                        return (
                            self.signals[inds, col]
                            - self.HF_dataset.C_total[inds, col]
                            - scale * base_valence[inds]
                            - background[inds]
                        )

                    x0 = [initial_scale, 0.0, 0.0]
                    result = optimize.leastsq(residual, x0, full_output=True)
                    params, cov_x, info, message, ier = result

                    scale, slope, intercept = params
                    shift = 0.0
                    fitted_valence = scale * base_valence

                background = np.polyval([slope, intercept], self.eloss)
                residual_signal = self.signals[:, col] - fitted_valence - background

                self.sqw[:, col] = residual_signal
                self.background[:, col] = background

                if update_valence:
                    self.valence[:, col] = fitted_valence

                fit_results[col] = {
                    "scale": float(scale),
                    "shift": float(shift),
                    "slope": float(slope),
                    "intercept": float(intercept),
                    "ier": int(ier),
                    "message": message,
                    "residual_norm": float(np.linalg.norm(residual(params))),
                }

                if make_plots:
                    full_model = self.HF_dataset.C_total[:, col] + fitted_valence + background

                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.plot(self.eloss, self.signals[:, col], label="signal")
                    ax.plot(self.eloss, full_model, label="C + valence + background")
                    ax.plot(self.eloss, background, label="background")
                    ax.plot(self.eloss, residual_signal, label="signal - valence - background")
                    ax.plot(self.eloss, self.HF_dataset.C_total[:, col], label="C")
                    ax.set_xlabel("Energy loss [eV]")
                    ax.set_ylabel("Intensity")
                    ax.set_title(f"Valence removal, q column {col}")
                    ax.legend()
                    ax.grid(True, linestyle='--', alpha=0.4, color='gray')
                    fig.tight_layout()
                    fig.canvas.draw_idle()

                    if wait_for_input:
                        input("Press [enter] to continue.")

                    plt.close(fig)

        finally:
            if make_plots and not was_interactive:
                plt.ioff()

        if return_fit:
            return fit_results

        return None
    
    def averageqs(
        self,
        whichq,
        error_weighting: bool = True,
        *,
        min_error: float = 1.0,
        legacy_unweighted_sum: bool = False,
        return_result: bool = False,
        **kwargs,
    ):
        """
        Average S(q,w) over selected q columns.

        Parameters
        ----------
        whichq:
            Column index or list/tuple/array of column indices.

        error_weighting:
            If True, use inverse-variance weighting.

        min_error:
            Replacement value for invalid, zero, or negative errors.

        legacy_unweighted_sum:
            If True, reproduce the old unweighted behavior:
                sqwav = sum(selected columns)
            If False, compute a real arithmetic average.

        return_result:
            If True, return a dictionary with averaged arrays.

        Notes
        -----
        The old misspelled keyword `errorweighing` is still accepted.
        """

        # Backward compatibility with the old misspelled keyword.
        if "errorweighing" in kwargs:
            error_weighting = kwargs.pop("errorweighing")

        if kwargs:
            unknown = ", ".join(kwargs.keys())
            raise TypeError(f"Unknown keyword argument(s): {unknown}")

        if not hasattr(self, "sqw"):
            raise AttributeError("self.sqw does not exist. Run valence removal first.")

        if not hasattr(self, "errors"):
            raise AttributeError("self.errors does not exist.")

        if self.sqw.shape != self.errors.shape:
            raise ValueError(
                f"self.sqw and self.errors must have the same shape. "
                f"Got {self.sqw.shape} and {self.errors.shape}."
            )

        n_energy, n_cols = self.sqw.shape
        columns = self._as_columns(whichq, n_cols)

        av = np.asarray(self.sqw[:, columns], dtype=float)
        averr = np.asarray(self.errors[:, columns], dtype=float).copy()

        # Do not mutate self.errors in place.
        bad_err = (~np.isfinite(averr)) | (averr <= 0.0)
        averr[bad_err] = min_error

        valid_signal = np.isfinite(av)
        valid_err = np.isfinite(averr) & (averr > 0.0)
        valid = valid_signal & valid_err

        if error_weighting:
            weights = np.zeros_like(averr, dtype=float)
            weights[valid] = 1.0 / averr[valid] ** 2.0

            weighted_sum = np.sum(np.where(valid, av * weights, 0.0), axis=1)
            weight_sum = np.sum(weights, axis=1)

            self.sqwav = np.divide(
                weighted_sum,
                weight_sum,
                out=np.zeros(n_energy, dtype=float),
                where=weight_sum > 0.0,
            )

            self.sqwaverr = np.divide(
                1.0,
                np.sqrt(weight_sum),
                out=np.full(n_energy, np.inf, dtype=float),
                where=weight_sum > 0.0,
            )

        else:
            signal_sum = np.sum(np.where(valid_signal, av, 0.0), axis=1)
            count = np.sum(valid_signal, axis=1)

            err_quadrature = np.sqrt(
                np.sum(np.where(valid_err, averr ** 2.0, 0.0), axis=1)
            )

            if legacy_unweighted_sum:
                self.sqwav = signal_sum
                self.sqwaverr = err_quadrature
            else:
                self.sqwav = np.divide(
                    signal_sum,
                    count,
                    out=np.zeros(n_energy, dtype=float),
                    where=count > 0,
                )

                self.sqwaverr = np.divide(
                    err_quadrature,
                    count,
                    out=np.full(n_energy, np.inf, dtype=float),
                    where=count > 0,
                )

        # Keep useful intermediate arrays for inspection/debugging.
        self.avsignals = av
        self.averrors = averr
        self.avcolumns = np.asarray(columns, dtype=int)

        if return_result:
            return {
                "columns": self.avcolumns,
                "sqwav": self.sqwav,
                "sqwaverr": self.sqwaverr,
                "avsignals": self.avsignals,
                "averrors": self.averrors,
            }

        return None

    def savetxtsqwav(
        self,
        filename,
        *,
        emin: float | None = None,
        emax: float | None = None,
        normrange: tuple[float, float] | list[float] | None = None,
        header: str = "eloss sqwav sqwaverr",
        return_data: bool = False,
    ):
        """
        Save averaged S(q,w) as text columns:

            energy loss, sqwav, sqwaverr
        """
        from pathlib import Path

        if not hasattr(self, "sqwav") or not hasattr(self, "sqwaverr"):
            raise AttributeError(
                "self.sqwav and self.sqwaverr must exist. Run averageqs first."
            )

        eloss = np.asarray(self.eloss, dtype=float)
        sqwav = np.asarray(self.sqwav, dtype=float)
        sqwaverr = np.asarray(self.sqwaverr, dtype=float)

        if not (len(eloss) == len(sqwav) == len(sqwaverr)):
            raise ValueError(
                "self.eloss, self.sqwav, and self.sqwaverr must have the same length."
            )

        mask = np.ones_like(eloss, dtype=bool)

        if emin is not None:
            mask &= eloss >= emin

        if emax is not None:
            mask &= eloss <= emax

        if not np.any(mask):
            raise ValueError("No data points selected by emin/emax.")

        data = np.column_stack(
            [
                eloss[mask],
                sqwav[mask],
                sqwaverr[mask],
            ]
        )

        if normrange is not None:
            if not isinstance(normrange, (list, tuple)) or len(normrange) != 2:
                raise ValueError("normrange must be a list or tuple of length 2.")

            nmin, nmax = float(normrange[0]), float(normrange[1])
            norm_mask = (data[:, 0] >= nmin) & (data[:, 0] <= nmax)

            if np.count_nonzero(norm_mask) < 2:
                raise ValueError(
                    f"Not enough points in normrange {normrange} for integration."
                )

            trapz = getattr(np, "trapezoid", np.trapz)
            norm = trapz(data[norm_mask, 1], data[norm_mask, 0])

            if not np.isfinite(norm) or norm == 0.0:
                raise ValueError(f"Invalid normalization factor: {norm}")

            data[:, 1] /= norm
            data[:, 2] /= abs(norm)

        path = Path(filename)
        if path.parent != Path("."):
            path.parent.mkdir(parents=True, exist_ok=True)

        np.savetxt(path, data, header=header)

        if return_data:
            return data

        return None
    
    def plotsqwav(
            self,
            *,
            emin=None,
            emax=None,
            show_error=True,
            normalize=False,
            normrange=None,
            title="Averaged S(q, ω)",
            savepath=None,
        ):
        """
        Plot averaged S(q,w) with optional error band.

        Parameters
        ----------
        emin, emax:
            Optional energy-loss range to plot.

        show_error:
            If True, plot sqwaverr as a shaded error band.

        normalize:
            If True, normalize sqwav and sqwaverr by area over normrange.

        normrange:
            Energy-loss range used for normalization, e.g. (10.0, 80.0).

        savepath:
            If provided, save figure to this path.
        """
        import numpy as np
        import matplotlib.pyplot as plt

        # 全局学术绘图配置（与你所有代码100%风格统一）
        plt.rcParams.update({
            'font.family': 'serif',
            'font.serif': ['Times New Roman'],
            'font.size': 12,
            'axes.labelsize': 14,
            'axes.titlesize': 16,
            'xtick.labelsize': 12,
            'ytick.labelsize': 12,
            'legend.fontsize': 12,
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
        })

        if not hasattr(self, "sqwav") or not hasattr(self, "sqwaverr"):
            raise AttributeError(
                "self.sqwav and self.sqwaverr do not exist. Run averageqs first."
            )

        eloss = np.asarray(self.eloss, dtype=float)
        sqwav = np.asarray(self.sqwav, dtype=float).copy()
        sqwaverr = np.asarray(self.sqwaverr, dtype=float).copy()

        if not (len(eloss) == len(sqwav) == len(sqwaverr)):
            raise ValueError(
                "self.eloss, self.sqwav, and self.sqwaverr must have the same length."
            )

        # Optional normalization.
        if normalize:
            if normrange is None:
                raise ValueError("normrange must be provided when normalize=True.")

            if not isinstance(normrange, (list, tuple)) or len(normrange) != 2:
                raise ValueError("normrange must be a list or tuple of length 2.")

            nmin, nmax = float(normrange[0]), float(normrange[1])
            norm_mask = (eloss >= nmin) & (eloss <= nmax)

            if np.count_nonzero(norm_mask) < 2:
                raise ValueError(f"Not enough points in normrange {normrange}.")

            trapz = getattr(np, "trapezoid", np.trapz)
            norm = trapz(sqwav[norm_mask], eloss[norm_mask])

            if not np.isfinite(norm) or norm == 0.0:
                raise ValueError(f"Invalid normalization factor: {norm}")

            sqwav /= norm
            sqwaverr /= abs(norm)

        # Plot range.
        mask = np.ones_like(eloss, dtype=bool)

        if emin is not None:
            mask &= eloss >= emin

        if emax is not None:
            mask &= eloss <= emax

        if not np.any(mask):
            raise ValueError("No data points selected by emin/emax.")

        x = eloss[mask]
        y = sqwav[mask]
        yerr = sqwaverr[mask]

        # 保留原画布尺寸(7,4.5)，保持原有宽高比
        fig, ax = plt.subplots(figsize=(7, 4.5))

        # 统一曲线样式：标准学术蓝+1.2线宽
        ax.plot(
            x, y,
            color='#1f77b4',
            linewidth=1.2,
            label="averaged S(q, ω)"
        )

        if show_error:
            # 误差带与曲线同色，视觉更统一
            ax.fill_between(
                x,
                y - yerr,
                y + yerr,
                color='#1f77b4',
                alpha=0.25,
                label="±1σ error",
            )

        # 标题与坐标轴（增加标题间距避免拥挤）
        ax.set_xlabel("Energy loss [eV]")
        ax.set_ylabel("S(q, ω)")
        ax.set_title(title, pad=15)

        # 无边框图例（期刊标准）+ 网格置于底层
        ax.legend(frameon=False)
        ax.grid(True, linestyle='--', alpha=0.3, color='lightgray', zorder=0)
        ax.set_axisbelow(True)

        fig.tight_layout()

        if savepath is not None:
            # 全局已配置dpi和bbox，无需重复传参
            fig.savefig(savepath)

        #plt.show()
        plt.close(fig)  # 自动释放内存，避免批量绘图时泄漏

        return fig, ax
