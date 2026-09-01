#!/usr/bin/python
# Filename: xrs_extraction.py

#/*##########################################################################
#
# The XRStools software package for XRS spectroscopy
#
# Copyright (c) 2013-2014 European Synchrotron Radiation Facility
#
# This file is part of the XRStools XRS spectroscopy package developed at
# the ESRF by the DEC and Software group.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
#############################################################################*/
__author__ = "Christoph J. Sahle - ESRF"
__contact__ = "christoph.sahle@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"

import os
from importlib.resources import files
import copy
import numpy as np
from xrsana import xrs_public, xrs_ComptonProfiles
import matplotlib.pyplot as plt

from scipy import interpolate, optimize
from xrsana.math_functions import pearson7, pearson7_zeroback

installation_dir = str(files("xrsana").joinpath("resources"))

HFCP_PATH = os.path.join(installation_dir, 'data', 'ComptonProfiles.dat')
LOGTABLE_PATH = os.path.join(installation_dir, 'data', 'logtable.dat')


def map_chamber_names(name):
    """
    **map_chamber_names**
    Maps names of chambers to range of ROI numbers.
    """
    chamberNames = {'VD':list(range(0,12)),
                    'VU':list(range(12,24)),
                    'VB':list(range(24,36)),
                    'HR':list(range(36,48)),
                    'HL':list(range(48,60)),
                    'HB':list(range(60,72))}
    key = name.upper()
    if key not in chamberNames:
        raise ValueError("Unknown chamber name '%s'. Use one of %s." % (name, sorted(chamberNames)))
    return chamberNames[key]


def _as_columns(whichq):
    if isinstance(whichq, str):
        return map_chamber_names(whichq)
    if np.isscalar(whichq):
        return [int(whichq)]
    return [int(col) for col in whichq]


def _range_indices(eloss, region):
    inds = np.where(np.logical_and(eloss >= region[0], eloss <= region[1]))[0]
    if len(inds) == 0:
        raise ValueError("Energy region %s does not contain any points." % (region,))
    return inds


def _combined_region(eloss, region1, region2=None, weights=None):
    inds1 = _range_indices(eloss, region1)
    if region2 is None:
        return inds1, np.ones(len(inds1))

    inds2 = _range_indices(eloss, region2)
    if weights is None:
        weights = [1.0, 1.0]
    if len(weights) != 2:
        raise ValueError("weights must contain two values.")
    region = np.concatenate((inds1, inds2))
    fit_weights = np.concatenate((
        np.full(len(inds1), float(weights[0])),
        np.full(len(inds2), float(weights[1])),
    ))
    return region, fit_weights


def _trapz(y, x):
    return np.trapezoid(y, x)


def subtract_stray_background(exp_data, ranges, order=1, columns=None):
    """
    Fit and subtract a polynomial stray background from reduced spectra.

    The input object is modified in place and the fitted background matrix is
    returned. ``ranges`` is a list of [emin, emax] windows used for the fit.
    """
    exp_data.signals = np.array(exp_data.signals, dtype=float, copy=True)
    exp_data.errors = np.array(exp_data.errors, dtype=float, copy=True)
    columns = _as_columns(columns if columns is not None else range(exp_data.signals.shape[1]))
    fit_inds = np.concatenate([_range_indices(exp_data.eloss, region) for region in ranges])
    background = np.zeros_like(exp_data.signals, dtype=float)

    for col in columns:
        coeff = np.polyfit(exp_data.eloss[fit_inds], exp_data.signals[fit_inds, col], int(order))
        background[:, col] = np.polyval(coeff, exp_data.eloss)
        exp_data.signals[:, col] = exp_data.signals[:, col] - background[:, col]

    return background


def normalize_to_hf_total(edge_obj, normrange=None, columns=None):
    """
    Normalize experimental spectra to the HF total Compton area.

    This puts the spectra on the same absolute-unit scale as the HF profiles
    held by ``edge_extraction.HF_dataset``. The object is modified in place and
    the applied scale factors are returned.
    """
    columns = _as_columns(columns if columns is not None else range(edge_obj.signals.shape[1]))
    if normrange is None:
        normrange = [edge_obj.eloss[0], edge_obj.eloss[-1]]
    inds = _range_indices(edge_obj.eloss, normrange)
    scales = np.ones(edge_obj.signals.shape[1])

    for col in columns:
        hf_area = _trapz(edge_obj.HF_dataset.J_total[inds, col], edge_obj.eloss[inds])
        exp_area = _trapz(edge_obj.signals[inds, col], edge_obj.eloss[inds])
        if np.isclose(exp_area, 0.0):
            raise ValueError("Cannot normalize analyzer %s: zero experimental area." % col)
        scale = hf_area / exp_area
        edge_obj.signals[:, col] *= scale
        edge_obj.errors[:, col] *= abs(scale)
        scales[col] = scale
    return scales


def _fit_ranges_to_indices(eloss, ranges):
    if ranges is None:
        return np.arange(len(eloss))
    return np.concatenate([_range_indices(eloss, region) for region in ranges])


def fill_profile_with_pearson(x, y, replace_above=None, fit_ranges=None, guess=None):
    """
    Replace the high-x part of a profile by a Pearson VII fit.

    This is useful when the near-edge oscillation masks part of the valence
    profile. If ``replace_above`` is None, the profile is returned unchanged.
    """
    y_filled = y.copy()
    if replace_above is None:
        return y_filled, None

    fit_inds = _fit_ranges_to_indices(x, fit_ranges)
    fit_inds = fit_inds[np.isfinite(y[fit_inds])]
    if len(fit_inds) < 5:
        raise ValueError("Not enough points to fit a Pearson VII valence patch.")

    if guess is None:
        local = fit_inds[np.argmax(y[fit_inds])]
        guess = [x[local], 4.0, 1.0, y[local], 0.0]

    fitfct = lambda a: y[fit_inds] - pearson7(x[fit_inds], a)
    params = optimize.leastsq(fitfct, guess)[0]
    pearson = pearson7(x, params)
    y_filled[x > replace_above] = pearson[x > replace_above]
    return y_filled, params


def fit_empirical_asymmetry(pz, profile, guess=None):
    """
    Fit the empirical odd asymmetry used by the legacy valence extraction.
    """
    if guess is None:
        guess = [0.0, 1.0, 1.0]

    pos = pz >= 0.0
    neg = pz < 0.0
    if not np.any(pos) or not np.any(neg):
        return np.zeros_like(profile), np.array(guess, dtype=float)

    pzp = pz[pos]
    fneg = interpolate.interp1d(-pz[neg], profile[neg], bounds_error=False, fill_value=0.0)
    jvalm = fneg(pzp)
    jvalp = profile[pos]
    model = lambda a, values: a[0]*(np.tanh(values/a[1])*np.exp(-(values/np.absolute(a[2]))**4.0))
    res = optimize.leastsq(lambda a: jvalp - jvalm - model(a, pzp), guess)[0]
    asym = -model(res, pz) / 2.0
    return asym, res


def extract_valence_profile_noninteractive(
    edge_obj,
    source_col,
    element,
    edge,
    linranges=None,
    core_scale=None,
    hfcore_shift=0.0,
    pearson_replace_pz=None,
    pearson_fit_ranges=None,
    pearson_guess=None,
    asymmetry_guess=None,
):
    """
    Extract an experimental valence Compton profile from one high-q spectrum.

    The result is stored in ``edge_obj.valencepz`` and ``edge_obj.valasymmetrypz``
    on the common pz scale, mirroring the legacy interactive method.
    """
    col = _as_columns(source_col)[0]
    pz = xrs_public.e2pz(edge_obj.eloss/1.0e3 + edge_obj.E0, edge_obj.E0, edge_obj.tth[col])[0]
    core = edge_obj._core_profile(element, edge, column=col, shift=hfcore_shift)

    if linranges is None:
        fit_inds = np.where(0.1*core > edge_obj.HF_dataset.V_total[:, col])[0]
        if len(fit_inds) == 0:
            fit_inds = np.arange(len(edge_obj.eloss))
    else:
        fit_inds = _fit_ranges_to_indices(edge_obj.eloss, linranges)

    if core_scale is None:
        fitfct = lambda a: (
            edge_obj.signals[fit_inds, col]
            - np.polyval(a[0:2], edge_obj.eloss[fit_inds])
            - a[2]*core[fit_inds]
        )
        res = optimize.leastsq(fitfct, [0.0, 0.0, 1.0])[0]
        background = np.polyval(res[0:2], edge_obj.eloss)
        core_scale = res[2]
    else:
        bg = np.polyfit(
            edge_obj.eloss[fit_inds],
            edge_obj.signals[fit_inds, col] - core_scale*core[fit_inds],
            1,
        )
        background = np.polyval(bg, edge_obj.eloss)

    raw_valence = edge_obj.signals[:, col] - background - core_scale*core
    filled_valence, pearson_params = fill_profile_with_pearson(
        pz,
        raw_valence,
        replace_above=pearson_replace_pz,
        fit_ranges=pearson_fit_ranges,
        guess=pearson_guess,
    )
    asymmetry, asymmetry_params = fit_empirical_asymmetry(pz, filled_valence, asymmetry_guess)

    order = np.argsort(pz)
    fval = interpolate.interp1d(
        pz[order],
        (filled_valence - asymmetry)[order],
        kind='linear',
        bounds_error=False,
        fill_value=0.0,
    )
    fasym = interpolate.interp1d(
        pz[order],
        asymmetry[order],
        kind='linear',
        bounds_error=False,
        fill_value=0.0,
    )
    edge_obj.valencepz[:, col] = fval(edge_obj.pzscale)
    edge_obj.valasymmetrypz[:, col] = fasym(edge_obj.pzscale)
    edge_obj.valence[:, col] = filled_valence - asymmetry
    edge_obj.valasymmetry[:, col] = asymmetry
    edge_obj.background[:, col] = background
    edge_obj._sync_valence_CP()

    return {
        "pz": pz,
        "raw_valence": raw_valence,
        "filled_valence": filled_valence,
        "asymmetry": asymmetry,
        "background": background,
        "core": core,
        "core_scale": core_scale,
        "pearson_params": pearson_params,
        "asymmetry_params": asymmetry_params,
    }


def transfer_valence_profile(edge_obj, source_col, target_cols=None, smoothgval=0.0):
    """
    Transfer an extracted valence profile from one q to target q values.
    """
    source = _as_columns(source_col)[0]
    columns = _as_columns(target_cols if target_cols is not None else range(edge_obj.signals.shape[1]))
    newvalence = np.zeros_like(edge_obj.signals)
    newasym = np.zeros_like(edge_obj.signals)

    for col in columns:
        newenergy = (xrs_public.pz2e1(edge_obj.E0, edge_obj.pzscale, edge_obj.tth[col]) - edge_obj.E0)*1.0e3
        order = np.argsort(newenergy)
        fval = interpolate.interp1d(newenergy[order], edge_obj.valencepz[:, source][order], bounds_error=False, fill_value=0.0)
        fasym = interpolate.interp1d(newenergy[order], edge_obj.valasymmetrypz[:, source][order], bounds_error=False, fill_value=0.0)
        raw_valence = fval(edge_obj.eloss)
        raw_asym = fasym(edge_obj.eloss)
        qvals = edge_obj.HF_dataset.q_vals[:, col]
        newvalence[:, col] = np.divide(raw_valence, qvals, out=np.zeros_like(raw_valence), where=~np.isclose(qvals, 0.0))
        newasym[:, col] = np.divide(raw_asym, qvals, out=np.zeros_like(raw_asym), where=~np.isclose(qvals, 0.0))

    if smoothgval > 0.0:
        for col in columns:
            edge_obj.valence[:, col] = xrs_public.convg(edge_obj.eloss, newvalence[:, col], smoothgval) + newasym[:, col]
    else:
        edge_obj.valence[:, columns] = newvalence[:, columns] + newasym[:, columns]
    edge_obj.valasymmetry[:, columns] = newasym[:, columns]
    edge_obj._sync_valence_CP()
    return edge_obj.valence[:, columns]


def subtract_valence_profiles(edge_obj, target_cols=None, background_order=1, fitrange=None):
    """
    Subtract transferred valence profiles from target spectra.

    A small polynomial residual background can be fitted in ``fitrange`` after
    valence subtraction. The extracted near-edge spectra are stored in
    ``edge_obj.sqw``.
    """
    columns = _as_columns(target_cols if target_cols is not None else range(edge_obj.signals.shape[1]))
    if fitrange is None:
        fitrange = edge_obj.prenormrange if edge_obj.prenormrange else [edge_obj.eloss[0], edge_obj.eloss[-1]]
    inds = _range_indices(edge_obj.eloss, fitrange)

    for col in columns:
        residual = edge_obj.signals[:, col] - edge_obj.valence[:, col]
        coeff = np.polyfit(edge_obj.eloss[inds], residual[inds], int(background_order))
        background = np.polyval(coeff, edge_obj.eloss)
        edge_obj.background[:, col] = background
        edge_obj.sqw[:, col] = residual - background
    return edge_obj.sqw[:, columns]


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
                edge_keyword = xrs_ComptonProfiles.mapShellNames(edge,xrs_public.element(key))
                for formula in self.formulas:
                    if key in formula:
                        # cp core-edge profile
                        atom_profile = self.HFProfile.FormulaProfiles[formula].AtomProfiles[key]
                        for jj in range(len(self.tth)):
                            self.C_edges[key][edge][:,jj] = np.interp(self.eloss,atom_profile.eloss,atom_profile.CperShell[edge_keyword][:,jj])
                    else:
                        print('Could not find ' + key + ' in any of the provided formulas.')

    def get_J_total_av(self,columns):
        return np.mean(self.J_total[:,_as_columns(columns)], axis=1)

    def get_C_total_av(self,columns):
        return np.mean(self.C_total[:,_as_columns(columns)], axis=1)

    def get_C_total(self,columns):
        return self.get_C_total_av(columns)

    def get_C_edges_av(self,element,edge,columns):
        return np.mean(self.C_edges[element][edge][:,_as_columns(columns)], axis=1)

class valence_CP:
    """
    **valence_CP**
    Class to organize information about extracted experimental valence Compton profiles.
    """
    def __init__(self):
        self.pzscale        = np.flipud(np.arange(-10,10,0.05)) # definition according to Huotari et al, JPCS 62 (2001) 2205
        self.valencepz      = np.array([])
        self.valasymmetrypz = np.array([])
        self.valence        = np.array([])
        self.valasymmetry   = np.array([])

    def get_asymmetry(self):
        return self.valasymmetry

    def get_pzscale(self):
        return self.pzscale

class functorObjectV:

    def __init__( self, y, eloss, hfcore, lam ):
        self.y      =  y
        self.eloss  =  eloss
        self.hfcore =  hfcore
        self.lam    =  lam

    def funct( self, a, eloss ):
        pear = pearson7_zeroback( eloss, a )
        poly = np.polyval( a[4:6], eloss )
        tot  = pear+poly
        return tot

    def __call__( self, x ):
        pea = pearson7_zeroback( self.eloss, x[0:4] )
        if len(x)==7:
            pol = np.polyval(x[4:6],self.eloss)
            hf  = self.hfcore
        else:
            pol = 0
            hf  = 0

        self.hf_fit = hf
        self.fit    = pea+pol+hf
        self.peapol = pea+pol

        diff = self.y*x[6]-self.fit
        # with extra cost for large linear background
        res  = diff/np.sqrt(self.y*x[6]) + self.lam*(x[4]**2+x[5]**2)

        return res

class edge_extraction:
    """
    **edge_extraction**
    Class to destill core edge spectra from x-ray Raman scattering experiments.
    """
    def __init__(self,exp_data, formulas, stoich_weights, edges ,prenormrange=[5,np.inf]):
        # input
        self.eloss   = copy.deepcopy( exp_data.eloss )
        self.signals = copy.deepcopy( exp_data.signals )
        self.errors  = copy.deepcopy( exp_data.errors )
        self.E0      = exp_data.E0
        self.tth     = exp_data.tth
        self.prenormrange = prenormrange
        self.HF_dataset = HF_dataset(exp_data, formulas, stoich_weights, edges)

        # output
        self.background     = np.zeros(np.shape(exp_data.signals))
        self.sqw            = np.zeros(np.shape(exp_data.signals))
        self.sqwav          = np.zeros(np.shape(exp_data.eloss))
        self.sqwaverr       = np.zeros(np.shape(exp_data.eloss))
        self.valence_CP     = valence_CP()
        self.pzscale        = self.valence_CP.pzscale.copy()
        self.valencepz      = np.zeros((len(self.pzscale), self.signals.shape[1]))
        self.valasymmetrypz = np.zeros((len(self.pzscale), self.signals.shape[1]))
        self.valence        = np.zeros(np.shape(exp_data.signals))
        self.valasymmetry   = np.zeros(np.shape(exp_data.signals))
        self._sync_valence_CP()

        # some variables for averaging rawdata over analyzers/whole chambers
        self.avsignals  = np.array([])
        self.averrors   = np.array([])
        self.av_C       = {}
        self.av_J       = np.array([])
        self.avqvals    = np.array([])

        # rough normalization over range given by prenormrange
        if prenormrange:
            for n in range(self.signals.shape[1]):
                inds    = np.where(np.logical_and(self.eloss >= prenormrange[0], self.eloss <= prenormrange[1]))[0]
                if len(inds) == 0:
                    continue
                HFnorm  = _trapz(self.HF_dataset.J_total[:,n], self.eloss)
                EXPnorm = np.trapezoid(self.signals[inds,n], self.eloss[inds])
                if not np.isfinite(EXPnorm) or np.isclose(EXPnorm, 0.0):
                    continue
                scale = HFnorm/EXPnorm
                self.signals[:,n] *= scale
                self.errors[:,n]  *= abs(scale)

    def _sync_valence_CP(self):
        self.valence_CP.pzscale = self.pzscale
        self.valence_CP.valencepz = self.valencepz
        self.valence_CP.valasymmetrypz = self.valasymmetrypz
        self.valence_CP.valence = self.valence
        self.valence_CP.valasymmetry = self.valasymmetry

    def _resolve_edge(self, element=None, edge=None):
        if element is None:
            elements = list(self.HF_dataset.edges.keys())
            if len(elements) != 1:
                raise ValueError("element must be provided when more than one element is configured.")
            element = elements[0]
        if element not in self.HF_dataset.edges:
            raise ValueError("Cannot find HF profiles for element '%s'." % element)

        if edge is None:
            edges = self.HF_dataset.edges[element]
            if len(edges) != 1:
                raise ValueError("edge must be provided when more than one edge is configured for '%s'." % element)
            edge = edges[0]
        if edge not in self.HF_dataset.edges[element]:
            raise ValueError("Cannot find HF core profiles for edge '%s' of '%s'." % (edge, element))
        return element, edge

    def _core_profile(self, element=None, edge=None, column=None, averaged=False, shift=0.0):
        element, edge = self._resolve_edge(element, edge)
        if averaged:
            if edge not in self.av_C.get(element, {}):
                raise ValueError("Use 'analyzerAverage' before requesting an averaged core profile.")
            core = self.av_C[element][edge]
        else:
            if column is None:
                raise ValueError("column must be provided for a per-analyzer core profile.")
            core = self.HF_dataset.C_edges[element][edge][:,column]
        if shift:
            core = np.interp(self.eloss, self.eloss + shift, core)
        return core

    def truncate(self, emin=None, emax=None, copy_obj=False):
        """
        Truncate spectra and fitted profiles to an energy-loss window.

        All arrays that share the ``eloss`` axis are sliced together, including
        the experimental spectra, extracted spectra, averaged spectra, and
        interpolated HF profiles. The window is inclusive.
        """
        obj = copy.deepcopy(self) if copy_obj else self
        if emin is None:
            emin = obj.eloss[0]
        if emax is None:
            emax = obj.eloss[-1]
        if emin > emax:
            raise ValueError("emin must be smaller than or equal to emax.")

        inds = _range_indices(obj.eloss, [emin, emax])
        old_len = len(obj.eloss)
        obj.eloss = obj.eloss[inds]

        def trim_array(value):
            if isinstance(value, np.ndarray) and value.ndim > 0 and value.shape[0] == old_len:
                return value[inds, ...]
            return value

        for attr in [
            "signals", "errors", "background", "sqw", "sqwav", "sqwaverr",
            "valence", "valasymmetry", "avsignals", "averrors", "av_J",
            "avqvals", "yres", "newspec",
        ]:
            if hasattr(obj, attr):
                setattr(obj, attr, trim_array(getattr(obj, attr)))

        hf = obj.HF_dataset
        hf.eloss = trim_array(hf.eloss)
        for attr in ["J_total", "C_total", "V_total", "q_vals"]:
            setattr(hf, attr, trim_array(getattr(hf, attr)))
        for element in hf.C_edges:
            for edge in hf.C_edges[element]:
                hf.C_edges[element][edge] = trim_array(hf.C_edges[element][edge])

        for element in obj.av_C:
            for edge in obj.av_C[element]:
                obj.av_C[element][edge] = trim_array(obj.av_C[element][edge])

        obj._sync_valence_CP()
        return obj

    def analyzerAverage(self,roi_numbers,errorweighing=True):
        """
        **analyzerAverage**
        Averages signals from several crystals before background subtraction.

        Args:

          * roi_numbers : list, str
              list of ROI numbers to average over of keyword for analyzer chamber
              (e.g. 'VD','VU','VB','HR','HL','HB')

          *  errorweighing : boolean (True by default)
                  keyword if error weighing should be used for the averaging or not

        """    
        columns = _as_columns(roi_numbers)
        if any(col < 0 or col >= self.signals.shape[1] for col in columns):
            raise IndexError("Requested analyzer outside available columns.")

        self.avsignals = np.zeros_like(self.eloss)
        self.averrors  = np.zeros_like(self.eloss)
        self.av_J      = np.zeros_like(self.eloss)
        self.avqvals   = np.zeros_like(self.eloss)

        # build matricies to sum over
        av      = np.zeros((len(self.eloss),len(columns)))
        averr   = np.zeros((len(self.eloss),len(columns)))
        avqvals = np.zeros((len(self.eloss),len(columns)))
        avJ     = np.zeros((len(self.eloss),len(columns)))
        avcvals = {}
        for key in self.HF_dataset.edges:
            avcvals[key] = {}
            for edge in self.HF_dataset.edges[key]:
                avcvals[key][edge] = np.zeros((len(self.eloss),len(columns)))

        # assign the signals to sum over
        for column,ii in zip(columns,list(range(len(columns)))):
            # find data points with error = 0.0 and replace error by 1.0
            inds = np.where(self.errors[:,column] == 0.0)[0]
            self.errors[inds,column] = 1.0
            av[:,ii]      = self.signals[:,column]
            averr[:,ii]   = self.errors[:,column]
            avqvals[:,ii] = self.HF_dataset.q_vals[:,column]
            for key in self.HF_dataset.edges:
                for edge in self.HF_dataset.edges[key]:
                    avcvals[key][edge][:,ii] = self.HF_dataset.C_edges[key][edge][:,column]

        # sum things up
        if errorweighing:
            self.avsignals = np.sum( av/(averr**2.0) ,axis=1)/( np.sum(1.0/(averr**2.0),axis=1))
            self.averrors  = np.sqrt( 1.0/np.sum(1.0/(averr**2.0),axis=1) )
        else: 
            self.avsignals = np.mean(av,axis=1)
            self.averrors  = np.sqrt(np.sum(np.absolute(averr)**2.0,axis=1))/averr.shape[1]

        # average over HF core profiles 
        self.avqvals = np.mean(avqvals,axis=1)
        self.av_J    = np.mean(self.HF_dataset.J_total[:,columns],axis=1)
        for key in self.HF_dataset.edges:
            self.av_C[key] = {}
            for edge in self.HF_dataset.edges[key]:
                self.av_C[key][edge] = np.mean(avcvals[key][edge],axis=1)

    def removePolyCoreAv(self,element,edge,range1,range2,weights=[1,1],guess=[1.0,0.0,0.0],ewindow=100.0):
        """
        **removePolyCoreAv**
        Subtract a polynomial from averaged data guided by the HF core Compton profile.

        Args

          * element : str
            String (e.g. 'Si') for the element you want to work on.
          * edge: str
            String (e.g. 'K' or 'L23') for the edge to extract.
          * range1 : list
            List with start and end value for fit-region 1.
          * range2 : list
            List with start and end value for fit-region 2.
          * weigths : list of ints
            List with weights for the respective fit-regions 1 and 2. Default is [1,1].
          * guess : list
            List of starting values for the fit. Default is [1.0,0.0,0.0] (i.e. a quadratic
            function. Change the number of guess values to get other degrees of polynomials
            (i.e. [1.0, 0.0] for a constant, [1.0,0.0,0.0,0.0] for a cubic, etc.).
            The first guess value passed is for scaling of the experimental data to the HF
            core Compton profile.
          * ewindow: float
            Width of energy window used in the plot. Default is 100.0.

        """
        # check that there are averaged signals available
        if not np.any(self.avsignals):
            print('Found no averaged signals. Use \'analyzerAverage\'-method first to create some averages.')
            return

        # check that desired edge is available
        if not element in list(self.HF_dataset.edges.keys()):
            print('Cannot find HF profiles for desired atom.')
            return
        if not edge in self.HF_dataset.edges[element]:
            print('Cannot find HF core profiles for desired edge.' )
            return

        # define fitting ranges
        region1 = _range_indices(self.eloss, range1)
        region2 = _range_indices(self.eloss, range2)
        region, fit_weights = _combined_region(self.eloss, range1, range2, weights)

        # prepare plotting window
        plt.ion()
        plt.cla()

        # get the HF core spectrum
        HF_core = self.av_C[element][edge]

        # estimate start value for scaling parameter
        HF_core_norm = _trapz(HF_core[region2],self.eloss[region2])
        exp_norm     = _trapz(self.avsignals[region2],self.eloss[region2])
        self.avsignals *= HF_core_norm/exp_norm

        # define fit-function, boundaries, and constraints
        cons    =  ({'type': 'eq',   'fun': lambda x: np.trapezoid(HF_core[region2],self.eloss[region2]) - 
                                                        np.trapezoid(x[0]*self.avsignals[region2]-HF_core[region2] - 
                                                                np.polyval(x[1::],self.eloss[region2]),self.eloss[region2] )  },
                    {'type': 'eq',   'fun': lambda x: np.trapezoid( np.abs( x[0]*self.avsignals[region1] - HF_core[region1] - 
                                                                np.polyval(x[1::],self.eloss[region1]),self.eloss[region1] )) },
                    {'type': 'ineq', 'fun': lambda x: x[0]})

        fitfct = lambda a: np.sum( fit_weights*(a[0]*self.avsignals[region] - HF_core[region] - np.polyval(a[1::],self.eloss[region]) )**2.0 )
        res    = optimize.minimize(fitfct, guess, method='SLSQP',constraints=cons).x
        print( 'The fit parameters are: ', res)

        yres = np.polyval(res[1::], self.eloss)
        plt.plot(self.eloss,HF_core)    
        plt.plot(self.eloss,self.avsignals*res[0], self.eloss,yres+HF_core, self.eloss,self.avsignals*res[0]-yres)
        plt.legend(['HF core','scaled signal','poly-fit + core','scaled signal - poly'])
        plt.xlabel('energy loss [eV]')
        plt.ylabel('signal [a.u.]')
        plt.xlim(range1[0]-ewindow,range2[1]+ewindow) 
        plt.autoscale(enable=True, axis='y')
        plt.draw()

        self.sqwav    = self.avsignals*res[0] - yres
        self.sqwaverr = self.averrors*res[0]

    def removeCorePearsonAv(self,element,edge,range1,range2,weights=[2,1],HFcore_shift=0.0,guess=None,scaling=None,return_background=False):
        """
        **removeCorePearsonAv**
        """
        # check that there are averaged signals available
        if not np.any(self.avsignals):
            print('Found no averaged signals. Use \'analyzerAverage\'-method first to create some averages.')
            return

        # check that desired edge is available
        if not element in list(self.HF_dataset.edges.keys()):
            print('Cannot find HF profiles for desired atom.') # @@@@@@@@@@@@@@@@@@@ raise
            return
        if not edge in self.HF_dataset.edges[element]:
            print('Cannot find HF core profiles for desired edge.' ) # @@@@@@@@@@@@ raise
            return

        # define fitting ranges
        region1 = _range_indices(self.eloss, range1)
        region2 = _range_indices(self.eloss, range2)
        region, fit_weights = _combined_region(self.eloss, range1, range2, weights)

        # find indices for guessing start values from HF J_total
        fitfct = lambda a: np.sum( fit_weights*(self.av_J[region] - pearson7_zeroback(self.eloss,a)[region] -
                                    np.polyval(a[4:6],self.eloss[region]) )**2.0 )
        guess1 = optimize.minimize(fitfct, [1.0,1.0,1.0,1.0], method='SLSQP').x
        #guessregion = np.where(np.logical_and(self.eloss>=self.prenormrange[0],self.eloss<=self.prenormrange[1]))[0]
        #if not guess: 
        #    guess = []
        #    ind   = self.avsignals[guessregion].argmax(axis=0) # find index of maximum of signal in "prenormrange" (defalt [5,inf])
        #    guess.append(self.eloss[guessregion][ind]) # max of signal (in range of prenorm from __init__)
        #    guess.append(guess[0]*1.0) # once the position of the peason maximum
        #    guess.append(1.0) # pearson shape, 1 = Lorentzian, infinite = Gaussian
        #    guess.append(self.avsignals[guessregion][ind]) # Peak intensity
        #    guess.append(0.0) # linear slope
        #    guess.append(0.0) # no background
        #    guess.append(1.0) # scaling factor for exp. data
        if not guess:
            guess = guess1
            guess = np.append(guess,[0.0,0.0,1.0]) # append starting values for linear and scaling

        # manage some plotting things
        plt.ion()
        plt.cla()

        # get the HF core spectrum
        HF_core = np.interp(self.eloss,self.eloss+HFcore_shift,self.av_C[element][edge])

        # approximately scale data in post-edge region
        HF_core_norm = _trapz(HF_core[region2],self.eloss[region2])
        exp_norm     = _trapz(self.avsignals[region2],self.eloss[region2])
        the_signals  = self.avsignals*HF_core_norm/exp_norm

        if scaling:
            bnds = ((None, None), (0, None), (0, None), (0, None), (0, None), (0, None), (scaling-scaling*0.001, scaling+scaling*0.001))
        else:
            scaling = 1.0
            bnds = ((None, None), (0, None), (0, None), (0, None), (0, None), (0, None), (0, None))

        cons    =  (#{'type': 'ineq', 'fun': lambda x:  x[2]  },
                    #{'type': 'ineq', 'fun': lambda x:  x[3]  },
                    #{'type': 'ineq', 'fun': lambda x:  x[6]  },
                    {'type': 'eq',   'fun': lambda x:  np.trapezoid(np.abs(scaling*the_signals[region1] - 
                                                                pearson7_zeroback(self.eloss[region1],x[0:4]) - 
                                                                np.polyval(x[4:6],self.eloss[region1]) - 
                                                                HF_core[region1] ),
                                                                self.eloss[region1]  ) },
                    {'type': 'eq',   'fun': lambda x:  np.trapezoid(scaling*the_signals[region2] - 
                                                                pearson7_zeroback(self.eloss[region2],x[0:4]) - 
                                                                np.polyval(x[4:6],self.eloss[region2])-HF_core[region2],
                                                                self.eloss[region2])})

        fitfct = lambda a: np.sum( fit_weights*(scaling*the_signals[region] -
                                    pearson7_zeroback(self.eloss[region],a[0:4]) - 
                                    np.polyval(a[4:6],self.eloss[region]) - 
                                    HF_core[region] )**2.0 )

        res    = optimize.minimize(fitfct, guess, method='SLSQP', bounds=bnds, constraints=cons).x
        print( 'The fit parameters are: ', res)

        yres    = pearson7_zeroback(self.eloss,res[0:4]) + np.polyval(res[4:6],self.eloss)
        plt.plot(self.eloss,the_signals*scaling,self.eloss,yres+HF_core,self.eloss,the_signals*scaling-yres,self.eloss,HF_core)
        plt.legend(('scaled data','pearson + linear + core','data - (pearson + linear)','core'))
        plt.draw()

        self.sqwav = the_signals*scaling - yres
        self.sqwaverr = self.averrors*scaling*HF_core_norm/exp_norm

        if return_background:
            return self.eloss, yres, HF_core

    def removePearsonAv(self,element,edge,range1,range2=None,weights=[2,1],guess=None,scale=1.0,HFcore_shift=0.0):
        """
        **removePearsonAv**
        """
        # check that there are averaged signals available
        if not np.any(self.avsignals):
            print('Found no averaged signals. Use \'analyzerAverage\'-method first to create some averages.')
            return

        # define fitting ranges
        region1 = _range_indices(self.eloss, range1)
        if range2:
            region, fit_weights = _combined_region(self.eloss, range1, range2, weights)
        else:
            region  = region1
            fit_weights = np.ones(len(region))

        guessregion = np.where(np.logical_and(self.eloss>=self.prenormrange[0],self.eloss<=self.prenormrange[1]))[0]
        if not guess: 
            guess = []
            ind   = self.avsignals[guessregion].argmax(axis=0) # find index of maximum of signal in "prenormrange" (defalt [5,inf])
            guess.append(self.eloss[guessregion][ind]) # max of signal (in range of prenorm from __init__)
            guess.append(guess[0]*2.0) # once the position of the peason maximum
            guess.append(1.0) # pearson shape, 1 = Lorentzian, infinite = Gaussian
            guess.append(self.avsignals[guessregion][ind]) # Peak intensity
            guess.append(0.0) # no background

        # scale data by hand
        thespec = self.avsignals * scale

        # get the HF core spectrum
        HF_core = np.interp(self.eloss,self.eloss+HFcore_shift,self.av_C[element][edge])

        # define fitfunction
        fitfct  = lambda a: np.sum( fit_weights*(thespec[region] - pearson7_zeroback(self.eloss[region],a[0:4]) - np.polyval(a[4:6],self.eloss[region]) - HF_core[region])**2.0 )

        res = optimize.minimize(fitfct,guess).x
        print( 'the fitting results are: ', res)

        yres = pearson7_zeroback(self.eloss,res[0:4]) + np.polyval(res[4:6],self.eloss)
        plt.cla()
        plt.plot(self.eloss,thespec,self.eloss,yres,self.eloss,thespec-yres,self.eloss,HF_core)
        plt.legend(('data','pearson fit','data - pearson','core'))
        plt.draw()

        self.sqwav = thespec-yres
        self.sqwaverr = self.averrors * scale

    def areanorm(self,whichq,emin=None,emax=None):
        """
        Normalize selected analyzer spectra to unit area.
        """
        columns = _as_columns(whichq)
        if emin is None:
            emin = self.eloss[0]
        if emax is None:
            emax = self.eloss[-1]
        inds = _range_indices(self.eloss, [emin, emax])

        for col in columns:
            norm = _trapz(self.signals[inds,col], self.eloss[inds])
            if np.isclose(norm, 0.0):
                raise ValueError("Cannot area-normalize analyzer %s: zero area." % col)
            self.signals[:,col] /= norm
            self.errors[:,col]  /= abs(norm)

    def energycorrect(self,whichq,alpha,densities,samthickness):
        """
        Apply absorption/self-absorption and Compton cross-section corrections.
        """
        columns = _as_columns(whichq)
        if np.isscalar(densities):
            densities = [densities]

        mu_in, mu_out = xrs_public.mpr_compds(
            self.eloss/1.0e3 + self.E0,
            self.HF_dataset.formulas,
            self.HF_dataset.stoich_weights,
            self.E0,
            densities,
        )

        for col in columns:
            if alpha >= 0:
                beta = abs(180.0 - alpha - self.tth[col])
            else:
                beta = -abs(abs(alpha) - self.tth[col])

            ac = xrs_public.abscorr2(mu_in, mu_out, alpha, beta, samthickness)
            _pz, cf = xrs_public.e2pz(self.E0 + self.eloss/1.0e3, self.E0, self.tth[col])
            correction = ac*cf
            self.signals[:,col] *= correction
            self.errors[:,col]  *= np.abs(correction)

            if self.prenormrange:
                inds = np.where(np.logical_and(
                    self.eloss >= self.prenormrange[0],
                    self.eloss <= self.prenormrange[1],
                ))[0]
                if len(inds) == 0:
                    continue
                hfnorm = _trapz(self.HF_dataset.J_total[:,col], self.eloss)
                expnorm = _trapz(self.signals[inds,col], self.eloss[inds])
                if np.isfinite(expnorm) and not np.isclose(expnorm, 0.0):
                    scale = hfnorm/expnorm
                    self.signals[:,col] *= scale
                    self.errors[:,col]  *= abs(scale)

    def removeelastic(self,whichq,range1,range2,guess=None,stoploop=True,overwrite=False):
        """
        Fit and optionally subtract an elastic Pearson tail from raw spectra.
        """
        columns = _as_columns(whichq)
        region = np.concatenate((
            _range_indices(self.eloss, range1),
            _range_indices(self.eloss, range2),
        ))

        plt.ion()
        for col in columns:
            col_guess = guess
            if col_guess is None:
                imax = self.signals[:,col].argmax(axis=0)
                col_guess = [self.eloss[imax], 1.0, 1.0, self.signals[imax,col], 0.0]

            fitfct = lambda a: self.signals[region,col] - pearson7(self.eloss[region], a)
            res = optimize.leastsq(fitfct, col_guess)[0]
            yres = pearson7(self.eloss, res)

            plt.cla()
            plt.plot(self.eloss, self.signals[:,col], self.eloss, yres, self.eloss, self.signals[:,col]-yres)
            plt.legend(('data','pearson fit','data - pearson'))
            plt.draw()

            self.background[:,col] = yres
            self.sqw[:,col] = self.signals[:,col] - yres
            if overwrite:
                self.signals[:,col] = self.sqw[:,col]
            if stoploop:
                input("Press [enter] to continue.")
            plt.close()
        plt.ioff()

    def removeconstav(self,emin,emax,ewindow=100.0):
        """
        Fit a constant background to averaged data.
        """
        if not np.any(self.avsignals):
            raise ValueError("Use 'analyzerAverage' first to create averages.")

        inds = _range_indices(self.eloss, [emin, emax])
        res = np.polyfit(self.eloss[inds], self.avsignals[inds], 0)
        yres = np.polyval(res, self.eloss)

        plt.ion()
        plt.cla()
        plt.plot(self.eloss,self.avsignals,self.eloss,yres,self.eloss,self.avsignals-yres)
        plt.legend(('signal','constant fit','signal - constant'))
        plt.xlabel('energy loss [eV]')
        plt.ylabel('signal [a.u.]')
        plt.xlim(emin-ewindow,emax+ewindow)
        plt.autoscale(enable=True, axis='y', tight=False)
        plt.draw()

        self.sqwav = self.avsignals - yres
        self.sqwaverr = self.averrors
        self.yres = yres
        self.newspec = self.sqwav

    def removeLinearAv(self,region1,region2=None,ewindow=100.0,scale=1,view=False):
        """
        Fit a linear background to averaged data.
        """
        if not np.any(self.avsignals):
            raise ValueError("Use 'analyzerAverage' first to create averages.")

        region, fit_weights = _combined_region(self.eloss, region1, region2)
        res = np.polyfit(self.eloss[region], self.avsignals[region], 1, w=np.sqrt(fit_weights))
        yres = np.polyval(res, self.eloss)
        newspec = (self.avsignals - yres)*scale
        newerrs = self.averrors*abs(scale)

        self.sqwav = newspec
        self.sqwaverr = newerrs
        self.yres = yres
        self.newspec = newspec

        if view:
            plt.ion()
            plt.cla()
            plt.plot(self.eloss,self.avsignals,self.eloss,yres,self.eloss,newspec)
            plt.legend(('signal','linear fit','signal - linear'))
            plt.xlabel('energy loss [eV]')
            plt.ylabel('signal [a.u.]')
            plt.grid(False)
            plt.xlim(region1[0]-ewindow,region1[1]+ewindow)
            plt.autoscale(enable=True, axis='y')
            plt.draw()

    def removepolyav1(self,polyregion1,polyregion2=None,polyorder=2.0,weights=[1,1]):
        """
        Fit a polynomial background to averaged data.
        """
        if not np.any(self.avsignals):
            raise ValueError("Use 'analyzerAverage' first to create averages.")

        region, fit_weights = _combined_region(self.eloss, polyregion1, polyregion2, weights)
        res = np.polyfit(self.eloss[region], self.avsignals[region], int(polyorder), w=np.sqrt(fit_weights))
        yres = np.polyval(res, self.eloss)

        self.sqwav = self.avsignals - yres
        self.sqwaverr = self.averrors
        self.yres = yres
        self.newspec = self.sqwav

        plt.ion()
        plt.cla()
        plt.plot(self.eloss,self.avsignals,self.eloss,yres,self.eloss,self.sqwav)
        plt.legend(('signal','poly fit','signal - poly'))
        plt.xlabel('energy loss [eV]')
        plt.ylabel('signal [a.u.]')
        plt.draw()

    def remquickval(self,whichq,corefitrange,interpolrange,convwidth,stoploop=True,element=None,edge=None):
        """
        Quickly estimate and subtract a valence profile from individual spectra.
        """
        columns = _as_columns(whichq)
        fitrange = _range_indices(self.eloss, corefitrange)
        left = np.where(self.eloss <= interpolrange[0])[0]
        right = np.where(self.eloss >= interpolrange[1])[0]
        interprange = np.concatenate((left, right))
        if len(interprange) == 0:
            raise ValueError("interpolrange leaves no points for interpolation.")

        plt.ion()
        for col in columns:
            core = self._core_profile(element, edge, column=col)
            fitfct = lambda a: np.sum((self.signals[fitrange,col] - a[0]*core[fitrange])**2.0)
            constr = lambda a: a[0]
            scale = optimize.fmin_cobyla(fitfct, [1.0], cons=constr, disp=0)[0]
            if np.isclose(scale, 0.0):
                raise ValueError("Cannot remove valence profile for analyzer %s: zero core scale." % col)

            f = interpolate.interp1d(
                self.eloss[interprange],
                self.signals[interprange,col] - scale*core[interprange],
                bounds_error=False,
                fill_value=0.0,
            )
            valdata = f(self.eloss)
            if convwidth > 0.0:
                valdata = xrs_public.convg(self.eloss, valdata, convwidth)
            subdata = self.signals[:,col] - valdata

            self.valence[:,col] = valdata
            self.sqw[:,col] = subdata/scale
            self.background[:,col] = valdata

            plt.cla()
            plt.plot(self.eloss,self.signals[:,col],self.eloss,core*scale,self.eloss,valdata,self.eloss,subdata)
            plt.legend(('data','scaled core compton','estimated valence','extracted data'))
            plt.draw()
            if stoploop:
                input("Press [enter] to continue.")
            plt.close()
        self._sync_valence_CP()
        plt.ioff()

    def extractval_test(self,whichq,mirror=False,linrange1=None,linrange2=None,element=None,edge=None):
        """
        Extract valence profiles on the common pz grid.
        """
        columns = _as_columns(whichq)

        plt.cla()
        plt.ion()
        for col in columns:
            pz_dmy = xrs_public.e2pz(self.eloss/1.0e3+self.E0, self.E0, self.tth[col])[0]
            
            core = self._core_profile(element, edge, column=col)

            if linrange1 and linrange2:
                linrange = np.concatenate((
                    _range_indices(self.eloss, linrange1),
                    _range_indices(self.eloss, linrange2),
                ))
            elif linrange1:
                linrange = _range_indices(self.eloss, linrange1)
            else:
                linrange = np.where(0.1*core > self.HF_dataset.V_total[:,col])[0]
                if len(linrange) == 0:
                    linrange = np.arange(len(self.eloss))

            fitfct = lambda a: (
                self.signals[linrange,col]
                - np.polyval(a[0:2], self.eloss[linrange])
                - a[2]*self.HF_dataset.J_total[linrange,col]
            )
            res = optimize.leastsq(fitfct, [0.0, 0.0, 1.0])[0]
            self.background[:,col] = np.polyval(res[0:2], self.eloss)
            val = self.signals[:,col] - self.background[:,col] - res[2]*core

            if mirror:
                neg = pz_dmy <= 0.0
                mirrorval = np.concatenate((val[neg], np.flipud(val[neg])))
                mirrorpz = np.concatenate((pz_dmy[neg], np.flipud(-pz_dmy[neg])))
                order = np.argsort(mirrorpz)
                f = interpolate.interp1d(mirrorpz[order], mirrorval[order], bounds_error=False, fill_value=0.0)
                extractedval = f(pz_dmy)
                asym = np.zeros_like(extractedval)
            else:
                print('select a point above which the valence profile should be replaced by a Pearson function')
                plt.cla()
                plt.plot(pz_dmy, val)
                finite = np.isfinite(val)
                if np.any(finite):
                    plt.ylim((np.nanmin(val[finite])*0.9, np.nanmax(val[finite])*1.1))
                plt.draw()
                xyval = plt.ginput(1)[0]
                edgeregion = np.where(pz_dmy < xyval[0])[0]
                if len(edgeregion) == 0:
                    raise ValueError("Selected pz point leaves no edge-fit region.")
                start_param = [
                    pz_dmy[edgeregion][np.argmax(val[edgeregion])],
                    4.0,
                    1.0,
                    np.max(val[edgeregion]),
                    0.0,
                ]
                fitfct = lambda a: val[edgeregion] - pearson7(pz_dmy[edgeregion], a)
                param = optimize.leastsq(fitfct, start_param)[0]
                pearson = pearson7(pz_dmy, param)
                val[pz_dmy > xyval[0]] = pearson[pz_dmy > xyval[0]]
                extractedval = val

                pzp = pz_dmy[pz_dmy >= 0.0]
                pzm = pz_dmy[pz_dmy < 0.0]
                jvalp = extractedval[pz_dmy >= 0.0]
                f = interpolate.interp1d(-pzm, extractedval[pz_dmy < 0.0], bounds_error=False, fill_value=0.0)
                jvalm = f(pzp)
                fitfct = lambda a: jvalp-jvalm - a[0]*(np.tanh(pzp/a[1])*np.exp(-(pzp/np.absolute(a[2]))**4.0))
                asym_res = optimize.leastsq(fitfct, [0.0, 1.0, 1.0])[0]
                asym = -(asym_res[0]*(np.tanh(pz_dmy/asym_res[1])*np.exp(-(pz_dmy/np.absolute(asym_res[2]))**4.0)))/2.0

            order = np.argsort(pz_dmy)
            f = interpolate.interp1d(pz_dmy[order], (extractedval-asym)[order], kind='linear', bounds_error=False, fill_value=0.0)
            self.valencepz[:,col] = f(self.pzscale)
            f = interpolate.interp1d(pz_dmy[order], asym[order], kind='linear', bounds_error=False, fill_value=0.0)
            self.valasymmetrypz[:,col] = f(self.pzscale)
            self.valence[:,col] = extractedval-asym
            self.valasymmetry[:,col] = asym
        self._sync_valence_CP()
        plt.ioff()
        return pz_dmy, val

    def getallvalprof(self,whichq,smoothgval=0.0,stoploop=True):
        """
        Transform an extracted valence profile onto all analyzer q-values.
        """
        source_cols = _as_columns(whichq)
        if len(source_cols) != 1:
            raise ValueError("getallvalprof expects one source analyzer.")
        source = source_cols[0]

        newvalence = np.zeros_like(self.signals)
        newasym = np.zeros_like(self.signals)
        plt.ion()
        for col in range(len(self.tth)):
            newenergy = (xrs_public.pz2e1(self.E0, self.pzscale, self.tth[col]) - self.E0)*1.0e3
            order = np.argsort(newenergy)
            f = interpolate.interp1d(newenergy[order], self.valencepz[:,source][order], bounds_error=False, fill_value=0.0)
            raw_valence = f(self.eloss)
            f = interpolate.interp1d(newenergy[order], self.valasymmetrypz[:,source][order], bounds_error=False, fill_value=0.0)
            raw_asym = f(self.eloss)

            qvals = self.HF_dataset.q_vals[:,col]
            newvalence[:,col] = np.divide(raw_valence, qvals, out=np.zeros_like(raw_valence), where=~np.isclose(qvals, 0.0))
            newasym[:,col] = np.divide(raw_asym, qvals, out=np.zeros_like(raw_asym), where=~np.isclose(qvals, 0.0))

            if stoploop:
                plt.cla()
                plt.plot(self.eloss,newvalence[:,col],self.eloss,newasym[:,col])
                plt.draw()
                input("Press [enter] to continue.")
                plt.close()

        if smoothgval > 0.0:
            for col in range(len(self.tth)):
                self.valence[:,col] = xrs_public.convg(self.eloss,newvalence[:,col],smoothgval) + newasym[:,col]
            self.valasymmetry = newasym
        else:
            self.valence = newvalence + newasym
            self.valasymmetry = newasym
        self._sync_valence_CP()
        plt.ioff()

    def remvalenceprof_test(self,whichq,eoffset=0.0,element=None,edge=None):
        """
        Remove extracted valence profiles from selected analyzers.
        """
        columns = _as_columns(whichq)
        inds = np.where(self.eloss >= self.prenormrange[0])[0]
        if len(inds) == 0:
            inds = np.arange(len(self.eloss))

        plt.ion()
        for col in columns:
            core = self._core_profile(element, edge, column=col)
            f = interpolate.interp1d(self.eloss+eoffset, self.valence[:,col], bounds_error=False, fill_value=0.0)
            thevalence = f(self.eloss)

            fitfct = lambda a: (
                self.signals[inds,col]
                - core[inds]
                - a[0]*np.interp(self.eloss, self.eloss+a[1], thevalence)[inds]
                - np.polyval(a[2:4], self.eloss[inds])
            )
            res = optimize.leastsq(fitfct, [1.0, 0.0, 0.0, 0.0])[0]
            shifted_valence = res[0]*np.interp(self.eloss, self.eloss+res[1], thevalence)
            background = np.polyval(res[2:4], self.eloss)

            self.sqw[:,col] = self.signals[:,col] - shifted_valence - background
            self.valence[:,col] = shifted_valence
            self.background[:,col] = background

            plt.cla()
            plt.plot(self.eloss,self.signals[:,col])
            plt.plot(self.eloss,core+shifted_valence+background)
            plt.plot(self.eloss,background)
            plt.plot(self.eloss,self.sqw[:,col],self.eloss,core)
            plt.draw()
        self._sync_valence_CP()
        plt.ioff()

    def averageqs(self,whichq,errorweighing=True):
        """
        Average extracted S(q,w) over analyzers or chambers.
        """
        columns = _as_columns(whichq)
        av = np.zeros((len(self.eloss), len(columns)))
        averr = np.zeros((len(self.eloss), len(columns)))
        for idx, col in enumerate(columns):
            zero_errs = np.where(self.errors[:,col] == 0.0)[0]
            self.errors[zero_errs,col] = 1.0
            av[:,idx] = self.sqw[:,col]
            averr[:,idx] = self.errors[:,col]

        if errorweighing:
            weights = 1.0/(averr**2.0)
            self.sqwav = np.sum(av*weights, axis=1)/np.sum(weights, axis=1)
            self.sqwaverr = np.sqrt(1.0/np.sum(weights, axis=1))
        else:
            self.sqwav = np.mean(av, axis=1)
            self.sqwaverr = np.sqrt(np.sum(np.absolute(averr)**2.0, axis=1))/len(columns)

    def save_state_hdf5(self,filename,groupname,comment=""):
        """
        Save extracted spectra and averaged HF profiles to an HDF5 group.
        """
        import h5py

        with h5py.File(filename, "a") as h5:
            if groupname in h5:
                del h5[groupname]
            h5group = h5.require_group(groupname)
            for key in [
                "eloss", "sqwav", "sqwaverr", "avsignals", "averrors",
                "avqvals", "av_J", "yres", "newspec", "valence",
                "valasymmetry", "valencepz", "valasymmetrypz", "pzscale",
            ]:
                if hasattr(self, key):
                    h5group[key] = getattr(self, key)

            if self.av_C:
                av_c_group = h5group.require_group("av_C")
                for element, edge_profiles in self.av_C.items():
                    element_group = av_c_group.require_group(element)
                    for edge, profile in edge_profiles.items():
                        element_group[edge] = profile
            h5group.attrs["comment"] = comment

    def savetxtsqwav(self,filename,emin=None,emax=None,normrange=None):
        """
        Compatibility alias for save_average_Sqw.
        """
        return self.save_average_Sqw(filename, emin=emin, emax=emax, normrange=normrange)

    def save_average_Sqw(self,filename, emin=None, emax=None, normrange=None):
        """
        **save_average_Sqw**
        Save the S(q,w) into a ascii file (energy loss, S(q,w), Poisson errors).

        Args:
          * filename : str
            Filename for the ascii file.
          * emin : float
            Use this to save only part of the spectrum.
          * emax : float
            Use this to save only part of the spectrum.
          * normrange : list of floats
            E_start and E_end for possible area-normalization before saving.

        """
        # check that there are is an extracted S(q,w) available
        if not np.any(self.sqwav):
            print('Found no extracted S(q,w).')
            return

        if emin is not None and emax is not None:
            inds = _range_indices(self.eloss, [emin, emax])
            data = np.zeros((len(inds),3))
            data[:,0] = self.eloss[inds]
            data[:,1] = self.sqwav[inds]
            data[:,2] = self.sqwaverr[inds]
        else:
            data = np.zeros((len(self.eloss),3))
            data[:,0] = self.eloss
            data[:,1] = self.sqwav
            data[:,2] = self.sqwaverr
        if normrange:
            #assert type(normrange) is list and len(normrange) is 2, "normrange has to be a list of length two!"
            assert isinstance(normrange, list) and len(normrange) == 2, "normrange has to be a list of length two!"
            inds = _range_indices(data[:,0], normrange)
            norm = _trapz(data[inds,1],data[inds,0])
            if np.isclose(norm, 0.0):
                raise ValueError("Cannot normalize saved S(q,w): zero area.")
            data[:,1] /= norm
            data[:,2] /= norm
        np.savetxt(filename,data)



    def removeCorePearsonAv_new( self, element, edge, range1, range2,
                                     HFcore_shift=0.0, guess=None, scaling=None,
                                     return_background=False, reg_lam=10):
        """
        **removeCorePearsonAv_new**
        """
        # check that there are averaged signals available
        if not np.any(self.avsignals):
            raise ValueError('Found no averaged signals. Use \'analyzerAverage\'-method first to create some averages.')

        # check that desired edge is available
        if not element in list(self.HF_dataset.edges.keys()):
            raise ValueError('Cannot find HF profiles for desired atom.')

        if not edge in self.HF_dataset.edges[element]:
            raise ValueError('Cannot find HF core profiles for desired edge.')

        # define fitting ranges
        region1 = _range_indices(self.eloss, range1)
        region2 = _range_indices(self.eloss, range2)
        region  = np.concatenate((region1, region2))

        # get the HF core spectrum
        HF_core = np.interp( self.eloss, self.eloss+HFcore_shift, self.av_C[element][edge] )
        
        y_reg1       = self.avsignals[region1]
        eloss_reg1   = self.eloss[region1]
        HF_core_reg1 = HF_core[region1]
        
        y_reg2       = self.avsignals[region2]
        eloss_reg2   = self.eloss[region2]
        HF_core_reg2 = HF_core[region2]
        
        y_reg       = self.avsignals[region]
        eloss_reg   = self.eloss[region]
        HF_core_reg = HF_core[region]

        HF_reg_max = HF_core[region].max()
        y_reg_max  = y_reg.max()
        fact       =  HF_reg_max/y_reg_max

        if not guess:
            fitfct   = functorObjectV( y_reg, eloss_reg,  HF_core_reg , reg_lam )
            bndsa    = [ -np.inf]+ [ 0  for tmp in range(6)]
            bndsa[5] = -np.inf
            bndsb    = [ np.inf for tmp in range(7)]
            solution = optimize.least_squares(fitfct, np.random.rand(7),method='trf', bounds=[bndsa,bndsb] )
            guess = solution.x
            # print ( "Using following guess parameters: ", solution.x)

        # manage some plotting things
        plt.ion()
        plt.cla()

        # do the actual fit using the guess parameters
        fitfct_res = functorObjectV( y_reg, eloss_reg,  HF_core_reg , reg_lam )  
        bndsa      = [ -np.inf]+ [ 0  for tmp in range(6)]
        bndsa[5]   = -np.inf
        bndsb      = [ np.inf for tmp in range(7)]
        solution   = optimize.least_squares(fitfct_res, guess, method='trf', bounds=[bndsa,bndsb] )
        print ( "Best fit parameters: ", solution.x)

        # plot results
        fitfct_res = functorObjectV( self.avsignals, self.eloss,  HF_core , reg_lam  )
        res = fitfct_res(solution.x)

        x  = self.eloss                   # eloss
        y1 = self.avsignals*solution.x[6] # scaled data
        y2 = fitfct_res.fit               # pearson + linear + core
        y3 = y1 - fitfct_res.peapol       # data - (pearson + linear)
        y4 = HF_core                      # core

        plt.plot( x, y1, x, y2, x, y3, x, y4 )
        plt.legend(('scaled data','pearson + linear + core','data - (pearson + linear)','core'))
        #plt.grid(True, linestyle='--', alpha=0.7, color='gray')
        plt.draw()

        self.sqwav = y3
        self.sqwaverr = self.averrors*solution.x[6]

        if return_background:
            yres = pearson7_zeroback(self.eloss, solution.x[0:4]) + np.polyval(solution.x[4:6], self.eloss)
            return self.eloss, yres, HF_core
