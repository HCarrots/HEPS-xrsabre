#!/usr/bin/python
# Filename: xrs_ComptonProfiles.py

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

import numpy as np
from xrsana import xrs_public
from . import xrs_fileIO

from scipy import interpolate, integrate, constants, optimize
from re import findall
from collections import defaultdict

# default valence energy cutoff value 
VAL_CUTOFF_DEFAULT = 20.0


def list_duplicates(seq):
    tally = defaultdict(list)
    for i,item in enumerate(seq):
        tally[item].append(i)
    return [(key,locs) for key,locs in tally.items()] # if len(locs)>1]

def parseChemFormula(ChemFormula):
    """
    """
    # parse single formula
    all_elements  = []
    all_stoichios = []
    splitted = findall(r'([A-Z][a-z]*)(\d*)',ChemFormula)
    all_elements.extend([element[0] for element in splitted])
    all_stoichios.extend([(int(element[1]) if element[1] else 1) for element in splitted])
    # sort out double appearances of elements
    elements = []
    stoichiometries = []
    duplicates = list_duplicates(all_elements)
    for pair in duplicates:
        elements.append(pair[0])
        stoich = 0
        stoichiometries.append(sum( [all_stoichios[ii] for ii in pair[1]]) )
    return elements, stoichiometries

def getAtomicWeight(Z):
    """Returns the atomic weight.
    """
    return xrs_public.myprho(1.0,Z)[2]

def getAtomicDensity(Z):
    """Returns the atomic density.
    """
    return xrs_public.myprho(1.0,Z)[1]

class AtomProfile:
    """
    **AtomProfile**

    Class to construct and handle Hartree-Fock atomic Compton Profile of a single atoms.

    Attributes:
      * filename : string 
        Path and filename to the HF profile table.
      * element : string
        Element symbol as in the periodic table.
      * elementNr : int
        Number of the element as in the periodic table.
      * shells : list of strings
        Names of the shells.
      * edges : list
        List of edge onsets (eV).
      * C_total : np.array
        Total core Compton profile.
      * J_total : np.array
        Total Compton profile.
      * V_total : np.array
        Total valence Compton profile.
      * CperShell : dict. of np.arrays
        Core Compton profile per electron shell.
      * JperShell : dict. of np.arrays
        Total Compton profile per electron shell.
      * VperShell : dict. of np.arrays
        Valence Compton profile per electron shell.
      * stoichiometry : float, optional
        Stoichiometric weight (default is 1.0).
      * atomic_weight : float
        Atomic weight.
      * atomic_density : float
        Density (g/cm**3).
      * twotheta : float
        Scattering angle 2Th (degrees).
      * alpha : float
        Incident angle (degrees).
      * beta : float
        Exit angle (degrees).
      * thickness : float
        Sample thickness (cm).

    """
    def __init__(self, element, filename, stoichiometry=1.0):
        self.filename  = filename
        self.element   = element
        self.elementNr = xrs_public.element(element)
        self.CP_profile, self.edges, self.occupation_num, self.shells = PzProfile(element,filename) 
        self.eloss     = []
        self.C_total   = []
        self.J_total   = []
        self.V_total   = []
        self.q_vals    = []
        self.CperShell = {}
        self.JperShell = {}
        self.VperShell = {}
        self.stoichiometry  = stoichiometry
        self.atomic_weight  = getAtomicWeight(element)
        self.atomic_density = getAtomicDensity(element)
        self.twotheta  = []
        self.alpha     = []
        self.beta      = []
        self.thickness = []

    def get_stoichiometry(self):
        return self.stoichiometry

    def get_elossProfiles(self,E0, twotheta,correctasym=None,valence_cutoff=VAL_CUTOFF_DEFAULT):
        """
        **get_elossProfiles**
        Convert the HF Compton profile on to energy loss scale.

        Args:
        E0 : float
            Analyzer energy, enery of the scattered r-rays.
        twotheta : float or list of floats
            Scattering angle 2Th.
        correctasym : float, optional
            Scaling factor to be multiplied to the asymmetry. 
        valence_cutoff : float, optional
            Energy cut off as to what is considered the boundary between core and valence.
        """
        # save the parameters
        self.E0 = E0
        # reset self.twotheta
        self.twotheta = []
        if isinstance(twotheta, list) or isinstance(twotheta, np.ndarray):
            self.twotheta.extend(twotheta)
        elif isinstance(twotheta, float):
            self.twotheta.append(twotheta)
        else:
            print('Unsupported type for twotheta argument')
            return

        # do the conversion for the first tth to get the size/shape of things
        enScale, J_total, C_total, V_total, q, J_shell, C_shell, V_shell = elossProfile(self.element,self.filename,E0,self.twotheta[0],correctasym,valence_cutoff)
        # save eloss scale
        self.eloss     = enScale

        # prepare output
        self.C_total = np.zeros((len(enScale),len(self.twotheta)))
        self.J_total = np.zeros((len(enScale),len(self.twotheta)))
        self.V_total = np.zeros((len(enScale),len(self.twotheta)))
        self.q_vals  = np.zeros((len(enScale),len(self.twotheta)))
        for key in C_shell:
            self.CperShell[key] = np.zeros((len(enScale),len(self.twotheta)))
            self.JperShell[key] = np.zeros((len(enScale),len(self.twotheta)))
            self.VperShell[key] = np.zeros((len(enScale),len(self.twotheta)))
            
        # convert everything to eloss scale
        for tth,ii in zip(self.twotheta,list(range(len(self.twotheta)))):
            enScale, J_total, C_total, V_total, q, J_shell, C_shell, V_shell = elossProfile(self.element,self.filename,E0,tth,correctasym,valence_cutoff)
        
            # save the results (all shell dicts have the same keys)
            self.C_total[:,ii] = np.interp(self.eloss,enScale,C_total)*self.stoichiometry
            self.J_total[:,ii] = np.interp(self.eloss,enScale,J_total)*self.stoichiometry
            self.V_total[:,ii] = np.interp(self.eloss,enScale,V_total)*self.stoichiometry
            self.q_vals[:,ii]  = np.interp(self.eloss,enScale,q)
            for key in self.CperShell:
                self.CperShell[key][:,ii] = np.interp(self.eloss,enScale,C_shell[key])
                self.JperShell[key][:,ii] = np.interp(self.eloss,enScale,C_shell[key])
                self.VperShell[key][:,ii] = np.interp(self.eloss,enScale,C_shell[key])

    def absorptionCorrectProfiles(self, alpha, thickness, geometry='transmission'):
        """
        **absorptionCorrectProfiles**

        Apply absorption correction to the Compton profiles on energy loss scale.

        Args:
          * alpha :float
            Angle of incidence (degrees).
          * beta : float
            Exit angle for the scattered x-rays (degrees). If 'beta' is negative, 
            transmission geometry is assumed, if 'beta' is positive, reflection geometry.
          * thickness : float
            Sample thickness.
        """
        # save the angles
        self.alpha = alpha
        if geometry == 'reflection':
            beta = 180 - alpha - self.twotheta
        if geometry == 'transmission':
            beta = alpha - self.twotheta
        if geometry == 'sphere':
            beta = alpha - self.twotheta
        self.beta = beta

        # set the sample thickness
        self.thickness = thickness # in [cm] now

        # get the mass absorption coefficients
        mu_in   = xrs_public.mpr(self.eloss/1.0e3+self.E0,self.element)[0]
        mu_out  = xrs_public.mpr(self.E0,self.element)[0]

        # calculate the absorption factor for several alpha values
        if isinstance(self.alpha,list) or isinstance(self.alpha,np.ndarray):
            for alpha,ii in zip(self.alpha,list(range(len(self.alpha)))):
                abs_corr = xrs_public.absCorrection(mu_in,mu_out,alpha,self.beta,self.thickness,geometry=geometry)
                # apply correction to all profiles
                self.C_total[:,ii] /= abs_corr
                self.J_total[:,ii] /= abs_corr
                self.V_total[:,ii] /= abs_corr
                for key in self.CperShell:
                    self.CperShell[key][:,ii] /= abs_corr
                    self.JperShell[key][:,ii] /= abs_corr
                    self.VperShell[key][:,ii] /= abs_corr

        # calculate the absorption factor for several beta values
        elif isinstance(self.beta,list) or isinstance(self.beta,np.ndarray):
            for beta,ii in zip(self.beta,list(range(len(self.beta)))):
                abs_corr = xrs_public.absCorrection(mu_in,mu_out,self.alpha,beta,self.thickness,geometry=geometry)
                # apply correction to all profiles
                self.C_total[:,ii] /= abs_corr
                self.J_total[:,ii] /= abs_corr
                self.V_total[:,ii] /= abs_corr
                for key in self.CperShell:
                    self.CperShell[key][:,ii] /= abs_corr
                    self.JperShell[key][:,ii] /= abs_corr
                    self.VperShell[key][:,ii] /= abs_corr

        # calculate the absorption factor for several sample thickness values
        elif isinstance(self.thickness,list) or isinstance(self.thickness,np.ndarray):
            for thick,ii in zip(self.thickness,list(range(len(self.thickness)))):
                abs_corr = xrs_public.absCorrection(mu_in,mu_out,self.alpha,self.beta,thick,geometry=geometry)
                # apply correction to all profiles
                self.C_total[:,ii] /= abs_corr
                self.J_total[:,ii] /= abs_corr
                self.V_total[:,ii] /= abs_corr
                for key in self.CperShell:
                    self.CperShell[key][:,ii] /= abs_corr
                    self.JperShell[key][:,ii] /= abs_corr
                    self.VperShell[key][:,ii] /= abs_corr

        # calculate the absorption factor for a single value
        else:
            abs_corr = xrs_public.absCorrection(mu_in,mu_out,self.alpha,self.beta,self.thickness,geometry=geometry)
            # apply correction to all profiles
            self.C_total /= abs_corr
            self.J_total /= abs_corr
            self.V_total /= abs_corr
            for key in self.CperShell:
                self.CperShell[key] /= abs_corr
                self.JperShell[key] /= abs_corr
                self.VperShell[key] /= abs_corr


class FormulaProfile:
    """
    **FormulaProfile**

    Class to construct and handle Hartree-Fock atomic Compton Profile of a single chemical compound.

    Attributes
      * filename : string
        Path and filename to Biggs database.
      * formula : string
        Chemical sum formula for the compound of interest (e.g. 'SiO2' or 'H2O').
      * elements : list of strings
        List of atomic symbols that make up the chemical sum formula.
      * stoichiometries : list of integers
        List of the stoichimetric weights for each of the elements in the list *elements*.
      * element_Nrs : list of integers
        List of atomic numbers for each element in the *elements* list.
      * AtomProfiles : list of *AtomProfiles* 
        List of instances of the *AtomProfiles* class for each element in the list.
      * eloss : np.ndarray
        Energy loss scale for the Compton profiles.
      * C_total : np.ndarray
        Core HF Compton profile (one column per 2Th).
      * J_total : np.ndarray
        Total HF Compton profile (one column per 2Th).
      * V_total :np.ndarray
        Valence HF Compton profile (one column per 2Th).
      * E0 : float
        Analyzer energy (keV).
      * twotheta : float, list, or np.ndarray
        Value or list/np.ndarray of the scattering angle.

    """
    def __init__(self, formula, filename, weight=1):
        assert type(formula) is str, "\'formula\' argument should be a string!"
        self.filename  = filename
        self.formula   = formula
        self.elements, self.stoichiometries = parseChemFormula(formula)
        self.element_Nrs = [xrs_public.element(element) for element in self.elements]
        self.AtomProfiles = {}
        for element,stoichio in zip(self.elements,self.stoichiometries):
            CP = AtomProfile(element,filename,stoichiometry=stoichio)
            self.AtomProfiles[element] = CP
        self.eloss     = []
        self.C_total   = []
        self.J_total   = []
        self.V_total   = []
        self.q_vals    = []
        self.E0        = 0.0
        self.twotheta  = []
        self.stoich_weight = weight

    def get_stoichWeight(self):
        return self.stoich_weight

    def get_elossProfiles(self,E0, twotheta,correctasym=None,valence_cutoff=VAL_CUTOFF_DEFAULT):
        self.E0 = E0

        # reset self.twotheta
        self.twotheta = []
        if isinstance(twotheta, list):
            self.twotheta.extend(twotheta)
        elif isinstance(twotheta, float):
            self.twotheta.append(twotheta)
        else:
            print('Unsupported type for twotheta argument')

        for key in self.AtomProfiles:
            self.AtomProfiles[key].get_elossProfiles(self.E0, self.twotheta,correctasym,valence_cutoff)

        self.eloss = self.AtomProfiles[list(self.AtomProfiles.keys())[0]].eloss

        # initialize the profiles
        self.C_total = np.zeros((len(self.eloss),len(self.twotheta)))
        self.J_total = np.zeros((len(self.eloss),len(self.twotheta)))
        self.V_total = np.zeros((len(self.eloss),len(self.twotheta)))
        self.q_vals  = np.zeros((len(self.eloss),len(self.twotheta)))

        # add up all AtomProfiles
        for key in self.AtomProfiles:
            AP = self.AtomProfiles[key]
            for ii in range(len(self.twotheta)):
                self.C_total[:,ii] += np.interp(self.eloss,AP.eloss,AP.C_total[:,ii])*AP.get_stoichiometry()
                self.J_total[:,ii] += np.interp(self.eloss,AP.eloss,AP.J_total[:,ii])*AP.get_stoichiometry()
                self.V_total[:,ii] += np.interp(self.eloss,AP.eloss,AP.V_total[:,ii])*AP.get_stoichiometry()
                self.q_vals[:,ii]  = np.interp(self.eloss,AP.eloss,AP.q_vals[:,ii])
 
class HFProfile:
    """
    *HFProfile*

    Class to construct and handle Hartree-Fock atomic Compton Profile of sample composed of several chemical compounds.

    Attributes
    

    """
    def __init__(self, formulas, stoich_weights, filename):

        if isinstance(formulas,list) and isinstance(stoich_weights,list):
            self.formulas = formulas
            self.stoich_weights = stoich_weights
        elif isinstance(formulas, str) and (isinstance(stoich_weights, int) or isinstance(stoich_weights, float)):
            self.formulas = []
            self.formulas.append(formulas)
            self.stoich_weights = []
            self.stoich_weights.append(stoich_weights)
        else:
            print('Unsupported/uncongruent types for formulas/stoich_weights arguments!')
            return

        self.filename  = filename
        self.FormulaProfiles = {}
        for formula,ii in zip(self.formulas,list(range(len(self.formulas)))):
            CP = FormulaProfile(formula,filename,weight=stoich_weights[ii])
            self.FormulaProfiles[formula] = CP

        self.eloss     = []
        self.C_total   = []
        self.J_total   = []
        self.V_total   = []
        self.q_vals    = []
        self.twotheta  = []
        self.E0        = 0.0

    def get_elossProfiles(self,E0, twotheta,correctasym=None,valence_cutoff=VAL_CUTOFF_DEFAULT):
        # save the E0 value
        self.E0 = E0

        # reset self.twotheta
        self.twotheta = []
        if isinstance(twotheta, list):
            self.twotheta.extend(twotheta)
        elif isinstance(twotheta, float):
            self.twotheta.append(twotheta)
        else:
            print('Unsupported type for twotheta argument')

        # get all Atomic and Formula unit profiles on eloss scale
        for key in self.FormulaProfiles:
            self.FormulaProfiles[key].get_elossProfiles(self.E0, self.twotheta,correctasym,valence_cutoff)

        # save the eloss-scale
        self.eloss = self.FormulaProfiles[list(self.FormulaProfiles.keys())[0]].eloss

        # initialize the profiles
        self.C_total = np.zeros((len(self.eloss),len(self.twotheta)))
        self.J_total = np.zeros((len(self.eloss),len(self.twotheta)))
        self.V_total = np.zeros((len(self.eloss),len(self.twotheta)))
        self.q_vals  = np.zeros((len(self.eloss),len(self.twotheta)))

        # add up all Compton Profiles from the sub-units
        for key,jj in zip(self.FormulaProfiles,list(range(len(self.twotheta)))):
            FP = self.FormulaProfiles[key]
            for ii in range(len(self.twotheta)):
                self.C_total[:,ii] += np.interp(self.eloss,FP.eloss,FP.C_total[:,ii])*FP.get_stoichWeight()
                self.J_total[:,ii] += np.interp(self.eloss,FP.eloss,FP.J_total[:,ii])*FP.get_stoichWeight()
                self.V_total[:,ii] += np.interp(self.eloss,FP.eloss,FP.V_total[:,ii])*FP.get_stoichWeight()
                self.q_vals[:,ii] = np.interp(self.eloss,FP.eloss,FP.q_vals[:,ii])


class ComptonProfiles:
    """Class for multiple HF Compton profiles.

    This class should hold one or more instances of the ComptonProfile class
    and have methods to return profiles from single atoms, single shells, all
    atoms. It should be able to apply corrections etc. on those... 

    Attributes:
       * element (string): Element symbol as in the periodic table.
       * elementNr (int) : Number of the element as in the periodic table.
       * shells (list)   :
       * edges (list)    :
       * C (np.array)    :
       * J (np.array)    :
       * V (np.array)    :
       * CperShell (dict. of np.arrays):
       * JperShell (dict. of np.arrays):
       * VperShell (dict. of np.arrays):
    """
    def __init__(self, element):
        self.element   = element
        self.elementNr = xrs_public.element(element)
        self.shells    = []
        self.edges     = []
        self.C         = []
        self.J         = []
        self.V         = []
        self.CperShell = {}
        self.JperShell = {}
        self.VperShell = {}


def trapz_weights(x):
    dx = np.diff(x)
    w = np.empty(x.shape)
    w[1:-1] = (dx[1:] + dx[:-1])/2.
    w[0] = dx[0] / 2.
    w[-1] = dx[-1] / 2.
    return w

def PzProfile(element,filename):
    """Returnes tabulated HF Compton profiles.

    Reads in tabulated HF Compton profiles from the Biggs paper,
    interpolates them, and normalizes them to the # of electrons 
    in the shell.

    Args:
        * element (string):  element symbol (e.g. 'Si', 'Al', etc.)
        * filename (string): absolute path and filename to tabulated profiles

    Returns:
      * CP_profile (np.array): Matrix of the Compton profile
        * 1. column: pz-scale
        * 2. ... n. columns: Compton profile of nth shell
      * binding_energy (list): binding energies of shells
      * occupation_num (list): number of electrons in the according shells

    """
    # load Biggs data, mirror at pz = 0.0
    CP_tab, occupation_num, binding_energies, shell_names = xrs_fileIO.readbiggsdata(filename,element)
    pz_tab = np.append(-1.0*np.flipud(CP_tab[1::,0]),CP_tab[:,0])
    CP_tab = np.append(np.flipud(CP_tab[1::,:]),CP_tab,axis=0)
    CP_tab[:,0] = pz_tab

    # pad with zeros as large pos. and large neg. pz for nicer spline
    CP_tab = np.append(np.zeros((1,CP_tab.shape[1])),CP_tab,axis=0)
    CP_tab = np.append(CP_tab,np.zeros((1,CP_tab.shape[1])),axis=0)
    CP_tab[0,0]  = -10000.0
    CP_tab[-1,0] = 10000.0

    # interpolate
    pz_scale  = np.arange(-100.0,100.0,0.01)
    CP_profile = np.zeros((len(pz_scale),len(binding_energies)+1))
    CP_profile[:,0] = pz_scale
    for n in range(len(binding_energies)):
        interp_func = interpolate.pchip(CP_tab[:,0], CP_tab[:,n+2])
        CP_profile[:,n+1]  = interp_func(pz_scale)

    # normalize to one electron, multiply by number of electrons
    for n in range(len(binding_energies)):
        norm              = np.trapezoid(CP_profile[:,n+1],CP_profile[:,0])
        CP_profile[:,n+1] = CP_profile[:,n+1]/norm*int(occupation_num[n])
    binding_energies    = [float(energy) for energy in binding_energies]
    occupation_num    = [float(value) for value in occupation_num]

    return CP_profile, binding_energies, occupation_num, shell_names

def elossProfile(element,filename,E0,tth,correctasym=None,valence_cutoff=20.0):
    """Returns HF Compton profiles on energy loss scale.

    Uses the PzProfile function to read read in Biggs HF profiles
    and converts them onto energy loss scale. The profiles are cut
    at the respective electron binding energies and are normalized
    to the f-sum rule (i.e. S(q,w) is in units of [1/eV]).

    Args:
      * element (string): element symbol.
      * filename (string): absolute path and filename to tabulated Compton profiles.
      * E0 (float): analyzer energy in [keV].
      * tth (float): scattering angle two theta in [deg].
      * correctasym (np.array): vector of scaling factors to be applied.
      * valence_cutoff (float): energy value below which edges are considered as valence

    Returns:
      * enScale (np.array): energy loss scale in [eV]
      * J_total (np.array): total S(q,w) in [1/eV]
      * C_total (np.array): core contribution to S(q,w) in [1/eV]
      * V_total (np.array): valence contribution to S(q,w) in [1/eV], the valence is defined by valence_cutoff
      * q (np.array): momentum transfer in [a.u]
      * J_shell (dict of np.arrays): dictionary of contributions for each shell, the key are defines as in Biggs table.
      * C_shell (dict of np.arrays): same as J_shell for core contribution
      * V_shell (dict of np.arrays): same as J_shell for valence contribution

    """
    # read in the Biggs data
    CP_profile, binding_energies, occupation_num, shell_names = PzProfile(element,filename)

    # convert pz to energy loss scale
    enScale = ((np.flipud(xrs_public.pz2e1(E0,CP_profile[:,0],tth))-E0)*1.0e3)

    # define the momentum transfer
    q = xrs_public.momtrans_au(enScale/1000.0 + E0, E0, tth)

    # calculate asymmetry after Holm and Ribberfors for filles 1s and 2p shells
    # if correctasym == True
    asymmetry = np.flipud(HRcorrect(CP_profile, occupation_num, q))
    if correctasym:
        CP_profile[:,1:4] = CP_profile[:1:4] + asymmetry * correctasym

    # discard profiles, q, and enScale for energy losses smaller than zero
    HF_profile = CP_profile[np.nonzero(enScale.T>=0.0)[0],:]
    q          = q[np.nonzero(enScale.T>=0)[0]]
    enScale    = enScale[np.nonzero(enScale.T>=0)[0]]
    HF_profile[:,0] = enScale 

    # discard profiles for energy losses  below according binding energies
    for n in range(len(binding_energies)):
        HF_profile[np.where(enScale<binding_energies[n]),n+1] = 0 

    # convert J(pz) to S(q,w) via J(pz)=N_electrons*hartree*q*S(q,w) and
    # normalize using the f-sum rule (sum(S(q,w)*w)=f)
    # 1. convert to a.u.
    hartree  = 1.0/constants.physical_constants['electron volt-hartree relationship'][0]
    enScaleH = enScale/hartree # eloss in a.u.

    # 2. normalize to one then multiply by N_el*q**2.0/2.0
    for n in range(len(binding_energies)):
        HF_profile[:,n+1] = HF_profile[:,n+1]/(integrate.trapezoid(np.multiply(HF_profile[:,n+1],enScaleH),enScaleH))
        HF_profile[:,n+1] = np.multiply(HF_profile[:,n+1],(q**2.0)/2.0)*occupation_num[n]

    # 3. convert back to [1/eV] and sum up
    J_total = np.zeros((len(enScale)))
    V_total = np.zeros((len(enScale)))
    for n in range(len(binding_energies)):
        if binding_energies[n] < enScale[-1]:
            J_total += HF_profile[:,n+1]/hartree
            if binding_energies[n] < valence_cutoff:
                V_total += HF_profile[:,n+1]/hartree
    C_total = J_total - V_total

    # make dictionaries for individual shells
    J_shell = {}
    C_shell = {}
    V_shell = {}
    counter = 1
    for name in shell_names:
        if 'Shell' in name:
            if binding_energies[counter-1] < enScale[-1]:
                J_shell[name] = HF_profile[:,counter]
                if binding_energies[counter-1] < valence_cutoff:
                    V_shell[name] = HF_profile[:,counter]
                else:
                    V_shell[name] = np.zeros_like(HF_profile[:,counter])
            else:
                J_shell[name] = np.zeros_like(HF_profile[:,counter])
                V_shell[name] = np.zeros_like(HF_profile[:,counter])

            C_shell[name] = J_shell[name] - V_shell[name]
            counter += 1

    return enScale, J_total, C_total, V_total, q, J_shell, C_shell, V_shell


def mapShellNames(shell_str,atomicNumber):
    """
    **mapShellNames**

    Translates to and from spectroscopic edge notation and the convention of the Biggs database.

    Args:
      * shell_str : string
        Spectroscopic symbol to be converted to Biggs database convention.
      * atomicNumber : int
        Z for the atom in question.
    """

    if atomicNumber <= 35: # for Z<=35 (Br), there is no 2p spin-orbit splitting in the Biggs data

        all_names   = ['pz', 'total', 'Shell_1', 'Shell_2', 'Shell_3', 'Shell_4', 'Shell_5', 'Shell_6', 'Shell_7', 'Shell_8']
        all_shells  = ['pz', 'total', '1s',      '2s',      '2p',      '3s',      '3p',      '3d',      '4s',      '4p'     ]
        all_spectro = ['pz', 'total', 'K',       'L1',      'L23',     'M1',      'M23',     'M45',     'N1',      'N23'    ]

        if shell_str in all_spectro:
            index = all_spectro.index(shell_str)
            return all_names[index]

        elif shell_str in all_shells:
            index = all_shells.index(shell_str)
            return all_names[index]

        elif shell_str in all_names:
            index = all_names.index(shell_str)
            return all_spectro[index]

        else:
            print('Following shells available for Z<=35: [\'K\',\'L1\',\'L23\',\'M1\',\'M23\',\'M45\',\'N1\',\'N23\']')
            return

    else:
        all_names   = ['pz', 'total', 'Shell_1', 'Shell_2', 'Shell_3', 'Shell_4', 'Shell_5', 
                  'Shell_6', 'Shell_7', 'Shell_8', 'Shell_9', 'Shell_10', 'Shell_11', 
                  'Shell_12', 'Shell_13', 'Shell_14', 'Shell_15', 'Shell_16', 'Shell_17', 
                  'Shell_18', 'Shell_19', 'Shell_20', 'Shell_21', 'Shell_22', 'Shell_23', 
                  'Shell_24']
        # all_shells  = ['pz', 'total', '1s', '2s', '2p1/2', '2p3/2', '3s', '3p1/2', '3p3/2', '3d1/2', '3d3/2', '3d5/2', '4s', '4p', '4d ']
        all_spectro = ['pz', 'total', 'K', 'L1', 'L2', 'L3', 'M1', 'M2', 'M3', 'M4', 'M5', 'N1', 'N2', 'N3', 'N4', 'N5', 'N6', 'N7', 'O1', 'O2', 'O3', 'O4', 'O5', 'P1', 'P2', 'P3']

        if shell_str in all_spectro:
            index = all_spectro.index(shell_str)
            return all_names[index]

        elif shell_str in all_names:
            index = all_names.index(shell_str)
            return all_spectro[index]

        else:
            print('Following shells available for Z>35: [\'K\', \'L1\', \'L2\', \'L3\', \'M1\', \'M2\', \'M3\', \'M4\', \'M5\', \'N1\', \'N2\', \'N3\', \'N4\', \'N5\', \'N6\', \'N7\', \'O1\', \'O2\', \'O3\', \'O4\', \'O5\', \'P1\', \'P2\', \'P3\']')
            return

# 1s, 2s, 2p, 3s, 3p, 4s, 3d, 4p, 5s, 4d, 5p, 6s, 4f, 5d, 6p, 7s, 5f, 6d, 7p, (8s, 5g, 6f, 7d, 8p, and 9s)

def HRcorrect(pzprofile,occupation,q):
    """ Returns the first order correction to filled 1s, 2s, and 2p Compton profiles.

    Implementation after Holm and Ribberfors (citation ...).

    Args: 
      * pzprofile (np.array): Compton profile (e.g. tabulated from Biggs) to be corrected (2D matrix). 
      * occupation (list): electron configuration.
      * q (float or np.array): momentum transfer in [a.u.].

    Returns:
       * asymmetry (np.array):  asymmetries to be added to the raw profiles (normalized to the number of electrons on pz scale)
    """
    # prepare output matrix
    if len(occupation) == 1:
        asymmetry = np.zeros((len(pzprofile[:,0]),1))
    elif len(occupation) == 2:
        asymmetry = np.zeros((len(pzprofile[:,0]),2))
    elif len(occupation) >= 3:
        asymmetry = np.zeros((len(pzprofile[:,0]),3))

    # take care for the cases where 2p levels have spin-orbit split taken into account in the Biggs table
    if len(occupation)>3 and occupation[2]==2 and occupation[3]==4:
        pzprofile[:,3] = pzprofile[:,3] + pzprofile[:,4]
        occupation[2] = 6
    
    # 1s 
    if occupation[0] < 2:
        pass
    else:
        # find gamma1s lambda x: (x[0] - 1)**2 + (x[1] - 2.5)**2
        fitfct  = lambda a: (np.absolute(np.max(pzprofile[:,1])-np.max(occupation[0]*8.0*a**5.0/3.0/np.pi/(a**2.0+pzprofile[:,0]**2.0)**3.0)))
        res = optimize.leastsq(fitfct,np.sum(occupation))
        gamma1s = res[0][0]
        # calculate j0 and j1
        j0 = occupation[0]*8.0*gamma1s**5.0/3.0/np.pi/((gamma1s**2.0+pzprofile[:,0]**2.0)**3.0)
        j1 = 2.0*gamma1s*np.arctan2(pzprofile[:,0],gamma1s)-3.0/2.0*pzprofile[:,0] 
        j1 = j1/q*j0
        asymmetry[:,0] = j1
    # 2s
    if len(occupation)>1:
        if occupation[1] < 2:
            pass
        else:
            # find gamma2s
            fitfct  = lambda a: (np.absolute(np.max(pzprofile[:,2])-np.max(occupation[1]*((a**4.0-10.0*a**2.0*pzprofile[:,0]**2 + 40.0*pzprofile[:,0]**4.0)*128.0*a**5.0/15.0/np.pi/(a**2.0 + 4.0*pzprofile[:,0]**2.0)**5.0))))
            res = optimize.leastsq(fitfct,np.sum(occupation)*2.0/3.0)
            gamma2s = res[0][0]
            # calculate j0 and j1
            j0 = occupation[1]*(gamma2s**4.0-10.0*gamma2s**2.0*pzprofile[:,0]**2.0+40.0*pzprofile[:,0]**4.0)*128.0*gamma2s**5.0/15.0/np.pi/(gamma2s**2.0 + 4.0*pzprofile[:,0]**2.0)**5.0
            j1 = 2.0*gamma2s*np.arctan2(2.0*pzprofile[:,0],gamma2s)-5.0/4.0*(gamma2s**4.0+48.0*pzprofile[:,0]**4.0)/(gamma2s**4.0-10.0*gamma2s**2.0*pzprofile[:,0]**2.0+40.0*pzprofile[:,0]**4.0)*pzprofile[:,0] 
            j1 = j1/q*j0
            asymmetry[:,1] = j1
    # 2p
    if len(occupation)>2:
        if occupation[2] < 6:
            pass
        else:
            forgamma = 3.0*pzprofile[:,3]/np.trapezoid(pzprofile[:,3],pzprofile[:,0]) # 2p correction is defined for 3 electrons in the 2p shell
            # find gamma2p
            fitfct = lambda a: (np.absolute(np.max(forgamma)-np.max(((a**2.0+20.0*pzprofile[:,0]**2.0)*64.0*a**7.0/5.0/np.pi/(a**2.0+4.0*pzprofile[:,0]**2.0)**5.0))))
            res = optimize.leastsq(fitfct,np.sum(occupation)*1.0/3.0)
            gamma2p = res[0][0]
            # calculate j0 and j1
            j0 = 2.0*(gamma2p**2.0+20.0*pzprofile[:,0]**2.0)*64.0*gamma2p**7.0/5.0/np.pi/(gamma2p**2.0+4.0*pzprofile[:,0]**2.0)**5.0
            j1 = 2.0*gamma2p*np.arctan2(2.0*pzprofile[:,0],gamma2p)-2.0/3.0*pzprofile[:,0]*(10.0*gamma2p**2.0+60.0*pzprofile[:,0]**2.0)/(gamma2p**2.0+20.0*pzprofile[:,0]**2.0)
            j1 = j1/q*j0
            asymmetry[:,2] = j1
    return asymmetry





