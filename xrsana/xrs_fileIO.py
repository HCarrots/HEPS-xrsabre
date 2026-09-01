#!/usr/bin/python


__author__ = "Christoph J. Sahle - ESRF"
__contact__ = "christoph.sahle@esrf.fr"
__license__ = "MIT"
__copyright__ = "European Synchrotron Radiation Facility, Grenoble, France"

import numpy as np


def readbiggsdata(filename,element):
    """
    Reads Hartree-Fock Profile of element 'element' from values tabulated 
    by Biggs et al. (Atomic Data and Nuclear Data Tables 16, 201-309 (1975))
    as provided by the DABAX library (http://ftp.esrf.eu/pub/scisoft/xop2.3/DabaxFiles/ComptonProfiles.dat).
    input:
    filename = path to the ComptonProfiles.dat file (the file should be distributed with this package)
    element  = string of element name
    returns:
    data     = the data for the according element as in the file:
        #UD  Columns: 
        #UD  col1: pz in atomic units 
        #UD  col2: Total compton profile (sum over the atomic electrons
        #UD  col3,...coln: Compton profile for the individual sub-shells
    occupation = occupation number of the according shells
    bindingen  = binding energies of the accorting shells
    colnames   = strings of column names as used in the file
    """
    elementid = '#S'
    sizeid    = '#N'
    occid     = '#UOCCUP'
    bindingid = '#UBIND'
    colnameid = '#L'
    data = []
    arraysize = occupation = bindingen = colnames = None
    with open(filename, 'r') as handle:
        while True:
            line = handle.readline()
            if line == '':
                break
            if line.startswith(elementid) and line.split()[-1] == element:
                line = handle.readline()
                while line and not line.startswith(elementid):
                    if line.startswith(sizeid):
                        arraysize = int(line.split()[-1])
                    elif line.startswith(occid):
                        occupation = line.split()[1:]
                    elif line.startswith(bindingid):
                        bindingen = line.split()[1:]
                    elif line.startswith(colnameid):
                        colnames = line.split()[1:]
                    elif line.startswith(' '):
                        data.append([float(n) for n in line.strip().split()])
                    line = handle.readline()
                break
    if arraysize is None or occupation is None or bindingen is None or colnames is None or not data:
        raise ValueError("No Biggs data for element %r in %s" % (element, filename))
    length = len(data)
    data = np.reshape(np.array(data),(length,arraysize))
    return data, occupation, bindingen, colnames







