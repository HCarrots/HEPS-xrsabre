#!/usr/bin/python
# Filename: math_functions.py

#/*##########################################################################
#
# The XRStools software package for XRS spectroscopy
#
# Copyright (c) 2013-2014 European Synchrotron Radiation Facility
#
# This file is part of the XRStools XRS spectroscopy package developed at
# the ESRF by the DEC and Software group and contains a collection of useful
# mathematical functions.
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

def flat2DGaussian(xxx_todo_changeme, amplitude, xo, yo, sigma_x, sigma_y, offset):
    """ **flat2DGaussian**

    Returns a flattened 2D Gaussian profile.

    Args: 
    -----
    (xx,yy) (tuple): tuple containing x-y-meshgrid for 2D object to be fitted.
    amplitude (float): amplitude for Gaussian.
    xo (float): x-offset for Gaussian.
    yo (float): y-offset for Gaussian.
    sigma_x (float): FWHM in x-direction.
    sigma_y (float): FWHM in y-direction.
    offset (float): Background/offset.

    Returns:
    --------
    g (np.array): flattened array of a 2D Gaussian profile.
    """
    (xx, yy) = xxx_todo_changeme
    g = offset + amplitude*np.exp( - (((xx-xo)**2/(2.0*sigma_x**2)) + ((yy-yo)**2/(2.0*sigma_y**2))))
    return g.ravel()

def gauss1(x,x0,fwhm):
    """
    area-normalized gaussian
    x    = x-axis vector
    x0   = position 
    fwhm = full width at half maximum
    """
    sigma = fwhm/(2*np.sqrt(2*np.log(2)));
    y = np.exp(-(x-x0)**2/2/sigma**2)/sigma/np.sqrt(2*np.pi)
    return y

def gauss_forcurvefit(x,amp,x0,fwhm):
    """
    area-normalized gaussian
    x    = x-axis vector
    amp  = amplitude
    x0   = position 
    fwhm = full width at half maximum
    """
    sigma = fwhm/(2*np.sqrt(2*np.log(2)))
    y = amp*np.exp(-(x-x0)**2/2/sigma**2)/sigma/np.sqrt(2*np.pi)
    return y

def two_gauss_forcurvefit(x, amp1,x01,fwhm1, amp2,x02,fwhm2):
	sigma1 = fwhm1/(2.0*np.sqrt(2.0*np.log(2.0)))
	sigma2 = fwhm2/(2.0*np.sqrt(2.0*np.log(2.0)))
	y = amp1*np.exp(-(x-x01)**2.0/2.0/sigma1**2)/sigma1/np.sqrt(2.0*np.pi) + amp2*np.exp(-(x-x02)**2.0/2.0/sigma2**2)/sigma2/np.sqrt(2.0*np.pi)
	return y

def gauss_areanorm(x,x0,fwhm):
    """
    area-normalized gaussian
    x    = x-axis vector
    x0   = position 
    fwhm = full width at half maximum
    """
    sigma = fwhm/(2.0*np.sqrt(2.0*np.log(2.0)))
    y = np.exp(-(x-x0)**2.0/2.0/sigma**2)/sigma/np.sqrt(2.0*np.pi)
    return y


def constant(x,a):
    """
    returns a constant
    x = x-axis
    a = value of the constant 
    returns vector with same length as x with entries a
    """
    x = np.array(x)
    y = np.zeros(len(x)) + a
    return y

def linear(x,a):
    """
    returns a linear function y = ax + b
    x = x-axis
    a = list of parameters, a[0] = slope, a[1] = y-offset
    it is better to use numpy's polyval
    """
    x = np.array(x)
    y = a[0]*x + a[1]
    return y

def lorentz(x,a):
    """
    % LORENTZ  The Lorentzian function
    %
    %          y  = lorentz(x,a)
    %          x      = x-vector
    %          a[0]   = peak position
    %          a[1]   = FWHM
    %          a[2]   = Imax (if not defined, Imax = 1)
    """
    y = a[2]*(((x-a[0])/(a[1]/2.0))**2.0+1.0)**(-1.0)
    return y

def pearson7(x,a):
    """
    returns a pearson function
    a[0] = Peak position
    a[1] = FWHM
    a[2] = Shape, 1 = Lorentzian, infinite = Gaussian
    a[3] = Peak intensity
    a[4] = Background
    """
    x = np.array(x)
    y = a[3] * (1.0+(2.0**(1.0/a[2])-1.0) * (2.0*(x-a[0])/a[1])**2.0)**(-a[2])+a[4]
    return y

def pearson7_forcurvefit(x,a0,a1,a2,a3,a4):
    """
    special version of person7 for use in constrained/bound minimizations
    returns a pearson function
    a[0] = Peak position
    a[1] = FWHM
    a[2] = Shape, 1 = Lorentzian, infinite = Gaussian
    a[3] = Peak intensity
    a[4] = Background
    """
    x = np.array(x)
    y = a3 * (1.0+(2.0**(1.0/a2)-1.0) * (2.0*(x-a0)/a1)**2.0)**(-a2)+a4
    return y

def pearson7_zeroback(x,a):
    """
    returns a pearson function but without y-offset
    a[0] = Peak position
    a[1] = FWHM
    a[2] = Shape, 1 = Lorentzian, infinite = Gaussian
    a[3] = Peak intensity
    """
    x = np.array(x)
    y = a[3] * (1.0+(2.0**(1.0/a[2])-1.0) * (2.0*(x-a[0])/a[1])**2.0)**(-a[2])
    return y

def pearson7_linear(x,a):
    """
    returns a pearson function but without y-offset
    a[0] = Peak position
    a[1] = FWHM
    a[2] = Shape, 1 = Lorentzian, infinite = Gaussian
    a[3] = Peak intensity
    a[4] = ax
    b[5] = +b
    """
    x = np.array(x)
    y = a[3] * (1.0+(2.0**(1.0/a[2])-1.0) * (2.0*(x-a[0])/a[1])**2.0)**(-a[2]) + a[4]*x + a[5]
    return y

def pearson7_linear_forcurvefit(x,a0,a1,a2,a3,a4,a5):
    """
    returns a pearson function but without y-offset
    a[0] = Peak position
    a[1] = FWHM
    a[2] = Shape, 1 = Lorentzian, infinite = Gaussian
    a[3] = Peak intensity
    a[4] = ax
    b[5] = +b
    """
    x = np.array(x)
    y = ( a3 * (1.0+(2.0**(1.0/a2)-1.0) * (2.0*(x-a0)/a1)**2.0)**(-a2) + a4*x + a5 )
    return y

def pearson7_linear_scaling_forcurvefit(x,a0,a1,a2,a3,a4,a5,a6):
    """
    returns a pearson function but without y-offset
    a[0] = Peak position
    a[1] = FWHM
    a[2] = Shape, 1 = Lorentzian, infinite = Gaussian
    a[3] = Peak intensity
    a[4] = ax
    b[5] = +b
    """
    x = np.array(x)
    y = a6*( a3 * (1.0+(2.0**(1.0/a2)-1.0) * (2.0*(x-a0)/a1)**2.0)**(-a2) + a4*x + a5 )
    return y

def gauss(x,a):
    """
    returns a gaussian with peak value normalized to unity
    a[0] = peak position
    a[1] = Full Width at Half Maximum
    a[2] = intensity
    """
    y = a[2]*np.exp(-np.log(2.0)*((x-a[0])/a[1]*2.0)**2.0)
    return y

def gauss2D(x, y, x0, y0, fwhm):
    return np.exp(-4*np.log(2) * ((x-x0)**2 + (y-y0)**2) / fwhm**2)

def arctan_forcurvefit(x, a0, a1, a2, a3):
    """ **arctan_forcurvefit**

    Returns an arctan function.

    Args:
        x  (array): Numpy array for the x-axis.
        a0 (float): amplitude
        a1 (float): squeezing factor
        a2 (float): x-offset 
        a3 (float): y-offset
    """
    return a0*np.arctan(a1*(x-a2) )+a3








