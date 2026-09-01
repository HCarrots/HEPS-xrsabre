import numpy as np
from scipy import special


# Math functions
def gauss(x, *params):
    amp, cen, wid, bkg = params
    # FWHM = 2.3548*wid
    return amp * np.exp(-(x - cen)**2 / (2. * wid**2)) + bkg


def lorentzian(x, *params):
    amp, cen, wid, bkg = params
    return amp * wid**2 / ((x - cen)**2 + wid**2) + bkg


def arctan(x, *params):
    amp, cen, wid, bkg = params
    return amp * np.arctan((x - cen) / wid) + bkg


def linear(x, *params):
    slope, intercept = params
    return slope * x + intercept


def erf_func(x, *params):
    amp, cen, wid, bkg = params
    return amp * special.erf((x - cen) / wid) + bkg


def pseudovoigt(x, *params):
    a, amp, cen, wid, bkg = params
    sig = wid / 2 / np.sqrt(2 * np.log(2))
    return (a * amp * np.exp(-(x - cen)**2 / (2. * sig**2))
            / np.sqrt(2 * np.pi) / sig
            + (1 - a) * amp * (wid / 2)**2 / np.pi
            / ((x - cen)**2 + (wid / 2)**2) + bkg)


def gauss2(x, *params):
    amp, cen, wid, bkg = params
    sig = wid / 2 / np.sqrt(2 * np.log(2))
    return (amp * np.exp(-(x - cen)**2 / (2. * sig**2))
            / np.sqrt(2 * np.pi) / sig + bkg)


def lorentzian2(x, *params):
    amp, cen, wid, bkg = params
    return (amp * (wid / 2)**2 / np.pi
            / ((x - cen)**2 + (wid / 2)**2) + bkg)


def pseudovoigt2(x, *params):
    a, amp, cen, wid, bkg = params
    return (a * amp * np.exp(-(x - cen)**2 / (2. * (wid / 2.3548)**2))
            + (1 - a) * amp * (wid / 2)**2
            / ((x - cen)**2 + (wid / 2)**2) + bkg)


# Calculate FWHM
def calc_fwhm(x_axis, y_axis):
    x_axis = np.asarray(x_axis, dtype=float)
    y_axis = np.asarray(y_axis, dtype=float)
    if x_axis.ndim != 1 or y_axis.ndim != 1 or x_axis.size != y_axis.size:
        raise ValueError('x_axis and y_axis must be one-dimensional arrays of equal length')
    if x_axis.size < 3 or not np.all(np.isfinite(x_axis)) or not np.all(np.isfinite(y_axis)):
        raise ValueError('FWHM requires at least three finite data points')
    if np.any(np.diff(x_axis) <= 0):
        raise ValueError('x_axis must be strictly increasing')

    half_max = np.max(y_axis) / 2
    peak = int(np.argmax(y_axis))
    crossings = np.flatnonzero(
        (y_axis[:-1] - half_max) * (y_axis[1:] - half_max) <= 0
    )
    left = crossings[crossings < peak]
    right = crossings[crossings >= peak]
    if left.size == 0 or right.size == 0:
        raise ValueError("Could not locate FWHM half-maximum crossings")

    def interpolate(index):
        x0, x1 = x_axis[index:index + 2]
        y0, y1 = y_axis[index:index + 2]
        if y1 == y0:
            return x0
        return x0 + (half_max - y0) * (x1 - x0) / (y1 - y0)

    return interpolate(int(right[0])) - interpolate(int(left[-1]))
