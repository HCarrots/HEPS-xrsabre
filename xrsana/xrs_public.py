import numpy as np
from scipy import constants
import math
from re import findall
import os
import numbers
from importlib.resources import files
from functools import lru_cache
from scipy import interpolate ,optimize
from scipy.integrate import trapezoid
from scipy.integrate import odeint
data_installation_dir = str(files("xrsana").joinpath("resources", "data"))

def element(z):
    """
    Convert element symbol to atomic number, or atomic number to element symbol.

    Examples:
        element("Si") -> 14
        element(14)   -> "Si"
    """

    symbols = [
        "H", "He", "Li", "Be", "B", "C", "N", "O", "F", "Ne",
        "Na", "Mg", "Al", "Si", "P", "S", "Cl", "Ar", "K", "Ca",
        "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
        "Ga", "Ge", "As", "Se", "Br", "Kr", "Rb", "Sr", "Y", "Zr",
        "Nb", "Mo", "Tc", "Ru", "Rh", "Pd", "Ag", "Cd", "In", "Sn",
        "Sb", "Te", "I", "Xe", "Cs", "Ba", "La", "Ce", "Pr", "Nd",
        "Pm", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb",
        "Lu", "Hf", "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg",
        "Tl", "Pb", "Bi", "Po", "At", "Rn", "Fr", "Ra", "Ac", "Th",
        "Pa", "U", "Np", "Pu", "Am", "Cm", "Bk", "Cf", "Es", "Fm",
        "Md", "No", "Lr", "Rf"
    ]

    symbol_to_z = {symbol: i + 1 for i, symbol in enumerate(symbols)}

    # Optional compatibility with old notation.
    symbol_to_z["Ku"] = 104

    if isinstance(z, str):
        symbol = z.strip()

        if not symbol:
            raise ValueError("Empty element symbol.")

        symbol = symbol[0].upper() + symbol[1:].lower()

        if symbol not in symbol_to_z:
            raise ValueError(f"Unknown element symbol: {z}")

        return symbol_to_z[symbol]

    if isinstance(z, numbers.Integral):
        atomic_number = int(z)

        if atomic_number < 1 or atomic_number > len(symbols):
            raise ValueError(f"Unknown atomic number: {z}")

        return symbols[atomic_number - 1]

    raise TypeError(f"Unsupported type for element(): {type(z)}")



@lru_cache(maxsize=None)
def _load_logtable(logtablefile):
    return np.loadtxt(logtablefile)


def _logpoly(le, coeffs):
    """
    Evaluate exp(a0 + a1*log(E) + a2*log(E)^2 + a3*log(E)^3).
    coeffs should contain four coefficients.
    """
    return np.exp(
        coeffs[0]
        + le * (
            coeffs[1]
            + le * (
                coeffs[2]
                + le * coeffs[3]
            )
        )
    )


def prho(
    energy,
    Z,
    logtablefile=os.path.join(data_installation_dir, "logtable.dat"),
):
    """
    Calculates photoelectric, elastic, and inelastic mass attenuation
    coefficients of an element.

    Args:
        energy:
            Energy in keV. Scalar or array-like.
        Z:
            Atomic number or element symbol.
        logtablefile:
            Path to logtable.dat.

    Returns:
        murho:
            Mass attenuation coefficients, shape = (len(energy), 3).
            Columns:
                0 = photoelectric
                1 = elastic
                2 = inelastic
            Unit is probably cm^2/g.
        rho:
            Element density, probably g/cm^3.
        m:
            Atomic mass, probably g/mol.
    """

    en = np.atleast_1d(energy).astype(float)

    if np.any(en <= 0):
        raise ValueError("energy must be strictly positive because log(energy) is used.")

    if isinstance(Z, numbers.Integral):
        atomic_number = int(Z)
    else:
        atomic_number = element(Z)

    logtable = _load_logtable(logtablefile)

    matches = np.where(logtable[:, 0] == atomic_number)[0]

    if len(matches) == 0:
        raise ValueError(f"No such element in logtable.dat: Z={Z}")

    ind = matches[0]

    if ind + 5 > logtable.shape[0]:
        raise ValueError(f"Incomplete logtable block for Z={Z}")

    c = np.array(logtable[ind:ind + 5, :], dtype=float)

    le = np.log(en)

    # Photoelectric absorption, piecewise.
    photo = _logpoly(le, c[1:5, 3])

    mask = en <= c[0, 3]
    photo[mask] = _logpoly(le[mask], c[1:5, 2])

    mask = en < c[0, 2]
    photo[mask] = _logpoly(le[mask], c[1:5, 1])

    mask = en < c[0, 1]
    photo[mask] = _logpoly(le[mask], c[1:5, 0])

    # Elastic and inelastic scattering.
    elastic = _logpoly(le, c[1:5, 4])
    inelastic = _logpoly(le, c[1:5, 5])

    mu = np.column_stack((photo, elastic, inelastic))

    atomic_mass = c[0, 4]
    density = c[0, 5]

    if atomic_mass <= 0:
        raise ValueError(f"Invalid atomic mass for Z={Z}: {atomic_mass}")

    murho = mu * 0.602252 / atomic_mass

    return murho, density, atomic_mass

def parseformula(formula):
    """Parses a chemical sum formula.

    Parses the constituing elements and stoichiometries from a given 
    chemical sum formula.

    Args:
      * formula (string): string of a chemical formula (e.g. 'SiO2', 'Ba8Si46', etc.)

    Returns:
      * elements (list): list of strings of constituting elemental symbols.
      * stoichiometries (list): list of according stoichiometries in the same order as 'elements'.
    """
    elements = []
    stoichiometries = []
    splitted = findall(r'([A-Z][a-z]*)(\d*)',formula)
    elements.extend([element[0] for element in splitted])
    stoichiometries.extend([(int(element[1]) if element[1] else 1) for element in splitted])
    return elements,stoichiometries

def mpr(energy, compound):
    """
    Calculates the mass attenuation coefficient of a chemical compound.

    Args:
        energy:
            Energy scale in keV. Can be scalar or array-like.
        compound:
            Chemical formula, for example 'SiO2'.

    Returns:
        murho_total:
            Total mass attenuation coefficient of the compound.
        murho_components:
            Separate contributions:
            column 0 = photoelectric,
            column 1 = elastic,
            column 2 = inelastic.
        element_densities:
            Element densities returned by myprho().
        element_mass_contributions:
            Stoichiometry-weighted atomic masses.
        molar_mass:
            Total formula mass of the compound.
    """

    en = np.atleast_1d(energy).astype(float)

    z, stoich = parseformula(compound)

    murho_components_weighted = np.zeros((len(en), 3), dtype=float)
    element_densities = np.zeros(len(z), dtype=float)
    element_mass_contributions = np.zeros(len(z), dtype=float)

    for i, zi in enumerate(z):
        tmp, rho, atomic_mass = prho(en, zi)

        mass_contribution = atomic_mass * stoich[i]

        element_mass_contributions[i] = mass_contribution
        element_densities[i] = rho

        murho_components_weighted += tmp * mass_contribution

    molar_mass = np.sum(element_mass_contributions)

    murho_components = murho_components_weighted / molar_mass
    murho_total = np.sum(murho_components, axis=1)

    return (
        murho_total,
        murho_components,
        element_densities,
        element_mass_contributions,
        molar_mass,
    )

def e2pz(w1, w2, th, *, eps=1e-14):
    """
    Calculates the momentum scale and the relativistic Compton cross section
    correction according to P. Holm, PRA 37, 3706 (1988).

    Args:
        w1 : float or array-like
            Incident energy in keV.
        w2 : float or array-like
            Scattered energy in keV.
        th : float or array-like
            Scattering angle 2 theta in degrees.
        eps : float
            Numerical tolerance for zero-division checks.

    Returns:
        pz : float or np.ndarray
            Momentum scale in atomic units.
        cf : float or np.ndarray
            Cross section correction factor.
    """

    scalar_input = np.isscalar(w1) and np.isscalar(w2) and np.isscalar(th)

    w1 = np.asarray(w1, dtype=float)
    w2 = np.asarray(w2, dtype=float)
    th = np.asarray(th, dtype=float)

    # Broadcast inputs to a common shape
    try:
        w1, w2, th = np.broadcast_arrays(w1, w2, th)
    except ValueError as exc:
        raise ValueError("w1, w2, and th must be broadcast-compatible.") from exc

    if np.any(w1 <= 0):
        raise ValueError("w1 must be positive, in keV.")

    if np.any(w2 <= 0):
        raise ValueError("w2 must be positive, in keV.")

    # Physical constants
    m = constants.value("electron mass energy equivalent in MeV") * 1e3  # keV
    alp = constants.value("fine-structure constant")
    r0 = constants.value("classical electron radius")  # meter

    th_rad = np.deg2rad(th)
    cos_th = np.cos(th_rad)
    one_minus_cos = 1.0 - cos_th

    if np.any(np.abs(one_minus_cos) < eps):
        raise ValueError("Scattering angle th must not be 0 degrees or too close to 0.")

    # Momentum transfer
    q2 = w1**2 + w2**2 - 2.0 * w1 * w2 * cos_th

    # Avoid tiny negative values caused by floating point roundoff
    q2 = np.where((q2 < 0) & (q2 > -eps), 0.0, q2)

    if np.any(q2 <= eps):
        raise ValueError("Momentum transfer q is zero or too close to zero.")

    q = np.sqrt(q2)

    sqrt_arg = 0.25 + m**2 / (2.0 * w1 * w2 * one_minus_cos)

    if np.any(sqrt_arg < 0):
        raise ValueError("Invalid square-root argument while computing pz.")

    # pz in natural units
    pz = q / 2.0 - (w1 - w2) * np.sqrt(sqrt_arg)

    E = np.sqrt(m**2 + pz**2)

    A = ((w1 - w2) * E - w1 * w2 * one_minus_cos) / q
    D = (w1 - w2 * cos_th) * A / q

    R = w1 * (E - D)
    R2 = R - w1 * w2 * one_minus_cos

    if np.any(np.abs(R) < eps):
        raise ValueError("R is zero or too close to zero; calculation is unstable.")

    if np.any(np.abs(R2) < eps):
        raise ValueError("R2 is zero or too close to zero; calculation is unstable.")

    chi = (
        R / R2
        + R2 / R
        + 2.0 * m**2 * (1.0 / R - 1.0 / R2)
        + m**4 * (1.0 / R - 1.0 / R2) ** 2
    )

    if np.any(np.abs(chi) < eps):
        raise ValueError("chi is zero or too close to zero; calculation is unstable.")

    cf = 2.0 * w1 * q * E / (m**2 * r0**2 * w2 * chi)

    # Unit conversions
    cf = cf * (1.0e-28 * (m * alp))
    pz = pz / (m * alp)

    if scalar_input:
        return float(pz), float(cf)

    return pz, cf










def mpr_compds(energy, formulas, concentrations, E0, rho_formu):
    """
    Calculate linear absorption coefficients for a mixture of compounds.

    Args:
        energy : float or array-like
            Incident/scanned energy scale in keV.
        formulas : list of str
            Chemical formulas, e.g. ["SiO2", "Al2O3"].
        concentrations : list of float
            Relative fractions of each compound.
        E0 : float or array-like
            Outgoing or fixed energy in keV.
        rho_formu : float or list of float
            Density for each compound in g/cm^3.

    Returns:
        mu_tot_in : np.ndarray
            Total linear absorption coefficient at `energy`.
        mu_tot_out : np.ndarray
            Total linear absorption coefficient at `E0`.
    """

    en = np.atleast_1d(np.asarray(energy, dtype=float))
    e0 = np.atleast_1d(np.asarray(E0, dtype=float))

    formulas = list(formulas)
    concentrations = np.atleast_1d(np.asarray(concentrations, dtype=float))
    rho_formu = np.atleast_1d(np.asarray(rho_formu, dtype=float))

    if not (len(formulas) == len(concentrations) == len(rho_formu)):
        raise ValueError(
            "formulas, concentrations, and rho_formu must have the same length."
        )

    if np.any(en <= 0):
        raise ValueError("All incident energies must be positive.")

    if np.any(e0 <= 0):
        raise ValueError("All outgoing energies must be positive.")

    mu_tot_in = np.zeros_like(en, dtype=float)
    mu_tot_out = np.zeros_like(e0, dtype=float)

    for formula, conc, rho in zip(formulas, concentrations, rho_formu):
        mu_tot_in += mpr(en, formula)[0] * conc * rho
        mu_tot_out += mpr(e0, formula)[0] * conc * rho

    # If outgoing energy is scalar but incident energy is an array,
    # broadcast mu_tot_out to the same shape as mu_tot_in.
    if mu_tot_out.size == 1 and mu_tot_in.size > 1:
        mu_tot_out = np.full_like(mu_tot_in, mu_tot_out.item())

    return mu_tot_in, mu_tot_out



def abscorr2(mu1, mu2, alpha, beta, samthick):
    """
    Calculates absorption correction for given mu1 and mu2.
    Multiply the measured spectrum with this correction factor.
    """

    mu1 = np.asarray(mu1, dtype=float)
    mu2 = np.asarray(mu2, dtype=float)

    cosa = math.cos(math.radians(alpha))
    cosb = math.cos(math.radians(beta))

    if beta >= 0:
        # Reflection geometry
        ac = (
            cosa * (mu1 / cosa + mu2 / cosb)
            / (
                1.0
                - np.exp(-mu1 * samthick / cosa - mu2 * samthick / cosb)
            )
        )

    else:
        # Transmission geometry
        x = mu1 / cosa
        y = mu2 / cosb

        general = (
            -cosa * (x - y)
            / (
                np.exp(-x * samthick)
                - np.exp(-y * samthick)
            )
        )

        limit = cosa / (
            samthick * np.exp(-mu1 * samthick / cosa)
        )

        close = np.isclose(x, y)

        ac = np.where(close, limit, general)

    return ac
def spline2(x,y,x2):
    """
    Extrapolates the smaller and larger valuea as a constant
    """
    xmin = np.min(x)
    xmax = np.max(x)
    imin = x == xmin
    imax = x == xmax
    f  = interpolate.interp1d(x,y, bounds_error=False, fill_value=0.0)
    y2 = f(x2)
    i     = np.where(x2<xmin)
    y2[i] = y[imin]
    i     = np.where(x2>xmax)
    y2[i] = y[imax]
    return y2

def pz2e1(w2,pz,th):
    """Calculates the incident energy for a specific scattered photon and momentum value.

    Returns the incident energy for a given photon energy and scattering angle.
    This function is translated from Keijo Hamalainen's Matlab implementation (KH 29.05.96).

    Args:
      * w2 (float): scattered photon energy in [keV]
      * pz (np.array): pz scale in [a.u.]
      * th (float): scattering angle two theta in [deg]

    Returns:
      * w1 (np.array): incident energy in [keV]
    """
    pz  = np.array(pz)
    w   = np.array(np.arange(np.array(w2)/4.0,4.0*np.array(w2),np.array(w2)/5000.0))
    p   = e2pz(w,w2,th)[0]
    if ( p[1]-p[0] <0) :
        tck = interpolate.UnivariateSpline(p[::-1],w[::-1])
    else:
        tck = interpolate.UnivariateSpline(p,w)
    w1  = tck(pz)
    return w1

def gauss(x,x0,fwhm):
    # area-normalized gaussian
    sigma = fwhm/(2*np.sqrt(2*np.log(2)));
    y = np.exp(-(x-x0)**2/2/sigma**2)/sigma/np.sqrt(2*np.pi)
    return y

def convg(x,y,fwhm):
    """
    Convolution with Gaussian
    x  = x-vector
    y  = y-vector
    fwhm = fulll width at half maximum of the gaussian with which y is convoluted
    """
    dx = np.min(np.absolute(np.diff(x)))
    x2 = np.arange(np.min(x)-1.5*fwhm, np.max(x)+1.5*fwhm, dx)
    xg = np.arange(-np.floor(2.0*fwhm/dx)*dx, np.floor(2.0*fwhm/dx)*dx, dx)
    yg = gauss(xg,0,fwhm)
    yg = yg/np.sum(yg)
    y2 = spline2(x,y,x2)
    c  = np.convolve(y2,yg, mode='full')
    n  = int( np.floor(np.max(np.shape(xg))/2))
    c  = c[n:len(c)-n+1] # not sure about the +- 1 here
    f  = interpolate.interp1d(x2,c)
    return f(x)

def taupgen(e, hkl = [6,6,0], crystals = 'Si', R = 1.0, dev = np.arange(-50.0,150.0,1.0), alpha = 0.0):
    """
    % TAUPGEN          Calculates the reflectivity curves of bent crystals
    %
    % function [refl,e,dev]=taupgen_new(e,hkl,crystals,R,dev,alpha);
    %
    %              e = fixed nominal energy in keV
    %            hkl = reflection order vector, e.g. [1 1 1]
    %       crystals = crystal string, e.g. 'si' or 'ge'
    %              R = bending radius in meters
    %            dev = deviation parameter for which the 
    %                  curve will be calculated (vector) (optional)
    %          alpha = asymmetry angle 
    % based on a FORTRAN program of Michael Krisch
    % Translitterated to Matlab by Simo Huotari 2006, 2007
    % Is far away from being good matlab writing - mostly copy&paste from
    % the fortran routines. Frankly, my dear, I don't give a damn. 
    % Complaints -> /dev/null
    """
    path =  os.path.join(data_installation_dir,'chitable_') # prefix + 'data/chitables/chitable_' # path to chitables
    # load the according chitable (tabulated)
    hkl_string = str(int(hkl[0])) + str(int(hkl[1])) + str(int(hkl[2]))
    filestring = path + crystals.lower() + hkl_string + '.dat'
    chi = np.loadtxt(filestring)

    # good for 1 m bent crystals in backscattering
    ystart = -50.0 # start value of angular range in arcsecs
    yend   = 150.0 # end value of angular range in arcsecs
    ystep  = 1.0   # step width in arcsecs

    if len(chi[:,0]) == 1:
        print( ' I will only  calculate for the following energy: ' + '%.4f' % chi[0,0] + ' keV!!!')
    else:
        if e < np.min(chi[:,0]) or e > np.max(chi[:,0]):
            print( 'Energy outside of the range in ' + filestring)
            return

        chi0r = np.interp(e,chi[:,0],chi[:,1])
        chi0i = np.interp(e,chi[:,0],chi[:,2])
        chihr = np.interp(e,chi[:,0],chi[:,3])
        chihi = np.interp(e,chi[:,0],chi[:,4])

    th = braggd(hkl,e,crystals)
    lam = 12.3984191/e/10.0 # wavelength in nm

    reflcorr = 0.0
    chi0 = complex(chi0r,chi0i)
    chih = complex(chihr,chihi)

    if crystals.upper() == 'SI':
        s13 = -0.278
    elif crystals.upper() == 'GE':
        s13 = -0.273
    else:
        print( 'Poisson ratio for this crystal not defined')
        return

    s15 = -0.0 # s15/s11
    dsp = dspace(hkl,crystals)/10.0 # dspace

    dwf    = 1.0 # dwf = 0.899577 # debye-waller factor
    radius = R # meridional bending radius
    rsag   = R*np.sin(np.radians(th))**2.0 # sagittal bending radius
    thick  = 500.0 # thickness in micrometers #rsag = R

    lam      = lam*1e-9
    dsp      = dsp*1e-9
    alpha    = np.radians(alpha) # alpha in rad
    thick    = thick*1e-6
    ystart   = ystart/3600.0/180.0*np.pi
    yend     = yend/3600.0/180.0*np.pi
    ystep    = ystep/3600.0/180*np.pi
    dev      = dev/3600.0/180.0*np.pi
    reflcorr = reflcorr/3600.0/180.0*np.pi

    thetab = np.arcsin(lam/(2.0*dsp))
    cpol   = 1.0 # cpol=0.5*(1+cos(2*thetab).^2) # cpol=cos(2*thetab).^2

    # gamma0 = sin(thetab+alpha) # normal convention
    # gammah = -sin(thetab-alpha) # normal convention
    gammah = -np.sin(thetab + alpha) # Krisch et al. convention (really!)
    gamma0 = np.sin(thetab - alpha) # Krisch et al. convention (I'm not kidding!!)

    beta  = gamma0/np.abs(gammah)
    gamma = gammah/gamma0

    a0 = np.sqrt(1-gamma0**2.0)
    ah = np.sqrt(1-gammah**2.0)

    mu = -2.0*np.pi/lam*chi0i

    tdepth = 1.0/mu/(1.0/np.abs(gamma0)+1.0/np.abs(gammah))

    lex = lam*np.sqrt(gamma0*np.abs(gammah))/(np.pi*chihr)

    y0 = chi0i*(1.0+beta)/(2.0*np.sqrt(beta)*chihr)

    pfried = -chihi/chihr

    c1 = cpol*dwf* complex(1.0,pfried)

    #abbreviation concerning the deviation parameter y
    abb0 = -np.sqrt(beta)/2.0/chihr
    abb1 = chi0r*(1.0+beta)/(2.0*np.sqrt(beta)*chihr)

    #abbreviations concerning the deformation field

    abb2 = gamma0*gammah*(gamma0-gammah)
    abb3 = 1.0 + 1.0/(gamma0*gammah)
    abb4 = s13*(1.0 + radius/rsag)
    abb5 = (ah - a0)/(gamma0 - gammah)*s15 
    abb6 = 1.0/(np.abs(cpol)*chihr*np.cos(thetab)*radius)
    abb7 = 2.0*np.abs(cpol)*chihr*np.cos(thetab)/gamma0

    #   a spectrometer based on a spherical diced analyzer crystal with a 1-m bending radius in nearly backscattering conditions utilizing a strain gradient beta
    sgbeta = abb6*(abb2*(abb3 - abb4 + abb5))

    nstep=len(dev)
    eta  = np.zeros_like(dev)
    abb8z = np.zeros_like(dev)
    refl  = np.zeros_like(dev)
    refl1 = np.zeros_like(dev)
    refl2 = np.zeros_like(dev)

    OLDMETHOD = 1

    if OLDMETHOD :
            for l in range(nstep):
                    # actual value of the deviation angle
                    # dev[l] = ystart + (l - 1)*ystep

                    # deviation parameter
                    abb8   = -2.0*np.sin(2.0*thetab)*dev[l]
                    eta[l] = (dev[l]*np.sin(2.0*thetab)+np.abs(chi0.real)/2.0*(1.0-gamma))/(np.abs(cpol)*np.sqrt(np.abs(gamma))*np.sqrt(chih*chih))
                    eta[l] = eta[l].real

                    ndiff = 2
                    xend = 0
                    x = np.max([-10.0*tdepth, -thick])
                    y = np.array([0.0, 0.0])
                    h = xend
                    abb8z[l] = abb8

                    # in this point call the subroutine
                    #     [T,Y] = ODE23(ODEFUN,TSPAN,Y0,OPTIONS,P1,P2,...) passes the additional
                    #    parameters P1,P2,... to the ODE function as ODEFUN(T,Y,P1,P2...), and to
                    #    all functions specified in OPTIONS. Use OPTIONS = [] as a place holder if
                    #    no options are set.   
                    #print 'the fucking shape of y is ', np.shape(y)
                    T = np.arange(x,xend,1e-8)
                    Y = odeint(odefctn,y,T,args=(abb0,abb1,abb7,abb8,lex,sgbeta,y0,c1)) 

                    # normalized reflectivity at this point
                    refl[l] = np.sum(Y[-1,:]**2.0)
                    refl1[l] = Y[-1,0]
                    refl2[l] = Y[-1,1]

    else:
            YY = np.zeros([nstep],"D")
            for l in range(nstep):
                    abb8   = -2.0*np.sin(2.0*thetab)*dev[l]
                    eta[l] = ((dev[l]*np.sin(2.0*thetab)+np.abs(chi0.real)/2.0*(1.0-gamma))/
                              (np.abs(cpol)*np.sqrt(np.abs(gamma))*np.sqrt(chih*chih)))
                    eta[l] = eta[l].real
                    abb8z[l] = abb8


            xend = 0
            x = np.max([-10.0*tdepth, -thick])
            ministep = tdepth/1000.0 ## when delta/beta is 100 we have still 10
            xpoints = np.arange(x,0, ministep)
            substep_RungeKutta = ministep/2
            ssrk  = substep_RungeKutta

            for xpos in xpoints[:-1]:
                    Yp0 = odefctn_CN( YY, xpos+0*ssrk ,      abb0,abb1,abb7,abb8z,lex,sgbeta,y0,c1)
                    Yp1 = odefctn_CN( YY+1*ssrk*Yp0, xpos+1*ssrk ,      abb0,abb1,abb7,abb8z,lex,sgbeta,y0,c1)
                    Yp2 = odefctn_CN( YY+1*ssrk*Yp1, xpos+1*ssrk ,      abb0,abb1,abb7,abb8z,lex,sgbeta,y0,c1)
                    Ypb = odefctn_CN( YY+2*ssrk*Yp2, xpos+2*ssrk ,      abb0,abb1,abb7,abb8z,lex,sgbeta,y0,c1)

                    YY =  YY  + ministep*(  Yp0 + 2*(Yp1+Yp2) + Ypb  )/6.0

            refl1 = YY.real
            refl2 = YY.imag
            refl  = refl1*refl1+refl2*refl2


    de = dev * e * 1.0e6 /np.tan(thetab)

    lam    = lam *1.0e+09        
    dsp    = dsp*1.0e+09        
    alpha  = alpha/np.pi*180.0        
    ystart = ystart/4.848136811e-06
    yend   = yend/4.848136811e-06   
    ystep  = ystep/4.848136811e-06
    dev    = dev/4.848136811e-06 # dev in arcsecs
    
    dev = dev/3600.0 # in degrees
    thb = th
    th  = thb + dev
    e0  = e
    e   = energy(dspace(hkl,crystals),th)-e0
    e = e*1e6

    dev = dev*3600.0 # back to arcsecs

    return refl,e,dev,e0

def dspace(hkl=[6,6,0],xtal='Si'):
    """
    % DSPACE Gives d-spacing for given xtal
    %     d=dspace(hkl,xtal)
    %     hkl can be a matrix i.e. hkl=[1,0,0 ; 1,1,1];
    %     xtal='Si','Ge','LiF','InSb','C','Dia','Li' (case insensitive)
    %     if xtal is number this is user as a d0
    %
    %     KH 28.09.93 
    %        SH 2005
    %
    """
    # create a database of lattice constants (could be a shelf)
    xtable = {}
    xtable['SI'] = 5.43102088
    xtable['GE'] = 5.657
    xtable['SIXOP'] = 5.430919
    xtable['SIKOH'] = 5.430707
    xtable['LIF'] = 4.027
    xtable['INSB'] = 6.4784
    xtable['C'] = 6.708
    xtable['DIA'] = 3.57
    xtable['LI'] = 3.41
    xtable['TCNE'] = 9.736
    xtable['CU'] = 3.61
    xtable['PB'] = 4.95
    xtable['NA'] = 4.2906
    xtable['AL'] = 4.0495

    if isinstance(xtal,str):
        try:
            a0 = xtable[xtal.upper()]
        except KeyError:
            print( 'Lattice constant is not in database')
            return
    else: 
        a0 = xtal # if number is provided, it's taken as lattice constant

    return a0/np.sqrt(np.sum(np.array(hkl)**2.0))

def energy(d,ba):
    """
    % ENERGY  Calculates energy corrresponing to Bragg angle for given d-spacing
    %         function e=energy(dspace,bragg_angle)
    %
    %      dspace for reflection
    %      bragg_angle in DEG
    %
    %         KH 28.09.93
    """
    hc = 12.3984191 # CODATA 2002 physics.nist.gov/constants
    return (2.0*d*np.sin(ba/180.0*np.pi)/hc)**(-1)
def bragg(hkl,e,xtal='Si'):
    """
    % BRAGG  Calculates Bragg angle for given reflection in RAD
    %      output=bangle(hkl,e,xtal)
    %        hkl can be a matrix i.e. hkl=[1,0,0 ; 1,1,1];
    %      e=energy in keV
    %      xtal='Si', 'Ge', etc. (check dspace.m) or d0 (Si default)
    %
    %      KH 28.09.93
    %
    """
    hc = 12.3984191 # CODATA 2002 recommended value, physics.nist.gov/constants
    return np.real(np.arcsin((2.0*dspace(hkl,xtal)*e/hc)**(-1.0)))


def braggd(hkl,e,xtal='Si'):
    """
    # BRAGGD  Calculates Bragg angle for given reflection in deg
    #      Call BRAGG.M
    #      output=bangle(hkl,e,xtal)
    #        hkl can be a matrix i.e. hkl=[1,0,0 ; 1,1,1];
    #      e=energy in keV
    #      xtal='Si', 'Ge', etc. (check dspace.m) or d0 (Si default)
    #
    #      KH 28.09.93
    """
    return bragg(hkl,e,xtal)/np.pi*180.0

def odefctn(y,t,abb0,abb1,abb7,abb8,lex,sgbeta,y0,c1):
    """
    #%    [T,Y] = ODE23(ODEFUN,TSPAN,Y0,OPTIONS,P1,P2,...) passes the additional
    #%    parameters P1,P2,... to the ODE function as ODEFUN(T,Y,P1,P2...), and to
    #%    all functions specified in OPTIONS. Use OPTIONS = [] as a place holder if
    #%    no options are set.   
    """
    #print 'shape of y is ' , np.shape(y), np.shape(t)
    fcomp = 1.0/(complex(0,-lex)) * (-2.0*((abb0*(abb8 + abb7*sgbeta*t) + abb1) + complex(0,y0))*(y[0] + complex(0,y[1])) + c1*(1.0 + (y[0] + complex(0,y[1]))**2.0))
    return fcomp.real,fcomp.imag


def odefctn_CN(yCN,t,abb0,abb1,abb7,abb8N,lex,sgbeta,y0,c1):
    fcomp = 1.0/(complex(0,-lex)) * (-2.0*((abb0*(abb8N + abb7*sgbeta*t) + abb1) + complex(0,y0))*(yCN) + c1*(1.0 + yCN* yCN) )
    return fcomp

def fwhm(x,y):
    """
    finds full width at half maximum of the curve y vs. x
    returns 
    f  = FWHM
    x0 = position of the maximum
    """
    if x[-1] < x[0]:
        x = np.flipud(x)
        y = np.flipud(y)

    y0 = np.amax(y)
    i0 = np.where(y == y0)[0]
    
    if len(i0)>1:
        i0 = i0[0]

    x0 = x[i0]

    i1 = np.where(np.logical_and(y>y/3.0, x<x0))[0]
    i2 = np.where(np.logical_and(y>y/3.0, x>x0))[0]

    if len(y[i1])==0 or len(y[i2])==0:
        return 0,0
    #f  = interpolate.interp1d(y[i1],x[i1], bounds_error=False, fill_value=0.0)
    #x1 = f(y0/2.0)
    #f  = interpolate.interp1d(y[i2],x[i2], bounds_error=False, fill_value=0.0)
    #x2 = f(y0/2.0)
    x1 = np.interp(y0/2.0,y[i1],x[i1])
    x2 = np.interp(y0/2.0,np.flipud(y[i2]),np.flipud(x[i2]))
    fwhm = x2 - x1
    x0 = np.mean([x2, x1])
    return fwhm, x0

def makeprofile_compds(formulas,concentrations=None,filename=os.path.join( data_installation_dir,'ComptonProfiles.dat'),E0=9.69,tth=35.0,correctasym=None):
    """
    returns sum of compton profiles from a lost of chemical compounds weighted by the given concentration
    """
    # if correctasym is not given, no HR correction is applied 
    if not np.any(concentrations):
        concentrations = np.ones(len(formulas))/len(formulas)
    if not np.any(correctasym):
        correctasym = []
        for formula in formulas:
            elements,stoichiometries = parseformula(formula)
            correctasym.append(np.zeros(len(elements)))
    
    eloss,J,C,V,q = makeprofile_comp(formulas[0],filename,E0,tth,correctasym[0])
    if len(formulas)>1:
        J = J*concentrations[0]
        C = C*concentrations[0]
        V = V*concentrations[0]
        for n in range(len(formulas[1:])):
            eloss,j,c,v,q = makeprofile_comp(formulas[n+1],filename,E0,tth,correctasym[n+1])
            J += j*concentrations[n+1]
            C += c*concentrations[n+1]
            V += v*concentrations[n+1]
    return eloss,J,C,V,q


def makeprofile_comp(formula,filename=os.path.join(data_installation_dir,'ComptonProfiles.dat'),E0=9.69,tth=35,correctasym=None):
    """
    returns the compton profile of a chemical compound with formula 'formula'
    input:
    formula = string of a chemical formula (e.g. 'SiO2', 'Ba8Si46', etc.)
    filename = path and filename to tabulated profiles
    E0       = scattering energy [keV]
    tth      = scattering angle  [deg]
    returns:
    eloss = energy loss scale
    J = total CP
    C = only core contribution to CP
    V = only valence contribution to CP
    q = momentum transfer [a.u.]
    """
    elements,stoichiometries = parseformula(formula)
    
    if not np.any(correctasym):
        correctasym = np.zeros(len(elements))
        
    eloss,J,C,V,q = makeprofile(elements[0],filename,E0,tth,correctasym[0])
    J *= stoichiometries[0]
    C *= stoichiometries[0]
    V *= stoichiometries[0]

    for n in range(len(elements[1:])):
        eloss,j,c,v,q = makeprofile(elements[n+1],filename,E0,tth,correctasym[n+1])
        J += j*stoichiometries[n+1]
        C += c*stoichiometries[n+1]
        V += v*stoichiometries[n+1]
    return eloss, J,C,V,q

def makeprofile(element,filename=os.path.join(data_installation_dir,'ComptonProfiles.dat'),E0=9.69,tth=35.0,correctasym=None):
    """
    takes the profiles from 'makepzprofile()', converts them onto eloss 
    scale and normalizes them to S(q,w) [1/eV]
    input:
    element  = element symbol (e.g. 'Si', 'Al', etc.)
    filename = path and filename to tabulated profiles
    E0       = scattering energy [keV]
    tth      = scattering angle  [deg]
    returns:
    enscale = energy loss scale
    J = total CP
    C = only core contribution to CP
    V = only valence contribution to CP
    q = momentum transfer [a.u.]
    """
    pzprofile,binden,occ = makepzprofile(element,filename)
    # convert to eloss scale
    enscale = ((np.flipud(pz2e1(E0,pzprofile[:,0],tth))-E0)*1e3)
    q = momtrans_au(enscale/1000.0+E0,E0,tth)
    # add asymmetry if needed (2p1/2 and 2p3/2 for Z > 35 (Br))
    asymmetry = np.flipud(HRcorrect(pzprofile,occ,q));  # asymmetry flipped for conversion to e-loss scale (???)
    if correctasym:
        pzprofile[:,1:4] = pzprofile[:,1:4] + asymmetry*correctasym

    # discard profiles below zero
    hfprofile = pzprofile[np.nonzero(enscale.T>=0)[0],:]
    q         = q[np.nonzero(enscale.T>=0)[0]] #q[:,np.nonzero(enscale.T>=0)[0]]
    enscale   = enscale[np.nonzero(enscale.T>=0)[0]] #enscale[:,np.nonzero(enscale.T>=0)[0]]
    hfprofile[:,0] = enscale
    # cut at edges
    for n in range(len(binden)):
        hfprofile[np.where(enscale<binden[n]),n+1] = 0 
    # convert J(pz) to S(q,w) via J(pz)=N_electrons*hartree*q*S(q,w) and
    # normalize using the f-sum rule (sum(S(q,w)*w)=f)
    # convert to a.u.
    hartree = 1.0/constants.physical_constants['electron volt-hartree relationship'][0]
    enscaleh = enscale/hartree # eloss in a.u.
    # normalize to one then multiply by N_el*q**2/2
    for n in range(len(binden)):
        hfprofile[:,n+1] = hfprofile[:,n+1]/(trapezoid(np.multiply(hfprofile[:,n+1],enscaleh),enscaleh))
        hfprofile[:,n+1] = np.multiply(hfprofile[:,n+1],(q**2.0)/2.0)*occ[n]
    # convert back to [1/eV] and sum up
    # total profile J and valence V (all edges )
    J = np.zeros((len(enscale)))
    V = np.zeros((len(enscale)))
    for n in range(len(binden)):
        if binden[n] < enscale[-1]:
            J += hfprofile[:,n+1]/hartree
            if binden[n] < 10:
                V += hfprofile[:,n+1]/hartree
    C = J - V
    return enscale,J,C,V,q

def makepzprofile(element,filename=os.path.join(data_installation_dir,'ComptonProfiles.dat')):
    """
    constructs compton profiles of element 'element' on pz-scale 
    (-100:100 a.u.) from the Biggs tables provided in 'filename'

    input:
      * element   = element symbol (e.g. 'Si', 'Al', etc.)
      * filename  = path and filename to tabulated profiles

    returns:
      * pzprofile = numpy array of the CP:
        *  1. column: pz-scale
        *  2. ... n. columns: compton profile of nth shell
        * binden     = binding energies of shells
        * occupation = number of electrons in the according shells
    """
    theory,occupation,binden,colnames = readbiggsdata(filename,element)
    # first spline onto a rough grid:
    roughpz = np.logspace(0.01,2,65)-1
    roughtheory      = np.zeros((len(roughpz),len(binden)+2))
    roughtheory[:,0] = roughpz
    for n in range(len(binden)+1):
        intf               = interpolate.pchip(theory[:,0], theory[:,2]) # interpolate.interp1d(theory[:,0],theory[:,n+1])
        roughtheory[:,n+1] = intf(roughpz)
    pzscale   = np.linspace(-100,100,num=4000)
    pzprofile      = np.zeros((len(pzscale),len(binden)+1))
    pzprofile[:,0] = pzscale     
    # mirror, spline onto fine grid
    for n in range(len(binden)):
        intf             = interpolate.splrep(roughtheory[:,0],roughtheory[:,n+2],s=0.000000001,k=2) # skip the column with the total J for now #try interp1d with bounds_error=False and fill_value=0.0
        pzprofile[:,n+1] = interpolate.splev(abs(pzscale),intf,der=0)
    # normalize to one electron, multiply by number of electrons
    for n in range(len(binden)):
        normval = trapezoid(pzprofile[:,n+1],pzprofile[:,0])
        pzprofile[:,n+1] = pzprofile[:,n+1]/normval*int(occupation[n])
    binden     = [float(en) for en in binden]
    occupation = [float(val) for val in occupation]
    return pzprofile, binden, occupation


def momtrans_au(e1,e2,tth):
    """ Returns the momentum transfer (in a.u.).

    Calculates the momentum transfer in atomic units for two given
    energies e1 and e1 (in keV) and the scattering angle tth (two theta).

    Args:
      *e1 (float or np.array): incident energy in [keV], can be a single value or a vector
      *e2 (float or np.array): scattered energy in [keV], can be a single value or a vector
      *tth (float): scattering angle two theta in [deg]

    Returns:
      * q (float or np.array): momentum transfer [a.u.], single value or vector depending on input
    """
    e1    = np.array(e1*1.0e3/13.60569172/2.0)
    e2    = np.array(e2*1.0e3/13.60569172/2.0)
    th    = math.radians(tth)#tth/180.0*np.pi
    hbarc = 137.03599976
    q     = 1/hbarc*np.sqrt(e1**2.0+e2**2.0-2.0*e1*e2*np.cos(th));
    return q

def HRcorrect(pzprofile,occupation,q):
    """ Returns the first order correction to filled 1s, 2s, and 2p Compton profiles.

    Implementation after Holm and Ribberfors (citation ...).

    Args: 
      * pzprofile (np.array): Compton profile (e.g. tabulated from Biggs) to be corrected (2D matrix). 
      * occupation (list): electron configuration.
      * q (float or np.array): momentum transfer in [a.u.].

    Returns:
       asymmetry (np.array):  asymmetries to be added to the raw profiles (normalized to the number of electrons on pz scale)
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
            forgamma = 3.0*pzprofile[:,3]/trapezoid(pzprofile[:,3],pzprofile[:,0]) # 2p correction is defined for 3 electrons in the 2p shell
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


def readbiggsdata(filename,element):
    """
    Reads Hartree-Fock Profile of element 'element' from values tabulated 
    by Biggs et al. (Atomic Data and Nuclear Data Tables 16, 201-309 (1975))
    as provided by the DABAX library (http://ftp.esrf.eu/pub/scisoft/xop2.3/DabaxFiles/ComptonProfiles.dat).
    input:
    filename = path to the ComptonProfiles.dat file (the file should be distributed with this package)
    element  = string of element name
    returns:

      * data     = the data for the according element as in the file:
          * #UD  Columns: 
          * #UD  col1: pz in atomic units 
          * #UD  col2: Total compton profile (sum over the atomic electrons
          * #UD  col3,...coln: Compton profile for the individual sub-shells

      * occupation = occupation number of the according shells
      * bindingen  = binding energies of the accorting shells
      * colnames   = strings of column names as used in the file
    """
    elementid = '#S'
    sizeid    = '#N'
    occid     = '#UOCCUP'
    bindingid = '#UBIND'
    colnameid = '#L'
    data = []
    f = open(filename,'r')
    istrue = True
    while istrue:
        line = f.readline()
        if line[0:2] == elementid:
            if line.split()[-1] == element:
                line = f.readline()
                while line[0:2] != elementid:
                    if line[0:2] == sizeid:
                        arraysize = int(line.split()[-1])
                        line = f.readline()
                    if line[0:7] == occid:
                        occupation = line.split()[1:]
                        line = f.readline()
                    if line[0:6] == bindingid:
                        bindingen = line.split()[1:]    
                        line = f.readline()
                    if line[0:2] == colnameid:
                        colnames = line.split()[1:]
                        line = f.readline()
                    if line[0]== ' ':
                        data.append([float(n) for n in line.strip().split()])
                        #data = np.zeros((31,arraysize))
                        line = f.readline()
                break
    length = len(data)
    data = (np.reshape(np.array(data),(length,arraysize)))
    return data, occupation, bindingen, colnames
