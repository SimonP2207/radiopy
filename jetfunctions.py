# -*- coding: utf-8 -*-
"""
Contains all methods associated with radio observations of ionised jets
"""
import numpy as np
import scipy.constants as con
from uncertainties import ufloat as uf
from uncertainties import umath as um

def blackbody_nu(freq, temp):
    """Calculate blackbody flux per steradian, :math:`B_{\\nu}(T)`.

    Parameters
    ----------
    freq : Frequency in Hz.
    temp : Blackbody temperature in Kelvin.

    Returns
    -------
    Blackbody monochromatic flux in W m^{-2} s^{-1} Hz^{-1} sr^{-1}
    """
    # Convert to units for calculations, also force double precision

    log_boltz = con.h * freq / (con.k * temp)
    boltzm1 = um.exp(log_boltz) - 1.

    # Calculate blackbody flux
    bb_nu = 2. * con.h * freq**3. / (con.c**2. * boltzm1)

    return bb_nu

def op_thin_jml(nu, flux, major, minor, dist, inc=uf(57.3, 0.),
                T_e=uf(1E4, 0.), x_i=uf(0.12, 0.), vel=uf(500, 0.),
                mu=1.3, geometry='Conical'):
    """
    Calculates the jet mass loss rate assuming optically thin jet
    
    INPUTS
    ------
    nu    : Frequency (in GHz)
    flux  : Flux (in mJy)
    major : Major axis (in arcsec)
    minor : Minor axis (in arcsec)
    dist  : Distance (in kpc)
    inc   : Inclination (in degrees)
    T_e   : Electron temperature (in K)
    x_i   : Ionisation fraction
    vel     : Jet velocity (in km/s)
    mu    : Mean atomic weight
    geometry : Geometrical assumption. Either 'Conical' or 'Gaussian'

    RETURNS
    -------
    Jet mass loss rate in M_sol / yr
    """
    print(nu, flux, major, minor, dist)

    nu *= 1E9  # From GHz to Hz
    vel *= 1000.  # km/s to m/s
    flux *= 1E-29  # From mJy to W m^-2 Hz^-1
    dist *= 1000. * con.parsec  # from kpc to m

    major_m = um.tan(major * con.arcsec) * dist / um.sin(um.radians(inc))
    minor_m = um.tan(minor * con.arcsec) * dist

    # Volume of observed, optically-thin jet (assuming bi-conical geometry)
    if geometry == 'Conical':
        vol = np.pi * (minor_m / 2.)**2. * (major_m / 2.) / 3.  # m^3
        vol = np.pi * minor_m**2. * major_m / 12.  # m^3
        # ang_area = (minor / 2.) * ((major / um.sin(um.radians(inc))) / 2.) *\
        #            con.arcsec**2. # sr
        ang_area = 0.5 * minor * major * con.arcsec**2. # sr

    # Volume of observed, optically-thin jet (assuming Gaussian ellipsoid approx.)
    elif geometry == 'Gaussian':
        # vol = 4. / 3. * np.pi * (minor_m / 2.)**2. * (major_m / 2.)  # m^3
        vol = minor_m**2. * major_m * (np.pi / (4. * np.log(2)))**1.5 # m^3
        # In steradians
        ang_area = np.pi / (4. * np.log(2.)) * minor * major * con.arcsec**2.

    else:
        raise ValueError("geometry must be one of 'Conical' or 'Gaussian'")

    # Calculate optical depth
    # tau = -um.log(1. - flux / (blackbody_nu(nu, T_e) * ang_area))
    T_b = flux / ang_area * con.c**2. / (2. * con.k * nu**2.)
    tau = -um.log(1. - T_b / T_e)

    # Calculate Gaunt factor
    g_ff =  um.log(4.955e-2 * (nu / 1e9)**-1.) + 1.5 * um.log(T_e)

    # Calculate emission measure
    em_pccm6 = tau / (3.014E-2 * T_e**-1.5 * (nu / 1e9)**-2. * g_ff)

    # Calculate electron number density where minor is los thickness of jet
    ds = minor_m / (2. * um.sin(um.radians(inc)))  # 2 for average thickness
    n_e = um.sqrt(em_pccm6 / (ds / con.parsec))

    # Calculate mass
    mass = vol * n_e * 1E6 * con.u * 1.00794 * mu / x_i  # mass in kg
    mass_msol = mass / 1.989E30  # mass in M_sol

    # Assuming velocity constant, calculate ejection time
    ejection_time = (major_m / 2.) / vel

    # Calculate the mass loss rate assuming constant MLR
    mlr = mass_msol / (ejection_time / con.year)
    # print(mlr,
    #       (vol * (n_e * 1e6) * con.u * 1.00794 * mu / x_i/ejection_time) /
    #       (1.989E30 * (con.year)**-1))

    return mlr

def JML(flux, freq, alpha, dist, opang, v=uf(500., 0.), x_0=uf(0.12, 0.0),
        mu=1.3, nu_m=50., i=uf(57.3, 0.), T_e=1E4):
    """
    Calculates the jet mass loss rate as per Equation 19 of Reynolds (1986)
    
    INPUTS
    ------
    flux  : Flux (in mJy)
    freq  : Frequency (in GHz)
    alpha : Spectral index
    dist  : Distance (in kpc)
    opang : Opening angle (in degrees)
    v     : Jet velocity (in km/s)
    x_0   : Ionisation fraction
    mu    : Mean atomic weight
    nu_m  : Turnover frequency (GHz)
    i     : Inclination (in degrees)
    T_e   : Electron temperature (in K)
    
    RETURNS
    -------
    Jet mass loss rate in M_sol / yr
    """
    if alpha < -0.1:
        alpha = uf(-0.1, 0.3)
        #alpha = uf(float('NaN'), float('NaN'))
        
    if alpha < 0.:
        nu_m = freq
        
    # Assume one of three cases, based values for alpha:
    # Case 1: The standard, spherical model
    if alpha < 0.8 and alpha >= 0.4:
        epsilon = 1.
        q_T = 0.
        
    # Case 2: The conical, recombining model
    elif alpha >= 0.8:
        epsilon = 1.
        q_T = -0.5
    
    # Case 3: The standard, collimated model
    else:
        epsilon = 2. / 3.
        q_T = 0.
    
    q_tau = (2.1 * (1. + epsilon + q_T)) / (alpha - 2.)
    
    try:
    #    F = 4.41 / (q_tau * (alpha - 2.) * (alpha + 0.1))
        F = 4.41 / (q_tau * (alpha**2. - 1.9 * alpha -0.2))
    except ZeroDivisionError:
        F = 1
        #F = uf(float('NaN'), float('NaN'))
    if um.isnan(opang.s) or um.isnan(opang.n):
        opang = uf(35.2, 16.2)  # Average opang of ATCA and VLA datasets
        # opang = uf(float('NaN'), float('NaN'))
    part1 = 0.938 * (v / 1E3) * x_0**-1. * mu * (flux * (freq / 10.)**(-1. * alpha))**0.75
    part2 = dist**1.5 * (nu_m / 10.)**(-0.45 + 3. * alpha / 4.) * um.radians(opang)**0.75
    part3 = (T_e / 1E4)**-0.075 * um.sin(um.radians(i))**-0.25 * F**-0.75

    return part1 * part2 * part3 * 1E-6

def gamma_from_alpha(alpha):
    # Assume one of three cases, based values for alpha:
    # Case 1: The standard, spherical model
    if alpha < 0.8 and alpha >= 0.4:
        epsilon = 1.
        q_T = 0.
        
    # Case 2: The conical, recombining model
    elif alpha >= 0.8:
        epsilon = 1.
        q_T = -0.5
    
    # Case 3: The standard, collimated model
    else:
        epsilon = 2. / 3.
        q_T = 0.
    return (alpha - 2.) / (1. + epsilon + q_T)
