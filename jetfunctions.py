# -*- coding: utf-8 -*-
"""
Contains all methods associated with radio observations of ionised jets
"""
from uncertainties import ufloat as uf
from uncertainties import umath as um


def JML(flux, freq, alpha, dist, opang, v=uf(500., 0.), x_0=uf(0.2, 0.),
        mu=1.3, nu_m=50., i=uf(39.54, 0.), T_e=1E4):
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
        F = 4.41 / (q_tau * (alpha - 2.) * (alpha + 0.1))
    except ZeroDivisionError:
        F = 1.
    
    if um.isnan(opang.s) or um.isnan(opang.n):
        opang = uf(40., 20.)
    
    part1 = 0.938 * (v / 1E3) * x_0**-1. * mu * (flux * (freq / 10.)**(-1. * alpha))**0.75
    part2 = dist**1.5 * (nu_m / 10.)**(-0.45 + 3. * alpha / 4.) * um.radians(opang)**0.75
    part3 = (T_e / 1E4)**-0.075 * um.sin(um.radians(i))**-0.25 * F**-0.75
    
    return part1 * part2 * part3 * 1E-6