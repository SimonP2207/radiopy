#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 16:40:34 2019

@author: purser
"""
import numpy
import scipy.optimize
import uncertainties
import uncertainties
import matplotlib.pylab
from . import jetfunctions as jf


# Attributes that are always exported:
__all__ = ['Flux', 'Size', 'Observation']


class Flux:
    """
    Class to handle radio fluxes
    """
    def __init__(self, flux, flux_err, freq, AFE=0.1, up_lim=False):
        """
        INPUTS
        ------
        flux     : Flux (float)
        flux_err : Error in radio flux (float)
        freq     : Frequency of radio flux (float)
        AFE      : Absolute flux uncertainty (float)
        up_lim   : Whether this is an upper limit (boolean)
        """
        assert type(flux) is float, "flux needs to be a float"
        assert type(flux_err) is float, "flux_err needs to be a float"
        assert type(freq) is float, "freq needs to be a float"
        assert type(AFE) is float, "AFE needs to be a float"
        assert type(up_lim) is bool, "up_lim needs to be a bool"
        self.flux = flux
        self.flux_e = flux_err
        self.freq = freq
        self.AFE = AFE
        self.upper_limit = up_lim

    def get_freq(self):
        return self.freq + 0

    def get_flux(self):
        return uncertainties.ufloat(self.flux, (self.flux_e**2. +
                       (self.flux * self.AFE)**2.)**0.5)

class Size:
    """
    Class to handle radio flux dimensions
    """
    def __init__(self, major, major_err, minor, minor_err, pa, pa_err, freq,
                 up_lim=False):
        """
        INPUTS
        ------
        major     : Flux (float or NaN)
        major_err : Error in radio flux (float or NaN)
        minor     : Flux (float or NaN)
        minor_err : Error in radio flux (float or NaN)
        pa        : Flux (float or NaN)
        pa_err    : Error in radio flux (float or NaN)
        freq      : Frequency of observation
        up_lim    : Whether this is an upper limit (boolean)
        """
        self.major = major
        self.major_e = major_err
        self.minor = minor
        self.minor_e = minor_err
        self.pa = pa
        self.pa_e = pa_err
        self.freq = freq
        self.upper_limit = up_lim
        oa = 2. * uncertainties.umath.atan(uncertainties.ufloat(self.minor,
                                                                self.minor_e) /
                                           uncertainties.ufloat(self.major,
                                                                self.major_e))
        self.opang = uncertainties.umath.degrees(oa)

    def get_freq(self):
        return self.freq + 0

    def get_major(self):
        return uncertainties.ufloat(self.major, self.major_e) + 0

    def get_minor(self):
        return uncertainties.ufloat(self.minor, self.minor_e) + 0

    def get_pa(self):
        return uncertainties.ufloat(self.pa, self.pa_e) + 0

class Observation:
    """
    Class to handle all data associated with observation of an MYSO
    """
    def __init__(self, name, comp, obj_type, sample, distance, dist_err):
        """
        INPUTS
        ------
        name     : MYSO name (string)
        comp     : Component name (string)
        obj_type : Type of object's component e.g. jet, Hii etc. (string)
        sample   : Name of sample to which observation belongs (string)
        distance : Distance to MYSO in kiloparsecs (float)
        dist_err : Error in distance to MYSO in kiloparsecs (float)
        """
        self.myso = name
        self.component = comp
        self.obj_type = obj_type
        self.sample = sample
        self.dist = distance
        self.dist_e = dist_err
        self.fluxes = []
        self.sizes = []
        self.spix = float('NaN')

    def get_dist(self):
        return uncertainties.ufloat(self.dist, self.dist_e)

    def lbol_from_fbol(self, fbol, fbol_err):
        """
        Calculate bolometric luminosity of MYSO from its bolometric flux
        """
        fbol_uf = uncertainties.ufloat(fbol, fbol_err)
        bol_lum = 31256404982224.164 * fbol_uf * self.get_dist()**2.
        self.lbol = bol_lum + 0
        return bol_lum

    def jml(self):
        # JML(flux, freq, alpha, dist, opang)
        """
        Returns the jet mass loss rate in units of M_sol yr^{-1}
        """
        d = uncertainties.ufloat(self.dist, self.dist_e)
        alpha = self.spix
        
        if uncertainties.umath.isnan(alpha) and 'jet' in self.obj_type.lower():
            alpha = uncertainties.ufloat(0.6, 0.2)
        
        jmls = []
        for flux in self.fluxes:
            if flux.upper_limit:
                continue
            f = uncertainties.ufloat(flux.flux, flux.flux_e) * 1E3
            freq = flux.freq / 1E9

            opang = uncertainties.ufloat(float('NaN'), float('NaN'))
            for size in self.sizes:
                if size.freq == freq:
                    opang = size.opang

            jmls.append((jf.JML(f, freq, alpha, d, opang),
                         (f, freq, alpha, d, opang)))
        return jmls
        

    def s_nu(self, desired_freq):
        # Get rid of values that are NaNs or upper limits
        keep_idxs = []
        for idx, f in enumerate(self.fluxes):
            if not uncertainties.umath.isnan(f.get_flux())
                if not f.upper_limit:
                    keep_idxs.append(idx)

        freqs = [[_.get_freq() for _ in self.fluxes][i] for i in keep_idxs]
        fluxes = [[_.get_flux() for _ in self.fluxes][i] for i in keep_idxs]

        # Calculate SPIX and flux function (as lambda)
        if len(fluxes) == 1:
            spix = uncertainties.ufloat(float('NaN'), float('NaN'))
            s_nu = lambda nu: uncertainties.ufloat(float('NaN'), float('NaN'))
            s_nu.ref_freq = float('NaN')
            rpr = "float('NaN')"
            if 'jet' in self.obj_type.lower():
                alpha = uncertainties.ufloat(0.6, 0.2)
                flux = fluxes[0]
                r_freq = freqs[0]
                s_nu = lambda nu, rf=r_freq, s=alpha, f=flux: (nu / rf)**s * f
                s_nu.ref_freq = r_freq
                rpr = '('
                rpr += format(flux.n, '.2e') + "+/-" + format(flux.s, '.2e')
                rpr += ") * (nu / " + format(r_freq, '.2e')
                rpr += ")**(" + format(alpha.n, '.2f') + "+/-"
                rpr += format(alpha.s, '.2f') + ')'
            elif 'hii' in self.obj_type.lower():
                alpha = uncertainties.ufloat(-0.1, 0.2)
                flux = fluxes[0]
                r_freq = freqs[0]
                s_nu = lambda nu, rf=r_freq, s=alpha, f=flux: (nu / rf)**s * f
                s_nu.ref_freq = r_freq
                rpr = '('
                rpr += format(flux.n, '.2e') + "+/-" + format(flux.s, '.2e')
                rpr += ") * (nu / " + format(r_freq, '.2e')
                rpr += ")**(" + format(alpha.n, '.2f') + "+/-"
                rpr += format(alpha.s, '.2f') + ')'

        # LSQ fit in case of more than 1 frequency
        elif len(fluxes) >= 2:
            # Power law whose reference frequency controls the errors more
            # sensibly, rather than y-intercept dominated errors
            def tailored_plaw(ref_freq):
                def plaw(freq, a, c):
                    return (freq / ref_freq)**a * c
                return plaw
            
            # Need to calculate weighted average frequency (r_freq)...            
            weights = numpy.array([_.s**-2. for _ in fluxes])
            tot_weight = numpy.sum(weights)
            weights /= tot_weight

            r_freq = 10**numpy.nansum(weights * numpy.log10(freqs))
            popt1, pcov1 = scipy.optimize.curve_fit(tailored_plaw(r_freq), freqs,
                                     [_.n for _ in fluxes],
                                     sigma=[_.s for _ in fluxes],
                                     absolute_sigma=True)
            std_err1 = pcov1.diagonal()**0.5
            spix = uncertainties.ufloat(popt1[0], std_err1[0])
            s_nu = lambda nu, rf=r_freq, p=popt1, se=std_err1: (nu / rf)**uncertainties.ufloat(p[0], se[0]) * uncertainties.ufloat(p[1], se[1])
            s_nu.ref_freq = r_freq
            rpr = '('
            rpr += format(popt1[1], '.2e') + "+/-" + format(std_err1[1], '.2e')
            rpr += ") * (nu / " + format(r_freq, '.2e')
            rpr += ")**(" + format(popt1[0], '.2f') + "+/-"
            rpr += format(std_err1[0], '.2f') + ')'
            del popt1
            del pcov1
            del std_err1
            del r_freq

        else:
            spix = uncertainties.ufloat(float('NaN'), float('NaN'))
            s_nu = lambda nu: uncertainties.ufloat(float('NaN'), float('NaN'))
            s_nu.ref_freq = float('NaN')
            rpr = "float('NaN')"
        s_nu.__repr__ = rpr        
        self.set_spix = spix
        return s_nu(desired_freq)

    def add_flux(self, flux):
        assert type(flux) is Flux, "Added flux must be instance of Flux class"
        self.fluxes.append(flux)
        
    def add_size(self, size):
        assert type(size) is Size, "Added size must be instance of Size class"
        self.sizes.append(size)

    def set_spix(self, spix):
        self.spix = spix

    def plot_fluxes(self, errorbars=True, save_pdf=False, log=True, **kwargs):
        """
        Plots fluxes of object using matplotlib
        
        INPUTS
        ------
        errorbars : Whether to plot as errorbars or points
        save_pdf  : Save copy of plot as .pdf? If a string, should be the full
                    path to save plot under (False or string)
        log       : Logarithmic axes? (boolean)
        **kwargs  : Passed to plotting method of matplotlib
        """
        nus = numpy.array([_.get_freq() for _ in self.fluxes])
        fs = numpy.array([_.get_flux().n for _ in self.fluxes])
        f_es = numpy.array([_.get_flux().s for _ in self.fluxes])

        uplim_mask = numpy.array([_.upper_limit for _ in self.fluxes])

        matplotlib.pylab.close('all')

        fig, ax = matplotlib.pylab.subplots(1, 1, figsize=(3.15, 3.15))

        if log:
            ax.set_xscale('log')
            ax.set_yscale('log')       

        # Plot detections
        if errorbars:
            ax.errorbar(nus[~numpy.array(uplim_mask)], fs[~numpy.array(uplim_mask)],
                        yerr=f_es[~numpy.array(uplim_mask)], fmt='None', capsize=2,
                        **kwargs)
        else:
            ax.plot(nus[~numpy.array(uplim_mask)], fs[~numpy.array(uplim_mask)],
                    fmt='None', mfc='k', marker='o', **kwargs)

        # Plot upper-limits
        ax.errorbar(nus[numpy.array(uplim_mask)], fs[numpy.array(uplim_mask)],
                    yerr=0.2 * fs[numpy.array(uplim_mask)], uplims=True,
                    fmt='None', **kwargs)

        matplotlib.pylab.show()
        
        if save_pdf:
            fig.savefig(save_pdf, dpi=300.)

        return fig, ax