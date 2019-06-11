#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 21 16:40:34 2019

@author: purser
"""
import numpy
import scipy.optimize
import uncertainties
import matplotlib.pylab
import datetime
import astropy.coordinates
from scipy.optimize import curve_fit


if __name__ == "__main__":
    import jetfunctions as jf
    import mathfunctions as mf
    import miscfunctions as miscf

else:
    from . import jetfunctions as jf
    from . import mathfunctions as mf
    from . import miscfunctions as miscf

# Attributes that are always exported:
__all__ = ['Coordinate', 'Flux', 'Size', 'RadioSED', 'Observation']

class Coordinate(astropy.coordinates.SkyCoord):
    def __init__(self, ra, dec, ra_err, dec_err, freq, obs_date=None,
                 *args, **kwargs):
        # super() method directly inherits method of parent class
        super().__init__(ra, dec, *args, **kwargs)
        if type(ra_err) is str and miscf.is_float(ra_err):
            ra_err = float(ra_err)
        if type(dec_err) is str and miscf.is_float(dec_err):
            dec_err = float(dec_err)
        if type(ra_err) is str and ra_err in ('', '-'):
            ra_err = 0.1
        if type(dec_err) is str and dec_err in ('', '-'):
            dec_err = 0.1
        self.raerr = ra_err
        self.decerr = dec_err
        self.freq = freq
        if obs_date is not None:
            assert type(obs_date) is datetime.datetime,\
                   "obs_date must be instance of datetime.datetime class"
            self.obsdate = obs_date
        else:
            self.obsdate = None

    def separation(self, coord2):
        if coord2 is None:
            return None, None
        # assert isinstance(coord2, Coordinate), "coord2 must be Coordinate instance"
        sep = mf.angsep(self, coord2, (self.raerr, self.decerr),
                        (coord2.raerr, coord2.decerr))
        pa = mf.posang(self, coord2, (self.raerr, self.decerr),
                       (coord2.raerr, coord2.decerr))

        return sep, pa

class Flux:
    """
    Class to handle radio fluxes
    """
    def __init__(self, flux, flux_err, freq, AFE=0.1, telescope=None,
                 up_lim=False, obs_date=None):
        """
        INPUTS
        ------
        flux      : Flux in Jy (float)
        flux_err  : Error in radio flux in Jy (float)
        freq      : Frequency of radio flux in Hz (float)
        AFE       : Fractional absolute flux uncertainty (float)
        telescope : Telescope used to record flux (None or str)
        up_lim    : Whether this is an upper limit (boolean)
        obs_date  : When the data was recorded. None by default, otherwise must
                   be a datetime.datetime instance
        """
        #assert type(flux) is float, "flux needs to be a float"
        #assert type(flux_err) is float, "flux_err needs to be a float"
        #assert type(freq) is float, "freq needs to be a float"
        #assert type(AFE) is float, "AFE needs to be a float"
        assert isinstance(flux, float), "flux needs to be a float"
        assert isinstance(flux_err, float), "flux_err needs to be a float"
        assert isinstance(freq, float), "freq needs to be a float"
        assert isinstance(AFE, float), "AFE needs to be a float"
        assert isinstance(telescope, (str, type(None))),\
               "telescope needs to be `None' or str"
        assert type(up_lim) is bool, "up_lim needs to be a bool"
        self.flux = flux
        self.flux_e = flux_err
        self.freq = freq
        self.AFE = AFE
        self._telescope = telescope
        self.upper_limit = up_lim
        if obs_date is not None:
            assert type(obs_date) is datetime.datetime,\
                   "obs_date must be instance of datetime.datetime class"
            self.obsdate = obs_date
        else:
            self.obsdate = None

    def wavelength(self):
        return 299792458.0 / self.get_freq()

    def get_freq(self):
        return self.freq + 0

    def get_flux(self):
        return uncertainties.ufloat(self.flux, (self.flux_e**2. +
                       (self.flux * self.AFE)**2.)**0.5)

    @property
    def telescope(self):
        return self._telescope

    @telescope.setter
    def telescope(self, instrument):
        self._telescope = instrument

class Size:
    """
    Class to handle radio flux dimensions
    """
    def __init__(self, major, major_err, minor, minor_err, pa, pa_err, freq,
                 up_lim=False, obs_date=None):
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
        obs_date  : When the data was recorded. None by default, otherwise must
                    be a datetime.datetime instance
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
        if obs_date is not None:
            assert type(obs_date) is datetime.datetime,\
                   "obs_date must be instance of datetime.datetime class"
            self.obsdate = obs_date
        else:
            self.obsdate = None

    def get_freq(self):
        return self.freq + 0

    def get_major(self):
        return uncertainties.ufloat(self.major, self.major_e) + 0

    def get_minor(self):
        return uncertainties.ufloat(self.minor, self.minor_e) + 0

    def get_pa(self):
        return uncertainties.ufloat(self.pa, self.pa_e) + 0

class RadioSED:
    """
    Class to handle all fluxes associated with radio source
    """
    def __init__(self, fluxes=None):
        if fluxes is None:
            self._fluxes = fluxes
        else:
            assert not isinstance(fluxes, str), "fluxes arg can not be str"
            if type(fluxes) is Flux:
                self._fluxes = numpy.array([fluxes])
            else:
                assert isinstance(fluxes, (list, tuple, numpy.ndarray)),\
                       "fluxes must not be of type " + type(fluxes)
                for flux in fluxes:
                    assert isinstance(flux, Flux),\
                           "fluxes elements must be instances of Flux class"
                self._fluxes = numpy.array(fluxes)

    @property
    def fluxes(self):
        return self._fluxes

    @fluxes.setter
    def fluxes(self, fluxes):
        self._fluxes = fluxes

    def add_flux(self, flux):
        """
        Appends measured flux to the SED. Must be Flux instance
        """
        assert isinstance(flux, Flux), "added flux must be Flux class instance"
        if self._fluxes is None:
            self.fluxes = numpy.array([flux])
        else:
            self.fluxes = numpy.append(self.fluxes, flux)

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
        self.coordinates = []
        self._spix = float('NaN')
        self._gamma = float('NaN')
        self._data = None

    @property
    def data(self):
        """
        Method to insert large datasets (e.g. from a .csv file) into this entry.
        data_entry should be a dictionary with keys being header values and 
        values being associated data values.
        """
        return self._data

    @data.setter
    def data(self, data_entry):
        self._data = data_entry

    @property
    def spix(self):
        """
        Get spectral index information
        """
        if uncertainties.umath.isnan(self._spix):
            self.spix = 'auto'
        return self._spix

    @spix.setter
    def spix(self, alpha):
        if alpha == 'auto':
            # Get rid of values that are NaNs or upper limits
            keep_idxs = []
            for idx, f in enumerate(self.fluxes):
                if not uncertainties.umath.isnan(f.get_flux()):
                    if not f.upper_limit:
                        keep_idxs.append(idx)
    
            nus = [[_.get_freq() for _ in self.fluxes][i] for i in keep_idxs]
            fs = [[_.get_flux() for _ in self.fluxes][i] for i in keep_idxs]
            if len(nus) < 2:
                self._spix = float('NaN')
            else:
                
                def log_power_law(log_x, m, c):
                    return log_x * m + c
                popt, pcov = curve_fit(log_power_law,
                                       [uncertainties.umath.log10(_) for _ in nus],
                                       [uncertainties.umath.log10(_).n for _ in fs],
                                       sigma=[uncertainties.umath.log10(_).s for _ in fs],
                                       absolute_sigma=True)
                stde = pcov.diagonal()**0.5
                if uncertainties.umath.isinf(stde[0]):
                    print('HALLP', uncertainties.ufloat(popt[0], stde[0]), self.myso, self.component)
                self._spix = uncertainties.ufloat(popt[0], stde[0])
        else:
            self._spix = alpha

    @property
    def gamma(self):
        """
        Get gamma (theta_maj power-law coefficient with frequency) information
        """
        if uncertainties.umath.isnan(self._gamma):
            self.gamma = 'auto'
        return self._gamma

    @gamma.setter
    def gamma(self, g):
        if g == 'auto':
            # Get rid of values that are NaNs or upper limits
            keep_idxs = []
            for idx, s in enumerate(self.sizes):
                if not uncertainties.umath.isnan(s.get_major()):
                    if not s.upper_limit:
                        keep_idxs.append(idx)
    
            nus = [[_.get_freq() for _ in self.sizes][i] for i in keep_idxs]
            ts = [[_.get_major() for _ in self.sizes][i] for i in keep_idxs]
            if len(nus) < 2:
                self._gamma = float('NaN')
            else:
                
                def log_power_law(log_x, m, c):
                    return log_x * m + c
                popt, pcov = curve_fit(log_power_law,
                                       [uncertainties.umath.log10(_) for _ in nus],
                                       [uncertainties.umath.log10(_).n for _ in ts],
                                       sigma=[uncertainties.umath.log10(_).s for _ in ts],
                                       absolute_sigma=True)
                stde = pcov.diagonal()**0.5
                if uncertainties.umath.isinf(stde[0]):
                    print('HALLP', uncertainties.ufloat(popt[0], stde[0]), self.myso, self.component)
                self._gamma = uncertainties.ufloat(popt[0], stde[0])
        else:
            self._gamma = g

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
        alpha = self.spix + 0
        if uncertainties.umath.isnan(alpha):
            self.spix = 'auto'
            alpha = self.spix

        if uncertainties.umath.isnan(alpha) and 'jet' in self.obj_type.lower():
            alpha = uncertainties.ufloat(0.6, 0.2)
        #print(alpha)
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
        """
        Method which returns a function with one argument (frequency) which
        return flux at that frequency.
        """
        # Get rid of values that are NaNs or upper limits
        keep_idxs = []
        for idx, f in enumerate(self.fluxes):
            if not uncertainties.umath.isnan(f.get_flux()):
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
        self.spix = spix
        return s_nu(desired_freq)

    def tmaj_nu(self, desired_freq):
        """
        Method which returns a function with one argument (frequency) which
        return major axis length at that frequency.
        """
        # Get rid of values that are NaNs or upper limits
        keep_idxs = []
        for idx, s in enumerate(self.sizes):
            if not uncertainties.umath.isnan(s.get_major()):
                if not s.upper_limit:
                    keep_idxs.append(idx)

        freqs = [[_.get_freq() for _ in self.sizes][i] for i in keep_idxs]
        t_majs = [[_.get_major() for _ in self.sizes][i] for i in keep_idxs]

        # Calculate SPIX and flux function (as lambda)
        if len(t_majs) == 1:
            gam = uncertainties.ufloat(float('NaN'), float('NaN'))
            tmaj_nu = lambda nu: uncertainties.ufloat(float('NaN'), float('NaN'))
            tmaj_nu.ref_freq = float('NaN')
            rpr = "float('NaN')"
            if 'jet' in self.obj_type.lower() or 'dw' in self.obj_type.lower():
                if not uncertainties.umath.isnan(self.spix):
                    gam = jf.gamma_from_alpha(self.spix)
                else:
                    gam = uncertainties.ufloat(-0.7, 0.3)
                    print(u"Assuming \u03B3 = -0.7 \u00B1 0.3 for " +
                          self.myso + ", component '" + self.component + "'")
                    # gam = uncertainties.ufloat(float('NaN'), float('NaN'))
                t_maj = t_majs[0]
                r_freq = freqs[0]
                tmaj_nu = lambda nu, rf=r_freq, g=gam, t=t_maj: (nu / rf)**g * t
                tmaj_nu.ref_freq = r_freq
                rpr = '('
                rpr += format(t_maj.n, '.2e') + "+/-" + format(t_maj.s, '.2e')
                rpr += ") * (nu / " + format(r_freq, '.2e')
                rpr += ")**(" + format(gam.n, '.2f') + "+/-"
                rpr += format(gam.s, '.2f') + ')'
            elif 'hii' in self.obj_type.lower():
                gam = uncertainties.ufloat(0., 0.)
                t_maj = t_majs[0]
                r_freq = freqs[0]
                tmaj_nu = lambda nu, rf=r_freq, g=gam, t=t_maj: (nu / rf)**g * t
                tmaj_nu.ref_freq = r_freq
                rpr = '('
                rpr += format(t_maj.n, '.2e') + "+/-" + format(t_maj.s, '.2e')
                rpr += ") * (nu / " + format(r_freq, '.2e')
                rpr += ")**(" + format(gam.n, '.2f') + "+/-"
                rpr += format(gam.s, '.2f') + ')'

        # LSQ fit in case of more than 1 frequency
        elif len(t_majs) >= 2:
            # Power law whose reference frequency controls the errors more
            # sensibly, rather than y-intercept dominated errors
            def tailored_plaw(ref_freq):
                def plaw(freq, a, c):
                    return (freq / ref_freq)**a * c
                return plaw
            
            # Need to calculate weighted average frequency (r_freq)...            
            weights = numpy.array([_.s**-2. for _ in t_majs])
            tot_weight = numpy.sum(weights)
            weights /= tot_weight

            r_freq = 10**numpy.nansum(weights * numpy.log10(freqs))
            popt1, pcov1 = scipy.optimize.curve_fit(tailored_plaw(r_freq), freqs,
                                     [_.n for _ in t_majs],
                                     sigma=[_.s for _ in t_majs],
                                     absolute_sigma=True)
            std_err1 = pcov1.diagonal()**0.5
            gam = uncertainties.ufloat(popt1[0], std_err1[0])
            tmaj_nu = lambda nu, rf=r_freq, p=popt1, se=std_err1: (nu / rf)**uncertainties.ufloat(p[0], se[0]) * uncertainties.ufloat(p[1], se[1])
            tmaj_nu.ref_freq = r_freq
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
            gam = uncertainties.ufloat(float('NaN'), float('NaN'))
            tmaj_nu = lambda nu: uncertainties.ufloat(float('NaN'), float('NaN'))
            tmaj_nu.ref_freq = float('NaN')
            rpr = "float('NaN')"
        tmaj_nu.__repr__ = rpr        
        self.gamma = gam
        return tmaj_nu(desired_freq)

    def add_flux(self, flux):
        assert type(flux) is Flux, "Added flux must be instance of Flux class"
        self.fluxes.append(flux)
        
    def add_size(self, size):
        assert type(size) is Size, "Added size must be instance of Size class"
        self.sizes.append(size)

    def add_coordinate(self, coord):
        assert type(coord) is Coordinate,\
               "Added coord must be instance of Coordinate class"
        self.coordinates.append(coord)

    def plot_fluxes(self, ax=None, errorbars=True, save_pdf=False, log=True, **kwargs):
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
        
        if ax is None:
            matplotlib.pylab.close('all')
            fig, ax = matplotlib.pylab.subplots(1, 1, figsize=(3.15, 3.15))

        if log:
            ax.set_xscale('log')
            ax.set_yscale('log')       

        # Plot detections
        if errorbars:
            print(nus[~numpy.array(uplim_mask)])
            ax.errorbar(nus[~numpy.array(uplim_mask)],
                        fs[~numpy.array(uplim_mask)],
                        yerr=f_es[~numpy.array(uplim_mask)], ls='None',
                        capsize=2, **kwargs)
#        else:
        ax.plot(nus[~numpy.array(uplim_mask)], fs[~numpy.array(uplim_mask)],
                ls='None', mfc='k', marker='o', **kwargs)

        # Plot upper-limits
        ax.errorbar(nus[numpy.array(uplim_mask)], fs[numpy.array(uplim_mask)],
                    yerr=0.2 * fs[numpy.array(uplim_mask)], uplims=True,
                    fmt='None', **kwargs)

        matplotlib.pylab.show()
        
        if save_pdf:
            fig.savefig(save_pdf, dpi=300.)

        return matplotlib.pylab.gcf(), ax

if __name__ == '__main__':
    import astropy.units as u
    import datetime as dt
    od = dt.datetime(2015, 4, 12)
    a = Coordinate(12.4, 0.21, 3, 3.2, 9e9, obs_date=od, unit=(u.rad, u.rad))
    b = Coordinate(12.3214, 0.21423, 4.53, 3.42, 9e9, unit=(u.rad, u.rad))
    print(a.obsdate)
    print(b.obsdate)

