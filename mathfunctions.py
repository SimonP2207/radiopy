import scipy.constants as con
from uncertainties import umath
from uncertainties import ufloat


def tailored_plaw(ref_freq):
    """
    Creates power law function with variable reference value for frequency
    (i.e. x_0)
    """
    def plaw(freq, a, c):
        return (freq / ref_freq)**a * c
    return plaw

def power_law(x, a, c):
    """
    Power law function
    """
    return x**a * c

def log_power_law(log_x, m, c):
    """
    Logarithmic power law function
    """
    return log_x * m + c

def log_power_law_odr(B, log_x):
    """
    Logarithmic power law function for use with scipy's ODR package.
    """
    return log_x * B[0] + B[1]

def posang(coord1, coord2, coord1e, coord2e):
    """
    Calculates the position angle between two coordinates and its errors

    INPUTS
    ------
    coord1  : SkyCoord instance of coordinate to calculate position angle FROM
    coord2  : SkyCoord instance of coordinate to calculate position angle TO
    coord1e : 2-tuple of errors as (ra_err, dec_err) in arcsecs for coord1
    coord2e : 2-tuple of errors as (ra_err, dec_err) in arcsecs for coord2

    RETURNS
    -------
    Position angle from coord1 to coord2 as a uncertainties.ufloat instance
    """
    lon1e = coord1e[0] * con.arcsec
    lat1e = coord1e[1] * con.arcsec
    lon2e = coord2e[0] * con.arcsec
    lat2e = coord2e[1] * con.arcsec
    lon1 = ufloat(coord1.ra.rad, lon1e)
    lat1 = ufloat(coord1.dec.rad, lat1e)
    lon2 = ufloat(coord2.ra.rad, lon2e)
    lat2 = ufloat(coord2.dec.rad, lat2e)
      
    if lon1 > lon2:
        ra_diff = (lon1 - lon2) * umath.cos(lat1)
        if lat2 > lat1:
            dec_diff = lat2 - lat1
            pa = (umath.atan(dec_diff / ra_diff)) / (2. * con.pi) * 360. + 270.
        else:
            dec_diff = lat1 - lat2
            pa = 270. - (umath.atan(dec_diff / ra_diff)) / (2. * con.pi) * 360.

    else:
        ra_diff = (lon2 - lon1) * umath.cos(lat1)
        if lat2 > lat1:
            dec_diff = lat1 - lat2
            pa = (umath.atan(dec_diff / ra_diff)) / (2. * con.pi) * 360. + 90.
                                            
        else: 
            dec_diff = lat2 - lat1
            pa = 90. - (umath.atan(dec_diff / ra_diff)) / (2. * con.pi) * 360.

    return pa

def angsep(coord1, coord2, coord1e, coord2e): 
    """
    Calculates angular separation between two coordinates on the sky using the
    Haversine formula

    INPUTS
    ------
    coord1  : SkyCoord instance of 1st coordinate
    coord2  : SkyCoord instance of 2nd coordinate
    coord1e : 2-tuple of errors as (ra_err, dec_err) in arcsecs for coord1
    coord2e : 2-tuple of errors as (ra_err, dec_err) in arcsecs for coord2

    RETURNS
    -------
    Separation between coord1 and coord2 in arcsecs as a uncertainties.ufloat
    instance
    """
    lon1e = coord1e[0] * con.arcsec
    lat1e = coord1e[1] * con.arcsec
    lon2e = coord2e[0] * con.arcsec
    lat2e = coord2e[1] * con.arcsec
    lon1 = ufloat(coord1.ra.rad, lon1e)
    lat1 = ufloat(coord1.dec.rad, lat1e)
    lon2 = ufloat(coord2.ra.rad, lon2e)
    lat2 = ufloat(coord2.dec.rad, lat2e)
    
    angle = umath.atan(((umath.cos(lat2)**2. * umath.sin(lon2 - lon1)**2.) +
                        (umath.cos(lat1) * umath.sin(lat2) - umath.sin(lat1) *
                         umath.cos(lat2) * umath.cos(lon2 - lon1))**2.)**.5 /
                       (umath.sin(lat1) * umath.sin(lat2) + umath.cos(lat1) *
                        umath.cos(lat2) * umath.cos(lon2 - lon1)))

    return angle / con.arcsec
