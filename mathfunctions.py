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