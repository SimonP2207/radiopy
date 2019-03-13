# -*- coding: utf-8 -*-
"""
Contains all miscellaneous python handling methods
"""
import numpy as np
from uncertainties import umath as um
from collections.abc import Iterable

def remove_nans(*args):
    """
    Remove NaNs from all supplied lists/arrays on the basis of NaN presence in
    the list supplied to the control_arr argument.
    """
    nan_idxs = []
    for arg in args:
        for idx, c in enumerate(arg):
            if not isinstance(e, Iterable) and um.isnan(c):
                nan_idxs.append(idx)
    nan_idxs = list(set(nan_idxs))
    not_nan_idxs = [_ for _ in range(len(args[0])) if _ not in nan_idxs]
    return np.array(args)[:, not_nan_idxs]

def is_float(x):
    """
    Check if the supplied argument, x, can be type-cast as a float.
    """
    try:
        float(x)
        return True
    except ValueError:
        return False

