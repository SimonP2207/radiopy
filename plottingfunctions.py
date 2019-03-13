import numpy as np


def equalise_axes(ax):
    """
    Assuming equal figure dimensions (i.e. in inches), equalise the data ranges
    of the x and y-axes.
    """
    xlims = np.array(ax.get_xlim())
    ylims = np.array(ax.get_ylim())

    if ax.get_xscale() == 'log':
        xlims = np.log10(xlims)
        ylims = np.log10(ylims)

    x_range = np.max(xlims) - np.min(xlims)
    y_range = np.max(ylims) - np.min(ylims)
    
    x_midpoint = np.min(xlims) + x_range / 2.
    y_midpoint = np.min(ylims) + y_range / 2.
    
    if x_range < y_range:
        xlims = np.array([x_midpoint - y_range / 2, x_midpoint + y_range / 2])
    else:
        ylims = np.array([y_midpoint - x_range / 2, y_midpoint + x_range / 2])
    
    if ax.get_xscale() == 'log':
        xlims = 10.**xlims
        ylims = 10.**ylims

    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    
    return xlims, ylims