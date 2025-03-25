import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from typing import List
import warnings
import numpy as np
from matplotlib.collections import LineCollection


def colored_line(x, y, c, ax, **lc_kwargs):
    """
    Plot a line with a color specified along the line by a third value.

    It does this by creating a collection of line segments. Each line segment is
    made up of two straight lines each connecting the current (x, y) point to the
    midpoints of the lines connecting the current point with its two neighbors.
    This creates a smooth line with no gaps between the line segments.

    Parameters
    ----------
    x, y : array-like
        The horizontal and vertical coordinates of the data points.
    c : array-like
        The color values, which should be the same size as x and y.
    ax : Axes
        Axis object on which to plot the colored line.
    **lc_kwargs
        Any additional arguments to pass to matplotlib.collections.LineCollection
        constructor. This should not include the array keyword argument because
        that is set to the color argument. If provided, it will be overridden.

    Returns
    -------
    matplotlib.collections.LineCollection
        The generated line collection representing the colored line.
    """
    if "array" in lc_kwargs:
        warnings.warn('The provided "array" keyword argument will be overridden')

    # Default the capstyle to butt so that the line segments smoothly line up
    default_kwargs = {"capstyle": "butt"}
    default_kwargs.update(lc_kwargs)

    # Compute the midpoints of the line segments. Include the first and last points
    # twice so we don't need any special syntax later to handle them.
    x = np.asarray(x)
    y = np.asarray(y)
    x_midpts = np.hstack((x[0], 0.5 * (x[1:] + x[:-1]), x[-1]))
    y_midpts = np.hstack((y[0], 0.5 * (y[1:] + y[:-1]), y[-1]))

    # Determine the start, middle, and end coordinate pair of each line segment.
    # Use the reshape to add an extra dimension so each pair of points is in its
    # own list. Then concatenate them to create:
    # [
    #   [(x1_start, y1_start), (x1_mid, y1_mid), (x1_end, y1_end)],
    #   [(x2_start, y2_start), (x2_mid, y2_mid), (x2_end, y2_end)],
    #   ...
    # ]
    coord_start = np.column_stack((x_midpts[:-1], y_midpts[:-1]))[:, np.newaxis, :]
    coord_mid = np.column_stack((x, y))[:, np.newaxis, :]
    coord_end = np.column_stack((x_midpts[1:], y_midpts[1:]))[:, np.newaxis, :]
    segments = np.concatenate((coord_start, coord_mid, coord_end), axis=1)

    lc = LineCollection(segments, **default_kwargs)
    lc.set_array(c)  # set the colors of each segment

    return ax.add_collection(lc)

# Plots


# state space plot grid; color by time
def plot_state_space(df: pd.DataFrame, time_variable='time', variables: List[str]= []):
    """Plot each state variable against each other, colored by time.

    Args:
        df (pd.DataFrame): solution of an ODE model.
        time_variable (str, optional): column name of time variable. Defaults to 'time'.
        variables (List[str], optional): list of columns names to plot. Defaults to [].
    """
    if len(variables) < 1:
        variables = [c for c in df.columns if c != time_variable]
    n = len(variables)
    plt.figure(figsize=(4*n, 4*n))
    for i in range(n):
        for j in range(n):
            plt.subplot(n, n, i*n + j + 1)
            if i == j:
                sns.histplot(df[variables[i]], kde=True)
            else:
                plt.scatter(df[variables[j]].values, df[variables[i]].values,
                            s=0)
                ax = plt.gca()
                colored_line(df[variables[j]].values, df[variables[i]].values,
                             df[time_variable].values, ax=ax, cmap='jet',linewidth=1)
                ax.set_xlabel(variables[j])
                ax.set_ylabel(variables[i])
    plt.tight_layout()
# time series plot for (choice of) state variables

def plot_solution(df: pd.DataFrame, time_variable='time', variables: List[str]= [], **kwargs):
    """Plot the time series of each state variable from an ODE solution.

    Args:
        df (pd.DataFrame): solution of an ODE model.
        time_variable (str, optional): name of time column. Defaults to 'time'.
        variables (List[str], optional): List of variables to plot. Defaults to [].
    """
    if len(variables) < 1:
        variables = [c for c in df.columns if c != time_variable]
    n = len(variables)
    for i in range(n):
        plt.subplot(n, 1, i + 1)
        plt.plot(df[time_variable].values, df[variables[i]].values,
                             linewidth=0.5, color='k', **kwargs)
        plt.ylabel(variables[i])
    plt.xlabel(time_variable)
