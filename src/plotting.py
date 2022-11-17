"""
Plotting utilities for experiments.
"""

import logging
import math

from pathlib import Path
from typing import Optional, Union

import numpy as np
import scipy.stats

from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import FigureBase

logging.getLogger('matplotlib').setLevel(logging.WARNING)

Pathlike = Union[str, Path]

def subplots(
    n_plots,
    scale_factor=4,
    sharex=True,
    sharey=True,
    **kwargs
    ) -> tuple[FigureBase, list[Axes]]:
    """
    Create nicely sized and laid-out subplots for a desired number of plots.
    """
    # essentially we want to make the subplots as square as possible
    # number of rows is the largest factor of n_plots less than sqrt(n_plots)
    options = range(1, int(math.sqrt(n_plots) + 1))
    n_rows = max(filter(lambda n: n_plots % n == 0, options))
    n_cols = int(n_plots / n_rows)
    # now generate the Figure and Axes pyplot objects
    # cosmetic scale factor to make larger plot
    figsize = (n_cols * scale_factor, n_rows * scale_factor)
    fig, axs = plt.subplots(
        n_rows, n_cols, figsize=figsize,
        sharex=sharex, sharey=sharey,
        **kwargs
        )
    flattened_axes = []
    for ax_row in axs:
        if isinstance(ax_row, np.ndarray):
            flattened_axes += list(ax_row)
        else:
            flattened_axes.append(ax_row)
    return fig, flattened_axes

def plot_gaussians(
    means,
    covs,
    arena_dims: Union[tuple, np.ndarray],
    save_path: Optional[Pathlike] = None
    ) -> tuple[FigureBase, list[Axes]]:
    """
    Create and save contour plots of the Gaussians parameterized
    by the given means and covariance matrices.
    """
    if len(means) != len(covs):
        raise ValueError(
            f'Number of means ({len(means)}) does not match '
            f'number of covariance matrices ({len(covs)})!'
            )

    xdim = arena_dims[0]
    ydim = arena_dims[1]
    # change xgrid / ygrid size to preserve aspect ratio
    ratio = ydim / xdim
    xs = np.linspace(0, xdim, 200)
    ys = np.linspace(0, ydim, int(ratio * 200))

    xgrid, ygrid = np.meshgrid(xs, ys)

    coord_grid = np.dstack((xgrid, ygrid))

    fig, axs = subplots(len(means))

    for ax, mean, cov in zip(axs, means, covs):
        distr = scipy.stats.multivariate_normal(mean, cov)
        pdf = distr.pdf(coord_grid)

        ax.contourf(xgrid, ygrid, pdf)
        ax.set_aspect('equal', 'box')

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)

    return (fig, axs)
