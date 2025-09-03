import numpy as np

def compute_drift_from_surfaces(surfaces, theta1, theta2):
    """
    Compute best-point drift metrics from a stack of 2D surfaces.

    Parameters
    ----------
    surfaces : list or array, shape (T, H, W)
        Sequence of 2D return surfaces.
    theta1 : array-like, shape (W,)
        Grid of x-axis (θ1) values.
    theta2 : array-like, shape (H,)
        Grid of y-axis (θ2) values.

    Returns
    -------
    coords : ndarray, shape (T, 2)
        (x, y) coordinates of the absolute max per surface.
    deltas : ndarray, shape (T-1, 2)
        Stepwise drift vectors (coords[t+1] - coords[t]).
    magnitudes : ndarray, shape (T-1,)
        Drift magnitudes per interval.
    angles : ndarray, shape (T-1,)
        Drift directions (radians).
    z_vals : ndarray, shape (T,)
        Max return value per surface.
    """
    best_points = []
    for Z in surfaces:
        iy, ix = np.unravel_index(np.nanargmax(Z), Z.shape)
        x_best, y_best, z_best = theta1[ix], theta2[iy], Z[iy, ix]
        best_points.append([x_best, y_best, z_best])

    best_points = np.array(best_points)
    coords = best_points[:, :2]
    z_vals = best_points[:, 2]

    deltas = np.diff(coords, axis=0)
    magnitudes = np.linalg.norm(deltas, axis=1)
    angles = np.arctan2(deltas[:, 1], deltas[:, 0])

    return coords, deltas, magnitudes, angles, z_vals

import matplotlib.pyplot as plt
import seaborn as sns

def plot_drift_magnitude(
    magnitudes,
    trading_periods=None,
    title="Max-point drift magnitude over time",
    save_path=None,
    dpi=300,
    show=True,
    color="royalblue",
    figsize=(9, 6),
):
    """
    Plot the drift magnitude vs time with seaborn darkgrid styling.

    Parameters
    ----------
    magnitudes : array-like
        Sequence of drift magnitudes (length T-1).
    trading_periods : array-like or None, optional
        Sequence of time period labels (length T).
        If provided, x-axis will be aligned to time indices (0..T-2).
    title : str, optional
        Title of the plot.
    save_path : str or None, optional
        If provided, path where the figure will be saved.
    dpi : int, default 300
        Resolution (dots per inch) when saving the figure.
    show : bool, default True
        Whether to display the plot interactively.
    color : str, default "royalblue"
        Line color.
    figsize : tuple, default (9, 6)
        Figure size.
    """
    if trading_periods is None:
        x_vals = np.arange(len(magnitudes))
    else:
        x_vals = np.arange(len(trading_periods) - 1)

    # Set seaborn theme
    sns.set_theme(style="darkgrid", palette="deep", context="talk")

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    sns.lineplot(x=x_vals, y=magnitudes, marker='o', lw=2, ax=ax, color=color)

    ax.set_xlabel("Time interval (t → t+1)")
    ax.set_ylabel("Drift magnitude")
    ax.set_title(title)

    # White grid lines
    ax.grid(True, color="white", alpha=0.6, linewidth=1.5)

    # Black outline
    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.5)

    fig.tight_layout()

    # Save if requested
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax


import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

def plot_drift_2d(
    coords, 
    deltas, 
    theta1, 
    theta2, 
    title="Max-point drift in θ-space",
    save_path=None,
    dpi=300,
    show=True
):
    """
    Plot 2D drift arrows with discrete time-colored points.

    Parameters
    ----------
    coords : array-like, shape (T, 2)
        Sequence of (x, y) best points across time.
    deltas : array-like, shape (T-1, 2)
        Stepwise drift vectors (coords[t+1] - coords[t]).
    theta1, theta2 : array-like
        Grids for θ1 and θ2 (used for axis limits).
    title : str, optional
        Plot title.
    save_path : str or None, optional
        If provided, path where the figure will be saved.
    dpi : int, default 150
        Resolution for saved figure.
    show : bool, default True
        Whether to display the plot interactively.
    """
    T = len(coords)
    time_idx = np.arange(T)

    fig, ax = plt.subplots(figsize=(6, 6))

    # Draw arrows (fade with time)
    for i in range(T - 1):
        alpha_val = 0.2 + 0.8 * (i / (T - 2)) if T > 1 else 1.0
        ax.quiver(
            coords[i, 0], coords[i, 1],
            deltas[i, 0], deltas[i, 1],
            angles='xy', scale_units='xy', scale=1,
            color='black', alpha=alpha_val, width=0.004
        )

    # Discrete colormap with T bins
    cmap = plt.colormaps["viridis"].resampled(T)
    bounds = np.arange(-0.5, T + 0.5, 1)
    norm = mcolors.BoundaryNorm(bounds, T)

    # Scatter points colored by time
    sc = ax.scatter(
        coords[:, 0], coords[:, 1],
        c=time_idx, cmap=cmap, norm=norm,
        marker="x", s=60, label="Best points"
    )

    ax.set_xlabel(r"$\theta_1$")
    ax.set_ylabel(r"$\theta_2$")
    ax.set_title(title)
    ax.set_xlim(theta1.min(), theta1.max())
    ax.set_ylim(theta2.min(), theta2.max())
    ax.set_aspect("equal", adjustable="box")
    ax.legend()

    # Discrete colorbar with blocks
    cb = fig.colorbar(sc, ticks=time_idx, boundaries=bounds, pad=0.02)
    cb.set_label("time index")

    fig.tight_layout()

    # Save if requested
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax
