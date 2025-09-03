import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter


def heatmap_from_df(
    df: pd.DataFrame,
    *,
    # Column names (can be pulled from your config later)
    y_col: str = "slcoef",
    x_col: str = "TPSLRatio",
    value_col: str = "predicted",
    # Smoothing (None = off). Sigma is in **grid steps**, not parameter units
    smoothing_sigma: float | None = None,
    smoothing_mode: str = "nearest",  # "nearest" | "reflect" | "mirror" | "constant" | "wrap"
    # Plot cosmetics
    title: str | None = None,
    xlabel: str | None = None,
    ylabel: str | None = None,
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    robust: bool = True,              # if True and vmin/vmax not given, use percentiles
    robust_pct: tuple[float, float] = (2, 98),
    annotate_best: bool = True,
    best_mode: str = "max",          # "max" or "min"
    contour: bool = False,
    contour_levels: int = 10,
    show_colorbar: bool = True,
    show_grid: bool = True,
    # I/O
    save_path: str | None = None,
    show: bool = True,
    ax: plt.Axes | None = None,
):
    """
    Plot a heatmap from a long-form DataFrame of parameter search results, with
    optional Gaussian smoothing, contour overlays, and annotation of the best point.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data in long-form, containing at least three columns: one for the
        y-axis parameter, one for the x-axis parameter, and one for the value to plot.

    y_col : str, default="slcoef"
        Column in `df` to use as the y-axis values.

    x_col : str, default="TPSLRatio"
        Column in `df` to use as the x-axis values.

    value_col : str, default="predicted"
        Column in `df` to use as the metric or score to plot as color values.

    smoothing_sigma : float or None, default=None
        Standard deviation for Gaussian smoothing, in grid steps. If None, no
        smoothing is applied.

    smoothing_mode : {"nearest", "reflect", "mirror", "constant", "wrap"}, default="nearest"
        Boundary mode passed to `scipy.ndimage.gaussian_filter`.

    title : str or None, default=None
        Title for the heatmap plot.

    xlabel : str or None, default=None
        Label for the x-axis. If None, uses the column name given by `x_col`.

    ylabel : str or None, default=None
        Label for the y-axis. If None, uses the column name given by `y_col`.

    cmap : str, default="viridis"
        Matplotlib colormap to use for the heatmap.

    vmin, vmax : float or None, default=None
        Minimum and maximum values for the colormap. If None, these are determined
        automatically. If `robust=True`, percentiles are used.

    robust : bool, default=True
        If True, ignores extreme values when computing color limits (uses the
        `robust_pct` percentiles).

    robust_pct : tuple of float, default=(2, 98)
        Percentiles to use when computing robust vmin/vmax.

    annotate_best : bool, default=True
        Whether to mark the best-performing parameter combination on the heatmap.

    best_mode : {"max", "min"}, default="max"
        Whether to annotate the maximum or minimum value in the grid.

    contour : bool, default=False
        Whether to overlay contour lines on top of the heatmap.

    contour_levels : int, default=10
        Number of contour levels to draw if `contour=True`.

    show_colorbar : bool, default=True
        If True, add a colorbar to the right of the heatmap.

    show_grid : bool, default=True
        If True, overlay a faint grid on the heatmap.

    save_path : str or None, default=None
        If given, save the figure to this path.

    show : bool, default=True
        If True, display the plot immediately with `plt.show()`.

    ax : matplotlib.axes.Axes or None, default=None
        Axis object to draw the heatmap on. If None, a new figure and axis are created.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.

    ax : matplotlib.axes.Axes
        The axis containing the plot.

    grid_df : pandas.DataFrame
        The pivoted (and optionally smoothed) 2D grid of values used to make the heatmap.
    """

    # 1) Pivot to 2D grid. Ensure sorted axes for consistent visuals.
    pivot_df = (
        df.pivot(index=y_col, columns=x_col, values=value_col)
          .sort_index(axis=0)
          .sort_index(axis=1)
    )

    y_vals = pivot_df.index.values.astype(float)
    x_vals = pivot_df.columns.values.astype(float)
    Z = pivot_df.values.astype(float)

    # 2) Optional smoothing on the numpy array
    if smoothing_sigma is not None:
        Z = gaussian_filter(Z, sigma=float(smoothing_sigma), mode=smoothing_mode)

    # Rewrap into a DataFrame so labels are preserved downstream
    grid_df = pd.DataFrame(Z, index=y_vals, columns=x_vals)

    # 3) Prepare for plotting
    Z_vals = grid_df.values
    Z_masked = np.ma.masked_invalid(Z_vals)

    # Robust color scaling if needed
    if (vmin is None or vmax is None) and robust:
        finite_vals = Z_vals[np.isfinite(Z_vals)]
        if finite_vals.size > 0:
            lo, hi = np.percentile(finite_vals, robust_pct)
            if vmin is None:
                vmin = lo
            if vmax is None:
                vmax = hi

    # 4) Plot
    if ax is None:
        fig, ax = plt.subplots(figsize=(10.8, 7.8))
    else:
        fig = ax.figure

    # extent maps array indices to the actual param values
    extent = [x_vals.min(), x_vals.max(), y_vals.min(), y_vals.max()]
    im = ax.imshow(
        Z_masked,
        origin="lower",
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=extent,
    )

    # Ticks: thin out for readability
    def _nice_ticks(vals, max_ticks=8):
        if len(vals) <= max_ticks:
            return vals
        idx = np.linspace(0, len(vals) - 1, num=max_ticks, dtype=int)
        return vals[idx]

    ax.set_xticks(_nice_ticks(x_vals))
    ax.set_yticks(_nice_ticks(y_vals))

    ax.set_xlabel(xlabel if xlabel is not None else x_col)
    ax.set_ylabel(ylabel if ylabel is not None else y_col)
    if title:
        ax.set_title(title)

    if show_grid:
        ax.grid(color="white", alpha=0.2, linewidth=0.5)

    if show_colorbar:
        cbar = fig.colorbar(im, ax=ax, pad=0.02)
        cbar.ax.set_ylabel(value_col, rotation=90, va="center")

    # Optional contour overlay
    if contour:
        Xg, Yg = np.meshgrid(x_vals, y_vals)
        cs = ax.contour(Xg, Yg, Z_vals, levels=contour_levels, linewidths=0.7, colors='black')
        ax.clabel(cs, inline=True, fontsize=8, fmt="%.2g")

    # Annotate best point
    if annotate_best and np.isfinite(Z_vals).any():
        if best_mode.lower() == "min":
            iy, ix = np.unravel_index(np.nanargmin(Z_vals), Z_vals.shape)
        else:
            iy, ix = np.unravel_index(np.nanargmax(Z_vals), Z_vals.shape)
        x_best, y_best, z_best = x_vals[ix], y_vals[iy], Z_vals[iy, ix]
        ax.scatter([x_best], [y_best], s=84, marker="o", edgecolor="black", facecolor="none", linewidths=1.8)
        ha = "left" if x_best < x_vals.mean() else "right"
        offset = (6, 6) if ha == "left" else (-6, 6)

        ax.annotate(
            f"{best_mode.upper()}: {z_best:.4g}\n({x_col}={x_best:.3g}, {y_col}={y_best:.3g})",
            xy=(x_best, y_best),
            xytext=offset,
            textcoords="offset points",
            fontsize=9,
            ha=ha,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.8, ec="none"),
        )

    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=350)

    if show:
        plt.show()

    return fig, ax, grid_df
