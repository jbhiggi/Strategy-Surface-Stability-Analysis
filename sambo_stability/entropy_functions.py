import numpy as np
import matplotlib.pyplot as plt

def bernoulli_entropy_per_cell(stack_binary: np.ndarray, base: float = 2.0):
    """
    Entropy per cell for a 0/1 stack over time.

    Parameters
    ----------
    stack_binary : np.ndarray
        Shape (T, H, W) with values in {0,1}. No NaNs.
    base : float, default 2.0
        Log base (2=bits, e=nats, 10=Hartleys).

    Returns
    -------
    Hmap : (H, W) ndarray
        Shannon entropy per cell in chosen base.
    pmap : (H, W) ndarray
        Fraction of 1s per cell (MLE): k/T.
    """
    if stack_binary.ndim != 3:
        raise ValueError("stack_binary must have shape (T, H, W)")
    T = stack_binary.shape[0]
    kmap = stack_binary.sum(axis=0).astype(float)
    pmap = kmap / T

    log_base = np.log(base)
    Hmap = np.zeros_like(pmap, dtype=float)

    # Compute only where needed, suppress harmless log(0) warnings
    with np.errstate(divide="ignore", invalid="ignore"):
        Hmap -= np.where(pmap > 0, pmap * (np.log(pmap) / log_base), 0.0)
        Hmap -= np.where(pmap < 1, (1 - pmap) * (np.log(1 - pmap) / log_base), 0.0)

    return Hmap, pmap

import matplotlib.pyplot as plt

def plot_entropy_heatmap(
    Hmap,
    theta1_grid=None,
    theta2_grid=None,
    title="Per-Cell Shannon Entropy",
    cbar_label="H (bits)",
    save_path=None,
    dpi=300,
    cmap="viridis_r",
    show=True
):
    if theta1_grid is not None and theta2_grid is not None:
        if theta1_grid.shape != Hmap.shape or theta2_grid.shape != Hmap.shape:
            raise ValueError("theta grids must match Hmap shape")
        x = np.unique(theta1_grid[0, :]); y = np.unique(theta2_grid[:, 0])
        extent = [x.min(), x.max(), y.min(), y.max()]
        aspect = "auto"; xlabel, ylabel = r"$\theta_1$", r"$\theta_2$"
    else:
        extent = None; aspect = "equal"; xlabel, ylabel = "col", "row"

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(Hmap, origin="lower", extent=extent, aspect=aspect, cmap=cmap)
    cbar = plt.colorbar(im, ax=ax); cbar.set_label(cbar_label)
    ax.set_title(title); ax.set_xlabel(xlabel); ax.set_ylabel(ylabel)
    plt.tight_layout()

    # Save if requested
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    # Show or suppress
    if show:
        plt.show()

    return fig, ax
