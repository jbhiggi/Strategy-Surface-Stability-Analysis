import numpy as np
import matplotlib.pyplot as plt

# ------------------------------
# Jaccard time series calculation
# ------------------------------

def _jaccard_mask(A, B):
    """Jaccard index between two boolean masks."""
    A = A.astype(bool)
    B = B.astype(bool)
    inter = np.logical_and(A, B).sum()
    union = np.logical_or(A, B).sum()
    return 0.0 if union == 0 else inter / union

def _jaccard_time_series_from_bool(stack_bool, mode="consecutive", baseline_index=0):
    """
    Compute Jaccard series given a pre-binarized 3D stack (T,H,W).
    """
    T = stack_bool.shape[0]
    J = np.full(T, np.nan)

    if mode == "consecutive":
        for t in range(1, T):
            J[t] = _jaccard_mask(stack_bool[t], stack_bool[t-1])
    elif mode == "baseline":
        base = stack_bool[baseline_index]
        for t in range(T):
            J[t] = _jaccard_mask(stack_bool[t], base)
    else:
        raise ValueError("Unknown mode")

    return J

import numpy as np
from typing import List, Literal, Optional, Dict

def jaccard_pipeline(
    smoothed_surfaces: List[np.ndarray],
    method: Literal["quantile", "absolute", "topk"] = "quantile",
    top_frac: float = 0.10,
    threshold: Optional[float] = None,
    k: Optional[int] = None,
    baseline_index: int = 0,
) -> Dict[str, np.ndarray]:
    """
    Convert rolling 2D surfaces -> boolean masks -> Jaccard time series.

    Parameters
    ----------
    smoothed_surfaces : list of 2D np.ndarray
        Sequence of return (or score) grids, one per rolling window. All must share shape (H, W).
    method : {"quantile", "absolute", "topk"}, default="quantile"
        How to select the region of interest (ROI) per grid:
          - "quantile": keep top `top_frac` of cells in each grid (scale-free)
          - "absolute": keep cells with value >= `threshold`
          - "topk":     keep top `k` cells in each grid
    top_frac : float, default=0.10
        Fraction for "quantile" (e.g., 0.10 = top 10%).
    threshold : float, optional
        Absolute cutoff for "absolute".
    k : int, optional
        Number of cells to keep for "topk".
    baseline_index : int, default=0
        Which window to use as the fixed baseline in baseline mode.

    Returns
    -------
    out : dict
        {
          "stack_bool": (T, H, W) boolean array of ROI masks,
          "J_consec":   (T,) Jaccard vs previous window (J[0]=nan),
          "J_base":     (T,) Jaccard vs baseline window,
          "roi_frac":   (T,) fraction of ones per window (sanity check)
        }
    """
    if len(smoothed_surfaces) == 0:
        raise ValueError("smoothed_surfaces must be non-empty")

    # --- binarize each grid -> mask
    masks = []
    roi_frac = []
    for Z in smoothed_surfaces:
        Z = np.asarray(Z)
        if Z.ndim != 2:
            raise ValueError("Each surface must be 2D (H, W)")

        if method == "quantile":
            if not (0.0 < top_frac < 1.0):
                raise ValueError("top_frac must be in (0, 1) for 'quantile'")
            cut = np.quantile(Z, 1.0 - top_frac)
            M = (Z >= cut)
        elif method == "absolute":
            if threshold is None:
                raise ValueError("threshold must be provided for 'absolute'")
            M = (Z >= threshold)
        elif method == "topk":
            if k is None or k <= 0:
                raise ValueError("k must be a positive integer for 'topk'")
            flat = Z.ravel()
            kk = min(k, flat.size)
            idx = np.argpartition(flat, flat.size - kk)[-kk:]
            mask_flat = np.zeros_like(flat, dtype=bool)
            mask_flat[idx] = True
            M = mask_flat.reshape(Z.shape)
        else:
            raise ValueError("Unknown method")

        masks.append(M.astype(bool))
        roi_frac.append(M.mean())

    stack_bool = np.stack(masks, axis=0)  # (T, H, W)
    roi_frac = np.asarray(roi_frac)

    # --- Jaccard series
    J_consec = _jaccard_time_series_from_bool(stack_bool, mode="consecutive")
    J_base   = _jaccard_time_series_from_bool(stack_bool, mode="baseline", baseline_index=baseline_index)

    return {
        "stack_bool": stack_bool,
        "J_consec": J_consec,
        "J_base": J_base,
        "roi_frac": roi_frac,
    }

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def plot_jaccard_series(
    J_consec: np.ndarray,
    J_base: np.ndarray,
    n_windows: int,
    title: str = "Jaccard Similarity Across Rolling Return Grids",
    baseline_index: int = 0,
    figsize=(9, 4.5),
    save_path: str = None,
    dpi: int = 300,
    show: bool = True,
):
    """
    Plot consecutive and baseline Jaccard time series with seaborn darkgrid styling.
    """
    # Apply seaborn theme (scoped here if you wrap with seaborn_theme contextmanager)
    sns.set_theme(style="darkgrid", palette="deep", context="talk")

    x = np.arange(n_windows)

    fig, ax = plt.subplots(figsize=figsize)
    sns.lineplot(x=x, y=J_consec, lw=2, label="Consecutive", ax=ax, color="royalblue")
    sns.lineplot(
        x=x,
        y=J_base,
        lw=2,
        linestyle="--",
        label=f"Vs baseline (t={baseline_index})",
        ax=ax,
        color="red"
    )

    ax.set_xlabel("Rolling window index")
    ax.set_ylabel("Jaccard index")
    ax.set_title(title)
    ax.set_ylim(-0.02, 1.0)

    # Black outline
    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.5)

    ax.legend(frameon=True, fancybox=True, shadow=True)
    fig.tight_layout()

    # Save if requested
    if save_path is not None:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)

    return fig, ax
