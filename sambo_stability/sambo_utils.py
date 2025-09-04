from pathlib import Path

def prepare_results_dirs(root: Path, basename: str):
    """
    Create results directory structure for a given run.

    Parameters
    ----------
    root : Path
        Repository root path (usually ROOT from your script).
    basename : str
        Name of the experiment/run. Subfolder will be created under results/.
    
    Returns
    -------
    dict
        Dictionary with keys 'results' and 'figures', containing Path objects.
    """
    results_dir = root / "results" / basename
    figures_dir = results_dir / "figures"

    # Create directories if they don't exist
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    return {"results": results_dir, "figures": figures_dir}


from backtesting import Backtest

def optimize_strategy(
    data,
    strategy_class,
    param_ranges,
    commission=0.0,
    initial_equity=250,
    margin=1.0,
    constraint=None,
    maximize='Return [%]',
    method=None,
    max_tries=None,
    random_state=42,
    return_heatmap=False
):
    """
    Run optimization on a given strategy and data with flexible parameter ranges.

    If using the 'sambo' method and return_heatmap=False, the full optimization result
    object is returned. For other methods, only stats and heatmap will be returned.

    Parameters:
    - data: pandas.DataFrame or similar price series.
    - strategy_class: Strategy class (e.g., Sma4Cross).
    - param_ranges: dict mapping parameter names to lower/upper bounds.
    - commission: float commission per trade.
    - initial_equity: float starting cash for backtest.
    - margin: float leverage margin.
    - constraint: function(p) -> bool to enforce parameter relationships.
    - maximize: string name of metric to maximize.
    - method: optimization method (e.g., 'grid', 'random', 'sambo').
    - max_tries: int for optimizer max attempts.
        * For 'sambo', a good rule of thumb is 20-50 x number of parameters.
    - random_state: seed for reproducibility.
    - return_heatmap: bool to return heatmap DataFrame.

    Returns:
    - (stats, heatmap, optimize_result)
    """
    # Build Backtest kwargs
    backtest_kwargs = {
        'commission': commission,
        'cash': initial_equity,
        'margin': margin
    }
    bt = Backtest(data, strategy_class, finalize_trades=True, **backtest_kwargs)

    # Build optimize arguments
    optimize_args = {
        **param_ranges,
        'constraint': constraint,
        'maximize': maximize,
        'method': method,
        'max_tries': max_tries,
        'random_state': random_state,
        'return_heatmap': return_heatmap,
        'return_optimization': method == 'sambo'
    }

    # Execute optimization
    if method == 'sambo':
        stats, heatmap, optimize_result = bt.optimize(**optimize_args)
        return stats, heatmap, optimize_result
    else:
        stats, heatmap = bt.optimize(**optimize_args)
        optimize_result = None
        return stats, heatmap, optimize_result
    
import pandas as pd

def split_df_time_chunks(
    df,
    window_size="90D",
    overlap=False,
    include_last_partial=False,
):
    """
    Split a DatetimeIndex DataFrame into fixed-duration time chunks.

    - Non-overlap: next chunk starts at the end of the last one.
    - 50% overlap: next chunk starts halfway through the previous chunk.

    Parameters
    ----------
    df : pd.DataFrame (DatetimeIndex required)
    window_size : str or pd.Timedelta
        Duration per chunk (e.g., '90D', '30min').
    overlap : bool
        If True, use 50% overlap; else consecutive non-overlapping windows.
    include_last_partial : bool
        If True, include a final partial chunk if the tail is shorter than window_size.

    Returns
    -------
    List[pd.DataFrame]
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame must have a DatetimeIndex.")
    if df.empty:
        return []

    df = df.sort_index()
    win = pd.Timedelta(window_size)
    step = win // 2 if overlap else win

    chunks = []
    t_start = df.index.min()
    t_end_data = df.index.max()

    while True:
        t_stop = t_start + win

        if t_stop <= t_end_data:
            chunk = df.loc[t_start:t_stop]
            if not chunk.empty:
                chunks.append(chunk)
            # advance: end of last vs halfway (50% overlap)
            t_start = t_start + step
        else:
            # Handle the tail
            if include_last_partial and t_start < t_end_data:
                chunk = df.loc[t_start:t_end_data]
                if not chunk.empty:
                    chunks.append(chunk)
            break

    return chunks

import os
import json
import pandas as pd
import numpy as np

# ——— Helper for JSON serialization —————————————————————

def _sanitize_for_json(obj):
    """
    Recursively convert numpy and pandas types to native Python types
    so that JSON serialization succeeds.
    """
    if isinstance(obj, dict):
        return {_sanitize_for_json(k): _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_sanitize_for_json(v) for v in obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, pd.Series):
        return _sanitize_for_json(obj.tolist())
    # Fallback for other types (int, float, str, etc.)
    return obj

# ——— Save/Load Sambo history as JSON ———————————————————————

def _save_sambo_result_json(optimize_result, filepath):
    """
    Save Sambo optimization history to a JSON file.

    Parameters:
    - optimize_result: result from bt.optimize(method='sambo')
        Must have attributes `xv` and `funv`.
    - filepath: target .json file path
    """
    data = {
        'xv': [list(x) for x in optimize_result.xv],
        'funv': list(optimize_result.funv)
    }
    # Convert any numpy/pandas types to Python natives
    sanitized = _sanitize_for_json(data)
    with open(filepath, 'w') as f:
        json.dump(sanitized, f, indent=2)


def _load_sambo_result_json(filepath):
    """
    Load Sambo optimization history from JSON and wrap in a dummy object
    with `x_iters` and `func_vals` attributes for use in interpolation.
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    class Result:
        pass
    res = Result()
    # match naming expected by interpolate_rf_grid
    res.x_iters = data['xv']
    res.func_vals = data['funv']
    return res

# ——— Master save function for all optimization outputs —————————————————

def save_optimization_outputs(
    stats: pd.Series,
    heatmap: pd.DataFrame,
    optimize_result,
    output_dir: str,
    basename: str = 'opt',
    store_heatmap_key: str = 'heatmap'
):
    """
    Save optimization outputs:
    - stats: best-run summary (CSV)
    - heatmap: full grid results (HDF5)
    - optimize_result: Sambo history (JSON)

    Parameters:
    - stats: pd.Series returned from Backtest.optimize
    - heatmap: pd.DataFrame returned from Backtest.optimize
    - optimize_result: result object from sambo; must have .xv and .funv
    - output_dir: folder to save files in
    - basename: file name prefix
    - store_heatmap_key: HDF5 key name for heatmap
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1) Save stats (CSV)
    stats_path = os.path.join(output_dir, f"{basename}_stats.csv")
    stats.to_csv(stats_path)

    # 2) Save heatmap (HDF5)
    heatmap_path = os.path.join(output_dir, f"{basename}_heatmap.h5")
    heatmap.to_hdf(heatmap_path, key=store_heatmap_key, mode='w')

    # 3) Save sambo result history (JSON)
    if optimize_result is not None:
        json_path = os.path.join(output_dir, f"{basename}_sambo_history.json")
        _save_sambo_result_json(optimize_result, json_path)

import os
import pandas as pd

def load_optimization_outputs(
    output_dir: str,
    basename: str = 'opt',
    heatmap_key: str = 'heatmap'
):
    """
    Load stats (CSV), heatmap (HDF5), and Sambo history (JSON)
    that were saved by save_optimization_outputs.

    Returns:
        stats: pd.Series
        heatmap: pd.DataFrame
        optimize_result: object with .x_iters and .func_vals, or None
    """
    # Paths
    stats_path    = os.path.join(output_dir, f"{basename}_stats.csv")
    heatmap_path  = os.path.join(output_dir, f"{basename}_heatmap.h5")
    json_path     = os.path.join(output_dir, f"{basename}_sambo_history.json")

    # 1) Load stats
    stats_df = pd.read_csv(stats_path, index_col=0)
    # If single-column, convert to Series
    if stats_df.shape[1] == 1:
        stats = stats_df.iloc[:, 0]
    else:
        stats = stats_df

    # 2) Load heatmap
    heatmap = pd.read_hdf(heatmap_path, key=heatmap_key)

    # 3) Load Sambo history
    optimize_result = None
    if os.path.exists(json_path):
        optimize_result = _load_sambo_result_json(json_path)

    return stats, heatmap, optimize_result

import numpy as np
import pandas as pd
from itertools import product
from sklearn.ensemble import RandomForestRegressor

def interpolate_rf_grid(
    optimize_result,
    param_lists,
    grid_resolution=50,
    rf_kwargs=None
):
    """
    From an skopt-style OptimizeResult and original parameter lists,
    fit a RandomForest surrogate and interpolate its predictions over
    a dense, regular grid.

    Parameters
    ----------
    optimize_result : OptimizeResult
        Result from bt.optimize(method='sambo').
        Must have attributes `x_iters` (list of param vectors)
        and `func_vals` (list/array of objective values).
    param_lists : dict[str, array-like]
        Original discrete values you passed for each parameter.
        Used only to compute min/max for each dimension.
    grid_resolution : int, default=50
        Number of points along each parameter axis.
    rf_kwargs : dict, optional
        Passed to RandomForestRegressor constructor (e.g. n_estimators).

    Returns
    -------
    df_grid : pandas.DataFrame
        Columns are all parameter names plus 'predicted'.
        One row per grid point.
    """
    # 1) Extract training data from optimize_result
    X = np.array(optimize_result.x_iters)
    y = np.array(optimize_result.func_vals)

    # 2) Fit a Random Forest surrogate
    rf_kwargs = rf_kwargs or {}
    rf = RandomForestRegressor(**rf_kwargs)
    rf.fit(X, y)

    # 3) Build high-res grid bounds
    bounds = {
        name: (min(vals), max(vals))
        for name, vals in param_lists.items()
    }
    grid_axes = {
        name: np.linspace(low, high, grid_resolution)
        for name, (low, high) in bounds.items()
    }

    # 4) Create mesh of all combinations
    names = list(grid_axes.keys())
    mesh = np.array(list(product(*(grid_axes[n] for n in names))))

    # 5) Predict on the mesh
    preds = rf.predict(mesh)

    # 6) Return a DataFrame
    df = pd.DataFrame(mesh, columns=names)
    df['predicted'] = preds
    return df

