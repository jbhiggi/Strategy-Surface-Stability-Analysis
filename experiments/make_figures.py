from sambo_stability.sambo_utils import split_df_time_chunks, load_optimization_outputs, interpolate_rf_grid
from sambo_stability.plot_heatmap import heatmap_from_df
from sambo_stability.centroid_drift import compute_drift_from_surfaces, plot_drift_magnitude, plot_drift_2d
from sambo_stability.jaccard import jaccard_pipeline, plot_jaccard_series
from sambo_stability.entropy_functions import bernoulli_entropy_per_cell, plot_entropy_heatmap
from sambo_stability.plot_utils import seaborn_theme
from sambo_stability.combine_pdfs import combine_pdfs

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
from pathlib import Path
import yaml
import sys

config_name = "config.yaml" # default if no argument is passed

# if user passed one argument, use that instead
if len(sys.argv) > 1:
    config_name = sys.argv[1]
    if not config_name.endswith(".yaml"):
        config_name += ".yaml"

ROOT = Path(__file__).resolve().parents[1] # location of *this* file and up one level
config_path = ROOT / "sambo_stability" / "config" / config_name

with open(config_path, "r") as f:
    cfg = yaml.safe_load(f)

print(f"🔧 Loaded config: {config_path}")

# ---------- Processing stock price data ---------- #
file_path_historical_data = cfg["file_path_historical_data"]
total_lookback_months = cfg["total_lookback_months"]

# ---------- Filepaths ---------- #
basename = cfg["basename"]
store_heatmap_key = cfg["store_heatmap_key"]

# ---------- Parameters to Optimize ---------- #
param_ranges = cfg["param_ranges"]

# ---------- Backtesting parameters ---------- #
commission = cfg["commission"]
initial_equity = cfg["initial_equity"]
margin = cfg["margin"]
max_tries = cfg["max_tries"]
maximize_parameter = cfg["maximize_parameter"]

# ---------- Time interval analysis ---------- #
window_size = cfg["window_size"]
overlap = cfg["overlap"]
include_last_partial = cfg["include_last_partial"]

# ---------- Post-backtesting Analysis ---------- #
grid_resolution = cfg["grid_resolution"]
smoothing_sigma = cfg["smoothing_sigma"]
smoothing_mode = cfg["smoothing_mode"]
top_fraction = cfg["top_fraction"]


#---------- Load OHLC Data ----------#
#------------------------------------#

# 1 - Import data
df_original = pd.read_csv(ROOT / file_path_historical_data)
df_original['Gmt time'] = pd.to_datetime(df_original['Gmt time'], errors='coerce')
#df_original["Gmt time"] = df_original["Gmt time"].str.replace(".000","")
#df_original['Gmt time'] = pd.to_datetime(df_original['Gmt time'], format='%d.%m.%Y %H:%M:%S')
#df_original = df_original[df_original.High!=df_original.Low]

# Filter to last 'total_lookback_months' months
latest_time = df_original['Gmt time'].max()
total_lookback_date = latest_time - pd.DateOffset(months=total_lookback_months)
df_original = df_original[df_original['Gmt time'] >= total_lookback_date]

# Set index and display
df_original.set_index("Gmt time", inplace=True)

total_period_start = df_original.index.min()
total_period_end = df_original.index.max()
total_num_days = (total_period_end - total_period_start).days

print(f"📂 Total backtest timeframe: {total_period_start:%Y-%m-%d} → {total_period_end:%Y-%m-%d} ({total_num_days} days)")


#---------- Interpolate Full Backtest ----------#
#-----------------------------------------------#

stats, heatmap, optimize_result = load_optimization_outputs(
    output_dir=ROOT / 'results' / basename / basename,
    basename=basename,
    heatmap_key=store_heatmap_key
)

# Interpolate over a high‐resolution grid
df_grid_full_backtest = interpolate_rf_grid(
    optimize_result=optimize_result,
    param_lists=param_ranges,
    grid_resolution=grid_resolution, # e.g. 100×100 grid
    rf_kwargs={'n_estimators': 500, 'random_state': 42}
)

df_grid_full_backtest["predicted"] *= -1 # Flip sign so that higher = better profit
df_grid_full_backtest = df_grid_full_backtest.rename(columns={"predicted": maximize_parameter}) # Rename column

print("⚙️ Completed interpolation onto uniform grid for full backtest")

#---------- Create Heatmap for Full Backtest ----------#
#------------------------------------------------------#

save_path_full_backtest_heatmap_figure = ROOT / 'results' / basename / 'figures' / f"{basename}_full_backtest_heatmap_figure.pdf"

# Plot from df_grid_full_backtest with Gaussian smoothing
fig, ax, grid_df_full_backtest = heatmap_from_df(
    df_grid_full_backtest,
    y_col=list(param_ranges.keys())[0],
    x_col=list(param_ranges.keys())[1],
    value_col=maximize_parameter,
    smoothing_sigma=smoothing_sigma, # Gaussian sigma in grid steps
    title=f"Full Backtest {total_period_start:%Y-%m-%d} → {total_period_end:%Y-%m-%d} ({total_num_days} days) (Gaussian $\sigma$={smoothing_sigma})",
    annotate_best=True,
    best_mode="max",
    contour=True,
    contour_levels=12,
    save_path=save_path_full_backtest_heatmap_figure,
    show=False
)

plt.close(fig)

print("✅ Created heatmap for full backtest")

#---------- Interpolate Time Interval Backtest ----------#
#--------------------------------------------------------#

# 1) Create time-based chunks
trading_periods = split_df_time_chunks(
    df_original,
    window_size=window_size,
    overlap=overlap,
    include_last_partial=include_last_partial
)

print("📈 Creating heatmap for time interval backtest:")

smoothed_surfaces = []
theta1 = None  # x-axis (columns)
theta2 = None  # y-axis (index)

time_interval_heatmap_figure_name_list = []

# 2) Loop chunks through your optimizer
for i, window_df in enumerate(trading_periods, 1):
    period_start = window_df.index.min()
    period_end   = window_df.index.max() # this will be t_start + window_size for full windows
    num_days = (period_end - period_start).days

    iterative_basename = f"{basename}_{i}_of_{len(trading_periods)}"

    stats, heatmap, optimize_result = load_optimization_outputs(
        output_dir=ROOT / 'results' / basename / basename,
        basename=iterative_basename,
        heatmap_key=store_heatmap_key
    )

    # 3) Interpolate over a high‐resolution grid
    df_grid = interpolate_rf_grid(
        optimize_result=optimize_result,
        param_lists=param_ranges,
        grid_resolution=grid_resolution,    # e.g. 100×100 grid
        rf_kwargs={'n_estimators': 500, 'random_state': 42}
    )

    df_grid["predicted"] *= -1 # Flip sign so that higher = better profit
    df_grid = df_grid.rename(columns={"predicted": maximize_parameter}) # Rename column

    # Create a heatmap with Guassian smoothing
    save_path_time_interval_heatmap_figure = ROOT / 'results' / basename / 'figures' / f"{iterative_basename}_heatmap_figure.pdf"
    time_interval_heatmap_figure_name_list.append(save_path_time_interval_heatmap_figure)

    # Plot from df_grid with Gaussian smoothing
    fig, ax, grid_df = heatmap_from_df(
        df_grid,
        y_col=list(param_ranges.keys())[0],
        x_col=list(param_ranges.keys())[1],
        value_col=maximize_parameter,
        smoothing_sigma=smoothing_sigma, # Gaussian sigma in grid steps
        smoothing_mode=smoothing_mode,
        title=f"[{i}/{len(trading_periods)}] {period_start:%Y-%m-%d} → {period_end:%Y-%m-%d} ({num_days} days) (Gaussian $\sigma$={smoothing_sigma})",
        annotate_best=True,
        best_mode="max",
        contour=True, # optional: add contours
        contour_levels=12,
        save_path=save_path_time_interval_heatmap_figure,
        show=False
    )

    plt.close(fig)

    Z_smooth = grid_df.values.astype(float)
    smoothed_surfaces.append(Z_smooth)

    # Capture theta axes once (consistent across iterations)
    if theta1 is None or theta2 is None:
        theta1 = grid_df.columns.to_numpy(dtype=float) # x-axis
        theta2 = grid_df.index.to_numpy(dtype=float) # y-axis

    print(f"  🔄 [{i}/{len(trading_periods)}] - Heatmap created - Gaussian smoothing completed")

print(f"✅ Completed heatmaps for time interval backtest")

#---------- Combine Time Interval Heatmaps ----------#
#----------------------------------------#

save_path_combined_pdfs = ROOT / 'results' / basename / 'figures' / f"{basename}_combined_time_interval_heatmap_figure.pdf"
combine_pdfs(time_interval_heatmap_figure_name_list, save_path_combined_pdfs)

print(f"✅ Completed combining heatmaps for time interval backtest")

#---------- Plot drift figures ----------#
#----------------------------------------#

# Compute drift
coords, deltas, magnitudes, angles, z_vals = compute_drift_from_surfaces(
    smoothed_surfaces, theta1, theta2
)

save_path_drift_magnitude = ROOT / 'results' / basename / 'figures' / f"{basename}_drift_magnitude_figure.pdf"
with seaborn_theme("darkgrid", "deep", "talk"):
    plot_drift_magnitude(
        magnitudes,
        trading_periods=range(len(smoothed_surfaces)),
        title="1D Drift Magnitude Plot",
        save_path=save_path_drift_magnitude,
        show=False
    )

save_path_drift_2d = ROOT / 'results' / basename / 'figures' / f"{basename}_drift_2d_figure.pdf"
plot_drift_2d(coords, deltas, theta1, theta2,
              save_path=save_path_drift_2d, show=False)

print("🎯 Centroid drift plots created")

#---------- Plot Jaccard figure ----------#
#-----------------------------------------#

out = jaccard_pipeline(
    smoothed_surfaces,
    method="quantile",
    top_frac=top_fraction,
    baseline_index=0,
)

stack_bool = out["stack_bool"]
J_consec = out["J_consec"]
J_base0 = out["J_base"]
roi_frac = out["roi_frac"]

save_path_jaccard_figure = ROOT / 'results' / basename / 'figures' / f"{basename}_jaccard_figure.pdf"
with seaborn_theme("darkgrid", "deep", "talk"):
    plot_jaccard_series(
        J_consec,
        J_base0,
        n_windows=len(smoothed_surfaces),
        baseline_index=0,
        save_path=save_path_jaccard_figure,
        show=False
    )

print("📊 Jaccard figures plotted")

#---------- Creating entropy plots ----------#
#-----------------------------------------#

# stack_bool: shape (T, H, W), entries in {0,1}
base = 2.0  # bits
Hmap, pmap = bernoulli_entropy_per_cell(stack_bool, base=base)

TH1, TH2 = np.meshgrid(theta1, theta2)  # shapes (H, W)

save_path_entropy = ROOT / 'results' / basename / 'figures' / f"{basename}_entropy_figure.pdf"
fig, ax = plot_entropy_heatmap(Hmap, TH1, TH2,
                    save_path=save_path_entropy,
                    cmap="viridis",
                    show=False
                    )
plt.close(fig)

save_path_top_percent_regions = ROOT / 'results' / basename / 'figures' / f"{basename}_top_percent_regions_figure.pdf"
fig, ax = plot_entropy_heatmap(pmap, TH1, TH2,
                           title="Percentage of time cell is in top region",
                           save_path=save_path_top_percent_regions,
                           cbar_label="$\%$ in top region",
                           cmap="viridis",
                           show=False)
plt.close(fig)

# --- 2) Consistency-weighted "avoidance" score: A = (1 - p) * (1 - H/Hmax) ---
# Hmax = log_b(2) because Bernoulli entropy is maximized at p=0.5
Hmax = np.log(2.0) / np.log(base)
Amap = (1.0 - pmap) * (1.0 - (Hmap / Hmax))

# (Optional) Clamp any tiny negative values due to floating error
Amap = np.clip(Amap, 0.0, 1.0)

save_path_modifed_entropy = ROOT / 'results' / basename / 'figures' / f"{basename}_modifed_entropy_figure.pdf"

# --- 3) Plot heatmap of Amap ---
fig, ax = plot_entropy_heatmap(
    Amap,
    TH1,
    TH2,
    title="Consistency-Weighted Avoidance Score A",
    cbar_label="A (unitless)",
    save_path=save_path_modifed_entropy,
    cmap="viridis_r",
    show=False
)

plt.close(fig)

print("🌀 Entropy plots generated")
print("\n📈 Post Backtest Analysis Completed ✅\n")
