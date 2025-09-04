# Methodology

This document describes the methodology used in the **SAMBO Rolling Stability Lab** in more detail. It expands upon the high-level overview in the main README.

---

# Part 1: Backtesting Pipeline

## 1. Configuration Loading

* **Config file:** Each experiment is driven by a YAML configuration file (`config.yaml`).
* **Purpose:** Keeps strategy, optimization, and analysis parameters separate from the codebase.
* **Variables defined:**

  * File paths to historical data and results directories.
  * Strategy module and class name (e.g., `Strat_SMA200`).
  * Parameter ranges to optimize.
  * Backtesting settings (commission, equity, margin).
  * Rolling window setup (size, overlap).
  * Analysis settings (grid resolution, smoothing, top fraction).
* **Usage:** The script loads the config at runtime and passes these values into the pipeline.

---

## 2. Data Preparation

* **Input:** Historical asset price series (e.g., SPY daily closes or BTC 5-minute candles).
* **Processing:**
  * Load raw CSV from `data/` directory.
  * Convert timestamps to standardized format.
  * Verify that required features (e.g., `SMA200`, sentiment factors) are precomputed and present in the CSV, as strategies may assume their existence.

---

## 3. Strategy Definition  

Strategies are implemented as **self-contained Python classes** (subclassing `backtesting.Strategy`), each saved in its own file for modularity. This design lets any rule-based system be swapped in without changing the optimization or reporting code.  

- **Structure:**  
  - Class attributes define hyperparameters (e.g., thresholds, lookbacks) and trading parameters (e.g., position size, stop-loss/take-profit ratios).  
  - `init()` registers indicators via `self.I(...)`.  
  - `next()` executes trades based on the latest signals.  

- **Usage:**  
  - The pipeline imports the strategy module specified in the config file.  
  - SAMBO optimizes declared parameters, recording per-trade returns, equity curves, and sampled performance surfaces.  
  - Results feed directly into stability analysis (heatmaps, Jaccard overlap, centroid drift).  

> **Flexibility:** To test a new idea, simply add a new strategy file (e.g., `sambo_stability/MyStrategy.py`) and point the config to it — the rest of the pipeline remains unchanged.

---

## 4. Full Backtest

* Run the strategy with fixed parameters across the **entire dataset**.
* Purpose: establish a baseline performance profile.
* Backtests automatically record key statistics:

  * **Return \[%]**
  * **Sharpe ratio**
  * **Sortino ratio**
  * **Max. Drawdown \[%]**
  * **Equity curve time series**

---

## 5. Time-Interval Backtesting (Rolling Windows)

* Partition the dataset into windows controlled by two settings:
  * **Window length (W):** e.g., 252 trading days.
  * **Overlap (O):** controls whether windows overlap (e.g., 50%) or run consecutively without overlap.
    * If `overlap=True` → exact 50% overlap between consecutive windows.
    * If `overlap=False` → completely consecutive, non-overlapping windows.
* Within each window:

  * Optimize parameters with SAMBO.
  * Save trial results, best parameters, per-window equity.
* Each backtest window automatically records the same set of metrics: Return \[%], Sharpe ratio, Sortino ratio, Max. Drawdown, and equity curve.

This captures how parameter performance shifts across regimes.

---

## 6. Parameter Optimization with SAMBO

* **Goal:** Efficiently explore parameter space and identify high-performing regions.
* **SAMBO (Sample-efficient Adaptive Model-Based Optimization):**

  * Iteratively samples parameter candidates.
  * Builds a surrogate model of the performance surface we want to optimize over.
  * Uses that model to balance exploration vs exploitation.
  * Vastly more efficient than random or brute-force grid search.
* **Configuration:**
  * Parameter ranges specified in YAML config.
  * Max iterations (e.g., 100) - a good rule of thumb is 20–50 iterations per parameter dimension.
Outputs from optimization:

* `stats.csv`, `sambo_history.json`: trial parameters + metrics (Return, Sharpe, Sortino, Max. Drawdown, SAMBO results).

---

# Part 2: Analysis & Stability

## 7. Grid Interpolation & Smoothing

* Interpolate trial results onto a **uniform grid** (e.g., 100×100).
* Interpolation enables direct comparisons between different time intervals, since all results are aligned to the same grid resolution.
* Apply **Gaussian smoothing** with configurable $\sigma$.
* The smoothing is designed to emphasize stable, broad regions of high performance rather than isolated noisy spikes.
* Purpose: reduce noise and highlight regions that are more likely to generalize.

Outputs:

* `heatmap_raw.npz`: interpolated raw grid.
* `heatmap_smooth.npz`: smoothed grid.

---

## 8. Region Extraction & Stability Metrics

* Define a **Top-X% region** (quantile cutoff).
* Extract binary mask of high-performing cells.
* Track the **top centroid** as the single best point after Gaussian smoothing, rather than the mean of the entire Top-X% region.

**Metrics:**

1. **Jaccard overlap** – persistence of Top-X% regions across windows (not similarity of the whole surface).
2. **Centroid drift** – distance of the smoothed best point across windows.
3. **Cell frequency / entropy** – computed using Bernoulli-Shannon entropy, modified to penalize regions that consistently perform poorly across time.

Outputs:

* Figures only (Jaccard time series, centroid drift plots, entropy maps). Raw npz or mask files are not saved at this stage.

---

## 9. Visualization & Reporting

1. **Heatmaps:** per-window smoothed Sharpe with top region outlined.
2. **Stability plots:** Jaccard overlap and centroid drift series.
3. **Gallery / composites:** multi-panel plots combining windows.

Outputs:

* `figures/` folder with PDFs in the experiment specific subfolder within `results/`.

---

## 10. Summary

This methodology combines efficient optimization (SAMBO), region extraction, and stability metrics to assess the robustness of trading strategies across regimes. The two-part approach ensures both **baseline backtesting** and **stability analysis**, emphasizing *parameter regions that remain robust across time* rather than fragile single-point optima.
