from sambo_stability.sambo_utils import optimize_strategy, split_df_time_chunks, save_optimization_outputs, prepare_results_dirs
import pandas as pd
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

# ---------- Strategy Class Information ---------- #
strategy_path = cfg['strategy_path']
strategy_class = cfg['strategy_class']

#from sambo_stability.NewStrat2 import NewStrat
from sambo_stability.strategy_loader import load_strategy
StrategyClass = load_strategy(ROOT/strategy_path, strategy_class)

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

dirs = prepare_results_dirs(ROOT, basename)

print("📁 Results will be saved in:", dirs["results"])
print("📁 Figures will be saved in:", dirs["figures"])

if __name__ == "__main__":

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

    #---------- Full backtest ----------#
    #-----------------------------------#

    print("🚀 Beginning full backtest...")

    stats, heatmap, optimize_result = optimize_strategy(
        data=df_original,
        strategy_class=StrategyClass,
        param_ranges=param_ranges,
        commission=commission,
        initial_equity=initial_equity,
        margin=margin,
        constraint=None,
        maximize=maximize_parameter, # 'Sharpe Ratio', 'Return [%]'
        method='sambo',
        max_tries=max_tries,  # Try 20–50x number of parameters
        return_heatmap=True
    )

    save_optimization_outputs(
        stats=stats,
        heatmap=heatmap,
        optimize_result=optimize_result,
        output_dir=ROOT / 'results' / basename / basename,
        basename=basename,
        store_heatmap_key=store_heatmap_key  # matches the key used in .to_hdf()
    )

    print("✅ Total backtest completed")

    #---------- Time Interval Backtest ----------#
    #--------------------------------------------#

    # 1) Create time-based chunks
    trading_periods = split_df_time_chunks(
        df_original,
        window_size=window_size,
        overlap=overlap,
        include_last_partial=include_last_partial
    )

    print("📈 Beginning time interval backtest:")

    # 2) Loop chunks through your optimizer
    for i, window_df in enumerate(trading_periods, 1):
        period_start = window_df.index.min()
        period_end = window_df.index.max() # this will be t_start + window_size for full windows
        num_days = (period_end - period_start).days

        try:
            stats, heatmap, optimize_result = optimize_strategy(
                window_df,
                strategy_class=StrategyClass,
                param_ranges=param_ranges,
                commission=commission,
                initial_equity=initial_equity,
                margin=margin,
                constraint=None,
                maximize=maximize_parameter,
                method='sambo',          # or 'grid'/'random' as you prefer
                max_tries=max_tries,  # Try 20–50x number of parameters
                return_heatmap=True      # or False if you don’t need it
            )

            save_optimization_outputs(
                stats=stats,
                heatmap=heatmap,
                optimize_result=optimize_result,
                output_dir=ROOT / 'results' / basename / basename,
                basename=f"{basename}_{i}_of_{len(trading_periods)}",
                store_heatmap_key="heatmap"
            )

            print(f"  🔄 [{i}/{len(trading_periods)}] {period_start:%Y-%m-%d} → {period_end:%Y-%m-%d} - ({num_days} days) done.")

        except Exception as e:
            # Log & continue, so one bad window doesn't kill the whole run
            print(f"Window {i} failed ({period_start} → {period_end}): {e}")

    print("\n📊 Backtest analysis complete ✅\n")
