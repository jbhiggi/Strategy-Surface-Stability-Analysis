#!/usr/bin/env python3
import sys, subprocess

# Edit this once; everything else reuses it
CONFIGS = ["config_SMA200_SPY", "config_SMA200_BTC", "config_SMA200_QQQ"]

def run(cmd):
    print("▶", " ".join(cmd))
    p = subprocess.run(cmd)
    if p.returncode != 0:
        sys.exit(p.returncode)

def main():
    py = sys.executable  # use current Python (safe for conda/venv)

    for cfg in CONFIGS:
        run([py, "experiments/run_experiments.py", cfg])
        run([py, "experiments/make_figures.py", cfg])

    print("✅ All experiments completed.")

if __name__ == "__main__":
    main()
