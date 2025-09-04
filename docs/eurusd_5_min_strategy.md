# Dataset Specification: EUR/USD 5-Minute Interval (2019–2022)

## Overview

This document outlines the dataset specifications to be used for the Adaptive Strategy Surface Analysis project. The focus is on EUR/USD exchange rate data with a 5-minute resolution spanning from 2019 to 2022.

---

## Dataset Details

- **Instrument**: EUR/USD (Euro to US Dollar)
- **Interval**: 5-minute OHLCV data
- **Date Range**: January 1, 2019 – December 31, 2022
- **Data Fields**:
  - Open
  - High
  - Low
  - Close
  - Volume

---

## Preprocessing Recommendations

- **Timezone normalization** (e.g., convert to UTC)
- **Missing data handling** (e.g., forward-fill small gaps, drop large holes)
- **Resampling** if needed to harmonize with other datasets
- **Feature Engineering**:
  - EMA fast and slow variants
  - Bollinger Bands
  - Rolling volatility
  - RSI, MACD (optional)

---

## Binning for Strategy Surface Analysis

- **Window size**: e.g., 3-month rolling or expanding windows
- **Backtest sampling**: Evaluate performance across parameter grids for each window
- **Grid Parameters** (example):
  - `EMA_fast`: 5 to 30 (step 5)
  - `EMA_slow`: 20 to 100 (step 10)

---

## Use Cases

- Build performance surfaces for each time bin
- Vectorize and analyze correlation of surfaces across bins
- Detect regime changes and forecast optimal parameter regions

---

## Notes

High-frequency data like 5-minute intervals can exhibit significant noise and microstructure effects. Smoothing techniques and robust metric choices (e.g., Sortino ratio) are recommended.

Let me know if you'd like this linked into the main methodology doc or if you'd prefer to integrate it into a full pipeline.

