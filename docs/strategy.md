## Strategy: SMA(200) Trend Filter + RSI(2) Mean Reversion (Long-Only)

**Idea.** Trade short-term mean reversion **only when the long-term trend is up**. In bullish regimes (price above its 200-day SMA), brief oversold dips tend to revert.

---

### Inputs & Notation
- Price series: close $C_t$
- Simple moving average (long-term filter):  
  $$\text{SMA}_{200}(t) = \frac{1}{200}\sum_{i=0}^{199} C_{t-i}$$
- RSI with Wilder smoothing (period $n=2$):
  - $U_t = \max(C_t - C_{t-1}, 0)$  
  - $D_t = \max(C_{t-1} - C_t, 0)$  
  - $AU_t = \text{WilderEMA}_n(U_t)$  
  - $AD_t = \text{WilderEMA}_n(D_t)$  
  - $\text{RSI}_n(t) = 100 - \frac{100}{1 + \frac{AU_t}{AD_t + \varepsilon}}$
    (with small $\varepsilon$ to avoid divide-by-zero)

---

### Entry/Exit Rules
**Trend filter (gate):** only consider longs when $C_t > \text{SMA}_{200}(t)$. Otherwise stay flat.

**Entry (mean reversion long):**
- If the gate is open **and** $\text{RSI}_2(t) < \theta_{\text{enter}}$ (e.g., 10), then **enter long** at next bar’s open/close per your backtest convention.

**Exit (rebound / take-profit):**
- While long, exit when $\text{RSI}_2(t) > \theta_{\text{exit}}$ (e.g., 70).

**Stop/Target (optional but recommended):**
- ATR-based protective stop and/or take-profit:
  - Compute $\text{ATR}_k(t)$ (e.g., $k=14$).
  - **Stop:** exit if price falls $s_\text{ATR}\cdot \text{ATR}_k$ below entry (e.g., $s_\text{ATR}=1.0$).
  - **Target:** exit if price rises $r_\text{ATR}\cdot \text{ATR}_k$ above entry (e.g., $r_\text{ATR}=1.5$).
- If both RSI exit and stop/target fire on same bar, define a clear priority (e.g., stop/target evaluated on intrabar, RSI on close).

---

### Parameters (for optimization / stability maps)
- $\theta_{\text{enter}}$ (RSI oversold): typical range **5–20** (default **10**)
- $\theta_{\text{exit}}$ (RSI rebound): typical range **60–85** (default **70**)
- $\text{ATR period } k$: e.g., **10–20** (default **14**)
- $s_\text{ATR}$ (stop multiplier): e.g., **0.5–2.0** (default **1.0**)
- $r_\text{ATR}$ (target multiplier): e.g., **1.0–3.0** (default **1.5**)
- (Fixed) $\text{SMA window} = 200$ days — keep fixed to isolate short-term mean-reversion dynamics

> For 2D stability plots, pick two to vary (e.g., $\theta_{\text{enter}}$ vs $\theta_{\text{exit}}$), hold others fixed.

---

### Execution & Sizing
- **Positioning:** 100% notional long when signaled (or fixed fraction of equity). Optionally cap exposure.
- **Costs:** include commission/slippage consistent with your config.
- **Signal timing:** generate signals on bar close; apply orders at next bar open (document this convention).
- **No shorts:** long-only by construction.

---

### Baselines to Report
- Buy-and-Hold (SPY) over the same sample.
- Strategy equity curve (in-sample and walk-forward).
- Metrics saved per backtest: Return [%], Sharpe, Sortino, MaxDD [%], exposure, turnover, #trades.

---

### Practical Notes
- RSI(2) is very fast; expect brief holding periods and occasional clustering of trades.
- The SMA(200) gate reduces whipsaws in downtrends and clarifies the regime.
- Use your SAMBO workflow to map $(\theta_{\text{enter}}, \theta_{\text{exit}})$ → Sharpe/Return surfaces, then apply Gaussian smoothing, top-X% masks, Jaccard, and centroid drift across rolling windows.
