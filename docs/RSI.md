## Relative Strength Index (RSI)

The Relative Strength Index (RSI) is a bounded momentum oscillator that
compares the magnitude of recent gains and losses over a given lookback period
(typically $n=14$). It is defined in several steps:

### 1. Price Changes
$$
\Delta P_t = P_t - P_{t-1}
$$
where $P_t$ is the closing price at time $t$.

### 2. Gains and Losses
$$
\text{Gain}_t = \max(\Delta P_t, 0), \quad 
\text{Loss}_t = \max(-\Delta P_t, 0)
$$

### 3. Average Gain and Loss
For the initial period, simple averages are taken:
$$
\text{AvgGain}_t = \frac{1}{n} \sum_{i=1}^{n} \text{Gain}_{t-i}, 
\quad 
\text{AvgLoss}_t = \frac{1}{n} \sum_{i=1}^{n} \text{Loss}_{t-i}
$$

Thereafter, Wilder’s smoothing method (a form of EMA with $\alpha = 1/n$) is used:
$$
\text{AvgGain}_t = \frac{(n-1)\cdot \text{AvgGain}_{t-1} + \text{Gain}_t}{n}
$$
$$
\text{AvgLoss}_t = \frac{(n-1)\cdot \text{AvgLoss}_{t-1} + \text{Loss}_t}{n}
$$

### 4. Relative Strength (RS)
$$
RS_t = \frac{\text{AvgGain}_t}{\text{AvgLoss}_t}
$$

### 5. RSI Formula
$$
RSI_t = 100 - \frac{100}{1 + RS_t}
$$

### Variables
- $P_t$: Closing price at time $t$  
- $\Delta P_t$: One-period change in closing price  
- $\text{Gain}_t, \text{Loss}_t$: Positive and negative components of $\Delta P_t$  
- $n$: Lookback period (commonly 14)  
- $\text{AvgGain}_t, \text{AvgLoss}_t$: Wilder-smoothed averages  
- $RS_t$: Relative Strength ratio  
- $RSI_t$: Final bounded momentum oscillator ($0 \leq RSI \leq 100$)  

---

Thus, RSI quantifies the balance of upward versus downward closes,
smoothed over $n$ periods, and normalizes the result to a scale between 0 and 100.
