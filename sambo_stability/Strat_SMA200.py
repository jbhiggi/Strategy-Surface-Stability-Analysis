import numpy as np
import pandas as pd
import pandas_ta as ta
from backtesting import Strategy

# ——— Helper signal functions —————————————————————————————

def sma_growing(x, max_len: int):
    s = pd.Series(x, dtype=float)
    return s.rolling(window=int(max_len), min_periods=1).mean().to_numpy()

def calculate_TotalSignal(sma_trend, close, rsi_trend, theta_enter, theta_exit):
    """
    Returns an int array of signals aligned to `close`:
        2  -> enter long
        1  -> exit long
        0  -> hold / no action

    Notes:
    - Long setup: close > sma_trend
    - Entry: RSI < theta_enter (under long setup)
    - Exit: RSI > theta_exit
    - Exits win if both happen on the same bar
    """
    sma_trend = np.asarray(sma_trend, dtype=float)
    close     = np.asarray(close, dtype=float)
    rsi_trend = np.asarray(rsi_trend, dtype=float)

    # Validity mask: require all inputs to be finite
    valid = np.isfinite(sma_trend) & np.isfinite(close) & np.isfinite(rsi_trend)

    # Core conditions
    long_setup   = (close > sma_trend)
    enter_signal = long_setup & (rsi_trend < theta_enter)
    exit_signal  = (rsi_trend > theta_exit)

    # Exits take priority if simultaneous
    enter_signal = enter_signal & ~exit_signal

    # Apply validity mask (no trades until indicators are defined)
    enter_signal &= valid
    exit_signal  &= valid

    # Encode signals
    sig = np.zeros_like(close, dtype=np.int8)
    sig[enter_signal] = 2
    sig[exit_signal]  = 1
    return sig

# ——— The unified, fully-optimizable strategy with pandas_ta —————————————————————

class Strat_SMA200(Strategy):
    #––– Hyperparameters to optimize –––#
    #sma_length = 200 # SMA period
    rsi_length = 2 # Canonical value for RSI period
    theta_enter = 30 # Enter when gate open and RSI is less than theta_enter
    theta_exit = 90 # Exit when long and RSI is greater than theta_exit

    atr_length  = 7 # ATR period

    #––– Trading parameters to optimize –––#
    mysize      = 20   # size per trade
    slcoef      = 1.1    # stop-loss = slcoef * ATR
    TPSLRatio   = 1.5    # take-profit to stop-loss ratio

    def init(self):
        # 1) 200-period SMA via pandas_ta
        #self.SMA = self.I(
        #    lambda x: ta.sma(pd.Series(x), length=self.sma_length).to_numpy(),
        #    self.data.Close
        #)
        #self.SMA = self.I(lambda x: sma_growing(x, self.sma_length), self.data.Close)

        # 2) RSI via pandas_ta
        self.RSI = self.I(
            lambda x: ta.rsi(pd.Series(x), length=self.rsi_length).to_numpy(),
            self.data.Close
        )

        # 3) ATR via pandas_ta
        self.ATR = self.I(
            lambda h, l, c: ta.atr(
                pd.Series(h), pd.Series(l), pd.Series(c), length=self.atr_length
            ).to_numpy(),
            self.data.High,
            self.data.Low,
            self.data.Close
        )

        # 4) Calculate TotalSignal
        self.signal1 = self.I(
            calculate_TotalSignal,
            self.data.SMA200,
            self.data.Close,
            self.RSI,
            self.theta_enter,
            self.theta_exit
        )

    def next(self):
        sig   = self.signal1[-1]
        price = self.data.Close[-1]
        slatr = self.slcoef * self.ATR[-1]

        # Long entry
        if sig == 2 and not self.position:
            self.buy(
                size=self.mysize,
                sl=price - slatr,
                tp=price + slatr * self.TPSLRatio
            )

        # Long exit only (no shorts)
        elif sig == 1 and self.position.is_long:
            self.position.close()