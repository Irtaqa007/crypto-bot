"""
Volume RSI — Standard RSI formula applied to volume instead of price.

Up volume   = volume when close > previous close
Down volume = volume when close < previous close
Period      = 10
"""

import pandas as pd
import numpy as np


def calculate_volume_rsi(df: pd.DataFrame, period: int = 10) -> pd.Series:
    """
    Compute Volume RSI over the given DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns 'close' and 'volume'.
    period : int
        Look-back period for the RSI smoothing (default 10).

    Returns
    -------
    pd.Series
        Volume RSI values in range [0, 100].
    """
    close = df["close"]
    volume = df["volume"]

    # Determine up-volume / down-volume based on price direction
    price_delta = close.diff()

    up_vol = volume.where(price_delta > 0, 0.0)
    down_vol = volume.where(price_delta < 0, 0.0)

    # Wilder's smoothed average (equivalent to EMA with alpha = 1/period)
    avg_up = up_vol.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
    avg_down = down_vol.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()

    # Relative Strength
    rs = avg_up / avg_down.replace(0, np.nan)  # avoid div-by-zero

    # RSI normalisation
    vol_rsi = 100.0 - (100.0 / (1.0 + rs))
    vol_rsi = vol_rsi.fillna(50.0)  # neutral when no down-volume

    return vol_rsi
