"""
Williams VIX Fix (WVF) — LazyBear / ChrisMoody style.

VIXFix = (Highest(Close, 22) - Low) / Highest(Close, 22) * 100
Bollinger Bands: 20-period SMA, 2.0 std dev
Percentile:     97th over 50-period lookback
"""

import pandas as pd
import numpy as np


def calculate_vixfix(
    df: pd.DataFrame,
    pd_val: int = 22,
    bbl: int = 20,
    mult: float = 2.0,
    lb: int = 50,
    ph: float = 0.97,
):
    """
    Compute the Williams VIX Fix indicator and its derived bands/percentile.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns 'high', 'low', 'close'.
    pd_val : int
        Look-back period for highest close (default 22).
    bbl : int
        Bollinger Band SMA period (default 20).
    mult : float
        Bollinger Band standard-deviation multiplier (default 2.0).
    lb : int
        Look-back period for percentile calculation (default 50).
    ph : float
        Percentile threshold (default 0.97 → 97th).

    Returns
    -------
    vixfix      : pd.Series  — raw VIXFix values
    upper_band  : pd.Series  — upper Bollinger Band of VIXFix
    lower_band  : pd.Series  — lower Bollinger Band of VIXFix
    percentile_97 : pd.Series — rolling 97th percentile of VIXFix
    """
    # ── Core VIXFix formula ──────────────────────────────────────────────
    highest_close = df["close"].rolling(window=pd_val).max()
    vixfix = ((highest_close - df["low"]) / highest_close) * 100.0

    # ── Bollinger Bands on VIXFix ────────────────────────────────────────
    basis = vixfix.rolling(window=bbl).mean()
    std = vixfix.rolling(window=bbl).std(ddof=0)  # population std
    upper_band = basis + (mult * std)
    lower_band = basis - (mult * std)

    # ── 97th percentile over look-back period ────────────────────────────
    percentile_97 = vixfix.rolling(window=lb).quantile(ph, interpolation="lower")

    return vixfix, upper_band, lower_band, percentile_97


def vixfix_signal(
    vixfix: pd.Series,
    upper_band: pd.Series,
    percentile_97: pd.Series,
) -> pd.Series:
    """
    Return a boolean Series that is True when VIXFix triggers a fear extreme
    (≥ upper band  OR  ≥ 97th percentile).
    """
    cond_band = vixfix >= upper_band
    cond_pct = vixfix >= percentile_97
    return cond_band | cond_pct
