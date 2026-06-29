"""
Signal Generator

Combines multiple indicators to produce LONG / SHORT / NONE signals.
In TEST_MODE, always returns LONG to force a trade for end-to-end testing.
"""

import logging
from typing import Literal

import pandas as pd
import numpy as np

from config.settings import (
    TRADING_PAIR,
    TIMEFRAME_BIAS,
    TIMEFRAME_ENTRY,
    RSI_PERIOD,
    RSI_LONG_FILTER,
    RSI_SHORT_FILTER,
    HMA_PERIOD,
    VIXFIX_PD,
    VIXFIX_BBL,
    VIXFIX_MULT,
    VIXFIX_LB,
    VIXFIX_PH,
    VOLUME_RSI_PERIOD,
    VOLUME_RSI_LONG_FILTER,
    VOLUME_RSI_SHORT_FILTER,
    TEST_MODE,
)
from utils.vixfix import calculate_vixfix
from utils.volume_rsi import calculate_volume_rsi
from utils.logger import log_signal, log_system_error

Signal = Literal["LONG", "SHORT", "NONE"]

logger = logging.getLogger("signal_generator")
signals_logger = logging.getLogger("signals")

KLINE_OPEN_TIME = 0
KLINE_OPEN = 1
KLINE_HIGH = 2
KLINE_LOW = 3
KLINE_CLOSE = 4
KLINE_VOLUME = 5


class SignalGenerator:
    """Fetches market data and produces trading signals."""

    def __init__(self, binance_client):
        self._client = binance_client
        self._current_4h_bias: str = "NEUTRAL"

    def fetch_4h_data(self, limit: int = 200) -> pd.DataFrame:
        raw = self._client.get_klines(TRADING_PAIR, TIMEFRAME_BIAS, limit=limit)
        df = self._klines_to_df(raw)
        signals_logger.debug("fetched_4h_candles [count=%d] [interval=%s]", len(df), TIMEFRAME_BIAS)
        return df

    def fetch_1h_data(self, limit: int = 200) -> pd.DataFrame:
        raw = self._client.get_klines(TRADING_PAIR, TIMEFRAME_ENTRY, limit=limit)
        df = self._klines_to_df(raw)
        signals_logger.debug("fetched_1h_candles [count=%d] [interval=%s]", len(df), TIMEFRAME_ENTRY)
        return df

    def generate_signal(self) -> Signal:
        """Evaluate all conditions and return the current signal."""

        # ── TEST MODE: force LONG every call ─────────────────────────────
        if TEST_MODE:
            logger.warning("TEST_MODE active — forcing LONG signal")
            log_signal("TEST_MODE_SIGNAL", signal="LONG", reason="test_mode_forced")
            self._current_4h_bias = "BULL"
            return "LONG"

        # ── Normal signal logic ───────────────────────────────────────────
        try:
            df_4h = self.fetch_4h_data()
            df_1h = self.fetch_1h_data()
        except Exception:
            logger.exception("Failed to fetch klines for signal generation")
            log_signal("signal_generation_failed", reason="klines_fetch_error")
            return "NONE"

        current_price = df_1h["close"].iloc[-1]

        vixfix, upper_bb, lower_bb, pct97 = calculate_vixfix(
            df_4h,
            pd_val=VIXFIX_PD,
            bbl=VIXFIX_BBL,
            mult=VIXFIX_MULT,
            lb=VIXFIX_LB,
            ph=VIXFIX_PH,
        )
        vixfix_short_pct = vixfix.rolling(window=VIXFIX_LB).quantile(1.0 - VIXFIX_PH, interpolation="lower")

        vixfix_val = vixfix.iloc[-1]
        upper_bb_val = upper_bb.iloc[-1]
        lower_bb_val = lower_bb.iloc[-1]
        pct97_val = pct97.iloc[-1]
        pct3_val = vixfix_short_pct.iloc[-1]

        long_vix = vixfix_val >= upper_bb_val or vixfix_val >= pct97_val
        short_vix = vixfix_val <= lower_bb_val or vixfix_val <= pct3_val

        if long_vix:
            self._current_4h_bias = "BULL"
        elif short_vix:
            self._current_4h_bias = "BEAR"
        else:
            self._current_4h_bias = "NEUTRAL"

        signals_logger.debug(
            "vixfix_calculation [vixfix_raw=%.4f] [upper_bb=%.4f] [lower_bb=%.4f] "
            "[pct97=%.4f] [pct3=%.4f] [long_triggered=%s] [short_triggered=%s] [bias=%s]",
            vixfix_val, upper_bb_val, lower_bb_val, pct97_val, pct3_val,
            long_vix, short_vix, self._current_4h_bias,
        )

        rsi = self._calc_rsi(df_1h["close"], RSI_PERIOD)
        rsi_val = rsi.iloc[-1] if not rsi.empty and pd.notna(rsi.iloc[-1]) else 50.0
        signals_logger.debug("rsi_calculation [rsi_14_value=%.2f]", rsi_val)

        hma = self._calc_hma(df_1h["close"], HMA_PERIOD)
        hma_val = hma.iloc[-1] if not hma.empty and pd.notna(hma.iloc[-1]) else None
        close_val = df_1h["close"].iloc[-1]
        close_vs_hma = "above" if hma_val is not None and close_val > hma_val else "below" if hma_val is not None else "unknown"

        signals_logger.debug(
            "hma_calculation [hma_200=%s] [current_price=%.2f] [close_vs_hma=%s]",
            f"{hma_val:.2f}" if hma_val is not None else "N/A", close_val, close_vs_hma,
        )

        vol_rsi = calculate_volume_rsi(df_1h, period=VOLUME_RSI_PERIOD)
        vol_rsi_val = vol_rsi.iloc[-1] if not vol_rsi.empty and pd.notna(vol_rsi.iloc[-1]) else 50.0

        signals_logger.debug(
            "volume_rsi_calculation [vol_rsi=%.2f] [threshold_long=<%d] [threshold_short=>%d]",
            vol_rsi_val, VOLUME_RSI_LONG_FILTER, VOLUME_RSI_SHORT_FILTER,
        )

        long_cond_vix = long_vix
        long_cond_rsi = rsi_val > RSI_LONG_FILTER
        long_cond_hma = hma_val is not None and close_val > hma_val
        long_cond_volrsi = vol_rsi_val < VOLUME_RSI_LONG_FILTER

        long_reasons = []
        if not long_cond_vix:
            long_reasons.append(f"vixfix_not_triggered(val={vixfix_val:.2f},upperBB={upper_bb_val:.2f},pct97={pct97_val:.2f})")
        if not long_cond_rsi:
            long_reasons.append(f"rsi_not_above_{RSI_LONG_FILTER}(val={rsi_val:.2f})")
        if not long_cond_hma:
            long_reasons.append(f"close_not_above_hma(close={close_val:.2f},hma={hma_val:.2f})" if hma_val is not None else "hma_not_available")
        if not long_cond_volrsi:
            long_reasons.append(f"volrsi_not_below_{VOLUME_RSI_LONG_FILTER}(val={vol_rsi_val:.2f})")

        signals_logger.debug(
            "long_condition_check [vixfix=%s] [rsi=%s] [hma=%s] [volrsi=%s] [all=%s] [rejections=%s]",
            long_cond_vix, long_cond_rsi, long_cond_hma, long_cond_volrsi,
            all([long_cond_vix, long_cond_rsi, long_cond_hma, long_cond_volrsi]),
            "; ".join(long_reasons) if long_reasons else "none",
        )

        if all([long_cond_vix, long_cond_rsi, long_cond_hma, long_cond_volrsi]):
            log_signal(
                "SIGNAL_GENERATED",
                signal="LONG",
                current_price=f"{current_price:.2f}",
                vixfix_raw=f"{vixfix_val:.4f}",
                vixfix_upper_bb=f"{upper_bb_val:.4f}",
                vixfix_percentile_97=f"{pct97_val:.4f}",
                volume_rsi_value=f"{vol_rsi_val:.2f}",
                hma_200=f"{hma_val:.2f}" if hma_val is not None else "N/A",
                rsi_14=f"{rsi_val:.2f}",
                bias_4h=self._current_4h_bias,
            )
            logger.info(">>> LONG signal generated (VIXFix=%.2f, RSI=%.2f, VolRSI=%.2f)", vixfix_val, rsi_val, vol_rsi_val)
            return "LONG"

        short_cond_vix = short_vix
        short_cond_rsi = rsi_val < RSI_SHORT_FILTER
        short_cond_hma = hma_val is not None and close_val < hma_val
        short_cond_volrsi = vol_rsi_val > VOLUME_RSI_SHORT_FILTER

        short_reasons = []
        if not short_cond_vix:
            short_reasons.append(f"vixfix_not_triggered_short(val={vixfix_val:.2f},lowerBB={lower_bb_val:.2f},pct3={pct3_val:.2f})")
        if not short_cond_rsi:
            short_reasons.append(f"rsi_not_below_{RSI_SHORT_FILTER}(val={rsi_val:.2f})")
        if not short_cond_hma:
            short_reasons.append(f"close_not_below_hma(close={close_val:.2f},hma={hma_val:.2f})" if hma_val is not None else "hma_not_available")
        if not short_cond_volrsi:
            short_reasons.append(f"volrsi_not_above_{VOLUME_RSI_SHORT_FILTER}(val={vol_rsi_val:.2f})")

        signals_logger.debug(
            "short_condition_check [vixfix=%s] [rsi=%s] [hma=%s] [volrsi=%s] [all=%s] [rejections=%s]",
            short_cond_vix, short_cond_rsi, short_cond_hma, short_cond_volrsi,
            all([short_cond_vix, short_cond_rsi, short_cond_hma, short_cond_volrsi]),
            "; ".join(short_reasons) if short_reasons else "none",
        )

        if all([short_cond_vix, short_cond_rsi, short_cond_hma, short_cond_volrsi]):
            log_signal(
                "SIGNAL_GENERATED",
                signal="SHORT",
                current_price=f"{current_price:.2f}",
                vixfix_raw=f"{vixfix_val:.4f}",
                vixfix_lower_bb=f"{lower_bb_val:.4f}",
                vixfix_percentile_3=f"{pct3_val:.4f}",
                volume_rsi_value=f"{vol_rsi_val:.2f}",
                hma_200=f"{hma_val:.2f}" if hma_val is not None else "N/A",
                rsi_14=f"{rsi_val:.2f}",
                bias_4h=self._current_4h_bias,
            )
            logger.info("<<< SHORT signal generated (VIXFix=%.2f, RSI=%.2f, VolRSI=%.2f)", vixfix_val, rsi_val, vol_rsi_val)
            return "SHORT"

        rejection_parts = []
        if not long_cond_vix and not short_cond_vix:
            rejection_parts.append(f"vixfix_no_trigger(val={vixfix_val:.2f},range=[{lower_bb_val:.2f},{upper_bb_val:.2f}])")
        elif long_cond_vix:
            rejection_parts.append("vixfix_long_ok_but_other_failed")
        elif short_cond_vix:
            rejection_parts.append("vixfix_short_ok_but_other_failed")

        if not long_cond_rsi and not short_cond_rsi:
            rejection_parts.append(f"rsi_neutral({rsi_val:.2f},between_{RSI_LONG_FILTER}_and_{RSI_SHORT_FILTER})")
        if not long_cond_hma and not short_cond_hma:
            rejection_parts.append(f"hma_no_signal(close={close_val:.2f},hma={hma_val:.2f})" if hma_val is not None else "hma_na")
        if not long_cond_volrsi and not short_cond_volrsi:
            rejection_parts.append(f"volrsi_neutral({vol_rsi_val:.2f},between_{VOLUME_RSI_LONG_FILTER}_and_{VOLUME_RSI_SHORT_FILTER})")

        if long_cond_vix and not all([long_cond_vix, long_cond_rsi, long_cond_hma, long_cond_volrsi]):
            if not long_cond_rsi:
                rejection_parts.append(f"long_rejected_rsi({rsi_val:.2f}_not_above_{RSI_LONG_FILTER})")
            if not long_cond_hma:
                rejection_parts.append("long_rejected_hma")
            if not long_cond_volrsi:
                rejection_parts.append(f"long_rejected_volrsi({vol_rsi_val:.2f}_not_below_{VOLUME_RSI_LONG_FILTER})")

        if short_cond_vix and not all([short_cond_vix, short_cond_rsi, short_cond_hma, short_cond_volrsi]):
            if not short_cond_rsi:
                rejection_parts.append(f"short_rejected_rsi({rsi_val:.2f}_not_below_{RSI_SHORT_FILTER})")
            if not short_cond_hma:
                rejection_parts.append("short_rejected_hma")
            if not short_cond_volrsi:
                rejection_parts.append(f"short_rejected_volrsi({vol_rsi_val:.2f}_not_above_{VOLUME_RSI_SHORT_FILTER})")

        rejection_reason = "; ".join(rejection_parts) if rejection_parts else "unknown"

        log_signal(
            "NO_SIGNAL",
            final_signal="NONE",
            current_price=f"{current_price:.2f}",
            bias_4h=self._current_4h_bias,
            vixfix_raw=f"{vixfix_val:.4f}",
            vixfix_upper_bb=f"{upper_bb_val:.4f}",
            vixfix_lower_bb=f"{lower_bb_val:.4f}",
            vixfix_percentile_97=f"{pct97_val:.4f}",
            vixfix_triggered=f"{long_vix or short_vix}",
            volume_rsi_value=f"{vol_rsi_val:.2f}",
            volume_rsi_threshold_long=f"<{VOLUME_RSI_LONG_FILTER}",
            volume_rsi_threshold_short=f">{VOLUME_RSI_SHORT_FILTER}",
            volume_rsi_passed_long=f"{long_cond_volrsi}",
            volume_rsi_passed_short=f"{short_cond_volrsi}",
            hma_200_value=f"{hma_val:.2f}" if hma_val is not None else "N/A",
            close_vs_hma=close_vs_hma,
            hma_passed_long=f"{long_cond_hma}",
            hma_passed_short=f"{short_cond_hma}",
            rsi_14_value=f"{rsi_val:.2f}",
            rsi_passed_long=f"{long_cond_rsi}",
            rsi_passed_short=f"{short_cond_rsi}",
            reasons_for_rejection=rejection_reason,
        )

        logger.debug(
            "No signal (VIXFix=%.2f, RSI=%.2f, VolRSI=%.2f) — reasons: %s",
            vixfix_val, rsi_val, vol_rsi_val, rejection_reason,
        )

        return "NONE"

    def log_4h_candle_close(self, df_4h: pd.DataFrame):
        if len(df_4h) < 2:
            return
        latest = df_4h.iloc[-1]
        prev = df_4h.iloc[-2]
        log_signal(
            "4H_CANDLE_CLOSE",
            open=f"{latest['open']:.2f}",
            high=f"{latest['high']:.2f}",
            low=f"{latest['low']:.2f}",
            close=f"{latest['close']:.2f}",
            volume=f"{latest['volume']:.4f}",
            previous_close=f"{prev['close']:.2f}",
            candle_direction="BULL" if latest['close'] >= prev['close'] else "BEAR",
            bias_after=self._current_4h_bias,
        )

    def get_4h_bias(self) -> str:
        return self._current_4h_bias

    @staticmethod
    def _klines_to_df(raw: list[list]) -> pd.DataFrame:
        df = pd.DataFrame(
            raw,
            columns=[
                "open_time", "open", "high", "low", "close", "volume",
                "close_time", "quote_vol", "count", "taker_buy_vol",
                "taker_buy_quote", "ignore",
            ],
        )
        for col in ("open", "high", "low", "close", "volume"):
            df[col] = pd.to_numeric(df[col], errors="coerce")
        return df[["open", "high", "low", "close", "volume"]]

    @staticmethod
    def _calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
        delta = series.diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1.0 / period, min_periods=period, adjust=False).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        rsi = 100.0 - (100.0 / (1.0 + rs))
        return rsi.fillna(50.0)

    @staticmethod
    def _calc_wma(series: pd.Series, period: int) -> pd.Series:
        weights = np.arange(1, period + 1, dtype=float)
        w_sum = weights.sum()
        return series.rolling(period).apply(lambda x: np.dot(x, weights) / w_sum, raw=True)

    def _calc_hma(self, series: pd.Series, period: int = 200) -> pd.Series:
        n = int(period)
        half_n = max(1, int(n / 2))
        sqrt_n = max(1, int(np.sqrt(n)))
        wma_n = self._calc_wma(series, n)
        wma_half = self._calc_wma(series, half_n)
        return self._calc_wma((2.0 * wma_half) - wma_n, sqrt_n)