"""
Centralized Logger Setup — Crypto Trading Bot
===============================================

Provides:
- Rotating file handlers per log category (trades, signals, performance, system, equity)
- Color-coded console output (GREEN/RED/YELLOW/CYAN/WHITE)
- Performance tracker with daily/weekly/milestone summaries
- Live status line updates (no-trade: 5min, trade-active: 10s)
- Text dashboard printed every 6 hours
- Bankroll snapshot tracking (hourly + per-trade)
"""

import logging
import logging.handlers
import os
import sys
import time
import math
from datetime import datetime, timedelta, timezone
from typing import Optional

import pandas as pd
import numpy as np

from config.settings import LOG_CONFIG, TARGET_CAPITAL_USDT

# ── ANSI color codes ────────────────────────────────────────────────────────
class Color:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    WHITE = "\033[97m"
    BOLD = "\033[1m"
    RESET = "\033[0m"
    MAGENTA = "\033[95m"


LEVEL_COLORS = {
    "DEBUG": Color.WHITE,
    "INFO": Color.GREEN,
    "WARNING": Color.YELLOW,
    "ERROR": Color.RED,
    "CRITICAL": f"{Color.RED}{Color.BOLD}",
}


class ColoredFormatter(logging.Formatter):
    """Console formatter that applies ANSI color codes per log level."""

    def format(self, record: logging.LogRecord) -> str:
        level_color = LEVEL_COLORS.get(record.levelname, Color.WHITE)
        # Colour the whole line based on level severity
        msg = super().format(record)
        return f"{level_color}{msg}{Color.RESET}"


# ── Logger registry ─────────────────────────────────────────────────────────
_loggers: dict[str, logging.Logger] = {}
_initialized = False


def setup_logging() -> dict[str, logging.Logger]:
    """
    Initialise all loggers with rotating file handlers and a shared
    coloured console handler.  Called once at startup.

    Returns a dict {category: logger} for convenient access.
    """
    global _initialized
    if _initialized:
        return _loggers

    os.makedirs(LOG_CONFIG["DIR"], exist_ok=True)

    detailed_fmt = logging.Formatter(
        LOG_CONFIG["FORMAT"],
        datefmt=LOG_CONFIG["DATE_FORMAT"],
    )

    # ── Console handler (shared) ───────────────────────────────────────────
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, LOG_CONFIG["LEVEL"]))
    console_handler.setFormatter(ColoredFormatter(
        LOG_CONFIG["FORMAT"],
        datefmt=LOG_CONFIG["DATE_FORMAT"],
    ))

    # ── Root logger: console only, level DEBUG ────────────────────────────
    root = logging.getLogger()
    root.setLevel(logging.DEBUG)
    # Remove any pre-existing handlers so we don't double-log
    root.handlers.clear()
    root.addHandler(console_handler)

    # ── Per-category loggers with their own rotating file handler ──────────
    categories = [
        ("trades",    LOG_CONFIG["TRADES_LOG"]),
        ("signals",   LOG_CONFIG["SIGNALS_LOG"]),
        ("performance", LOG_CONFIG["PERFORMANCE_LOG"]),
        ("system",    LOG_CONFIG["SYSTEM_LOG"]),
        ("equity",    LOG_CONFIG["EQUITY_LOG"]),
    ]

    for cat_name, file_path in categories:
        logger = logging.getLogger(cat_name)
        logger.setLevel(logging.DEBUG)
        logger.propagate = True  # also go to root → console

        # Rotating file handler
        fh = logging.handlers.RotatingFileHandler(
            file_path,
            maxBytes=LOG_CONFIG["MAX_BYTES"],
            backupCount=LOG_CONFIG["BACKUP_COUNT"],
        )
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(detailed_fmt)
        logger.handlers.clear()
        logger.addHandler(fh)

        _loggers[cat_name] = logger

    # Suppress noisy libs
    logging.getLogger("binance").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    _initialized = True
    return _loggers


def get_logger(category: str) -> logging.Logger:
    """Return a pre-configured logger by category name."""
    if not _initialized:
        setup_logging()
    return _loggers.get(category, logging.getLogger(category))


# ── Convenience log helpers with structured key=value pairs ─────────────────

def _fmt_kv(**kwargs) -> str:
    """Format keyword arguments as ``[key=value ...]``."""
    if not kwargs:
        return ""
    parts = []
    for k, v in kwargs.items():
        if isinstance(v, float):
            parts.append(f"{k}={v:.4f}")
        elif isinstance(v, bool):
            parts.append(f"{k}={str(v).lower()}")
        elif v is None:
            parts.append(f"{k}=None")
        else:
            parts.append(f"{k}={v}")
    return "[" + " ".join(parts) + "]"


def log_trade(msg: str, **kv):
    get_logger("trades").info("%s %s", msg, _fmt_kv(**kv))


def log_trade_debug(msg: str, **kv):
    get_logger("trades").debug("%s %s", msg, _fmt_kv(**kv))


def log_signal(msg: str, **kv):
    get_logger("signals").info("%s %s", msg, _fmt_kv(**kv))


def log_signal_debug(msg: str, **kv):
    get_logger("signals").debug("%s %s", msg, _fmt_kv(**kv))


def log_performance(msg: str, **kv):
    get_logger("performance").info("%s %s", msg, _fmt_kv(**kv))


def log_system(msg: str, **kv):
    get_logger("system").info("%s %s", msg, _fmt_kv(**kv))


def log_system_warning(msg: str, **kv):
    get_logger("system").warning("%s %s", msg, _fmt_kv(**kv))


def log_system_error(msg: str, **kv):
    get_logger("system").error("%s %s", msg, _fmt_kv(**kv))


def log_system_critical(msg: str, **kv):
    get_logger("system").critical("%s %s", msg, _fmt_kv(**kv))


def log_equity(msg: str, **kv):
    get_logger("equity").info("%s %s", msg, _fmt_kv(**kv))


# ── Performance Dashboard (text, printed to console every 6H) ──────────────

class PerformanceTracker:
    """
    Aggregates trade/performance data and produces:
    - Daily summaries (every 24H)
    - Weekly summaries (every 168H)
    - Milestone summaries (every 50 trades)
    - Console dashboard (every 6H)
    - Hourly equity snapshots
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.start_time: float = time.time()
        self.day_number: int = 0
        self.starting_bankroll: float = 0.0
        self.bankroll: float = 0.0
        self.peak_bankroll: float = 0.0
        self.trades_taken: int = 0
        self.wins: int = 0
        self.losses: int = 0
        self.win_pcts: list[float] = []
        self.loss_pcts: list[float] = []
        self.largest_win: float = 0.0
        self.largest_loss: float = 0.0
        self.max_drawdown_day: float = 0.0
        self.max_drawdown_peak: float = 0.0  # peak-to-trough
        self.current_streak: int = 0  # positive = wins, negative = losses
        self.best_streak: int = 0
        self.worst_streak: int = 0
        self.total_fees: float = 0.0
        self.time_in_trades: list[float] = []
        self.last_day_write: int = -1
        self.last_week_write: int = -1
        self.last_6h_dashboard: float = 0.0
        self.last_hourly_equity: float = 0.0
        self.last_equity_hour: int = -1
        self.daily_trades: list[dict] = []
        self.weekly_trades: list[dict] = []

        # For live status line
        self.last_trade_time: float = 0.0
        self.last_trade_result: str = ""
        self.last_trade_pnl: float = 0.0
        self.current_4h_bias: str = "NEUTRAL"
        self.next_signal_time: float = 0.0

        # Position tracking
        self.has_position: bool = False
        self.position_side: str = ""
        self.position_entry: float = 0.0
        self.position_entry_time: float = 0.0

    # ── Trade recording ────────────────────────────────────────────────────

    def record_trade(self, pnl_usdt: float, pnl_pct: float, fees: float,
                     time_in_trade_min: float, bankroll_after: float):
        self.trades_taken += 1
        self.total_fees += fees
        self.time_in_trades.append(time_in_trade_min)
        self.bankroll = bankroll_after

        if pnl_usdt > 0:
            self.wins += 1
            self.win_pcts.append(pnl_pct)
            self.largest_win = max(self.largest_win, pnl_pct)
            self.current_streak = self.current_streak + 1 if self.current_streak >= 0 else 1
        else:
            self.losses += 1
            self.loss_pcts.append(pnl_pct)
            self.largest_loss = min(self.largest_loss, pnl_pct)
            self.current_streak = self.current_streak - 1 if self.current_streak <= 0 else -1

        self.best_streak = max(self.best_streak, self.current_streak)
        self.worst_streak = min(self.worst_streak, self.current_streak)

        if bankroll_after > self.peak_bankroll:
            self.peak_bankroll = bankroll_after

        # Track drawdown
        dd = (self.peak_bankroll - bankroll_after) / self.peak_bankroll * 100 if self.peak_bankroll > 0 else 0
        self.max_drawdown_peak = max(self.max_drawdown_peak, dd)

        # Daily aggregation
        day_rec = {
            "pnl_usdt": pnl_usdt,
            "pnl_pct": pnl_pct,
            "fees": fees,
            "time_min": time_in_trade_min,
            "bankroll_after": bankroll_after,
            "is_win": pnl_usdt > 0,
        }
        self.daily_trades.append(day_rec)
        self.weekly_trades.append(day_rec)

        # Update last trade info
        self.last_trade_time = time.time()
        self.last_trade_result = "WIN" if pnl_usdt > 0 else "LOSS"
        self.last_trade_pnl = pnl_usdt

        # Milestone check
        if self.trades_taken % 50 == 0:
            self._write_milestone()

    # ── Periodically called checks ─────────────────────────────────────────

    def check_hourly(self, bankroll: float, open_position_value: float,
                     unrealized_pnl: float, margin_used: float,
                     available_margin: float, current_leverage: float,
                     distance_to_liquidation_pct: float):
        """Write equity snapshot if an hour has passed since last write."""
        now = datetime.now(timezone.utc)
        hour = now.hour
        if hour != self.last_equity_hour:
            self.last_equity_hour = hour
            dd_peak = 0.0
            if self.peak_bankroll > 0:
                dd_peak = (self.peak_bankroll - bankroll) / self.peak_bankroll * 100

            log_equity(
                "hourly_snapshot",
                bankroll=f"{bankroll:.2f}",
                open_position_value=f"{open_position_value:.2f}",
                total_equity=f"{bankroll + unrealized_pnl:.2f}",
                unrealized_pnl=f"{unrealized_pnl:.2f}",
                margin_used=f"{margin_used:.2f}",
                available_margin=f"{available_margin:.2f}",
                current_leverage=f"{current_leverage:.2f}",
                distance_to_liquidation_pct=f"{distance_to_liquidation_pct:.2f}",
                peak_bankroll=f"{self.peak_bankroll:.2f}",
                drawdown_from_peak_pct=f"{dd_peak:.2f}",
                trades_count_total=self.trades_taken,
                win_count=self.wins,
                loss_count=self.losses,
            )

    def check_daily_summary(self, bankroll: float):
        """Write daily performance summary if a new calendar day."""
        now = datetime.now(timezone.utc)
        day_num = now.timetuple().tm_yday
        if day_num != self.last_day_write and self.trades_taken > 0:
            self.last_day_write = day_num
            self.day_number += 1
            self._write_daily_summary(bankroll)

    def check_weekly_summary(self, bankroll: float):
        """Write weekly performance summary (every 7 days)."""
        now = datetime.now(timezone.utc)
        week_num = now.isocalendar()[1]
        if week_num != self.last_week_write and self.trades_taken > 0:
            self.last_week_write = week_num
            self._write_weekly_summary(bankroll)

    def check_6h_dashboard(self, bankroll: float):
        """Print formatted dashboard to console every 6 hours."""
        now = time.time()
        if now - self.last_6h_dashboard < 6 * 3600:
            return
        self.last_6h_dashboard = now
        self.print_dashboard(bankroll)

    # ── Internal summary writers ───────────────────────────────────────────

    def _write_daily_summary(self, bankroll: float):
        """Write daily aggregates to performance.log."""
        if not self.daily_trades:
            return

        day_trades = self.daily_trades
        wins_d = [t for t in day_trades if t["is_win"]]
        losses_d = [t for t in day_trades if not t["is_win"]]
        n = len(day_trades)

        win_rate_day = len(wins_d) / n * 100 if n > 0 else 0.0
        win_rate_cum = self.wins / self.trades_taken * 100 if self.trades_taken > 0 else 0.0
        avg_win = sum(t["pnl_pct"] for t in wins_d) / len(wins_d) if wins_d else 0.0
        avg_loss = sum(t["pnl_pct"] for t in losses_d) / len(losses_d) if losses_d else 0.0
        largest_win = max((t["pnl_pct"] for t in wins_d), default=0.0)
        largest_loss = min((t["pnl_pct"] for t in losses_d), default=0.0)
        avg_time = sum(t["time_min"] for t in day_trades) / n if n > 0 else 0.0
        total_fees_day = sum(t["fees"] for t in day_trades)

        # Drawdown
        peak_d = max(self.peak_bankroll, bankroll)
        dd_d = (peak_d - bankroll) / peak_d * 100 if peak_d > 0 else 0.0

        # Geometric growth rate for the day
        day_pnl_pcts = [t["pnl_pct"] / 100.0 for t in day_trades]
        geo_growth = 1.0
        for p in day_pnl_pcts:
            geo_growth *= (1.0 + p)
        geo_rate = (geo_growth - 1.0) * 100.0 if geo_growth > 0 else 0.0

        # Estimate days to target
        days_to_1b = self._estimate_days_to_target(bankroll, geo_rate)

        starting = self.starting_bankroll if self.starting_bankroll > 0 else bankroll
        bankroll_pct = bankroll / TARGET_CAPITAL_USDT * 100 if TARGET_CAPITAL_USDT > 0 else 0.0
        start_mult = bankroll / starting if starting > 0 else 1.0

        log_performance(
            "DAILY_SUMMARY",
            day_number=self.day_number,
            starting_bankroll=f"{starting:.2f}",
            ending_bankroll=f"{bankroll:.2f}",
            trades_taken=n,
            wins=len(wins_d),
            losses=len(losses_d),
            win_rate_day=f"{win_rate_day:.2f}%",
            win_rate_cumulative=f"{win_rate_cum:.2f}%",
            avg_win_pct=f"{avg_win:.4f}%",
            avg_loss_pct=f"{avg_loss:.4f}%",
            largest_win=f"{largest_win:.4f}%",
            largest_loss=f"{largest_loss:.4f}%",
            max_drawdown_day=f"{dd_d:.2f}%",
            max_drawdown_peak_to_trough=f"{self.max_drawdown_peak:.2f}%",
            current_streak=self.current_streak,
            best_streak=self.best_streak,
            worst_streak=self.worst_streak,
            avg_time_in_trade=f"{avg_time:.1f}min",
            total_fees_paid=f"{total_fees_day:.4f}",
            geometric_growth_rate=f"{geo_rate:.4f}%",
            estimated_days_to_1b=days_to_1b,
            bankroll_as_pct_of_target=f"{bankroll_pct:.6f}%",
            distance_from_start_multiple=f"{start_mult:.4f}x",
        )

        # Reset daily aggregation
        self.daily_trades = []

    def _write_weekly_summary(self, bankroll: float):
        """Write weekly aggregates to performance.log."""
        if not self.weekly_trades:
            return

        week_trades = self.weekly_trades
        wins_w = [t for t in week_trades if t["is_win"]]
        losses_w = [t for t in week_trades if not t["is_win"]]
        n = len(week_trades)

        win_rate_w = len(wins_w) / n * 100 if n > 0 else 0.0
        avg_win = sum(t["pnl_pct"] for t in wins_w) / len(wins_w) if wins_w else 0.0
        avg_loss = sum(t["pnl_pct"] for t in losses_w) / len(losses_w) if losses_w else 0.0
        avg_time = sum(t["time_min"] for t in week_trades) / n if n > 0 else 0.0
        total_fees_w = sum(t["fees"] for t in week_trades)

        # Approximate Sharpe (using daily returns if we assume ~3 trades/day avg)
        pcts = [t["pnl_pct"] / 100.0 for t in week_trades]
        sharpe = 0.0
        if len(pcts) > 1 and np.std(pcts) > 0:
            sharpe = np.mean(pcts) / np.std(pcts) * np.sqrt(365)  # annualised approx

        # Risk of ruin estimate
        wr = self.wins / self.trades_taken if self.trades_taken > 0 else 0.5
        avg_r = abs(avg_loss / avg_win) if avg_win != 0 else 1.0
        ror = self._risk_of_ruin(wr, avg_r, bankroll / TARGET_CAPITAL_USDT if TARGET_CAPITAL_USDT > 0 else 1.0)

        # 50% WR comparison
        expected_at_50wr = sum(t["pnl_usdt"] for t in week_trades)  # simplified

        log_performance(
            "WEEKLY_SUMMARY",
            week_number=self.last_week_write,
            trades_taken=n,
            wins=len(wins_w),
            losses=len(losses_w),
            win_rate_week=f"{win_rate_w:.2f}%",
            cumulative_win_rate=f"{self.wins / self.trades_taken * 100:.2f}%" if self.trades_taken > 0 else "N/A",
            avg_win_pct=f"{avg_win:.4f}%",
            avg_loss_pct=f"{avg_loss:.4f}%",
            avg_time_in_trade=f"{avg_time:.1f}min",
            total_fees_paid=f"{total_fees_w:.4f}",
            sharpe_ratio_approx=f"{sharpe:.4f}",
            risk_of_ruin_estimate=f"{ror:.6f}%",
            current_bankroll=f"{bankroll:.2f}",
            peak_bankroll=f"{self.peak_bankroll:.2f}",
            comparison_to_50wr_expectation="above" if expected_at_50wr > 0 else "below",
        )

        self.weekly_trades = []

    def _write_milestone(self):
        """Write milestone summary every 50 trades."""
        days_elapsed = (time.time() - self.start_time) / 86400.0
        days_elapsed = max(1.0, days_elapsed)
        trades_per_day = self.trades_taken / days_elapsed

        # Projected completion at current pace
        if self.trades_taken > 0 and days_elapsed > 0:
            daily_bankroll_growth = (self.bankroll / self.starting_bankroll) ** (1.0 / days_elapsed) if self.starting_bankroll > 0 else 1.0
            if daily_bankroll_growth > 1.0:
                days_to_target = math.log(TARGET_CAPITAL_USDT / self.bankroll) / math.log(daily_bankroll_growth) if self.bankroll > 0 else float("inf")
            else:
                days_to_target = float("inf")
        else:
            days_to_target = float("inf")

        projected_date = ""
        if math.isfinite(days_to_target):
            est_date = datetime.now(timezone.utc) + timedelta(days=int(days_to_target))
            projected_date = est_date.strftime("%Y-%m-%d")
        else:
            projected_date = "N/A"

        log_performance(
            "MILESTONE",
            milestone_number=self.trades_taken,
            bankroll=f"{self.bankroll:.2f}",
            days_elapsed=f"{days_elapsed:.1f}",
            trades_per_day_avg=f"{trades_per_day:.2f}",
            projected_completion_date=projected_date,
            current_win_rate=f"{self.wins / self.trades_taken * 100:.2f}%" if self.trades_taken > 0 else "N/A",
            peak_bankroll=f"{self.peak_bankroll:.2f}",
        )

    def print_dashboard(self, bankroll: float):
        """Print a formatted text dashboard to the console. (Public method.)"""
        dd_peak = 0.0
        if self.peak_bankroll > 0:
            dd_peak = (self.peak_bankroll - bankroll) / self.peak_bankroll * 100

        wr = self.wins / self.trades_taken * 100 if self.trades_taken > 0 else 0.0
        avg_w = sum(self.win_pcts) / len(self.win_pcts) if self.win_pcts else 0.0
        avg_l = sum(self.loss_pcts) / len(self.loss_pcts) if self.loss_pcts else 0.0

        # Estimate time to 1B
        days_elapsed = max(1.0, (time.time() - self.start_time) / 86400.0)
        if self.trades_taken > 0 and bankroll > self.starting_bankroll > 0:
            daily_growth = (bankroll / self.starting_bankroll) ** (1.0 / days_elapsed)
            days_at_current = math.log(TARGET_CAPITAL_USDT / bankroll) / math.log(daily_growth) if daily_growth > 1 else float("inf")
        else:
            days_at_current = float("inf")

        # Days at 50% WR with R:R = 4.85
        if self.trades_taken > 0:
            avg_trade_pct = (avg_w * wr + avg_l * (100 - wr)) / 100.0 / 100.0 if wr > 0 else 0.0
            if avg_trade_pct > 0:
                trades_needed = math.log(TARGET_CAPITAL_USDT / bankroll) / math.log(1 + avg_trade_pct) if bankroll > 0 else float("inf")
                tpd = self.trades_taken / days_elapsed
                days_at_50wr = trades_needed / tpd if tpd > 0 else float("inf")
            else:
                days_at_50wr = float("inf")
        else:
            days_at_50wr = float("inf")

        bankroll_pct = bankroll / TARGET_CAPITAL_USDT * 100 if TARGET_CAPITAL_USDT > 0 else 0.0
        last_trade_str = "N/A"
        if self.last_trade_time > 0:
            mins_ago = int((time.time() - self.last_trade_time) / 60)
            last_trade_str = f"[{self.last_trade_result}] {'+' if self.last_trade_pnl >= 0 else ''}{self.last_trade_pnl:.2f} USDT  |  {mins_ago} min ago"

        # Determine color for key metrics
        dd_color = Color.RED if dd_peak >= 20 else Color.YELLOW if dd_peak >= 10 else Color.WHITE
        wr_color = Color.GREEN if wr >= 50 else Color.RED

        sep = f"{Color.CYAN}{'═' * 68}{Color.RESET}"
        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")

        print(f"\n{Color.CYAN}╔{'═' * 68}╗{Color.RESET}")
        print(f"{Color.CYAN}║{Color.RESET}  {Color.BOLD}CRYPTO BOT PERFORMANCE — Day {self.day_number}, {now_str}{Color.RESET}")
        print(f"{Color.CYAN}╠{'═' * 68}╣{Color.RESET}")
        print(f"{Color.CYAN}║{Color.RESET}  BANKROLL      ${bankroll:>10,.2f}  /  ${TARGET_CAPITAL_USDT:>13,.2f}  ({bankroll_pct:.6f}%)")
        print(f"{Color.CYAN}║{Color.RESET}  PEAK          ${self.peak_bankroll:>10,.2f}  |  DRAWDOWN  {dd_color}{dd_peak:>7.2f}%{Color.RESET}")
        print(f"{Color.CYAN}║{Color.RESET}  TRADES        {self.trades_taken:>3d}  |  WINS {self.wins:>3d}  |  LOSSES {self.losses:>3d}  |  WR {wr_color}{wr:>5.1f}%{Color.RESET}")
        print(f"{Color.CYAN}║{Color.RESET}  AVG WIN       {Color.GREEN}+{avg_w:>6.2f}%{Color.RESET}  |  AVG LOSS  {Color.RED}{avg_l:>7.2f}%{Color.RESET}")
        print(f"{Color.CYAN}║{Color.RESET}  LAST TRADE    {last_trade_str}")
        print(f"{Color.CYAN}║{Color.RESET}  NEXT SIGNAL   Checking in --:--  |  4H bias: {Color.CYAN}{self.current_4h_bias}{Color.RESET}")

        days_curr_str = f"{int(days_at_current):,} days" if math.isfinite(days_at_current) else "∞"
        days_50_str = f"{int(days_at_50wr):,} days" if math.isfinite(days_at_50wr) else "∞"
        print(f"{Color.CYAN}║{Color.RESET}  EST. TO 1B    {days_curr_str} at current rate / {days_50_str} at 50% WR")
        print(f"{Color.CYAN}╚{'═' * 68}╝{Color.RESET}\n")

    # ── Statistical helpers ────────────────────────────────────────────────

    @staticmethod
    def _estimate_days_to_target(bankroll: float, daily_growth_pct: float) -> str:
        if daily_growth_pct <= 0 or bankroll <= 0:
            return "N/A"
        daily_mult = 1.0 + daily_growth_pct / 100.0
        if daily_mult <= 1.0:
            return "N/A"
        days = math.log(TARGET_CAPITAL_USDT / bankroll) / math.log(daily_mult)
        if math.isfinite(days) and days > 0:
            return f"{int(days):,}"
        return "N/A"

    @staticmethod
    def _risk_of_ruin(win_rate: float, avg_risk_reward: float, capital_ratio: float) -> float:
        """
        Approximate risk of ruin using Kelly-derived formula.
        Simplified: ROR = ((1 - WR) / WR) ^ (capital / avg_trade_risk)
        """
        if win_rate <= 0 or win_rate >= 1:
            return 100.0 if win_rate <= 0 else 0.0
        if avg_risk_reward <= 0:
            return 100.0
        # b = payout ratio (w/l), p = win rate
        b = 1.0 / avg_risk_reward  # reward per unit risk
        p = win_rate
        q = 1.0 - p
        if p <= 0 or b <= 0:
            return 100.0
        try:
            term = q / p
            ror = term ** capital_ratio
            return min(max(ror * 100.0, 0.0), 100.0)
        except (ValueError, OverflowError):
            return 100.0


# ── Live status line ────────────────────────────────────────────────────────

class StatusLine:
    """
    Prints a live-updating status line using carriage-return technique.
    - No active trade: updates every 5 minutes
    - Active trade: updates every 10 seconds
    """

    def __init__(self):
        self.last_print: float = 0.0
        self.no_trade_interval: float = 300.0  # 5 min
        self.trade_interval: float = 10.0      # 10 sec

    def render_no_trade(self, bankroll: float, peak: float, trades: int,
                        wins: int, losses: int, days_running: float,
                        next_signal_in_secs: int):
        dd = (peak - bankroll) / peak * 100 if peak > 0 else 0.0
        wr = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0.0
        mins = next_signal_in_secs // 60
        secs = next_signal_in_secs % 60

        dd_color = Color.RED if dd >= 20 else Color.YELLOW if dd >= 10 else Color.WHITE
        line = (
            f"{Color.CYAN}┤{Color.RESET} "
            f"Balance: {Color.GREEN}${bankroll:>8.2f}{Color.RESET} "
            f"| Peak: ${peak:>8.2f} "
            f"| DD: {dd_color}{dd:>5.1f}%{Color.RESET} "
            f"| Trades: {trades:>3d} "
            f"| WR: {Color.GREEN if wr >= 50 else Color.RED}{wr:>4.1f}%{Color.RESET} "
            f"| Days: {days_running:.1f} "
            f"| Next signal: {mins:>2d}:{secs:02d} "
            f"{Color.CYAN}├{Color.RESET}"
        )
        self._print(line)

    def render_with_position(self, side: str, entry: float, current: float,
                             unrealized_pct: float, stop: float, target: float,
                             time_in_trade_min: float):
        pnl_color = Color.GREEN if unrealized_pct >= 0 else Color.RED
        line = (
            f"{Color.MAGENTA}┤{Color.RESET} "
            f"Position: {Color.BOLD}{side}{Color.RESET} "
            f"| Entry: {entry:.2f} "
            f"| Current: {current:.2f} "
            f"| PnL: {pnl_color}{unrealized_pct:>+6.2f}%{Color.RESET} "
            f"| Stop: {Color.RED}{stop:.2f}{Color.RESET} "
            f"| Target: {Color.GREEN}{target:.2f}{Color.RESET} "
            f"| Time: {time_in_trade_min:.0f}m "
            f"{Color.MAGENTA}├{Color.RESET}"
        )
        self._print(line)

    def _print(self, line: str):
        # Overwrite current line with carriage return
        print(f"\r{' ' * 120}\r{line}", end="", flush=True)

    def clear(self):
        print(f"\r{' ' * 120}\r", end="", flush=True)
