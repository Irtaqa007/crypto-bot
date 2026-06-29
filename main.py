#!/usr/bin/env python3
"""
Crypto Trading Bot – Binance Futures (USDT-M)

Orchestrates:
  1. Check balance
  2. Check for open position
  3. If no position → generate signal
  4. If signal → execute trade
  5. Sleep until next 1H candle
  6. Max 5 trades/day
  7. Deep logging to trades.log, signals.log, performance.log, system.log, equity.log
  8. Performance dashboard every 6H on console
  9. Live status line (no-trade: 5min, trade-active: 10s)
"""

import datetime
import logging
import os
import sys
import time
from typing import Optional


from config.settings import (
    TARGET_CAPITAL_USDT,
    MAX_TRADES_PER_DAY,
    LOG_CONFIG,
    EXCEPTION_COOLDOWN_SECONDS,
    SLEEP_BEFORE_NEXT_CANDLE_BUFFER,
    TRADING_PAIR,
    COMPOUNDING_ENABLED,
    LEVERAGE,
    STARTING_CAPITAL_USDT,
)
from core.binance_client import BinanceFuturesClient
from core.risk_manager import RiskManager
from core.signal_generator import SignalGenerator
from core.trade_executor import TradeExecutor
from utils.logger import (
    setup_logging,
    get_logger,
    log_trade,
    log_trade_debug,
    log_signal,
    log_system,
    log_system_warning,
    log_system_error,
    log_system_critical,
    log_equity,
    PerformanceTracker,
    StatusLine,
)

# ── Initialise logging framework ───────────────────────────────────────────
setup_logging()

logger = logging.getLogger("main")
system_logger = logging.getLogger("system")
trades_logger = logging.getLogger("trades")
equity_logger = logging.getLogger("equity")

# ── State tracking ──────────────────────────────────────────────────────────

class BotState:
    """Simple runtime state — tracks bankroll, trades, peak capital etc."""

    def __init__(self):
        self.capital: float = 0.0          # latest available balance
        self.peak_capital: float = 0.0
        self.trades_today: int = 0
        self.current_date: str = ""        # YYYY-MM-DD
        self.consecutive_errors: int = 0

    def reset_daily_counter(self):
        """Reset trade counter if a new calendar day has started."""
        today = datetime.datetime.utcnow().strftime("%Y-%m-%d")
        if today != self.current_date:
            logger.info("New day %s — resetting trade counter (was %d)", today, self.trades_today)
            log_system(
                "day_reset",
                date=today,
                previous_trades=self.trades_today,
            )
            self.trades_today = 0
            self.current_date = today

    def check_drawdown(self):
        """Log critical alert if bankroll dropped ≥30% from peak."""
        if self.peak_capital <= 0:
            return
        dd_pct = (self.peak_capital - self.capital) / self.peak_capital * 100.0
        if dd_pct >= 30.0:
            log_system_critical(
                "DRAWDOWN_ALERT",
                peak=f"{self.peak_capital:.2f}",
                current=f"{self.capital:.2f}",
                drawdown_pct=f"{dd_pct:.1f}%",
            )
            logger.critical(
                "DRAWDOWN ALERT — Peak: %.2f USDT | Current: %.2f USDT | Drawdown: %.1f%%",
                self.peak_capital, self.capital, dd_pct,
            )


# ── Main loop ───────────────────────────────────────────────────────────────

def main_loop():
    """
    Core trading loop — runs indefinitely until KeyboardInterrupt.
    """
    client = BinanceFuturesClient()
    risk_mgr = RiskManager(client)
    signal_gen = SignalGenerator(client)
    executor = TradeExecutor(client, risk_mgr)
    state = BotState()
    perf_tracker = PerformanceTracker()
    status_line = StatusLine()

    # ── Initialise exchange settings ──────────────────────────────────────
    try:
        client.set_leverage(TRADING_PAIR, LEVERAGE)
        client.set_margin_type(TRADING_PAIR, "ISOLATED")
    except Exception:
        logger.exception("Failed to initialise leverage/margin — continuing anyway")
        log_system_error("initialisation_failed", component="leverage_margin_setup")

    # Ensure no leftover orders
    executor.cancel_pending_orders()

    # Log startup
    log_system(
        "BOT_STARTED",
        trading_pair=TRADING_PAIR,
        leverage=LEVERAGE,
        margin_type="ISOLATED",
        max_trades_per_day=MAX_TRADES_PER_DAY,
        risk_per_trade_pct="5.08%",
        risk_reward_ratio="4.85",
        starting_capital=f"{STARTING_CAPITAL_USDT:.2f}",
        target_capital=f"{TARGET_CAPITAL_USDT:.2f}",
        compounding=f"{COMPOUNDING_ENABLED}",
        log_level=LOG_CONFIG["LEVEL"],
    )

    logger.info("=" * 60)
    logger.info("BOT STARTED — %s | %dx isolated | %d trades/day max",
                TRADING_PAIR, LEVERAGE, MAX_TRADES_PER_DAY)
    logger.info("=" * 60)

    start_time = time.time()

    while True:
        try:
            state.reset_daily_counter()

            # ── 1. Fetch balance ──────────────────────────────────────────
            state.capital = client.get_balance("USDT")
            if state.capital <= 0:
                logger.warning("Balance is 0 or negative (%.2f); sleeping", state.capital)
                log_system_warning("balance_zero_or_negative", balance=f"{state.capital:.2f}")
                time.sleep(60)
                continue

            # Track peak
            if state.capital > state.peak_capital:
                state.peak_capital = state.capital
                log_system_debug("new_peak", peak=f"{state.peak_capital:.2f}")

            # Initialise starting bankroll for performance tracker
            if perf_tracker.starting_bankroll == 0:
                perf_tracker.starting_bankroll = state.capital
                perf_tracker.bankroll = state.capital
                perf_tracker.peak_bankroll = state.capital

            # Update perf tracker bankroll
            perf_tracker.bankroll = state.capital
            perf_tracker.peak_bankroll = max(perf_tracker.peak_bankroll, state.capital)

            logger.info("─── Current capital: %.2f USDT (peak: %.2f) ───",
                        state.capital, state.peak_capital)

            # Check drawdown
            state.check_drawdown()

            # ── Perf tracker periodic tasks ───────────────────────────────
            days_running = (time.time() - start_time) / 86400.0

            # Hourly equity snapshot
            perf_tracker.check_hourly(
                bankroll=state.capital,
                open_position_value=0.0,
                unrealized_pnl=0.0,
                margin_used=0.0,
                available_margin=state.capital,
                current_leverage=LEVERAGE,
                distance_to_liquidation_pct=100.0,
            )

            # Daily summary
            perf_tracker.check_daily_summary(state.capital)

            # Weekly summary
            perf_tracker.check_weekly_summary(state.capital)

            # 6H dashboard
            perf_tracker.check_6h_dashboard(state.capital)

            # ── 2. Check position ─────────────────────────────────────────
            position = executor.has_open_position()
            if position:
                summary = executor.get_position_summary()
                logger.info("Open position: %s", summary)
                log_system(
                    "position_check",
                    has_position=True,
                    side=summary.get("side", "?") if summary else "?",
                    entry_price=summary.get("entry_price", 0) if summary else 0,
                    size=summary.get("size", 0) if summary else 0,
                    unrealized_pnl=summary.get("unrealized_pnl", 0) if summary else 0,
                    liquidation_price=summary.get("liquidation_price", 0) if summary else 0,
                    mark_price=summary.get("mark_price", 0) if summary else 0,
                )

                # Update trade drawdown tracking
                if summary:
                    executor.update_trade_drawdown(summary.get("mark_price", 0))

                # Status line (trade active: 10s interval)
                now = time.time()
                if now - status_line.last_print >= status_line.trade_interval:
                    side_s = summary.get("side", "?").upper() if summary else "?"
                    entry_px = summary.get("entry_price", 0) if summary else 0
                    mark_px = summary.get("mark_price", 0) if summary else 0
                    unrel = summary.get("unrealized_pnl", 0) if summary else 0
                    unrel_pct = (unrel / state.capital) * 100 if state.capital > 0 else 0
                    liq = summary.get("liquidation_price", 0) if summary else 0
                    time_in = (time.time() - executor._entry_time) / 60.0 if executor._entry_time else 0

                    sl, tp = 0.0, 0.0
                    # Recalculate from side if entry price available
                    if entry_px > 0 and side_s in ("LONG", "SHORT"):
                        if side_s == "LONG":
                            sl = entry_px * (1 - 0.00254)
                            tp = entry_px * (1 + 0.01232)
                        else:
                            sl = entry_px * (1 + 0.00254)
                            tp = entry_px * (1 - 0.01232)

                    status_line.render_with_position(
                        side=side_s,
                        entry=entry_px,
                        current=mark_px,
                        unrealized_pct=unrel_pct,
                        stop=sl,
                        target=tp,
                        time_in_trade_min=time_in,
                    )
                    status_line.last_print = now

                time.sleep(30)
                continue

            # ── 3. Enforce max daily trades ──────────────────────────────
            if state.trades_today >= MAX_TRADES_PER_DAY:
                logger.info("Max trades/day (%d) reached; waiting for next day", MAX_TRADES_PER_DAY)
                log_system("max_trades_reached", trades_today=state.trades_today, max_per_day=MAX_TRADES_PER_DAY)
                _sleep_until_next_hour()
                continue

            # ── 4. Generate signal ────────────────────────────────────────
            signal = signal_gen.generate_signal()
            logger.info("Signal: %s (trade #%d today)", signal, state.trades_today + 1)

            # Update perf tracker bias
            perf_tracker.current_4h_bias = signal_gen.get_4h_bias()

            # Status line (no trade: 5min interval)
            now = time.time()
            if now - status_line.last_print >= status_line.no_trade_interval:
                next_signal = _seconds_until_next_hour()
                status_line.render_no_trade(
                    bankroll=state.capital,
                    peak=state.peak_capital,
                    trades=state.trades_today,
                    wins=perf_tracker.wins,
                    losses=perf_tracker.losses,
                    days_running=days_running,
                    next_signal_in_secs=next_signal,
                )
                status_line.last_print = now

            if signal == "NONE":
                logger.info("No valid signal; sleeping until next candle")
                _sleep_until_next_hour()
                continue

            # ── 5. Execute trade ──────────────────────────────────────────
            bankroll = state.capital  # compounding — use current balance
            success = False

            if signal == "LONG":
                success = executor.execute_long(bankroll)
            elif signal == "SHORT":
                success = executor.execute_short(bankroll)

            if success:
                state.trades_today += 1
                state.consecutive_errors = 0
                entry_price = executor._current_entry_price or 0.0
                qty = executor._current_quantity or 0.0
                logger.info(
                    "%s Entry — Capital: %.2f USDT | Size: %.6f ETH | Price: %.2f | Trade #%d today",
                    signal, bankroll, qty, entry_price, state.trades_today,
                )

                # ── Wait for position to close ───────────────────────────
                _wait_for_position_close(
                    client, executor, signal, state, entry_price,
                    perf_tracker, status_line,
                )
            else:
                logger.error("Trade execution failed — %s signal could not be executed", signal)
                log_system_error("trade_execution_failed", signal=signal)
                state.consecutive_errors += 1

                if state.consecutive_errors >= 5:
                    log_system_critical(
                        "CONSECUTIVE_ERRORS",
                        count=state.consecutive_errors,
                        action="shutting_down",
                    )
                    logger.critical("5 consecutive errors — shutting down")
                    break

            # ── Sleep until next 1H candle ────────────────────────────────
            _sleep_until_next_hour()

        except KeyboardInterrupt:
            _shutdown_gracefully(client, executor, state, perf_tracker, signal_gen)
            break

        except Exception:
            logger.exception("Unhandled exception in main loop")
            log_system_error("unhandled_exception", loop="main")
            state.consecutive_errors += 1
            if state.consecutive_errors >= 5:
                log_system_critical(
                    "CONSECUTIVE_ERRORS",
                    count=state.consecutive_errors,
                    action="shutting_down",
                )
                logger.critical("5 consecutive errors — shutting down")
                break
            logger.critical(
                "Unhandled exception on %s — restarting in %ds",
                TRADING_PAIR, EXCEPTION_COOLDOWN_SECONDS,
            )
            time.sleep(EXCEPTION_COOLDOWN_SECONDS)


# ── Helper: wait for position close ─────────────────────────────────────────

def _wait_for_position_close(
    client: BinanceFuturesClient,
    executor: TradeExecutor,
    signal: str,
    state: BotState,
    entry_price: float,
    perf_tracker: PerformanceTracker,
    status_line: StatusLine,
):
    """
    Poll until the position is closed (no remaining amount).
    Logs outcome with full PnL breakdown on win/loss.
    """
    logger.info("Waiting for position to close …")
    log_system("waiting_for_position_close")
    poll_interval = 10  # seconds

    bankroll_at_entry = state.capital  # snapshot before trade

    while True:
        try:
            pos = client.get_position(TRADING_PAIR)
            if pos is None:
                # Position closed — determine outcome
                exit_px = client.get_price(TRADING_PAIR)
                new_balance = client.get_balance("USDT")
                pnl = new_balance - bankroll_at_entry
                pnl_pct = (pnl / bankroll_at_entry) * 100.0 if bankroll_at_entry > 0 else 0.0

                # Determine exit reason by checking which TP/SL order was filled
                exit_reason = "STOP"
                if pnl > 0:
                    exit_reason = "TP"
                elif abs(pnl) < 0.001:
                    exit_reason = "TIMEOUT"

                outcome = "WIN 🟢" if pnl > 0 else "LOSS 🔴"

                # Log full trade exit via executor
                slippage = abs(exit_px - executor._current_entry_price) if executor._current_entry_price else 0.0
                executor.log_trade_exit(
                    exit_price=exit_px,
                    exit_reason=exit_reason,
                    bankroll_before=bankroll_at_entry,
                    bankroll_after=new_balance,
                    fees_paid=0.0,  # Binance fees deducted from balance already
                    slippage=slippage / entry_price * 100 if entry_price > 0 else 0.0,
                )

                # Record in performance tracker
                time_in_trade = 0.0
                if executor._entry_time is not None:
                    time_in_trade = (time.time() - executor._entry_time) / 60.0
                perf_tracker.record_trade(
                    pnl_usdt=pnl,
                    pnl_pct=pnl_pct,
                    fees=0.0,
                    time_in_trade_min=time_in_trade,
                    bankroll_after=new_balance,
                )

                logger.info(
                    "Position Closed — %s | Signal: %s | Entry: %.2f | Exit: %.2f | PnL: %+.2f USDT (%+.2f%%) | Balance: %.2f USDT",
                    outcome, signal, entry_price, exit_px, pnl, pnl_pct, new_balance,
                )

                state.capital = new_balance
                if COMPOUNDING_ENABLED:
                    logger.info("Compounding: new bankroll = %.2f USDT", new_balance)
                    log_system("compounding_applied", new_bankroll=f"{new_balance:.2f}")

                # Equity snapshot on trade event
                dd_peak = 0.0
                if perf_tracker.peak_bankroll > 0:
                    dd_peak = (perf_tracker.peak_bankroll - new_balance) / perf_tracker.peak_bankroll * 100
                is_new_peak = new_balance > perf_tracker.peak_bankroll
                log_equity(
                    "trade_event_snapshot",
                    bankroll=f"{new_balance:.2f}",
                    open_position_value="0.00",
                    total_equity=f"{new_balance:.2f}",
                    unrealized_pnl="0.00",
                    margin_used="0.00",
                    available_margin=f"{new_balance:.2f}",
                    current_leverage="0",
                    distance_to_liquidation_pct="100.00",
                    peak_bankroll=f"{perf_tracker.peak_bankroll:.2f}",
                    drawdown_from_peak_pct=f"{dd_peak:.2f}",
                    trades_count_total=perf_tracker.trades_taken,
                    win_count=perf_tracker.wins,
                    loss_count=perf_tracker.losses,
                    realized_pnl=f"{pnl:+.4f}",
                    is_new_peak=f"{is_new_peak}",
                )

                log_system(
                    "position_closed",
                    signal=signal,
                    entry_price=f"{entry_price:.2f}",
                    exit_price=f"{exit_px:.2f}",
                    pnl=f"{pnl:+.4f}",
                    pnl_pct=f"{pnl_pct:+.4f}%",
                    exit_reason=exit_reason,
                    bankroll_after=f"{new_balance:.2f}",
                )

                # Check if target reached
                if new_balance >= TARGET_CAPITAL_USDT:
                    logger.info("🎯 TARGET CAPITAL REACHED: %.2f USDT", new_balance)
                    log_system_critical("TARGET_REACHED", bankroll=f"{new_balance:.2f}")
                    sys.exit(0)

                executor.reset()
                status_line.clear()
                return

            # Still in position — update drawdown and status line
            unrel = float(pos.get("unRealizedProfit", 0))
            mark_px = float(pos.get("markPrice", 0))

            # Update executor drawdown tracking
            executor.update_trade_drawdown(mark_px)

            # Status line (trade active: 10s interval)
            now = time.time()
            if now - status_line.last_print >= status_line.trade_interval:
                unrel_pct = (unrel / state.capital) * 100 if state.capital > 0 else 0
                time_in = (time.time() - executor._entry_time) / 60.0 if executor._entry_time else 0
                side_s = "LONG" if signal == "LONG" else "SHORT"

                # Recalc SL/TP
                if side_s == "LONG":
                    sl = entry_price * (1 - 0.00254)
                    tp = entry_price * (1 + 0.01232)
                else:
                    sl = entry_price * (1 + 0.00254)
                    tp = entry_price * (1 - 0.01232)

                status_line.render_with_position(
                    side=side_s,
                    entry=entry_price,
                    current=mark_px,
                    unrealized_pct=unrel_pct,
                    stop=sl,
                    target=tp,
                    time_in_trade_min=time_in,
                )
                status_line.last_print = now

            logger.debug("Position still open — unrealised PnL: %.2f USDT", unrel)
            time.sleep(poll_interval)

        except Exception:
            logger.exception("Error while waiting for position close")
            time.sleep(poll_interval)


# ── Helper: sleep until next 1H candle ──────────────────────────────────────

def _sleep_until_next_hour():
    from config.settings import TEST_MODE
    if TEST_MODE:
        logger.warning("TEST_MODE: sleeping 60s instead of waiting for next candle")
        time.sleep(60)
        return
    now = datetime.datetime.utcnow()
    next_candle = now.replace(minute=0, second=0, microsecond=0) + datetime.timedelta(hours=1)
    delay = (next_candle - now).total_seconds() + SLEEP_BEFORE_NEXT_CANDLE_BUFFER
    delay = max(1.0, delay)
    logger.info("Sleeping %.0f seconds until next 1H candle (%s) …", delay, next_candle.strftime("%H:%M:%S"))
    log_system_debug("sleeping_until_candle", delay_seconds=f"{delay:.0f}", next_candle=next_candle.strftime("%H:%M:%S"))
    time.sleep(delay)


def _seconds_until_next_hour() -> int:
    """Return seconds until the next 1H candle starts."""
    now = datetime.datetime.utcnow()
    next_candle = now.replace(minute=0, second=0, microsecond=0) + datetime.timedelta(hours=1)
    return int((next_candle - now).total_seconds()) + SLEEP_BEFORE_NEXT_CANDLE_BUFFER


# ── Graceful shutdown ──────────────────────────────────────────────────────

def _shutdown_gracefully(
    client: BinanceFuturesClient,
    executor: TradeExecutor,
    state: BotState,
    perf_tracker: PerformanceTracker,
    signal_gen: SignalGenerator,
):
    """Log graceful shutdown with final stats."""
    logger.info("KeyboardInterrupt received — shutting down gracefully")

    try:
        final_balance = client.get_balance("USDT")
    except Exception:
        final_balance = state.capital

    log_system(
        "BOT_SHUTDOWN",
        reason="user_interrupt",
        final_bankroll=f"{final_balance:.2f}",
        peak_bankroll=f"{perf_tracker.peak_bankroll:.2f}",
        trades_taken=perf_tracker.trades_taken,
        wins=perf_tracker.wins,
        losses=perf_tracker.losses,
        win_rate=f"{perf_tracker.wins / max(perf_tracker.trades_taken, 1) * 100:.2f}%",
        days_running=f"{(time.time() - perf_tracker.start_time) / 86400.0:.1f}",
    )

    logger.info("─" * 60)
    logger.info("SHUTDOWN — Final bankroll: %.2f USDT | Trades: %d | Wins: %d | Losses: %d",
                final_balance, perf_tracker.trades_taken, perf_tracker.wins, perf_tracker.losses)
    logger.info("─" * 60)

    # Cancel pending orders for safety
    try:
        executor.cancel_pending_orders()
    except Exception:
        pass

    # Print final dashboard
    perf_tracker.print_dashboard(final_balance)


# ── Helper for DEBUG-level system logging ───────────────────────────────────

def log_system_debug(msg: str, **kv):
    """Log a DEBUG-level message to the system logger."""
    parts = [f"{msg}"]
    for k, v in kv.items():
        parts.append(f"[{k}={v}]")
    system_logger.debug(" ".join(parts))


# ── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        main_loop()
    except Exception as e:
        import traceback
        traceback.print_exc()          # ← shows the real error
        logger.critical("Fatal error — bot exiting: %s", e)
        log_system_critical("FATAL_ERROR", action="bot_exiting")
        sys.exit(1)