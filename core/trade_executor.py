"""
Trade Executor

Handles market entry, stop-loss, and take-profit order placement.
- Market entry via MARKET order
- Stop-loss via STOP_MARKET (direct signed request via place_stop_order)
- Take-profit via TAKE_PROFIT_MARKET (direct signed request via place_stop_order)
- Verifies fills before placing contingent orders
- Maximum 1 simultaneous position
"""

import logging
import time
from typing import Literal, Optional

from config.settings import (
    TRADING_PAIR,
    LEVERAGE,
    STOP_LOSS_PRICE_PCT,
    TAKE_PROFIT_PRICE_PCT,
    TARGET_CAPITAL_USDT,
)
from core.binance_client import BinanceFuturesClient
from core.risk_manager import RiskManager
from utils.logger import log_trade, log_trade_debug, log_system, log_system_error

logger = logging.getLogger("trade_executor")
trades_logger = logging.getLogger("trades")
system_logger = logging.getLogger("system")

Side = Literal["BUY", "SELL"]
Signal = Literal["LONG", "SHORT"]


class TradeExecutor:
    """Executes LONG and SHORT trades with fixed stop-loss and take-profit."""

    FILL_POLL_INTERVAL = 0.5
    FILL_POLL_TIMEOUT = 30

    def __init__(self, client: BinanceFuturesClient, risk_manager: RiskManager):
        self._client = client
        self._risk = risk_manager
        self._current_entry_price: Optional[float] = None
        self._current_quantity: Optional[float] = None
        self._current_side: Optional[Signal] = None
        self._current_signal_id: Optional[str] = None
        self._stop_order_id: Optional[int] = None
        self._tp_order_id: Optional[int] = None
        self._entry_time: Optional[float] = None
        self._max_drawdown_during_trade: float = 0.0
        self._peak_during_trade: float = 0.0
        self._trade_counter: int = 0

    # ── Public API ─────────────────────────────────────────────────────────

    def execute_long(self, bankroll: float) -> bool:
        self._trade_counter += 1
        logger.info("══════ EXECUTING LONG (trade #%d) ══════", self._trade_counter)
        log_trade("TRADE_START", direction="LONG", trade_number=self._trade_counter)
        return self._execute("LONG", "BUY", bankroll)

    def execute_short(self, bankroll: float) -> bool:
        self._trade_counter += 1
        logger.info("══════ EXECUTING SHORT (trade #%d) ══════", self._trade_counter)
        log_trade("TRADE_START", direction="SHORT", trade_number=self._trade_counter)
        return self._execute("SHORT", "SELL", bankroll)

    def has_open_position(self) -> bool:
        pos = self._client.get_position(TRADING_PAIR)
        return pos is not None

    def cancel_pending_orders(self):
        try:
            open_orders = self._client.get_open_orders(symbol=TRADING_PAIR)
            for order in open_orders:
                oid = order.get("orderId")
                self._client.cancel_order(symbol=TRADING_PAIR, order_id=oid)
                logger.info("Cancelled order %s", oid)
                log_system("pending_order_cancelled", order_id=oid)
        except Exception:
            logger.exception("Error cancelling pending orders")
            log_system_error("cancel_pending_orders_failed")

    def get_position_summary(self) -> Optional[dict]:
        pos = self._client.get_position(TRADING_PAIR)
        if pos is None:
            return None
        return {
            "symbol": pos.get("symbol"),
            "side": "LONG" if float(pos.get("positionAmt", 0)) > 0 else "SHORT",
            "size": abs(float(pos.get("positionAmt", 0))),
            "entry_price": float(pos.get("entryPrice", 0)),
            "unrealized_pnl": float(pos.get("unRealizedProfit", 0)),
            "liquidation_price": float(pos.get("liquidationPrice", 0)),
            "mark_price": float(pos.get("markPrice", 0)),
        }

    def reset(self):
        self._current_entry_price = None
        self._current_quantity = None
        self._current_side = None
        self._current_signal_id = None
        self._stop_order_id = None
        self._tp_order_id = None
        self._entry_time = None
        self._max_drawdown_during_trade = 0.0
        self._peak_during_trade = 0.0

    # ── Internal ───────────────────────────────────────────────────────────

    def _execute(self, signal: Signal, side: Side, bankroll: float) -> bool:
        signal_id = f"{signal}_{int(time.time())}_{self._trade_counter}"
        self._current_signal_id = signal_id
        self._entry_time = time.time()

        try:
            # ── 1. Fetch current price ────────────────────────────────────
            entry_price = self._client.get_price(TRADING_PAIR)
            logger.info("Current mark price: %.2f", entry_price)
            log_trade_debug("entry_price_fetched", signal_id=signal_id, current_price=f"{entry_price:.2f}")

            # ── 2. Position sizing ────────────────────────────────────────
            quantity = self._risk.calculate_position_size(bankroll, entry_price)
            if quantity <= 0:
                logger.error("Calculated quantity is <= 0; aborting trade")
                log_trade("TRADE_ABORTED", signal_id=signal_id, reason="quantity_zero_or_negative", bankroll=f"{bankroll:.2f}")
                return False

            # ── 3. Place MARKET entry ─────────────────────────────────────
            entry_order = self._client.place_order(
                symbol=TRADING_PAIR,
                side=side,
                type="MARKET",
                quantity=quantity,
            )
            entry_order_id = entry_order.get("orderId", "?")
            logger.info("Market %s order placed: %s", side, entry_order_id)
            log_trade(
                "ENTRY_ORDER_PLACED",
                signal_id=signal_id,
                direction=signal,
                order_id=entry_order_id,
                order_type="MARKET",
                quantity=f"{quantity:.6f}",
                side=side,
            )

            if entry_order_id == "?":
                logger.error("No orderId in market order response; aborting")
                log_trade("ENTRY_ORDER_FAILED", signal_id=signal_id, reason="no_order_id_in_response")
                return False

            # ── 4. Wait for fill ──────────────────────────────────────────
            filled_qty, avg_price = self._wait_for_fill(entry_order_id, quantity)

            # Fallback: testnet sometimes returns avgPrice=0
            if avg_price == 0 and filled_qty > 0:
                logger.warning("avg_price=0 from fill — using mark price %.2f as fallback", entry_price)
                avg_price = entry_price

            if filled_qty <= 0:
                logger.error("Market order was not filled; aborting")
                log_trade("ENTRY_ORDER_NOT_FILLED", signal_id=signal_id, order_id=entry_order_id)
                self.cancel_pending_orders()
                return False

            self._current_entry_price = avg_price
            self._current_quantity = filled_qty
            self._current_side = signal
            self._peak_during_trade = avg_price

            # ── 5. Compute SL / TP prices ─────────────────────────────────
            stop_price, tp_price = self._calc_stop_tp(avg_price, signal)

            if signal == "LONG":
                risk_amount = abs(avg_price - stop_price) * filled_qty
                reward_amount = abs(tp_price - avg_price) * filled_qty
            else:
                risk_amount = abs(stop_price - avg_price) * filled_qty
                reward_amount = abs(avg_price - tp_price) * filled_qty

            risk_pct = (risk_amount / bankroll) * 100 if bankroll > 0 else 0
            reward_pct = (reward_amount / bankroll) * 100 if bankroll > 0 else 0
            actual_rr = reward_amount / risk_amount if risk_amount > 0 else 0

            log_trade(
                "TRADE_ENTRY",
                signal_id=signal_id,
                direction=signal,
                entry_price=f"{avg_price:.2f}",
                bankroll_before=f"{bankroll:.2f}",
                position_size_usdt=f"{filled_qty * avg_price:.2f}",
                leverage_used=LEVERAGE,
                stop_price=f"{stop_price:.2f}",
                target_price=f"{tp_price:.2f}",
                risk_amount=f"{risk_amount:.4f}",
                reward_amount=f"{reward_amount:.4f}",
                expected_risk_pct=f"{risk_pct:.2f}%",
                expected_reward_pct=f"{reward_pct:.2f}%",
                actual_rr_ratio=f"{actual_rr:.4f}",
                quantity=f"{filled_qty:.6f}",
            )

            logger.info(
                "Entry filled: %.6f @ %.2f | Stop: %.2f | Target: %.2f | R:R = %.2f",
                filled_qty, avg_price, stop_price, tp_price, actual_rr,
            )

            # ── 6. Place SL order via direct signed request ───────────────
            close_side = "SELL" if side == "BUY" else "BUY"

            sl_order = self._client.place_stop_order(
                symbol=TRADING_PAIR,
                side=close_side,
                type="STOP_MARKET",
                quantity=filled_qty,
                stopPrice=round(stop_price, 2),
                reduceOnly="true",
                workingType="MARK_PRICE",
            )
            self._stop_order_id = sl_order.get("orderId")
            logger.info("Stop-loss order placed: %s at %.2f", self._stop_order_id, stop_price)
            log_trade("STOP_LOSS_PLACED", signal_id=signal_id, order_id=self._stop_order_id, stop_price=f"{stop_price:.2f}")

            # ── 7. Place TP order via direct signed request ───────────────
            tp_order = self._client.place_stop_order(
                symbol=TRADING_PAIR,
                side=close_side,
                type="TAKE_PROFIT_MARKET",
                quantity=filled_qty,
                stopPrice=round(tp_price, 2),
                reduceOnly="true",
                workingType="MARK_PRICE",
            )
            self._tp_order_id = tp_order.get("orderId")
            logger.info("Take-profit order placed: %s at %.2f", self._tp_order_id, tp_price)
            log_trade("TAKE_PROFIT_PLACED", signal_id=signal_id, order_id=self._tp_order_id, target_price=f"{tp_price:.2f}")

            log_system(
                "trade_execution_complete",
                signal_id=signal_id,
                direction=signal,
                entry_price=f"{avg_price:.2f}",
                quantity=f"{filled_qty:.6f}",
                stop_price=f"{stop_price:.2f}",
                target_price=f"{tp_price:.2f}",
                rr_ratio=f"{actual_rr:.4f}",
            )

            return True

        except Exception:
            logger.exception("Trade execution failed")
            log_trade("TRADE_EXECUTION_FAILED", signal_id=self._current_signal_id or "?", direction=signal)
            log_system_error("trade_execution_exception", signal_id=self._current_signal_id or "?", direction=signal)
            self.cancel_pending_orders()
            return False

    def _wait_for_fill(self, order_id: int, expected_qty: float) -> tuple[float, float]:
        """Poll order status until filled or timeout. Returns (filled_qty, avg_price)."""
        deadline = time.time() + self.FILL_POLL_TIMEOUT
        while time.time() < deadline:
            try:
                order = self._client.get_order(symbol=TRADING_PAIR, order_id=order_id)
                status = order.get("status", "")
                executed_qty = float(order.get("executedQty", 0))

                if status == "FILLED":
                    avg_price = float(order.get("avgPrice", 0))
                    if avg_price == 0:
                        cum_quote = float(order.get("cumQuote", 0) or order.get("cummQuoteQty", 0))
                        avg_price = cum_quote / executed_qty if executed_qty > 0 else 0.0
                    trades_logger.debug(
                        "ORDER_FILLED [order_id=%s] [executed_qty=%.6f] [avg_price=%.2f]",
                        order_id, executed_qty, avg_price,
                    )
                    return (executed_qty, avg_price)

                if status in ("CANCELED", "EXPIRED", "REJECTED"):
                    logger.warning("Entry order %s status: %s", order_id, status)
                    trades_logger.warning(
                        "ORDER_NOT_FILLED [order_id=%s] [status=%s] [executed_qty=%.6f]",
                        order_id, status, executed_qty,
                    )
                    return (executed_qty, 0.0)

                logger.debug("Waiting for fill… status=%s exec=%.6f", status, executed_qty)
            except Exception:
                logger.exception("Error polling order fill")

            time.sleep(self.FILL_POLL_INTERVAL)

        logger.error("Timeout waiting for order %s to fill", order_id)
        log_trade("ORDER_FILL_TIMEOUT", order_id=order_id, timeout_seconds=self.FILL_POLL_TIMEOUT)
        return (0.0, 0.0)

    def _calc_stop_tp(self, entry_price: float, signal: Signal) -> tuple[float, float]:
        sl_mult = STOP_LOSS_PRICE_PCT / 100.0
        tp_mult = TAKE_PROFIT_PRICE_PCT / 100.0
        if signal == "LONG":
            return entry_price * (1.0 - sl_mult), entry_price * (1.0 + tp_mult)
        else:
            return entry_price * (1.0 + sl_mult), entry_price * (1.0 - tp_mult)

    # ── Position monitoring ─────────────────────────────────────────────────

    def update_trade_drawdown(self, current_price: float):
        if self._current_entry_price is None or self._current_side is None:
            return
        if self._current_side == "LONG":
            self._peak_during_trade = max(self._peak_during_trade, current_price)
            dd_pct = (self._peak_during_trade - current_price) / self._peak_during_trade * 100
        else:
            self._peak_during_trade = min(self._peak_during_trade, current_price) if self._peak_during_trade > 0 else current_price
            dd_pct = (current_price - self._peak_during_trade) / current_price * 100 if current_price > 0 else 0
        self._max_drawdown_during_trade = max(self._max_drawdown_during_trade, dd_pct)

    def log_trade_exit(
        self,
        exit_price: float,
        exit_reason: str,
        bankroll_before: float,
        bankroll_after: float,
        fees_paid: float = 0.0,
        slippage: float = 0.0,
    ):
        signal_id = self._current_signal_id or "?"
        entry_price = self._current_entry_price or 0.0
        side = self._current_side or "?"
        quantity = self._current_quantity or 0.0
        time_in_trade = (time.time() - self._entry_time) / 60.0 if self._entry_time else 0.0

        realized_pnl_usdt = bankroll_after - bankroll_before
        realized_pnl_pct = (realized_pnl_usdt / bankroll_before) * 100 if bankroll_before > 0 else 0.0
        distance_to_target_pct = (bankroll_after / TARGET_CAPITAL_USDT) * 100 if TARGET_CAPITAL_USDT > 0 else 0.0

        log_trade(
            "TRADE_EXIT",
            signal_id=signal_id,
            exit_price=f"{exit_price:.2f}",
            exit_reason=exit_reason,
            bankroll_after=f"{bankroll_after:.2f}",
            realized_pnl_usdt=f"{realized_pnl_usdt:+.4f}",
            realized_pnl_pct=f"{realized_pnl_pct:+.4f}%",
            max_drawdown_during_trade=f"{self._max_drawdown_during_trade:.2f}%",
            time_in_trade_minutes=f"{time_in_trade:.1f}",
            slippage_from_expected=f"{slippage:.4f}",
            fees_paid=f"{fees_paid:.4f}",
            new_bankroll=f"{bankroll_after:.2f}",
            distance_to_target_1b_pct=f"{distance_to_target_pct:.6f}%",
            entry_price=f"{entry_price:.2f}",
            direction=side,
            quantity=f"{quantity:.6f}",
        )

        outcome = "WIN" if realized_pnl_usdt > 0 else "LOSS"
        color = "\033[92m" if realized_pnl_usdt > 0 else "\033[91m"
        logger.info(
            "%sTRADE EXIT (%s) — %s — PnL: %+.4f USDT (%+.4f%%) — Time: %.1f min\033[0m",
            color, exit_reason, outcome, realized_pnl_usdt, realized_pnl_pct, time_in_trade,
        )

        log_system(
            "trade_exit",
            signal_id=signal_id,
            exit_reason=exit_reason,
            pnl=f"{realized_pnl_usdt:+.4f}",
            pnl_pct=f"{realized_pnl_pct:+.4f}%",
            time_in_trade_min=f"{time_in_trade:.1f}",
        )

        self.reset()