"""
Risk Manager

Calculates position sizes based on the sacred parameters:
- 5.08% risk per trade
- 20x leverage on isolated margin
- Compounding: position sized on current bankroll
- Leverage-tier awareness
- Binance lot-size rounding
"""

import logging
import math
from decimal import Decimal, ROUND_DOWN

from config.settings import (
    RISK_PER_TRADE_PCT,
    LEVERAGE,
    STOP_LOSS_PRICE_PCT,
    TRADING_PAIR,
    MARGIN_BUFFER_PCT,
)
from utils.logger import log_system, log_system_warning

logger = logging.getLogger("risk_manager")
system_logger = logging.getLogger("system")


class RiskManager:
    def __init__(self, binance_client):
        self._client = binance_client

    def calculate_position_size(self, current_bankroll: float, entry_price: float) -> float:
        # ── 1. Risk amount ────────────────────────────────────────────────
        risk_amount = current_bankroll * (RISK_PER_TRADE_PCT / 100.0)
        logger.debug(
            "POSITION SIZING STEP 1: risk_amount=%.4f USDT (bankroll=%.2f × %.2f%%)",
            risk_amount, current_bankroll, RISK_PER_TRADE_PCT,
        )
        log_system(
            "position_sizing_step1",
            bankroll=f"{current_bankroll:.2f}",
            risk_pct=f"{RISK_PER_TRADE_PCT:.2f}%",
            risk_amount=f"{risk_amount:.4f}",
        )

        # ── 2. Position notional ──────────────────────────────────────────
        stop_loss_decimal = STOP_LOSS_PRICE_PCT / 100.0
        position_value = risk_amount / stop_loss_decimal
        logger.debug(
            "POSITION SIZING STEP 2: position_value=%.4f USDT (risk=%.4f / sl_pct=%.4f%%)",
            position_value, risk_amount, STOP_LOSS_PRICE_PCT,
        )
        log_system(
            "position_sizing_step2",
            risk_amount=f"{risk_amount:.4f}",
            stop_loss_pct=f"{STOP_LOSS_PRICE_PCT:.4f}%",
            position_value=f"{position_value:.4f}",
        )

        # ── 3. Base-asset quantity ────────────────────────────────────────
        quantity = position_value / entry_price
        logger.debug(
            "POSITION SIZING STEP 3: raw_quantity=%.8f ETH (position_value=%.4f / entry=%.2f)",
            quantity, position_value, entry_price,
        )
        log_system(
            "position_sizing_step3",
            position_value=f"{position_value:.4f}",
            entry_price=f"{entry_price:.2f}",
            raw_quantity=f"{quantity:.8f}",
        )

        # ── 4. Margin cap ─────────────────────────────────────────────────
        max_notional = current_bankroll * LEVERAGE * MARGIN_BUFFER_PCT
        max_qty = max_notional / entry_price
        if quantity > max_qty:
            logger.warning(
                "POSITION SIZING STEP 4: Clamping %.6f → %.6f (margin cap %.2f USDT)",
                quantity, max_qty, max_notional,
            )
            log_system_warning(
                "position_sizing_step4_clamp",
                original_quantity=f"{quantity:.6f}",
                clamped_quantity=f"{max_qty:.6f}",
                max_notional=f"{max_notional:.2f}",
                bankroll=f"{current_bankroll:.2f}",
                leverage=LEVERAGE,
            )
            quantity = max_qty
        else:
            logger.debug(
                "POSITION SIZING STEP 4: quantity=%.6f within margin cap (max=%.6f)",
                quantity, max_qty,
            )

        # ── 5. Leverage-tier clamp ────────────────────────────────────────
        quantity_before_tier = quantity
        quantity = self._clamp_to_tier(quantity, entry_price)
        if quantity != quantity_before_tier:
            logger.info(
                "POSITION SIZING STEP 5: tier clamp: %.6f → %.6f",
                quantity_before_tier, quantity,
            )

        # ── 6. Lot-size rounding ──────────────────────────────────────────
        quantity_before_round = quantity
        quantity = self._round_to_lot_step(quantity)
        if quantity != quantity_before_round:
            logger.debug(
                "POSITION SIZING STEP 6: lot rounding: %.8f → %.8f",
                quantity_before_round, quantity,
            )

        # ── 7. Min lot floor ──────────────────────────────────────────────
        quantity_before_floor = quantity
        quantity = self._floor_to_min_lot(quantity)
        if quantity != quantity_before_floor:
            logger.warning(
                "POSITION SIZING STEP 7: floored to min lot: %.8f → %.8f",
                quantity_before_floor, quantity,
            )

        final_notional = quantity * entry_price
        final_risk = final_notional * stop_loss_decimal
        final_risk_pct = (final_risk / current_bankroll) * 100 if current_bankroll > 0 else 0.0

        logger.info(
            "POSITION SIZING FINAL: quantity=%.6f ETH | notional=%.2f USDT | "
            "risk=%.4f USDT (%.2f%%) | entry=%.2f | leverage=%dx",
            quantity, final_notional, final_risk, final_risk_pct, entry_price, LEVERAGE,
        )
        log_system(
            "position_sizing_final",
            quantity=f"{quantity:.6f}",
            notional=f"{final_notional:.2f}",
            risk_amount=f"{final_risk:.4f}",
            risk_pct=f"{final_risk_pct:.2f}%",
            entry_price=f"{entry_price:.2f}",
            leverage=LEVERAGE,
            bankroll=f"{current_bankroll:.2f}",
        )

        return quantity

    def _clamp_to_tier(self, quantity: float, entry_price: float) -> float:
        try:
            brackets = self._client.get_leverage_brackets(TRADING_PAIR)
        except Exception:
            logger.warning("Could not fetch leverage brackets; skipping tier clamp")
            log_system_warning("tier_clamp_skipped", reason="fetch_failed")
            return quantity

        if isinstance(brackets, list) and brackets:
            brackets_data = brackets[0].get("brackets", [])
        else:
            return quantity

        position_notional = quantity * entry_price

        capped = False
        for tier in brackets_data:
            tier_num = tier.get("bracket", "?")
            max_leverage = tier.get("initialLeverage", "?")
            notional_cap = float(tier.get("notionalCap", float("inf")))
            if position_notional > notional_cap:
                clamped_notional = notional_cap
                new_qty = clamped_notional / entry_price
                logger.info(
                    "Tier clamp: notional %.2f > cap %.2f (tier=%s, maxLvg=%s) → qty %.6f",
                    position_notional, notional_cap, tier_num, max_leverage, new_qty,
                )
                log_system(
                    "tier_clamp_applied",
                    tier=tier_num,
                    max_leverage=max_leverage,
                    notional=f"{position_notional:.2f}",
                    notional_cap=f"{notional_cap:.2f}",
                    original_quantity=f"{quantity:.6f}",
                    clamped_quantity=f"{new_qty:.6f}",
                )
                quantity = new_qty
                position_notional = quantity * entry_price
                capped = True

        if not capped:
            logger.debug(
                "Position fits within all leverage tiers (notional=%.2f, qty=%.6f)",
                position_notional, quantity,
            )
            system_logger.debug(
                "tier_clamp_ok [notional=%.2f] [quantity=%.6f]",
                position_notional, quantity,
            )

        return quantity

    def _round_to_lot_step(self, quantity: float) -> float:
        filters = self._client.get_lot_size_filters(TRADING_PAIR)
        if filters is None:
            logger.warning("Could not fetch LOT_SIZE filters; returning raw quantity")
            log_system_warning("lot_size_fetch_failed")
            return quantity

        step_size = float(filters.get("stepSize", 0.001))
        min_qty = float(filters.get("minQty", 0))
        max_qty_f = float(filters.get("maxQty", float("inf")))

        if step_size <= 0:
            return quantity

        precision = self._precision_from_step(step_size)
        quantized = math.floor(quantity / step_size) * step_size
        quantized = round(quantized, precision)

        logger.debug(
            "Lot rounding: raw=%.8f step=%.6f min=%.6f max=%.6f → quantized=%.8f",
            quantity, step_size, min_qty, max_qty_f, quantized,
        )

        return quantized

    def _floor_to_min_lot(self, quantity: float) -> float:
        filters = self._client.get_lot_size_filters(TRADING_PAIR)
        if filters is None:
            return quantity

        min_qty = float(filters.get("minQty", 0))
        if quantity < min_qty:
            logger.warning(
                "Quantity %.8f < minQty %.8f → returning 0 (cannot trade)",
                quantity, min_qty,
            )
            log_system_warning(
                "quantity_below_min_lot",
                quantity=f"{quantity:.8f}",
                min_qty=f"{min_qty:.8f}",
            )
            return 0.0
        return quantity

    @staticmethod
    def _precision_from_step(step_size: float) -> int:
        s = f"{step_size:.10f}".rstrip("0").rstrip(".")
        if "." in s:
            return len(s.split(".")[1])
        return 0