"""
Binance Futures API Client

Thread-safe wrapper around python-binance with:
- Exponential-backoff retry on transient API errors
- Rate-limit awareness
- Testnet / live toggle
- Deep logging of every API call (timing, params masked, success/failure)
"""

import time
import logging
import hashlib
import hmac
import urllib.parse
from typing import Any, Optional

import requests
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceRequestException

from config.settings import (
    BINANCE_API_KEY,
    BINANCE_API_SECRET,
    USE_TESTNET,
    FUTURES_LIVE_URL,
    FUTURES_TESTNET_URL,
)
from utils.logger import (
    log_system,
    log_system_warning,
    log_system_error,
)

logger = logging.getLogger("binance_client")
system_logger = logging.getLogger("system")

MAX_RETRIES = 5
BASE_DELAY_SECONDS = 1.0
BACKOFF_MULTIPLIER = 2.0
SENSITIVE_PARAMS = {"apiKey", "signature", "timestamp", "api_key", "secret"}


class BinanceFuturesClient:
    """Thin, resilient wrapper around the Binance Futures API."""

    def __init__(self):
        self._base_url = FUTURES_TESTNET_URL if USE_TESTNET else FUTURES_LIVE_URL

        # Use testnet=True so python-binance sets the correct futures base URL
        # and uses the testnet signing endpoint. We override the spot ping URL
        # after construction to avoid DNS failure on testnet.binance.vision.
        self._client = Client(
            BINANCE_API_KEY,
            BINANCE_API_SECRET,
            testnet=USE_TESTNET,
        )

        # Override the futures base URL explicitly
        self._client.futures_base_url = self._base_url

        # Patch the spot API URL to point to futures testnet so that any
        # accidental spot calls don't hit the unreachable testnet.binance.vision
        if USE_TESTNET:
            self._client.API_URL = self._base_url

        self._symbol_info_cache = None
        self._leverage_bracket_cache = None

        log_system(
            "BinanceFuturesClient_initialised",
            testnet=f"{USE_TESTNET}",
            url=self._base_url,
        )

    # ── Signed request helper (bypasses python-binance routing) ────────────

    def _signed_request(self, method: str, path: str, params: dict) -> dict:
        """
        Make a direct HMAC-signed REST request to the futures base URL.
        Used for STOP_MARKET / TAKE_PROFIT_MARKET orders that python-binance
        routes to the wrong internal handler (causing -4120 on testnet).
        """
        params = dict(params)  # copy so we don't mutate caller's dict
        params["timestamp"] = int(time.time() * 1000)
        query = urllib.parse.urlencode(params)
        signature = hmac.new(
            BINANCE_API_SECRET.encode("utf-8"),
            query.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        params["signature"] = signature

        url = f"{self._base_url}/fapi/v1/{path}"
        headers = {"X-MBX-APIKEY": BINANCE_API_KEY}

        resp = requests.request(method, url, headers=headers, params=params, timeout=10)
        data = resp.json()

        if isinstance(data, dict) and data.get("code", 0) < 0:
            class _FakeResp:
                status_code = resp.status_code
                text = resp.text
            exc = BinanceAPIException(_FakeResp(), resp.status_code, resp.text)
            exc.code = data.get("code", -1)
            exc.message = data.get("msg", "Unknown error")
            raise exc

        return data

    # ── Retry helper ────────────────────────────────────────────────────────

    def _retry(self, fn, *args, endpoint: str = "unknown", **kwargs):
        masked_kwargs = self._mask_sensitive(kwargs)
        last_exc: Optional[Exception] = None

        for attempt in range(1, MAX_RETRIES + 1):
            start_ts = time.perf_counter()
            try:
                result = fn(*args, **kwargs)
                elapsed_ms = (time.perf_counter() - start_ts) * 1000.0
                system_logger.debug(
                    "API_CALL_OK [endpoint=%s] [params=%s] [response_time_ms=%.1f] [retry_count=%d]",
                    endpoint, str(masked_kwargs)[:200], elapsed_ms, attempt - 1,
                )
                return result

            except BinanceAPIException as exc:
                elapsed_ms = (time.perf_counter() - start_ts) * 1000.0

                if exc.code in (429,) or (exc.status_code and 500 <= exc.status_code < 600):
                    logger.warning("Retryable API error (attempt %d/%d): code=%s msg=%s", attempt, MAX_RETRIES, exc.code, exc.message)
                    log_system_warning("API_CALL_RETRYABLE", endpoint=endpoint, response_time_ms=f"{elapsed_ms:.1f}", error_code=exc.code, error_message=exc.message[:200], retry_attempt=attempt)
                    last_exc = exc
                    if attempt < MAX_RETRIES:
                        delay = BASE_DELAY_SECONDS * (BACKOFF_MULTIPLIER ** (attempt - 1))
                        logger.info("Retrying in %.1f seconds …", delay)
                        log_system("API_RETRY_BACKOFF", endpoint=endpoint, retry_attempt=attempt + 1, backoff_seconds=f"{delay:.1f}")
                        time.sleep(delay)
                else:
                    # -4046 = margin type already set; caller handles it silently
                    if exc.code != -4046:
                        logger.error("Non-retryable API error (attempt %d/%d): code=%s msg=%s", attempt, MAX_RETRIES, exc.code, exc.message)
                        log_system_error("API_CALL_FAILED", endpoint=endpoint, response_time_ms=f"{elapsed_ms:.1f}", error_code=exc.code, error_message=exc.message[:200], retry_count=attempt - 1)
                    raise

            except BinanceRequestException as exc:
                elapsed_ms = (time.perf_counter() - start_ts) * 1000.0
                logger.warning("Network error (attempt %d/%d): %s", attempt, MAX_RETRIES, exc)
                log_system_warning("API_CALL_NETWORK_ERROR", endpoint=endpoint, response_time_ms=f"{elapsed_ms:.1f}", error_message=str(exc)[:200], retry_attempt=attempt)
                last_exc = exc
                if attempt < MAX_RETRIES:
                    delay = BASE_DELAY_SECONDS * (BACKOFF_MULTIPLIER ** (attempt - 1))
                    logger.info("Retrying in %.1f seconds …", delay)
                    log_system("API_RETRY_BACKOFF", endpoint=endpoint, retry_attempt=attempt + 1, backoff_seconds=f"{delay:.1f}")
                    time.sleep(delay)

            except Exception as exc:
                elapsed_ms = (time.perf_counter() - start_ts) * 1000.0
                logger.error("Unexpected error (attempt %d/%d): %s", attempt, MAX_RETRIES, exc)
                log_system_error("API_CALL_UNEXPECTED_ERROR", endpoint=endpoint, response_time_ms=f"{elapsed_ms:.1f}", error_message=str(exc)[:200], retry_attempt=attempt)
                last_exc = exc
                if attempt == MAX_RETRIES:
                    raise
                delay = BASE_DELAY_SECONDS * (BACKOFF_MULTIPLIER ** (attempt - 1))
                log_system("API_RETRY_BACKOFF", endpoint=endpoint, retry_attempt=attempt + 1, backoff_seconds=f"{delay:.1f}")
                time.sleep(delay)

        raise last_exc  # type: ignore[misc]

    @staticmethod
    def _mask_sensitive(params: dict) -> dict:
        return {k: "***MASKED***" if k in SENSITIVE_PARAMS else v for k, v in params.items()}

    # ── Public API ──────────────────────────────────────────────────────────

    def get_balance(self, asset: str = "USDT") -> float:
        balances = self._retry(self._client.futures_account_balance, endpoint="futures_account_balance")
        for entry in balances:
            if entry["asset"] == asset:
                bal = float(entry["balance"])
                system_logger.debug("balance_check [asset=%s] [balance=%.2f]", asset, bal)
                return bal
        logger.warning("Asset %s not found in futures account balances; returning 0.0", asset)
        log_system_warning("balance_check_missing", asset=asset)
        return 0.0

    def get_open_orders(self, symbol: str = "ETHUSDT") -> list[dict]:
        return self._retry(self._client.futures_get_open_orders, symbol=symbol, endpoint="futures_get_open_orders")

    def get_balance_all(self) -> list[dict]:
        return self._retry(self._client.futures_account_balance, endpoint="futures_account_balance")

    def get_klines(self, symbol: str, interval: str, limit: int = 200) -> list[list]:
        system_logger.debug("fetching_klines [symbol=%s] [interval=%s] [limit=%d]", symbol, interval, limit)
        return self._retry(self._client.futures_klines, symbol=symbol, interval=interval, limit=limit, endpoint="futures_klines")

    def place_order(self, **params) -> dict:
        """Place a standard futures order (MARKET, LIMIT)."""
        masked = self._mask_sensitive(params)
        logger.info("Placing order: %s", masked)
        log_system(
            "placing_order",
            symbol=params.get("symbol", "?"),
            side=params.get("side", "?"),
            type=params.get("type", "?"),
            quantity=params.get("quantity", "?"),
        )
        return self._retry(self._client.futures_create_order, **params, endpoint="futures_create_order")

    def place_stop_order(self, **params) -> dict:
        """
        Place a STOP_MARKET or TAKE_PROFIT_MARKET order via direct signed request.
        python-binance routes these incorrectly causing -4120 on testnet.
        """
        masked = self._mask_sensitive(params)
        logger.info("Placing stop/tp order: %s", masked)
        log_system(
            "placing_stop_order",
            symbol=params.get("symbol", "?"),
            side=params.get("side", "?"),
            type=params.get("type", "?"),
            stopPrice=params.get("stopPrice", "?"),
        )

        last_exc: Optional[Exception] = None
        for attempt in range(1, MAX_RETRIES + 1):
            start_ts = time.perf_counter()
            try:
                result = self._signed_request("POST", "order", params)
                elapsed_ms = (time.perf_counter() - start_ts) * 1000.0
                system_logger.debug(
                    "API_CALL_OK [endpoint=futures_stop_order] [response_time_ms=%.1f] [retry_count=%d]",
                    elapsed_ms, attempt - 1,
                )
                return result
            except BinanceAPIException as exc:
                elapsed_ms = (time.perf_counter() - start_ts) * 1000.0
                if exc.code in (429,) or (exc.status_code and 500 <= exc.status_code < 600):
                    logger.warning("Retryable stop order error (attempt %d/%d): %s", attempt, MAX_RETRIES, exc.message)
                    last_exc = exc
                    if attempt < MAX_RETRIES:
                        time.sleep(BASE_DELAY_SECONDS * (BACKOFF_MULTIPLIER ** (attempt - 1)))
                else:
                    logger.error("Stop order failed: code=%s msg=%s", exc.code, exc.message)
                    log_system_error("STOP_ORDER_FAILED", error_code=exc.code, error_message=exc.message[:200])
                    raise
            except Exception as exc:
                logger.error("Stop order unexpected error (attempt %d/%d): %s", attempt, MAX_RETRIES, exc)
                last_exc = exc
                if attempt == MAX_RETRIES:
                    raise
                time.sleep(BASE_DELAY_SECONDS * (BACKOFF_MULTIPLIER ** (attempt - 1)))

        raise last_exc

    def cancel_order(self, symbol: str, order_id: Optional[int] = None, orig_client_order_id: Optional[str] = None) -> dict:
        logger.info("Cancelling order %s on %s", order_id or orig_client_order_id, symbol)
        log_system("cancelling_order", symbol=symbol, order_id=order_id)
        return self._retry(self._client.futures_cancel_order, symbol=symbol, orderId=order_id, origClientOrderId=orig_client_order_id, endpoint="futures_cancel_order")

    def get_order(self, symbol: str, order_id: Optional[int] = None, orig_client_order_id: Optional[str] = None) -> dict:
        return self._retry(self._client.futures_get_order, symbol=symbol, orderId=order_id, origClientOrderId=orig_client_order_id, endpoint="futures_get_order")

    def get_position(self, symbol: str = "ETHUSDT") -> dict | None:
        positions = self._retry(self._client.futures_position_information, symbol=symbol, endpoint="futures_position_information")
        for pos in positions:
            if pos["symbol"] == symbol:
                amt = float(pos.get("positionAmt", 0))
                if abs(amt) > 1e-12:
                    system_logger.debug(
                        "position_found [symbol=%s] [side=%s] [size=%.6f] [entry_price=%s] "
                        "[unrealized_pnl=%s] [liquidation_price=%s] [margin_ratio=%s]",
                        symbol, "LONG" if amt > 0 else "SHORT", amt,
                        pos.get("entryPrice", 0), pos.get("unRealizedProfit", 0),
                        pos.get("liquidationPrice", 0), pos.get("marginRatio", 0),
                    )
                    return pos
                break
        system_logger.debug("no_open_position [symbol=%s]", symbol)
        return None

    def get_all_positions(self) -> list[dict]:
        return self._retry(self._client.futures_position_information, endpoint="futures_position_information")

    def set_leverage(self, symbol: str, leverage: int):
        result = self._retry(self._client.futures_change_leverage, symbol=symbol, leverage=leverage, endpoint="futures_change_leverage")
        logger.info("Leverage set to %dx for %s (result: %s)", leverage, symbol, result)

    def set_margin_type(self, symbol: str, margin_type: str = "ISOLATED"):
        try:
            result = self._retry(self._client.futures_change_margin_type, symbol=symbol, marginType=margin_type, endpoint="futures_change_margin_type")
            logger.info("Margin type set to %s for %s (result: %s)", margin_type, symbol, result)
            log_system("margin_type_set", symbol=symbol, margin_type=margin_type)
        except BinanceAPIException as exc:
            if exc.code == -4046:
                logger.info("Margin type already %s for %s", margin_type, symbol)
                log_system("margin_type_already_set", symbol=symbol, margin_type=margin_type)
            else:
                log_system_error("margin_type_failed", symbol=symbol, margin_type=margin_type, error_code=exc.code, error_message=exc.message[:200])
                raise

    def get_leverage_brackets(self, symbol: str = "ETHUSDT") -> list[dict]:
        return self._retry(self._client.futures_leverage_bracket, symbol=symbol, endpoint="futures_leverage_bracket")

    def get_exchange_info(self) -> dict:
        return self._retry(self._client.futures_exchange_info, endpoint="futures_exchange_info")

    def get_symbol_info(self, symbol: str = "ETHUSDT") -> dict | None:
        info = self.get_exchange_info()
        for s in info.get("symbols", []):
            if s["symbol"] == symbol:
                return s
        return None

    def get_lot_size_filters(self, symbol: str = "ETHUSDT") -> dict | None:
        sym_info = self.get_symbol_info(symbol)
        if sym_info is None:
            return None
        for f in sym_info.get("filters", []):
            if f["filterType"] == "LOT_SIZE":
                return f
        return None

    def get_price(self, symbol: str = "ETHUSDT") -> float:
        ticker = self._retry(self._client.futures_mark_price, symbol=symbol, endpoint="futures_mark_price")
        price = float(ticker.get("markPrice", 0))
        system_logger.debug("mark_price [symbol=%s] [price=%.2f]", symbol, price)
        return price

    def get_account_trades(self, symbol: str = "ETHUSDT", limit: int = 10):
        return self._retry(self._client.futures_account_trades, symbol=symbol, limit=limit, endpoint="futures_account_trades")