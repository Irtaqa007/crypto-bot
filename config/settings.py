"""
Trading Bot Configuration — Sacred Parameters
DO NOT MODIFY any constant values unless explicitly instructed.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ─── TEST MODE ───────────────────────────────────────────────────────────────
# Set to True to force a trade signal every loop and sleep only 60s between checks.
# Switch back to False for real trading logic.
TEST_MODE = True

# ─── Sacred Trading Parameters ──────────────────────────────────────────────
STARTING_CAPITAL_USDT = 10.00
TARGET_CAPITAL_USDT = 1_000_000_000.00
RISK_PER_TRADE_PCT = 5.08
RISK_REWARD_RATIO = 4.85
WIN_RATE_ASSUMPTION = 50.0
LEVERAGE = 20
MAX_TRADES_PER_DAY = 5
MAX_SIMULTANEOUS_TRADES = 1
TRADING_PAIR = "ETHUSDT"
TIMEFRAME_BIAS = "4h"
TIMEFRAME_ENTRY = "1h"

MARGIN_BUFFER_PCT = 0.95  # use only 95% of available margin

STOP_LOSS_PRICE_PCT = 0.254
TAKE_PROFIT_PRICE_PCT = 1.232

COMPOUNDING_ENABLED = True
PARTIAL_PROFITS_ENABLED = False
TRAILING_STOPS_ENABLED = False

# ─── VIXFix Parameters ──────────────────────────────────────────────────────
VIXFIX_PD = 22
VIXFIX_BBL = 20
VIXFIX_MULT = 2.0
VIXFIX_LB = 50
VIXFIX_PH = 0.97

# ─── Volume RSI Parameters ──────────────────────────────────────────────────
VOLUME_RSI_PERIOD = 10
VOLUME_RSI_LONG_FILTER = 25
VOLUME_RSI_SHORT_FILTER = 75

# ─── RSI Parameters ─────────────────────────────────────────────────────────
RSI_PERIOD = 14
RSI_LONG_FILTER = 35
RSI_SHORT_FILTER = 65

# ─── HMA Parameters ─────────────────────────────────────────────────────────
HMA_PERIOD = 200

# ─── Binance API Configuration ──────────────────────────────────────────────
BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")
USE_TESTNET = os.getenv("USE_TESTNET", "true").lower() == "true"

FUTURES_LIVE_URL = "https://fapi.binance.com"
FUTURES_TESTNET_URL = "https://testnet.binancefuture.com"
FUTURES_BASE_URL = FUTURES_TESTNET_URL if USE_TESTNET else FUTURES_LIVE_URL

# ─── Logging Configuration ──────────────────────────────────────────────────
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-20s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_FILE = "crypto_bot.log"
LOG_LEVEL = "DEBUG"

LOG_CONFIG = {
    "LEVEL": "DEBUG",
    "FORMAT": LOG_FORMAT,
    "DATE_FORMAT": LOG_DATE_FORMAT,
    "DIR": "logs",
    "TRADES_LOG": "logs/trades.log",
    "SIGNALS_LOG": "logs/signals.log",
    "PERFORMANCE_LOG": "logs/performance.log",
    "SYSTEM_LOG": "logs/system.log",
    "EQUITY_LOG": "logs/equity.log",
    "MAX_BYTES": 10_485_760,
    "BACKUP_COUNT": 10,
}

# ─── Graceful Shutdown / Restart ────────────────────────────────────────────
EXCEPTION_COOLDOWN_SECONDS = 60
# In test mode sleep only 60s; real mode uses candle buffer
SLEEP_BEFORE_NEXT_CANDLE_BUFFER = 5