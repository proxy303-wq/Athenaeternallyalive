"""
ATHENA-X Configuration
Single source of truth for trading, risk, indicators, instruments,
ML, execution, and external-service settings.

IMPORTANT:
- Keep LIVE_TRADING = False until paper testing is complete.
- Credentials must come from environment variables; never hard-code secrets.
"""

import os
from dotenv import load_dotenv

load_dotenv()

from datetime import time as dt_time


# ============================================================
# 1. TRADING MODE & CAPITAL
# ============================================================

LIVE_TRADING = False

# Used only when Dhan balance cannot be read.
FALLBACK_CAPITAL = 500000.0


# ============================================================
# 2. RISK MANAGEMENT
# ============================================================

TARGET_PCT = 0.0100                 # 1.0% option-price target
RISK_PCT = 0.0050                   # Legacy capital-risk alias
OPTION_STOP_PCT = 0.0050            # 0.5% option-price stop
MAX_RISK_PER_TRADE_PCT = 0.0050     # Hard account-risk ceiling
EV_MIN_PER_RISK = 0.10              # Minimum EV measured in units of risk
KELLY_FRACTION = 0.25               # Fractional Kelly; never full Kelly
MAX_KELLY_RISK_PCT = 0.0050         # Kelly cannot exceed hard risk cap
TRANSACTION_COST_PCT = 0.0005       # Conservative round-trip cost allowance
SLIPPAGE_PCT = 0.0005               # Conservative entry/exit slippage allowance
MAX_DRAWDOWN_RISK_REDUCTION = 0.75  # Maximum risk reduction at drawdown limit
DRAWDOWN_FULL_RISK_PCT = 0.02       # Start reducing risk after 2% drawdown
DRAWDOWN_STOP_PCT = 0.08             # Stop new entries at 8% drawdown
HIGH_VOL_ATR_PCT = 1.50
LOW_VOL_ATR_PCT = 0.35
TREND_ADX_THRESHOLD = 25.0
RANGE_ADX_THRESHOLD = 18.0
TREND_EMA_SPREAD_PCT = 0.15
DAILY_LOSS_PCT = 0.0100             # 1.0% max daily loss
DAILY_TARGET_PCT = 0.0100            # 1.0% daily portfolio profit objective
MONTHLY_LOSS_PCT = 0.0500           # 5.0% max monthly loss

MAX_TRADES_PER_DAY = 3
MAX_POSITIONS = 1

# Option liquidity / execution-quality gates.
# These are deliberately conservative for PAPER validation.
MIN_OPTION_VOLUME = 100
MIN_OPTION_OI = 1000
MAX_OPTION_SPREAD_PCT = 0.02

# Market-data sanity checks.
MAX_OPTION_LTP_STALENESS_SECONDS = 15

MIN_WIN_RATE_REQUIRED = 60


# ============================================================
# 3. SIGNAL / QUALITY FILTERS
# ============================================================

MIN_SIGNAL_SCORE = 55
MIN_CONFIDENCE = 0.45
MIN_QUALITY_SCORE = 50
MIN_RISK_REWARD = 2.0

# Volatility expressed as ATR / underlying price * 100.
MIN_VOLATILITY = 0.3
MAX_VOLATILITY = 2.5

# Relative volume threshold.
MIN_VOLUME_RATIO = 0.8

# Number of completed bars used for directional confirmation.
CONFIRMATION_BARS = 1


# ============================================================
# 4. TIME FILTERS
# ============================================================

OPTIMAL_TRADE_TIMES = [
    (9, 30), (9, 45),
    (10, 0), (10, 15), (10, 30), (10, 45),
    (11, 0), (11, 15), (11, 30), (11, 45),
    (12, 0), (12, 15), (12, 30), (12, 45),
    (13, 0), (13, 15), (13, 30), (13, 45),
    (14, 0), (14, 15), (14, 30),
]

NO_TRADE_BEFORE = dt_time(9, 25)
NO_NEW_ENTRY_AFTER = dt_time(14, 45)
FORCE_EXIT_TIME = dt_time(15, 15)

# Main loop interval.
LOOP_SECONDS = 60


# ============================================================
# 5. TRAILING STOP
# ============================================================

TRAILING_STOP_ENABLED = True

# Activation and distance are based on the traded OPTION price.
TRAILING_STOP_ACTIVATION = 0.005    # Activate after 0.5% favorable move
TRAILING_STOP_DISTANCE = 0.003      # Trail by 0.3%
TRAILING_STOP_MIN_PROFIT = 0.002    # Preserve at least 0.2% profit


# ============================================================
# 6. TECHNICAL INDICATORS
# ============================================================

USE_ICHIMOKU = True
USE_SMA_CROSSOVER = True
TRACK_SHARPE_RATIO = True
RISK_FREE_RATE = 0.07

# Ichimoku
ICHIMOKU_CONVERSION = 9
ICHIMOKU_BASE = 26
ICHIMOKU_LAGGING = 52

# SMA
SMA_FAST = 20
SMA_SLOW = 50

# EMA
EMA_FAST = 20
EMA_SLOW = 50
EMA_LONG = 200

RSI_PERIOD = 14
ATR_PERIOD = 14


# ============================================================
# 7. OPTION FILTERS
# ============================================================

MIN_DELTA = 0.35
MAX_DELTA = 0.70

# Historical underlying data used for indicator calculations.
HISTORY_DAYS = 5


# ============================================================
# 8. INSTRUMENT DEFINITIONS
# ============================================================
#
# IMPORTANT:
# Each instrument carries its own exchange segment.
# Do NOT use NIFTY_EXCHANGE_SEGMENT for BANKNIFTY data.
#

NIFTY_SECURITY_ID = "13"
NIFTY_EXCHANGE_SEGMENT = "IDX_I"
NIFTY_LOT_SIZE = 65

BANKNIFTY_SECURITY_ID = "25"
BANKNIFTY_EXCHANGE_SEGMENT = "IDX_I"
BANKNIFTY_LOT_SIZE = 15

FINNIFTY_SECURITY_ID = "26034"
FINNIFTY_EXCHANGE_SEGMENT = "IDX_I"
FINNIFTY_LOT_SIZE = 40

INSTRUMENTS = {
    "NIFTY": {
        "security_id": NIFTY_SECURITY_ID,
        "exchange_segment": NIFTY_EXCHANGE_SEGMENT,
        "lot_size": NIFTY_LOT_SIZE,
        "enabled": True,
        "name": "NIFTY",
    },
    "BANKNIFTY": {
        "security_id": BANKNIFTY_SECURITY_ID,
        "exchange_segment": BANKNIFTY_EXCHANGE_SEGMENT,
        "lot_size": BANKNIFTY_LOT_SIZE,
        "enabled": True,
        "name": "BANKNIFTY",
    },
    "FINNIFTY": {
        "security_id": FINNIFTY_SECURITY_ID,
        "exchange_segment": FINNIFTY_EXCHANGE_SEGMENT,
        "lot_size": FINNIFTY_LOT_SIZE,
        "enabled": False,
        "name": "FINNIFTY",
    },
}


# ============================================================
# 9. XGBOOST ML
# ============================================================

ML_ENABLED = True
ML_MODEL_TYPE = "XGBOOST"

# Minimum historical closed trades before Athena trains its first model.
ML_MIN_TRADES_TO_TRAIN = 10

# Once trained, retrain after this many additional closed trades.
ML_RETRAIN_INTERVAL = 10

# Probability threshold used by the ML engine.
ML_YES_THRESHOLD = 0.65
ML_MAYBE_THRESHOLD = 0.50

# XGBoost parameters.
XGB_N_ESTIMATORS = 100
XGB_MAX_DEPTH = 6
XGB_LEARNING_RATE = 0.1
XGB_SUBSAMPLE = 0.8
XGB_COLSAMPLE_BYTREE = 0.8
XGB_RANDOM_STATE = 42

# Required ML features. These must be populated with real entry-time values.
ML_FEATURE_NAMES = [
    "rsi_at_entry",
    "ema_cross",
    "vwap_ratio",
    "atr_percent",
    "volume_ratio",
    "delta",
    "oi_at_entry",
    "market_score",
    "distance_from_high",
    "distance_from_low",
]


# ============================================================
# 10. DEEPSEEK AI — SECOND OPINION ONLY
# ============================================================
#
# DeepSeek must NOT directly place orders.
# It will be added after the quantitative + XGBoost layers are stable.
#

DEEPSEEK_ENABLED = False
DEEPSEEK_MODEL = os.getenv("DEEPSEEK_MODEL", "deepseek-chat")
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

# Maximum time allowed for an AI request before Athena skips the AI layer.
DEEPSEEK_TIMEOUT_SECONDS = 10

# AI must not override hard risk controls.
DEEPSEEK_REQUIRED_AGREEMENT = True


# ============================================================
# 11. DHAN CREDENTIALS
# ============================================================
#
# Credentials are loaded from environment variables.
# Example:
#   DHAN_CLIENT_ID=...
#   DHAN_ACCESS_TOKEN=...
#

CLIENT_ID = os.getenv("DHAN_CLIENT_ID", "")
ACCESS_TOKEN = os.getenv("DHAN_ACCESS_TOKEN", "")


# ============================================================
# 12. TELEGRAM
# ============================================================

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")


# ============================================================
# 13. PERSISTENCE
# ============================================================

DATABASE_PATH = os.getenv("ATHENA_DATABASE_PATH", "data/athena.db")
MODEL_PATH = os.getenv("ATHENA_MODEL_PATH", "athena_xgb_model.json")
SCALER_PATH = os.getenv("ATHENA_SCALER_PATH", "athena_scaler.pkl")


# ============================================================
# 14. EXECUTION SAFETY
# ============================================================

# Never allow an external AI response to bypass these controls.
ENFORCE_RISK_LIMITS = True

# Do not open another trade while an Athena position is active.
ONE_ACTIVE_TRADE_ONLY = True

# Require option LTP data before calculating option exits.
REQUIRE_OPTION_LTP_FOR_EXIT = True


# ============================================================
# 15. VALIDATION
# ============================================================

def validate_config():
    """Validate critical configuration before Athena starts."""

    if TARGET_PCT <= 0:
        raise ValueError("TARGET_PCT must be greater than 0.")

    if RISK_PCT <= 0 or OPTION_STOP_PCT <= 0:
        raise ValueError("Risk percentages must be greater than 0.")
    if MAX_RISK_PER_TRADE_PCT <= 0:
        raise ValueError("MAX_RISK_PER_TRADE_PCT must be greater than 0.")

    if TARGET_PCT / OPTION_STOP_PCT < MIN_RISK_REWARD:
        raise ValueError(
            "Configured target/stop does not satisfy MIN_RISK_REWARD."
        )

    if MIN_DELTA < 0 or MAX_DELTA > 1 or MIN_DELTA > MAX_DELTA:
        raise ValueError("Invalid option delta range.")

    if MAX_TRADES_PER_DAY < 1:
        raise ValueError("MAX_TRADES_PER_DAY must be at least 1.")

    if MAX_POSITIONS < 1:
        raise ValueError("MAX_POSITIONS must be at least 1.")

    if not INSTRUMENTS:
        raise ValueError("No instruments configured.")

    for name, instrument in INSTRUMENTS.items():
        required = ("security_id", "exchange_segment", "lot_size", "name")
        missing = [key for key in required if key not in instrument]
        if missing:
            raise ValueError(
                f"{name} missing configuration fields: {missing}"
            )
        if instrument["lot_size"] <= 0:
            raise ValueError(f"{name} lot size must be positive.")

    if ML_ENABLED and ML_MODEL_TYPE != "XGBOOST":
        raise ValueError("Athena-X ML engine is configured for XGBoost.")

    return True


# Validate on import so configuration errors fail early.
validate_config()