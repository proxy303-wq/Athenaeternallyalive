"""
ATHENA-X Main Controller — V2

Coordinates:
    market data -> indicators -> market score -> XGBoost ->
    option chain -> option selection -> trade execution ->
    OPTION-LTP monitoring -> exit -> ML learning.

Hard rule:
    Underlying/index price is NEVER used as an option exit price.
    Option target/stop/trailing/P&L always use the traded option LTP.
"""

import time
from datetime import datetime, timedelta, time as dt_time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

try:
    from dhanhq import DhanContext, dhanhq
    DHAN_AVAILABLE = True
except Exception:
    DhanContext = None
    dhanhq = None
    DHAN_AVAILABLE = False

from config import *
from logger import log, AthenaLogger
from meta_label_engine import MetaLabelEngine

META_ENGINE = MetaLabelEngine()

from database import AthenaDatabase
from indicators import calculate_indicators
from ml_engine import ml_engine
from telegram import telegram
from risk_engine import (
    classify_regime,
    effective_probability,
    evaluate_trade,
)
from orders import (
    select_best_option,
    calculate_trade_params,
    execute_trade,
    get_option_ltp,
    get_option_delta,
    get_option_oi,
    get_option_security_id,
    update_trailing_stop,
    check_exit_levels,
)

IST = ZoneInfo("Asia/Kolkata")


# ============================================================
# GLOBAL STATE
# ============================================================

STATE = {
    "trades_today": 0,
    "realized_pnl_today": 0.0,
    "realized_pnl_month": 0.0,
    "realized_pnl_year": 0.0,
    "wins": 0,
    "losses": 0,
    "active_trade": None,
    "date": None,
    "month_start": None,
    "year_start": None,
    "trade_history": [],
    "daily_returns": [],
    "sharpe_ratio": 0.0,
    "last_status_time": None,
    "last_report_date": None,
}



def restore_persistent_state():
    """Restore durable counters, trade history and any open trade."""
    try:
        saved_state = database.get_state("athena_state", {})

        if isinstance(saved_state, dict):
            for key in (
                "trades_today",
                "realized_pnl_today",
                "realized_pnl_month",
                "realized_pnl_year",
                "wins",
                "losses",
                "sharpe_ratio",
                "date",
                "month_start",
                "year_start",
            ):
                if key in saved_state:
                    STATE[key] = saved_state[key]

        active = database.get_active_trade()

        if active is not None:
            STATE["active_trade"] = active
            log("Database: restored active trade.")

        persisted_trades = database.get_trades()

        if persisted_trades:
            STATE["trade_history"] = persisted_trades

        log(
            "Database: restored "
            f"{len(persisted_trades)} completed trades."
        )

    except Exception as exc:
        # Persistence failure must not silently create a new trade state.
        log("Database restore failed: " + str(exc))


def persist_state():
    """Persist counters/state and the current active trade."""
    try:
        state_to_save = {
            key: STATE.get(key)
            for key in (
                "trades_today",
                "realized_pnl_today",
                "realized_pnl_month",
                "realized_pnl_year",
                "wins",
                "losses",
                "sharpe_ratio",
                "date",
                "month_start",
                "year_start",
            )
        }

        database.set_state("athena_state", state_to_save)
        database.save_active_trade(STATE.get("active_trade"))

    except Exception as exc:
        log("Database state save failed: " + str(exc))




# ============================================================
# CONNECTIONS
# ============================================================

dhan = None

if DHAN_AVAILABLE and CLIENT_ID and ACCESS_TOKEN:
    try:
        dhan = dhanhq(DhanContext(CLIENT_ID, ACCESS_TOKEN))
        log("Dhan connected")
    except Exception as exc:
        log("Dhan connection failed: " + str(exc))
else:
    if not DHAN_AVAILABLE:
        log("Dhan SDK unavailable.")
    else:
        log("Dhan credentials unavailable. Running without broker connection.")

logger = AthenaLogger()
database = AthenaDatabase()

if TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID:
    telegram.setup(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID)
else:
    log("Telegram credentials unavailable; Telegram alerts disabled.")


# ============================================================
# MARKET HOURS
# ============================================================

def is_trading_day():
    """Weekday check. Exchange holidays are handled by broker/data availability."""
    return datetime.now(IST).weekday() < 5


def is_market_open():
    now = datetime.now(IST).time()
    return dt_time(9, 15) <= now <= dt_time(15, 30)


def get_next_market_open():
    now = datetime.now(IST)

    for offset in range(0, 8):
        candidate = now + timedelta(days=offset)

        if candidate.weekday() >= 5:
            continue

        candidate_open = candidate.replace(
            hour=9,
            minute=15,
            second=0,
            microsecond=0,
        )

        if candidate_open > now:
            return candidate_open

    return None


def should_exit_market_hours():
    now = datetime.now(IST)

    if not is_trading_day():
        return True

    if now.time() < dt_time(9, 15):
        return True

    if now.time() > dt_time(15, 30):
        return True

    return False


def market_time_allowed():
    now = datetime.now(IST).time()
    return NO_TRADE_BEFORE <= now < NO_NEW_ENTRY_AFTER


def is_optimal_time():
    current = datetime.now(IST).time()
    current_minutes = current.hour * 60 + current.minute

    if current_minutes < 9 * 60 + 30:
        return False

    if current_minutes > 15 * 60:
        return False

    for hour, minute in OPTIMAL_TRADE_TIMES:
        if current.hour == hour and abs(current.minute - minute) <= 5:
            return True

    # Preserve the original broad intraday window.
    return 10 * 60 <= current_minutes <= 14 * 60 + 30


# ============================================================
# CAPITAL / RISK
# ============================================================

def get_capital():
    if not dhan:
        return float(FALLBACK_CAPITAL)

    try:
        funds = dhan.get_fund_limits()

        if isinstance(funds, dict):
            data = funds.get("data", funds)

            # Dhan has historically returned this field with a typo.
            for key in (
                "availabelBalance",
                "availableBalance",
                "available_balance",
            ):
                balance = data.get(key)

                if balance is not None:
                    value = float(balance)

                    if value > 0:
                        return value

    except Exception as exc:
        log("Capital lookup failed: " + str(exc))

    return float(FALLBACK_CAPITAL)


def get_equity_drawdown_pct(capital):
    """
    Estimate current drawdown from the configured capital anchor.

    This is intentionally conservative: realized cumulative P&L is used,
    without pretending that unrealized P&L is known when it is not.
    """
    try:
        capital = float(capital)
        if capital <= 0:
            return 0.0

        cumulative_pnl = float(
            STATE.get("realized_pnl_year", 0.0)
        )

        if cumulative_pnl >= 0:
            return 0.0

        return abs(cumulative_pnl) / capital

    except (TypeError, ValueError):
        return 0.0


def get_estimated_trade_cost(entry, quantity):
    """Conservative cost allowance used only for EV screening."""
    try:
        entry = float(entry)
        quantity = int(quantity)
        if entry <= 0 or quantity <= 0:
            return 0.0

        gross_notional = entry * quantity

        return gross_notional * (
            TRANSACTION_COST_PCT + SLIPPAGE_PCT
        )

    except (TypeError, ValueError):
        return 0.0


def check_limits(capital):
    daily_target = capital * DAILY_TARGET_PCT
    max_daily_loss = capital * DAILY_LOSS_PCT
    max_monthly_loss = capital * MONTHLY_LOSS_PCT

    # Daily profit objective: stop opening new trades once the portfolio
    # has reached +1% of the day's capital objective. Existing positions
    # may still be managed to their normal exits.
    if STATE["realized_pnl_today"] >= daily_target:
        log(
            f"DAILY TARGET REACHED: ₹{STATE['realized_pnl_today']:.2f} "
            f"/ ₹{daily_target:.2f}. No new entries."
        )
        return False

    if STATE["realized_pnl_today"] <= -max_daily_loss:
        log("DAILY LOSS LIMIT REACHED")
        telegram.send_error("DAILY LOSS LIMIT REACHED")
        return False

    if STATE["realized_pnl_month"] <= -max_monthly_loss:
        log("MONTHLY LOSS LIMIT REACHED")
        telegram.send_error("MONTHLY LOSS LIMIT REACHED")
        return False

    return True


def has_open_position():
    """
    Broker-side safety check.

    If broker state cannot be verified, return True so Athena does not
    accidentally open another position.
    """
    if not dhan:
        return False

    try:
        response = dhan.get_positions()

        positions = (
            response.get("data", [])
            if isinstance(response, dict)
            else response
        )

        if not isinstance(positions, list):
            return True

        for position in positions:
            try:
                net_qty = float(
                    position.get(
                        "netQty",
                        position.get("net_qty", 0),
                    )
                    or 0
                )

                if net_qty != 0:
                    return True

            except (TypeError, ValueError):
                continue

        return False

    except Exception as exc:
        log("Position verification failed: " + str(exc))
        return True


# ============================================================
# MARKET DATA
# ============================================================

def _parse_market_response(response):
    """Normalize common Dhan candle response formats."""
    if not isinstance(response, dict):
        return pd.DataFrame()

    data = response.get("data")

    if not data:
        return pd.DataFrame()

    if isinstance(data, dict):
        # Dhan commonly returns column -> list.
        if all(isinstance(value, list) for value in data.values()):
            df = pd.DataFrame(data)
        else:
            return pd.DataFrame()

    elif isinstance(data, list):
        if not data:
            return pd.DataFrame()

        if isinstance(data[0], (list, tuple)):
            df = pd.DataFrame(
                data,
                columns=[
                    "timestamp",
                    "open",
                    "high",
                    "low",
                    "close",
                    "volume",
                ],
            )
        else:
            df = pd.DataFrame(data)

    else:
        return pd.DataFrame()

    df.columns = [str(column).strip().lower() for column in df.columns]

    required = {"open", "high", "low", "close"}

    if not required.issubset(df.columns):
        return pd.DataFrame()

    for column in ("open", "high", "low", "close", "volume"):
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    if "timestamp" in df.columns:
        timestamps = pd.to_datetime(
            df["timestamp"],
            unit="s",
            errors="coerce",
        )

        # If timestamps were not epoch seconds, try normal parsing.
        if timestamps.notna().sum() == 0:
            timestamps = pd.to_datetime(
                df["timestamp"],
                errors="coerce",
            )

        df["timestamp"] = timestamps
        df = df.dropna(subset=["timestamp"])
        df = df.set_index("timestamp")

    df = df.dropna(subset=["high", "low", "close"])

    if isinstance(df.index, pd.DatetimeIndex):
        df = df.sort_index()

    return df


def get_instrument_data(instrument_config):
    """
    Fetch underlying INDEX candles.

    Critical:
        exchange_segment comes from the individual instrument config.
        It is NOT globally forced to NIFTY.
    """
    if not dhan:
        return pd.DataFrame()

    end_date = datetime.now(IST).date()
    start_date = end_date - timedelta(days=HISTORY_DAYS)

    try:
        response = dhan.intraday_minute_data(
            security_id=str(instrument_config["security_id"]),
            exchange_segment=instrument_config["exchange_segment"],
            instrument_type="INDEX",
            from_date=start_date.isoformat(),
            to_date=end_date.isoformat(),
        )

        return _parse_market_response(response)

    except Exception as exc:
        log(
            "Data error for "
            + str(instrument_config.get("name", "UNKNOWN"))
            + ": "
            + str(exc)
        )
        return pd.DataFrame()


# ============================================================
# MARKET ANALYSIS
# ============================================================

def calculate_quality_score(market):
    score = 0
    direction = market["direction"]

    if (
        direction == "BULLISH"
        and market["ema20"] > market["ema50"] > market["ema200"]
    ) or (
        direction == "BEARISH"
        and market["ema20"] < market["ema50"] < market["ema200"]
    ):
        score += 30

    elif (
        direction == "BULLISH"
        and market["ema20"] > market["ema50"]
    ) or (
        direction == "BEARISH"
        and market["ema20"] < market["ema50"]
    ):
        score += 20

    if direction == "BULLISH" and 45 <= market["rsi"] <= 60:
        score += 20
    elif direction == "BEARISH" and 40 <= market["rsi"] <= 55:
        score += 20

    if (
        direction == "BULLISH"
        and market["macd_hist"] > 0
    ) or (
        direction == "BEARISH"
        and market["macd_hist"] < 0
    ):
        score += 20

    if market["adx"] > 30:
        score += 15
    elif market["adx"] > 25:
        score += 10

    if (
        direction == "BULLISH"
        and market["price"] > market["vwap"]
    ) or (
        direction == "BEARISH"
        and market["price"] < market["vwap"]
    ):
        score += 15

    return score


def get_price_confirmation(df, market):
    wait_bars = max(1, int(CONFIRMATION_BARS))

    if len(df) < wait_bars + 1:
        return False

    latest = float(df.iloc[-1]["close"])
    previous = float(df.iloc[-1 - wait_bars]["close"])

    if market["direction"] == "BULLISH":
        return latest > previous

    return latest < previous


def _safe_float(value):
    try:
        value = float(value)

        if np.isfinite(value):
            return value

    except (TypeError, ValueError):
        pass

    return None


def analyze_market(df, instrument_name="NIFTY"):
    """
    Convert underlying OHLCV into Athena's market snapshot.

    No option-chain information is used here.
    """
    if df is None or df.empty:
        return None

    df = calculate_indicators(df)

    if len(df) < 50:
        return None

    latest = df.iloc[-1]
    previous = df.iloc[-2]

    # Core fields required by the scoring engine.
    required = [
        "close",
        "ema20",
        "ema50",
        "ema200",
        "sma20",
        "sma50",
        "rsi",
        "macd",
        "macd_signal",
        "macd_hist",
        "bb_upper",
        "bb_middle",
        "bb_lower",
        "adx",
        "atr",
        "vwap",
        "resistance",
        "support",
        "tenkan",
        "kijun",
        "senkou_a",
        "senkou_b",
    ]

    values = {}

    for field in required:
        value = _safe_float(latest.get(field))

        if value is None:
            return None

        values[field] = value

    price = values["close"]

    bullish_score = 0
    bearish_score = 0

    # EMA trend
    if values["ema20"] > values["ema50"] > values["ema200"]:
        bullish_score += 20
    elif values["ema20"] > values["ema50"]:
        bullish_score += 12
    elif values["ema20"] < values["ema50"] < values["ema200"]:
        bearish_score += 20
    elif values["ema20"] < values["ema50"]:
        bearish_score += 12

    # SMA crossover
    if USE_SMA_CROSSOVER:
        if values["sma20"] > values["sma50"]:
            bullish_score += 10
        else:
            bearish_score += 10

    # Ichimoku
    if USE_ICHIMOKU:
        if price > values["senkou_a"] and price > values["senkou_b"]:
            bullish_score += 15
        elif price < values["senkou_a"] and price < values["senkou_b"]:
            bearish_score += 15

        if values["tenkan"] > values["kijun"]:
            bullish_score += 8
        else:
            bearish_score += 8

    # MACD
    if values["macd"] > values["macd_signal"] and values["macd_hist"] > 0:
        bullish_score += 15
    elif values["macd"] > values["macd_signal"]:
        bullish_score += 8
    elif values["macd"] < values["macd_signal"] and values["macd_hist"] < 0:
        bearish_score += 15
    elif values["macd"] < values["macd_signal"]:
        bearish_score += 8

    # RSI
    rsi_value = values["rsi"]

    if 45 <= rsi_value <= 65:
        bullish_score += 10
    elif 30 <= rsi_value < 45:
        bullish_score += 15
    elif 65 < rsi_value <= 75:
        bearish_score += 8
    elif rsi_value > 75:
        bearish_score += 15
    elif rsi_value < 30:
        bullish_score += 20

    # Bollinger
    if price <= values["bb_lower"]:
        bullish_score += 10
    elif price >= values["bb_upper"]:
        bearish_score += 10
    elif price > values["bb_middle"]:
        bullish_score += 5
    else:
        bearish_score += 5

    # VWAP
    if price > values["vwap"]:
        bullish_score += 5
    else:
        bearish_score += 5

    # ADX direction reinforcement
    if values["adx"] > 25:
        if bullish_score > bearish_score:
            bullish_score += 5
        elif bearish_score > bullish_score:
            bearish_score += 5

    # Support/resistance
    if price <= values["support"] * 1.002:
        bullish_score += 5

    if price >= values["resistance"] * 0.998:
        bearish_score += 5

    # Latest candle direction
    previous_close = _safe_float(previous["close"])

    if previous_close is not None:
        if price > previous_close:
            bullish_score += 5
        elif price < previous_close:
            bearish_score += 5

    if bullish_score >= bearish_score:
        direction = "BULLISH"
        score = bullish_score
    else:
        direction = "BEARISH"
        score = bearish_score

    total_score = bullish_score + bearish_score

    confidence = (
        score / total_score
        if total_score > 0
        else 0.5
    )

    atr_value = values["atr"]
    vwap_value = values["vwap"]

    # Real volume ratio instead of the old hard-coded 1.
    volume_ratio = 1.0

    if "volume" in df.columns and len(df) >= 21:
        current_volume = _safe_float(df.iloc[-1]["volume"])
        average_volume = _safe_float(
            df["volume"].iloc[-21:-1].mean()
        )

        if (
            current_volume is not None
            and average_volume is not None
            and average_volume > 0
        ):
            volume_ratio = current_volume / average_volume

    # Distances are percentage distances from recent S/R.
    distance_from_high = (
        abs(values["resistance"] - price) / price * 100
        if price > 0
        else 0
    )

    distance_from_low = (
        abs(price - values["support"]) / price * 100
        if price > 0
        else 0
    )

    market = {
        "price": price,
        "direction": direction,
        "score": score,
        "bullish_score": bullish_score,
        "bearish_score": bearish_score,
        "confidence": confidence,
        "rsi": rsi_value,
        "atr": atr_value,
        "vwap": vwap_value,
        "support": values["support"],
        "resistance": values["resistance"],
        "expected_move": atr_value * 1.25,
        "ema20": values["ema20"],
        "ema50": values["ema50"],
        "ema200": values["ema200"],
        "sma20": values["sma20"],
        "sma50": values["sma50"],
        "macd": values["macd"],
        "macd_hist": values["macd_hist"],
        "adx": values["adx"],
        "tenkan": values["tenkan"],
        "kijun": values["kijun"],
        "senkou_a": values["senkou_a"],
        "senkou_b": values["senkou_b"],
        "instrument": instrument_name,
    }

    market["atr_percent"] = (
        atr_value / price * 100
        if price > 0
        else 0.0
    )

    market["regime"] = classify_regime(market)

    market["quality_score"] = calculate_quality_score(market)

    # These fields feed XGBoost. They are captured from the actual
    # underlying market snapshot, not fabricated at trade close.
    market["ml_features"] = {
        "rsi_at_entry": rsi_value,
        "ema_cross": 1 if values["ema20"] > values["ema50"] else -1,
        "vwap_ratio": (
            price / vwap_value
            if vwap_value > 0
            else 1.0
        ),
        "atr_percent": (
            atr_value / price * 100
            if price > 0
            else 0.0
        ),
        "volume_ratio": volume_ratio,
        # Delta/OI are filled after option selection.
        "delta": 0.5,
        "oi_at_entry": 10000.0,
        "market_score": score,
        "distance_from_high": distance_from_high,
        "distance_from_low": distance_from_low,
    }

    log(
        f"{instrument_name} | {direction} | "
        f"Score={score} | Confidence={confidence:.2f} | "
        f"Quality={market['quality_score']}"
    )

    return market


# ============================================================
# OPTION CHAIN
# ============================================================

def get_nearest_expiry(security_id):
    if not dhan:
        return None

    try:
        response = dhan.get_expiry_list(
            underlying_security_id=int(security_id),
            underlying_type="INDEX",
        )

        if isinstance(response, dict):
            data = response.get("data", [])

            if isinstance(data, list) and data:
                return data[0]

    except Exception as exc:
        log("Expiry lookup failed: " + str(exc))

    return None


def get_option_chain(security_id, expiry):
    if not dhan:
        return None

    try:
        return dhan.get_option_chain(
            underlying_security_id=int(security_id),
            expiry=expiry,
        )

    except Exception as exc:
        log("Option-chain lookup failed: " + str(exc))
        return None


def parse_option_chain(response):
    if response is None:
        return pd.DataFrame()

    data = response.get("data", response) if isinstance(
        response, dict
    ) else response

    rows = []

    if isinstance(data, dict):
        for strike_key, item in data.items():
            if not isinstance(item, dict):
                continue

            try:
                strike = float(
                    item.get(
                        "strike_price",
                        item.get("strikePrice", strike_key),
                    )
                )
            except (TypeError, ValueError):
                continue

            ce = (
                item.get("ce")
                or item.get("CE")
                or item.get("call")
                or {}
            )

            pe = (
                item.get("pe")
                or item.get("PE")
                or item.get("put")
                or {}
            )

            rows.append({
                "strike": strike,
                "ce": ce,
                "pe": pe,
            })

    elif isinstance(data, list):
        for item in data:
            if not isinstance(item, dict):
                continue

            try:
                strike = float(
                    item.get(
                        "strike_price",
                        item.get("strikePrice"),
                    )
                )
            except (TypeError, ValueError):
                continue

            rows.append({
                "strike": strike,
                "ce": (
                    item.get("ce")
                    or item.get("CE")
                    or item.get("call")
                    or {}
                ),
                "pe": (
                    item.get("pe")
                    or item.get("PE")
                    or item.get("put")
                    or {}
                ),
            })

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values("strike").reset_index(drop=True)


# ============================================================
# TRADE MONITORING
# ============================================================

def get_live_option_price(active_trade):
    """
    Fetch the traded OPTION's LTP.

    This function is intentionally separate from get_instrument_data().
    """
    if not active_trade:
        return None

    security_id = active_trade.get("security_id")

    if not security_id:
        return None

    if not dhan:
        # Paper mode cannot obtain a broker quote without a data source.
        # Do not substitute the underlying index price.
        return None

    try:
        # Dhan quote API response shapes vary by SDK version.
        response = dhan.ohlc_data(
            securities={
                "NSE_FNO": [str(security_id)]
            }
        )

        if not isinstance(response, dict):
            return None

        data = response.get("data", response)

        if isinstance(data, dict):
            instrument_data = data.get("NSE_FNO")

            # Dhan commonly nests the security ID one level below the
            # exchange segment: {"NSE_FNO": {"22222": {...}}}.
            if isinstance(instrument_data, dict):
                nested = (
                    instrument_data.get(str(security_id))
                    or instrument_data.get(int(security_id))
                    if str(security_id).isdigit()
                    else instrument_data.get(str(security_id))
                )

                if isinstance(nested, dict):
                    instrument_data = nested

                for key in ("last_price", "ltp", "LTP", "close"):
                    value = _safe_float(instrument_data.get(key))

                    if value is not None and value > 0:
                        return value

            # Also support a flat response keyed directly by security ID.
            flat = data.get(str(security_id))

            if isinstance(flat, dict):
                for key in ("last_price", "ltp", "LTP", "close"):
                    value = _safe_float(flat.get(key))

                    if value is not None and value > 0:
                        return value

        return None

    except Exception as exc:
        log("Option LTP lookup failed: " + str(exc))
        return None


def close_trade(exit_price, exit_reason, current_time):
    """
    Close the current paper trade.

    P&L is always:
        (option exit LTP - option entry LTP) * quantity

    because Athena buys options rather than shorting their premium.
    """
    active = STATE.get("active_trade")

    if active is None:
        return

    try:
        entry = float(active["entry"])
        exit_price = float(exit_price)
        quantity = int(active["quantity"])
    except (KeyError, TypeError, ValueError):
        log("Unable to close trade: invalid active trade state.")
        return

    pnl = (exit_price - entry) * quantity

    STATE["realized_pnl_today"] += pnl
    STATE["realized_pnl_month"] += pnl
    STATE["realized_pnl_year"] += pnl

    if pnl > 0:
        STATE["wins"] += 1
    else:
        STATE["losses"] += 1

    entry_time = active.get("entry_time")

    holding_minutes = 0.0

    try:
        if isinstance(entry_time, str):
            entry_time = datetime.fromisoformat(entry_time)

        if entry_time:
            holding_minutes = (
                current_time - entry_time
            ).total_seconds() / 60.0

    except Exception:
        pass

    # Preserve the actual ML entry features captured by orders.py.
    ml_features = active.get("ml_features", {}).copy()

    # Inject actual option values.
    ml_features["delta"] = active.get("delta", 0.5)
    ml_features["oi_at_entry"] = active.get("oi_at_entry", 10000.0)

    canonical_reason = {
        "TARGET": "TAKE_PROFIT",
        "STOP": "STOP_LOSS",
        "TIME_EXIT": "TIME_EXIT",
        "TAKE_PROFIT": "TAKE_PROFIT",
        "STOP_LOSS": "STOP_LOSS",
        "FORCE_EXIT": "FORCE_EXIT",
        "SIGNAL_REVERSAL": "SIGNAL_REVERSAL",
    }.get(str(exit_reason).upper(), str(exit_reason).upper())

    trade_record = {
        "timestamp": current_time.isoformat(),
        "instrument": active.get("instrument", "NIFTY"),
        "option_type": active.get("option_type", "CE"),
        "strike": active.get("strike", 0),
        "security_id": active.get("security_id"),
        "entry": entry,
        "exit": exit_price,
        "quantity": quantity,
        "pnl": pnl,
        "win": pnl > 0,
        "exit_reason": canonical_reason,
        "rsi_at_entry": ml_features.get("rsi_at_entry"),
        "ema_cross": ml_features.get("ema_cross"),
        "vwap_ratio": ml_features.get("vwap_ratio"),
        "atr_percent": ml_features.get("atr_percent"),
        "volume_ratio": ml_features.get("volume_ratio"),
        "delta": ml_features.get("delta"),
        "oi_at_entry": ml_features.get("oi_at_entry"),
        "market_score": ml_features.get(
            "market_score",
            active.get("score", 0),
        ),
        "distance_from_high": ml_features.get(
            "distance_from_high",
            0,
        ),
        "distance_from_low": ml_features.get(
            "distance_from_low",
            0,
        ),
        "holding_minutes": holding_minutes,
        "ml_features": ml_features,
        "regime": active.get("regime"),
        "win_probability_at_entry": active.get(
            "win_probability_at_entry"
        ),
        "expected_value_at_entry": active.get(
            "expected_value_at_entry"
        ),
        "expected_value_per_risk": active.get(
            "expected_value_per_risk"
        ),
    }

    STATE["trade_history"].append(trade_record)

    try:
        database.save_trade(trade_record)
        database.clear_active_trade()
        # Persist counters immediately after a completed trade so a restart
        # cannot lose the latest P&L/win-loss state.
        persist_state()
    except Exception as exc:
        log("Database trade save failed: " + str(exc))

    # Feed the exact same feature schema to XGBoost.
    ml_engine.add_trade_result(trade_record)

    try:
        logger.log_trade(trade_record)
    except Exception as exc:
        log("Trade logging failed: " + str(exc))

    log(
        f"TRADE CLOSED | {trade_record['instrument']} "
        f"{trade_record['option_type']} "
        f"{trade_record['strike']} | "
        f"Exit={exit_price:.2f} | "
        f"Reason={exit_reason} | "
        f"P&L={pnl:+.2f}"
    )

    telegram.send_status(
        f"TRADE CLOSED\n"
        f"{trade_record['instrument']} "
        f"{trade_record['option_type']} "
        f"{trade_record['strike']}\n"
        f"Exit: ₹{exit_price:.2f}\n"
        f"Reason: {exit_reason}\n"
        f"P&L: ₹{pnl:+.2f}"
    )

    STATE["active_trade"] = None


def monitor_trade():
    """
    Monitor the OPTION LTP.

    Underlying/index candles are not used for target, stop or P&L.
    """
    active = STATE.get("active_trade")

    if active is None:
        return

    option_price = get_live_option_price(active)

    if option_price is None:
        log("Waiting for traded option LTP — no index-price fallback.")
        return

    update_trailing_stop(active, option_price)

    exit_reason = check_exit_levels(
        active,
        option_price,
    )

    if exit_reason:
        close_trade(
            option_price,
            exit_reason,
            datetime.now(IST),
        )


def check_signal_reversal():
    """Close a trade if the underlying directional signal fully reverses."""
    active = STATE.get("active_trade")

    if active is None:
        return False

    instrument_name = active.get("instrument", "NIFTY")
    instrument_config = INSTRUMENTS.get(instrument_name)

    if not instrument_config:
        return False

    df = get_instrument_data(instrument_config)

    if df.empty:
        return False

    market = analyze_market(
        df,
        instrument_name,
    )

    if market is None:
        return False

    entry_direction = active.get("direction")
    current_direction = market["direction"]

    reversed_signal = (
        entry_direction == "BULLISH"
        and current_direction == "BEARISH"
    ) or (
        entry_direction == "BEARISH"
        and current_direction == "BULLISH"
    )

    if reversed_signal:
        log(
            f"Signal reversal detected: "
            f"{entry_direction} -> {current_direction}"
        )
        telegram.send_status(
            f"Signal reversal on {instrument_name}: "
            f"{entry_direction} -> {current_direction}"
        )
        return True

    return False


def force_exit():
    """
    Force-close at FORCE_EXIT_TIME.

    Paper mode requires an actual option LTP. It never substitutes
    the underlying index price.
    """
    active = STATE.get("active_trade")

    if active is None:
        return

    log("FORCE EXIT")
    telegram.send_status("FORCE EXIT")

    if not LIVE_TRADING:
        option_price = get_live_option_price(active)

        if option_price is None:
            log(
                "Paper force-exit deferred: "
                "traded option LTP unavailable."
            )
            return

        close_trade(
            option_price,
            "FORCE_EXIT",
            datetime.now(IST),
        )
        return

    if not dhan:
        log("Live force-exit failed: Dhan unavailable.")
        telegram.send_error(
            "Live force-exit failed: Dhan unavailable."
        )
        return

    try:
        response = dhan.exit_all_positions()
        log("Broker exit response: " + str(response))

        # Broker-side position should be verified before clearing state.
        if not has_open_position():
            STATE["active_trade"] = None
            telegram.send_status("Live positions exited.")
        else:
            telegram.send_error(
                "Broker still reports an open position after force exit."
            )

    except Exception as exc:
        log("Live force-exit error: " + str(exc))
        telegram.send_error(
            "Live force-exit error: " + str(exc)
        )


# ============================================================
# REPORTING / STATE
# ============================================================

def calculate_sharpe_ratio(returns):
    if len(returns) < 2:
        return 0.0

    values = np.asarray(returns, dtype=float)
    std = np.std(values)

    if std == 0:
        return 0.0

    return float(
        (np.mean(values) - RISK_FREE_RATE) / std
    )


def update_sharpe_ratio(pnl):
    if not TRACK_SHARPE_RATIO:
        return

    capital = get_capital()

    if capital <= 0:
        return

    STATE["daily_returns"].append(pnl / capital)

    STATE["daily_returns"] = STATE["daily_returns"][-365:]

    STATE["sharpe_ratio"] = calculate_sharpe_ratio(
        STATE["daily_returns"]
    )


def generate_report(capital):
    total = STATE["wins"] + STATE["losses"]

    win_rate = (
        STATE["wins"] / total * 100
        if total
        else 0
    )

    monthly_target = capital * 0.125

    progress = (
        STATE["realized_pnl_month"]
        / monthly_target
        * 100
        if monthly_target > 0
        else 0
    )

    report = {
        "date": datetime.now(IST).strftime("%Y-%m-%d"),
        "today_pnl": STATE["realized_pnl_today"],
        "month_pnl": STATE["realized_pnl_month"],
        "wins": STATE["wins"],
        "losses": STATE["losses"],
        "win_rate": win_rate,
        "progress": progress,
        "sharpe_ratio": STATE["sharpe_ratio"],
    }

    log(
        "PERFORMANCE | "
        f"Today P&L={STATE['realized_pnl_today']:.2f} | "
        f"Month P&L={STATE['realized_pnl_month']:.2f} | "
        f"Win rate={win_rate:.1f}%"
    )

    try:
        telegram.send_daily_report(report)
    except Exception:
        pass


def reset_state():
    current_date = datetime.now(IST).date()
    current_month = current_date.replace(day=1)
    current_year = current_date.replace(
        month=1,
        day=1,
    )

    if STATE["date"] != current_date:
        STATE["date"] = current_date
        STATE["trades_today"] = 0
        STATE["realized_pnl_today"] = 0.0
        STATE["active_trade"] = None
        STATE["daily_returns"] = []
        STATE["sharpe_ratio"] = 0.0
        STATE["last_report_date"] = None

        log("New trading day")

    if STATE["month_start"] != current_month:
        STATE["month_start"] = current_month
        STATE["realized_pnl_month"] = 0.0

        log("New month")

    if STATE["year_start"] != current_year:
        STATE["year_start"] = current_year
        STATE["realized_pnl_year"] = 0.0

        log("New year")


# ============================================================
# MAIN DECISION LOOP
# ============================================================

def run():
    reset_state()

    if should_exit_market_hours():
        return

    now = datetime.now(IST)
    current_time = now.time()

    # Periodic status.
    last_status = STATE.get("last_status_time")

    if (
        last_status is None
        or (now - last_status).total_seconds() > 1800
    ):
        STATE["last_status_time"] = now
        log(
            "ATHENA-X STATUS | "
            f"{now.strftime('%Y-%m-%d %H:%M:%S')} IST | "
            f"Mode={'LIVE' if LIVE_TRADING else 'PAPER'}"
        )

    # Mandatory force exit window.
    if current_time >= FORCE_EXIT_TIME:
        force_exit()

        if (
            current_time >= FORCE_EXIT_TIME
            and current_time < dt_time(15, 30)
            and STATE.get("last_report_date") != now.date()
        ):
            generate_report(get_capital())
            STATE["last_report_date"] = now.date()

        return

    # --------------------------------------------------------
    # ACTIVE TRADE
    # --------------------------------------------------------

    if STATE["active_trade"] is not None:
        monitor_trade()

        if STATE["active_trade"] is not None:
            if check_signal_reversal():
                force_exit()

        return

    # --------------------------------------------------------
    # BROKER POSITION SAFETY
    # --------------------------------------------------------

    if has_open_position():
        log(
            "Broker reports an open position while Athena has "
            "no local active trade. No new trade will be opened."
        )
        return

    # --------------------------------------------------------
    # ENTRY FILTERS
    # --------------------------------------------------------

    if not market_time_allowed():
        return

    if STATE["trades_today"] >= MAX_TRADES_PER_DAY:
        return

    capital = get_capital()

    if not check_limits(capital):
        return

    # --------------------------------------------------------
    # INSTRUMENT SCAN
    # --------------------------------------------------------

    for instrument_name, instrument_config in INSTRUMENTS.items():

        if not instrument_config.get("enabled", False):
            continue

        df = get_instrument_data(instrument_config)

        if df.empty:
            continue

        market = analyze_market(
            df,
            instrument_name,
        )

        if market is None:
            continue

        if market["score"] < MIN_SIGNAL_SCORE:
            log(
                f"{instrument_name}: score "
                f"{market['score']} below threshold"
            )
            continue

        if market["confidence"] < MIN_CONFIDENCE:
            log(
                f"{instrument_name}: confidence "
                f"{market['confidence']:.2f} below threshold"
            )
            continue

        if market["quality_score"] < MIN_QUALITY_SCORE:
            log(
                f"{instrument_name}: quality "
                f"{market['quality_score']} below threshold"
            )
            continue

        atr_percent = (
            market["atr"] / market["price"] * 100
            if market["price"] > 0
            else 0
        )

        if (
            atr_percent > MAX_VOLATILITY
            or atr_percent < MIN_VOLATILITY
        ):
            log(
                f"{instrument_name}: volatility "
                f"{atr_percent:.2f}% outside limits"
            )
            continue

        if not is_optimal_time():
            continue

        if not get_price_confirmation(df, market):
            log(
                f"{instrument_name}: waiting for "
                "price confirmation"
            )
            continue

        # ----------------------------------------------------
        # XGBOOST
        # ----------------------------------------------------

        ml_prediction = ml_engine.predict(
            market["ml_features"]
        )

        if ml_engine.is_trained:
            log(
                f"XGBoost | {instrument_name} | "
                f"Win Prob={ml_prediction['win_probability']:.3f} | "
                f"{ml_prediction['recommendation']}"
            )

            if ml_prediction["recommendation"] == "NO":
                telegram.send_status(
                    f"XGBoost rejected {instrument_name}"
                )
                continue

        # ----------------------------------------------------
        # OPTION CHAIN
        # ----------------------------------------------------

        expiry = get_nearest_expiry(
            instrument_config["security_id"]
        )

        if not expiry:
            log(
                f"{instrument_name}: no expiry available"
            )
            continue

        chain = get_option_chain(
            instrument_config["security_id"],
            expiry,
        )

        if not chain:
            continue

        chain_df = parse_option_chain(chain)

        if chain_df.empty:
            continue

        candidate = select_best_option(
            chain_df,
            market,
            instrument_name,
        )

        if not candidate:
            log(
                f"{instrument_name}: no valid option candidate"
            )
            continue

        # Update ML features with the actual selected option.
        market["ml_features"]["delta"] = candidate.get(
            "delta",
            0.5,
        )
        market["ml_features"]["oi_at_entry"] = candidate.get(
            "oi",
            10000,
        )

        # Re-run XGBoost with the actual option delta/OI.
        final_ml_prediction = ml_engine.predict(
            market["ml_features"]
        )

        if ml_engine.is_trained:
            if final_ml_prediction["recommendation"] == "NO":
                log(
                    f"{instrument_name}: XGBoost rejected "
                    "candidate after option data."
                )
                continue

        probability = effective_probability(
            ml_probability=final_ml_prediction.get(
                "win_probability",
                0.5,
            ),
            ml_trained=bool(
                final_ml_prediction.get("is_trained", False)
            ),
            market_confidence=market.get("confidence", 0.5),
        )

        drawdown_pct = get_equity_drawdown_pct(capital)

        # First calculate exits using one unit so EV can be evaluated
        # independently of position size.
        provisional_trade = calculate_trade_params(
            candidate,
            capital,
            instrument_name,
            probability=probability,
            market=market,
            drawdown_pct=drawdown_pct,
        )

        if not provisional_trade:
            continue

        estimated_costs = get_estimated_trade_cost(
            provisional_trade["entry"],
            provisional_trade["quantity"],
        )

        math_decision = evaluate_trade(
            probability=probability,
            reward=provisional_trade["reward"],
            risk=provisional_trade["risk"],
            market=market,
            drawdown_pct=drawdown_pct,
            estimated_costs=estimated_costs,
        )

        log(
            f"QUANT | {instrument_name} | "
            f"Regime={math_decision['regime']} | "
            f"P={probability:.3f} | "
            f"EV/Risk={math_decision['expected_value_per_risk']:.3f} | "
            f"Kelly={math_decision['fractional_kelly']:.3f} | "
            f"RiskMult={math_decision['risk_multiplier']:.2f} | "
            f"Decision={math_decision['reason']}"
        )

        if not math_decision["approved"]:
            continue

        # Optional second-stage meta-label filter.
        # It runs only after the primary XGBoost + quantitative layer have
        # approved the candidate, so the meta model is genuinely a
        # "should we take this setup?" filter.
        ml_features = market.get("ml_features", {})
        if not isinstance(ml_features, dict):
            ml_features = {}

        meta_row = {
            "primary_probability": probability,
            "delta_abs": abs(float(
                candidate.get("delta", 0.0) or 0.0
            )),
            "atr_percent": float(
                market.get("atr_percent", 0.0) or 0.0
            ),
            "volume_ratio": float(
                ml_features.get("volume_ratio", 0.0) or 0.0
            ),
            "market_score": float(
                ml_features.get("market_score", 0.0) or 0.0
            ),
            "distance_from_high": float(
                ml_features.get("distance_from_high", 0.0) or 0.0
            ),
            "distance_from_low": float(
                ml_features.get("distance_from_low", 0.0) or 0.0
            ),
            "expected_value_per_risk": float(
                math_decision["expected_value_per_risk"]
            ),
            "risk_pct_at_entry": float(
                provisional_trade.get("risk_pct", 0.0) or 0.0
            ),
            "regime_score": float(
                math_decision.get("risk_multiplier", 0.0) or 0.0
            ),
            "risk_reward": float(
                provisional_trade.get("risk_reward", 0.0) or 0.0
            ),
        }

        meta_decision = META_ENGINE.approve(meta_row)

        if meta_decision["available"] and not meta_decision["approved"]:
            log(
                f"Meta-label rejected {instrument_name} | "
                f"p={meta_decision['probability']:.3f}"
            )
            continue

        # Recalculate sizing using the approved probability/regime.
        trade = calculate_trade_params(
            candidate,
            capital,
            instrument_name,
            probability=probability,
            market=market,
            drawdown_pct=drawdown_pct,
        )

        if not trade:
            continue

        trade["expected_value"] = math_decision["expected_value"]
        trade["expected_value_per_risk"] = (
            math_decision["expected_value_per_risk"]
        )
        trade["regime"] = math_decision["regime"]
        trade["win_probability"] = probability
        trade["win_probability_at_entry"] = probability
        trade["expected_value_at_entry"] = (
            math_decision["expected_value"]
        )
        trade["meta_probability_at_entry"] = (
            meta_decision.get("probability")
        )
        trade["meta_model_available"] = (
            meta_decision.get("available", False)
        )

        # Execute through the centralized order module.
        result = execute_trade(
            dhan,
            market,
            candidate,
            trade,
            logger,
            STATE,
            ml_engine,
        )

        if result is not None:
            # execute_trade builds the canonical active state.
            # Ensure the exact ML features are retained.
            if STATE.get("active_trade") is not None:
                STATE["active_trade"]["ml_features"] = (
                    market["ml_features"].copy()
                )
                STATE["active_trade"]["regime"] = (
                    trade.get("regime", market.get("regime"))
                )
                STATE["active_trade"]["win_probability_at_entry"] = (
                    trade.get("win_probability")
                )
                STATE["active_trade"]["expected_value_at_entry"] = (
                    trade.get("expected_value")
                )
                STATE["active_trade"]["expected_value_per_risk"] = (
                    trade.get("expected_value_per_risk")
                )
                STATE["active_trade"]["risk_pct_at_entry"] = (
                    trade.get("risk_pct")
                )
                STATE["active_trade"]["risk_reward"] = (
                    trade.get("risk_reward")
                )
                STATE["active_trade"]["meta_probability_at_entry"] = (
                    meta_decision.get("probability")
                )
                STATE["active_trade"]["meta_model_available"] = (
                    meta_decision.get("available", False)
                )
                STATE["active_trade"]["barrier_target"] = (
                    trade.get("barrier_target", trade.get("target"))
                )
                STATE["active_trade"]["barrier_stop"] = (
                    trade.get("barrier_stop", trade.get("stop"))
                )
                STATE["active_trade"]["barrier_expiry"] = (
                    trade.get("barrier_expiry")
                )
                STATE["active_trade"]["barrier_volatility"] = (
                    trade.get("barrier_volatility")
                )

                try:
                    database.save_active_trade(
                        STATE["active_trade"]
                    )
                except Exception as exc:
                    log(
                        "Database active-trade save failed: "
                        + str(exc)
                    )

            break


persist_state()

# ============================================================
# STARTUP
# ============================================================

def startup():
    restore_persistent_state()
    capital = get_capital()

    enabled = [
        name
        for name, cfg in INSTRUMENTS.items()
        if cfg.get("enabled", False)
    ]

    log("=" * 60)
    log("ATHENA-X STARTING")
    log("=" * 60)
    log(f"Mode: {'LIVE' if LIVE_TRADING else 'PAPER'}")
    log(f"Capital: ₹{capital:,.2f}")
    log(f"XGBoost trained: {ml_engine.is_trained}")
    log(f"Enabled instruments: {enabled}")
    log(f"Trailing stop: {TRAILING_STOP_ENABLED}")
    log("=" * 60)

    telegram.send_status(
        "ATHENA-X STARTUP\n"
        f"Mode: {'LIVE' if LIVE_TRADING else 'PAPER'}\n"
        f"Capital: ₹{capital:,.2f}\n"
        f"XGBoost: {'READY' if ml_engine.is_trained else 'NOT TRAINED'}\n"
        f"Instruments: {enabled}"
    )


def main():
    startup()

    while True:
        try:
            run()
            persist_state()

        except KeyboardInterrupt:
            log("ATHENA-X stopped manually.")
            break

        except Exception as exc:
            error = "Main loop error: " + str(exc)
            log(error)
            telegram.send_error(error)
            time.sleep(30)

        if should_exit_market_hours():
            log("ATHENA-X stopping outside market hours.")
            break

        time.sleep(LOOP_SECONDS)


if __name__ == "__main__":
    main()