"""
ATHENA-X Live Market Feed
-------------------------
Read-only live market data for NIFTY and BANKNIFTY.

Uses DhanHQ MarketFeed WebSocket in a background daemon thread.

Important:
- This module does NOT place orders.
- WebSocket failures must not stop Athena.
- HTTP 429 responses trigger progressively longer reconnect delays.
- Only one WebSocket worker is started per Python process.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime

from config import (
    BANKNIFTY_EXCHANGE_SEGMENT,
    BANKNIFTY_SECURITY_ID,
    NIFTY_EXCHANGE_SEGMENT,
    NIFTY_SECURITY_ID,
)


# ============================================================
# TIMEZONE
# ============================================================

try:
    from zoneinfo import ZoneInfo

    IST = ZoneInfo("Asia/Kolkata")

except Exception:
    IST = None


# ============================================================
# SHARED STATE
# ============================================================

_lock = threading.Lock()

_worker_started = False
_worker_thread = None

_error = None

_state = {
    "NIFTY": {
        "ltp": None,
        "previous_close": None,
        "timestamp": None,
    },
    "BANKNIFTY": {
        "ltp": None,
        "previous_close": None,
        "timestamp": None,
    },
}


# ============================================================
# HELPERS
# ============================================================

def _now():
    """Return current IST timestamp."""
    if IST is not None:
        return datetime.now(IST)

    return datetime.now()


def _safe_float(value, default=None):
    """Safely convert a value to float."""

    try:
        if value is None:
            return default

        result = float(value)

        if result != result:
            return default

        return result

    except (TypeError, ValueError):
        return default


def _normalise_symbol(value):
    """
    Convert a feed symbol/name into Athena's internal symbol.
    """

    if value is None:
        return None

    text = str(value).upper().strip()

    if "BANKNIFTY" in text:
        return "BANKNIFTY"

    if "NIFTY" in text:
        return "NIFTY"

    return None


# ============================================================
# PACKET PARSER
# ============================================================

def _update_from_packet(packet):
    """
    Extract market data from a Dhan MarketFeed packet.

    Parsing is intentionally defensive because packet structures
    can differ between DhanHQ versions/feed modes.
    """

    if packet is None:
        return

    if not isinstance(packet, dict):
        return

    data = packet.get("data", packet)

    if not isinstance(data, dict):
        return

    # --------------------------------------------------------
    # Identify symbol
    # --------------------------------------------------------

    symbol = None

    possible_symbol_keys = (
        "symbol",
        "trading_symbol",
        "tradingSymbol",
        "security_id",
        "securityId",
        "securityID",
        "name",
    )

    for key in possible_symbol_keys:

        if key not in data:
            continue

        symbol = _normalise_symbol(data.get(key))

        if symbol:
            break

    # --------------------------------------------------------
    # Identify by security ID if symbol was not available
    # --------------------------------------------------------

    if symbol is None:

        security_id = (
            data.get("security_id")
            or data.get("securityId")
            or data.get("securityID")
        )

        if str(security_id) == str(NIFTY_SECURITY_ID):
            symbol = "NIFTY"

        elif str(security_id) == str(BANKNIFTY_SECURITY_ID):
            symbol = "BANKNIFTY"

    if symbol is None:
        return

    # --------------------------------------------------------
    # Extract LTP
    # --------------------------------------------------------

    ltp = None

    possible_ltp_keys = (
        "ltp",
        "LTP",
        "last_price",
        "lastPrice",
        "last_traded_price",
        "lastTradedPrice",
    )

    for key in possible_ltp_keys:

        if key not in data:
            continue

        ltp = _safe_float(data.get(key))

        if ltp is not None:
            break

    # --------------------------------------------------------
    # Extract previous close
    # --------------------------------------------------------

    previous_close = None

    possible_previous_close_keys = (
        "prev_close",
        "previous_close",
        "previous_close_price",
        "previousClose",
        "prevClose",
    )

    for key in possible_previous_close_keys:

        if key not in data:
            continue

        previous_close = _safe_float(data.get(key))

        if previous_close is not None:
            break

    # --------------------------------------------------------
    # Update shared state
    # --------------------------------------------------------

    with _lock:

        if symbol not in _state:
            return

        if ltp is not None and ltp > 0:
            _state[symbol]["ltp"] = ltp

        if previous_close is not None and previous_close > 0:
            _state[symbol]["previous_close"] = previous_close

        _state[symbol]["timestamp"] = _now()


# ============================================================
# WEBSOCKET WORKER
# ============================================================

def _worker():
    """
    Background Dhan WebSocket worker.

    Reconnection strategy:

        normal failure
            -> 10 second minimum delay

        HTTP 429 / rate limit
            -> 60 seconds
            -> 120 seconds
            -> 240 seconds
            -> maximum 300 seconds

    A successful connection resets the backoff.
    """

    global _error

    try:

        from config import CLIENT_ID, ACCESS_TOKEN
        from dhanhq import DhanContext, MarketFeed

        # ----------------------------------------------------
        # Credentials
        # ----------------------------------------------------

        if not CLIENT_ID or not ACCESS_TOKEN:
            raise RuntimeError(
                "Dhan credentials unavailable."
            )

        context = DhanContext(
            CLIENT_ID,
            ACCESS_TOKEN,
        )

        # ----------------------------------------------------
        # Correct instrument-specific segments
        # ----------------------------------------------------

        instruments = [
            (
                NIFTY_EXCHANGE_SEGMENT,
                NIFTY_SECURITY_ID,
                MarketFeed.Ticker,
            ),
            (
                BANKNIFTY_EXCHANGE_SEGMENT,
                BANKNIFTY_SECURITY_ID,
                MarketFeed.Ticker,
            ),
        ]

        # ----------------------------------------------------
        # Initial reconnect delay
        # ----------------------------------------------------

        reconnect_delay = 10

        # ----------------------------------------------------
        # Persistent reconnect loop
        # ----------------------------------------------------

        while True:

            feed = None

            try:

                # --------------------------------------------
                # Always create a fresh feed object after failure
                # --------------------------------------------

                feed = MarketFeed(
                    context,
                    instruments,
                    "v2",
                )

                # --------------------------------------------
                # Establish WebSocket connection
                # --------------------------------------------

                feed.run_forever()

                # --------------------------------------------
                # Connection established.
                # Reset retry delay.
                # --------------------------------------------

                reconnect_delay = 10

                with _lock:
                    _error = None

                # --------------------------------------------
                # Receive packets
                # --------------------------------------------

                while True:

                    packet = feed.get_data()

                    if packet is not None:
                        _update_from_packet(packet)

            except Exception as exc:

                error_text = str(exc)

                with _lock:
                    _error = error_text

                # --------------------------------------------
                # Safely close failed connection
                # --------------------------------------------

                try:

                    if feed is not None:
                        feed.close_connection()

                except Exception:
                    pass

                # --------------------------------------------
                # Detect rate limiting
                # --------------------------------------------

                lowered = error_text.lower()

                rate_limited = (
                    "429" in lowered
                    or "too many" in lowered
                    or "rate limit" in lowered
                    or "rate-limit" in lowered
                )

                # --------------------------------------------
                # Rate-limit backoff
                # --------------------------------------------

                if rate_limited:

                    reconnect_delay = min(
                        max(
                            reconnect_delay * 2,
                            60,
                        ),
                        300,
                    )

                # --------------------------------------------
                # Normal connection failure
                # --------------------------------------------

                else:

                    reconnect_delay = min(
                        max(
                            reconnect_delay,
                            10,
                        ),
                        60,
                    )

                # --------------------------------------------
                # Wait before reconnecting
                # --------------------------------------------

                time.sleep(reconnect_delay)

    except Exception as exc:

        with _lock:
            _error = str(exc)


# ============================================================
# START LIVE FEED
# ============================================================

def start_live_feed():
    """
    Start exactly one background WebSocket worker.

    Calling this function multiple times in the same process
    will not create additional workers.
    """

    global _worker_started
    global _worker_thread

    if _worker_started:
        return

    _worker_started = True

    _worker_thread = threading.Thread(
        target=_worker,
        name="athena-dhan-marketfeed",
        daemon=True,
    )

    _worker_thread.start()


# ============================================================
# MARKET SNAPSHOT
# ============================================================

def get_market_snapshot():
    """
    Return a safe copy of the latest market state.

    Returns:

        {
            "data": {
                "NIFTY": {
                    "ltp": ...,
                    "previous_close": ...,
                    "timestamp": ...
                },
                "BANKNIFTY": {
                    "ltp": ...,
                    "previous_close": ...,
                    "timestamp": ...
                }
            },
            "error": ...,
            "connected": True/False
        }
    """

    with _lock:

        snapshot = {
            symbol: dict(values)
            for symbol, values in _state.items()
        }

        error = _error

    connected = any(
        values["timestamp"] is not None
        for values in snapshot.values()
    )

    return {
        "data": snapshot,
        "error": error,
        "connected": connected,
    }