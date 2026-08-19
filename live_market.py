"""
ATHENA-X Dhan Live Market Feed
--------------------------------
Read-only live market data for NIFTY and BANKNIFTY.

Uses DhanHQ MarketFeed WebSocket in a background daemon thread.
No order API is called here.
"""

from __future__ import annotations

import threading
import time
from datetime import datetime
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")

NIFTY_SECURITY_ID = "13"
BANKNIFTY_SECURITY_ID = "25"
EXCHANGE_SEGMENT = "IDX_I"

_state = {
    "NIFTY": {
        "security_id": NIFTY_SECURITY_ID,
        "ltp": None,
        "previous_close": None,
        "timestamp": None,
    },
    "BANKNIFTY": {
        "security_id": BANKNIFTY_SECURITY_ID,
        "ltp": None,
        "previous_close": None,
        "timestamp": None,
    },
}

_lock = threading.Lock()
_worker_started = False
_worker_thread = None
_error = None


def _update_from_packet(packet):
    if not isinstance(packet, dict):
        return

    # DhanHQ-py returns normalized dictionaries from MarketFeed.
    # Keep parsing deliberately defensive because packet shapes can
    # differ between feed modes/library versions.
    data = packet.get("data", packet)

    if isinstance(data, list):
        packets = data
    else:
        packets = [data]

    for item in packets:
        if not isinstance(item, dict):
            continue

        security_id = str(
            item.get("security_id")
            or item.get("securityId")
            or item.get("securityID")
            or ""
        )

        if security_id == NIFTY_SECURITY_ID:
            symbol = "NIFTY"
        elif security_id == BANKNIFTY_SECURITY_ID:
            symbol = "BANKNIFTY"
        else:
            # Some SDK versions expose exchange/security as nested fields.
            security_id = str(
                item.get("securityId")
                or item.get("security_id")
                or ""
            )
            if security_id == NIFTY_SECURITY_ID:
                symbol = "NIFTY"
            elif security_id == BANKNIFTY_SECURITY_ID:
                symbol = "BANKNIFTY"
            else:
                continue

        ltp = (
            item.get("LTP")
            if item.get("LTP") is not None
            else item.get("ltp")
        )

        prev_close = (
            item.get("PrevClose")
            if item.get("PrevClose") is not None
            else item.get("prev_close")
        )

        if ltp is None:
            continue

        with _lock:
            _state[symbol]["ltp"] = float(ltp)

            if prev_close is not None:
                _state[symbol]["previous_close"] = float(prev_close)

            _state[symbol]["timestamp"] = datetime.now(IST)


def _worker():
    global _error

    try:
        from config import CLIENT_ID, ACCESS_TOKEN
        from dhanhq import DhanContext, MarketFeed

        if not CLIENT_ID or not ACCESS_TOKEN:
            raise RuntimeError("Dhan credentials unavailable.")

        context = DhanContext(CLIENT_ID, ACCESS_TOKEN)

        instruments = [
            (EXCHANGE_SEGMENT, NIFTY_SECURITY_ID, MarketFeed.Ticker),
            (EXCHANGE_SEGMENT, BANKNIFTY_SECURITY_ID, MarketFeed.Ticker),
        ]

        feed = MarketFeed(
            context,
            instruments,
            "v2",
        )

        while True:
            try:
                # DhanHQ's documented synchronous flow establishes the
                # connection with run_forever(), then receives packets
                # through get_data().
                feed.run_forever()

                while True:
                    packet = feed.get_data()

                    if packet is not None:
                        _update_from_packet(packet)

            except Exception as exc:
                _error = str(exc)

                try:
                    feed.close_connection()
                except Exception:
                    pass

                time.sleep(5)

    except Exception as exc:
        _error = str(exc)


def start_live_feed():
    """Start one background WebSocket worker and return immediately."""
    global _worker_started, _worker_thread

    if _worker_started:
        return

    _worker_started = True

    _worker_thread = threading.Thread(
        target=_worker,
        name="athena-dhan-marketfeed",
        daemon=True,
    )
    _worker_thread.start()


def get_market_snapshot():
    """Return a safe copy of the latest market state."""
    with _lock:
        snapshot = {
            symbol: values.copy()
            for symbol, values in _state.items()
        }

    return {
        "data": snapshot,
        "error": _error,
        "connected": any(
            values["timestamp"] is not None
            for values in snapshot.values()
        ),
    }