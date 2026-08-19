"""
ATHENA-X Logging
----------------
Console logging + structured trade history.

The logger is deliberately independent of broker/API code.
It should never be able to crash the trading loop because a log
message or one optional field is malformed.
"""

from __future__ import annotations

import csv
import json
import logging
from datetime import datetime
from pathlib import Path
from threading import Lock
from zoneinfo import ZoneInfo

IST = ZoneInfo("Asia/Kolkata")

LOG_DIR = Path("logs")
TRADE_LOG_FILE = LOG_DIR / "trades.csv"
APP_LOG_FILE = LOG_DIR / "athena.log"

TRADE_FIELDS = [
    "timestamp",
    "instrument",
    "symbol",
    "option_type",
    "strike",
    "security_id",
    "entry",
    "exit",
    "quantity",
    "pnl",
    "win",
    "exit_reason",
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
    "holding_minutes",
]

_FILE_LOCK = Lock()
_LOGGER = None


def _ensure_log_dir() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)


def _get_logger() -> logging.Logger:
    """Create the file logger once."""
    global _LOGGER

    if _LOGGER is not None:
        return _LOGGER

    _ensure_log_dir()

    logger = logging.getLogger("athena")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    if not logger.handlers:
        handler = logging.FileHandler(
            APP_LOG_FILE,
            encoding="utf-8",
        )
        handler.setFormatter(
            logging.Formatter(
                "[ATHENA %(asctime)s] %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        logger.addHandler(handler)

    _LOGGER = logger
    return logger


def _timestamp() -> str:
    return datetime.now(IST).strftime("%Y-%m-%d %H:%M:%S")


def log(message) -> None:
    """
    Log to stdout and the persistent application log.

    Logging failures are swallowed intentionally so the logger cannot
    stop Athena's trading loop.
    """
    timestamp = _timestamp()
    text = str(message)
    line = f"[ATHENA {timestamp}] {text}"

    try:
        print(line, flush=True)
    except Exception:
        pass

    try:
        _get_logger().info(text)
    except Exception:
        pass


def _normalise_trade(trade_data: dict) -> dict:
    """Convert a trade dictionary into the stable CSV schema."""
    if not isinstance(trade_data, dict):
        raise TypeError("trade_data must be a dictionary")

    row = {}

    # Preserve compatibility with both older and V2 field names.
    aliases = {
        "instrument": ["instrument", "name"],
        "symbol": ["symbol", "option_symbol"],
        "option_type": ["option_type"],
        "strike": ["strike"],
        "security_id": ["security_id"],
        "entry": ["entry", "entry_price"],
        "exit": ["exit", "exit_price"],
        "quantity": ["quantity", "qty"],
        "pnl": ["pnl"],
        "win": ["win"],
        "exit_reason": ["exit_reason", "reason"],
        "rsi_at_entry": ["rsi_at_entry", "rsi"],
        "ema_cross": ["ema_cross"],
        "vwap_ratio": ["vwap_ratio"],
        "atr_percent": ["atr_percent"],
        "volume_ratio": ["volume_ratio"],
        "delta": ["delta"],
        "oi_at_entry": ["oi_at_entry", "oi"],
        "market_score": ["market_score", "score"],
        "distance_from_high": ["distance_from_high"],
        "distance_from_low": ["distance_from_low"],
        "holding_minutes": ["holding_minutes"],
    }

    for field in TRADE_FIELDS:
        if field == "timestamp":
            row[field] = trade_data.get(
                "timestamp",
                datetime.now(IST).isoformat(),
            )
            continue

        value = None

        for key in aliases.get(field, [field]):
            if key in trade_data:
                value = trade_data[key]
                break

        row[field] = value

    return row


def _append_trade_csv(row: dict) -> None:
    """Append one normalized trade to the persistent CSV history."""
    _ensure_log_dir()

    with _FILE_LOCK:
        file_exists = TRADE_LOG_FILE.exists()
        needs_header = (
            not file_exists
            or TRADE_LOG_FILE.stat().st_size == 0
        )

        with TRADE_LOG_FILE.open(
            "a",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=TRADE_FIELDS,
                extrasaction="ignore",
            )

            if needs_header:
                writer.writeheader()

            writer.writerow(row)


class AthenaLogger:
    """Structured in-memory + persistent trade logger."""

    def __init__(self):
        self.trades = []

    def log_trade(self, trade_data):
        """
        Normalize, retain and persist a completed trade.

        Returns the normalized trade row on success.
        """
        try:
            row = _normalise_trade(trade_data)
            self.trades.append(dict(row))
            _append_trade_csv(row)

            log(
                "TRADE LOGGED: "
                + str(row.get("instrument") or row.get("symbol"))
                + " "
                + str(row.get("option_type"))
                + " "
                + str(row.get("strike"))
                + " | P&L="
                + str(row.get("pnl"))
            )

            return row

        except Exception as exc:
            # Logging must never take down the trading engine.
            log("TRADE LOGGING ERROR: " + str(exc))
            return None

    def get_trades(self):
        """Return a copy of the current in-memory trade history."""
        return list(self.trades)

    def get_trade_count(self):
        return len(self.trades)

    def clear_memory(self):
        """Clear only in-memory trades; persistent CSV remains untouched."""
        self.trades.clear()

    def export_json(self, filepath=None):
        """Export the current in-memory trade history as JSON."""
        path = (
            Path(filepath)
            if filepath is not None
            else LOG_DIR / "trades_export.json"
        )

        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as handle:
            json.dump(
                self.trades,
                handle,
                indent=2,
                default=str,
            )

        return path


# Backward-compatible helper for code that imports `logger`.
logger = AthenaLogger()
