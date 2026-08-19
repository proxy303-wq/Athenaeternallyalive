"""
ATHENA-X Persistent Database
----------------------------
SQLite persistence for:
- completed trades
- ML training history
- application state
- active trade state

This module is intentionally independent of Dhan, Telegram and XGBoost.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from threading import Lock
from datetime import datetime

from config import DATABASE_PATH
from logger import log


class AthenaDatabase:
    def __init__(self, db_path=None):
        self.db_path = Path(db_path or DATABASE_PATH)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        self._initialise()

    def _connect(self):
        connection = sqlite3.connect(
            str(self.db_path),
            timeout=10,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    def _initialise(self):
        with self._lock:
            with self._connect() as conn:
                conn.executescript(
                    """
                    CREATE TABLE IF NOT EXISTS trades (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT NOT NULL,
                        instrument TEXT,
                        symbol TEXT,
                        option_type TEXT,
                        strike REAL,
                        security_id TEXT,
                        entry REAL,
                        exit REAL,
                        quantity INTEGER,
                        pnl REAL,
                        win INTEGER,
                        exit_reason TEXT,
                        rsi_at_entry REAL,
                        ema_cross REAL,
                        vwap_ratio REAL,
                        atr_percent REAL,
                        volume_ratio REAL,
                        delta REAL,
                        oi_at_entry REAL,
                        market_score REAL,
                        distance_from_high REAL,
                        distance_from_low REAL,
                        holding_minutes REAL,
                        regime TEXT,
                        win_probability_at_entry REAL,
                        expected_value_at_entry REAL,
                        expected_value_per_risk REAL,
                        risk_pct_at_entry REAL,
                        risk_reward REAL,
                        barrier_target REAL,
                        barrier_stop REAL,
                        barrier_volatility REAL,
                        barrier_expiry TEXT,
                        created_at TEXT NOT NULL
                    );

                    CREATE INDEX IF NOT EXISTS idx_trades_timestamp
                    ON trades(timestamp);

                    CREATE INDEX IF NOT EXISTS idx_trades_instrument
                    ON trades(instrument);

                    CREATE TABLE IF NOT EXISTS ml_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        trade_id INTEGER,
                        payload TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        FOREIGN KEY(trade_id)
                            REFERENCES trades(id)
                            ON DELETE SET NULL
                    );

                    CREATE TABLE IF NOT EXISTS app_state (
                        key TEXT PRIMARY KEY,
                        value TEXT,
                        updated_at TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS active_trade (
                        id INTEGER PRIMARY KEY CHECK (id = 1),
                        payload TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );
                    """
                )

                # Backward-compatible migration for databases created by
                # earlier Athena versions.
                existing = {
                    row["name"]
                    for row in conn.execute(
                        "PRAGMA table_info(trades)"
                    ).fetchall()
                }
                migrations = {
                    "regime": "TEXT",
                    "win_probability_at_entry": "REAL",
                    "expected_value_at_entry": "REAL",
                    "expected_value_per_risk": "REAL",
                    "risk_pct_at_entry": "REAL",
                    "risk_reward": "REAL",
                    "barrier_target": "REAL",
                    "barrier_stop": "REAL",
                    "barrier_volatility": "REAL",
                    "barrier_expiry": "TEXT",
                    "meta_probability_at_entry": "REAL",
                    "meta_model_available": "INTEGER",
                }
                for column, dtype in migrations.items():
                    if column not in existing:
                        conn.execute(
                            f"ALTER TABLE trades ADD COLUMN {column} {dtype}"
                        )

    @staticmethod
    def _now():
        return datetime.now().astimezone().isoformat()

    # ========================================================
    # TRADES
    # ========================================================

    def save_trade(self, trade_data):
        """Persist one completed trade and return its database ID."""
        if not isinstance(trade_data, dict):
            raise TypeError("trade_data must be a dictionary")

        fields = [
            "timestamp", "instrument", "symbol", "option_type",
            "strike", "security_id", "entry", "exit", "quantity",
            "pnl", "win", "exit_reason", "rsi_at_entry", "ema_cross",
            "vwap_ratio", "atr_percent", "volume_ratio", "delta",
            "oi_at_entry", "market_score", "distance_from_high",
            "distance_from_low", "holding_minutes",
            "regime", "win_probability_at_entry",
            "expected_value_at_entry", "expected_value_per_risk",
            "risk_pct_at_entry", "risk_reward",
            "barrier_target", "barrier_stop",
            "barrier_volatility", "barrier_expiry",
            "meta_probability_at_entry", "meta_model_available",
        ]

        values = [
            trade_data.get(field)
            for field in fields
        ]

        # SQLite stores booleans as integers.
        win_index = fields.index("win")
        if values[win_index] is not None:
            values[win_index] = int(bool(values[win_index]))

        with self._lock:
            with self._connect() as conn:
                cursor = conn.execute(
                    """
                    INSERT INTO trades (
                        timestamp, instrument, symbol, option_type,
                        strike, security_id, entry, exit, quantity,
                        pnl, win, exit_reason, rsi_at_entry,
                        ema_cross, vwap_ratio, atr_percent,
                        volume_ratio, delta, oi_at_entry,
                        market_score, distance_from_high,
                        distance_from_low, holding_minutes,
                        regime, win_probability_at_entry,
                        expected_value_at_entry, expected_value_per_risk,
                        risk_pct_at_entry, risk_reward,
                        barrier_target, barrier_stop,
                        barrier_volatility, barrier_expiry,
                        meta_probability_at_entry,
                        meta_model_available, created_at
                    )
                    VALUES (
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?, ?, ?
                    )
                    """,
                    values + [self._now()],
                )

                trade_id = cursor.lastrowid

                # Keep the exact feature payload used for ML.
                self._save_ml_history_locked(
                    conn,
                    trade_id,
                    trade_data,
                )

                return trade_id

    def _save_ml_history_locked(self, conn, trade_id, trade_data):
        # Store the entry-time feature snapshot exactly as JSON so future
        # dataset rebuilds do not depend on today's Python schema.
        payload = {
            "ml_features": trade_data.get("ml_features", {}),
            "win_probability_at_entry": trade_data.get(
                "win_probability_at_entry"
            ),
            "expected_value_at_entry": trade_data.get(
                "expected_value_at_entry"
            ),
            "expected_value_per_risk": trade_data.get(
                "expected_value_per_risk"
            ),
            "risk_pct_at_entry": trade_data.get(
                "risk_pct_at_entry"
            ),
            "risk_reward": trade_data.get("risk_reward"),
            "regime": trade_data.get("regime"),
            "barrier_target": trade_data.get("barrier_target"),
            "barrier_stop": trade_data.get("barrier_stop"),
            "barrier_volatility": trade_data.get(
                "barrier_volatility"
            ),
            "meta_probability_at_entry": trade_data.get(
                "meta_probability_at_entry"
            ),
            "meta_model_available": trade_data.get(
                "meta_model_available"
            ),
            "barrier_expiry": trade_data.get("barrier_expiry"),
            "meta_probability_at_entry": trade_data.get(
                "meta_probability_at_entry"
            ),
            "meta_model_available": trade_data.get(
                "meta_model_available"
            ),
        }

        conn.execute(
            """
            INSERT INTO ml_history (
                trade_id, payload, created_at
            )
            VALUES (?, ?, ?)
            """,
            (
                trade_id,
                json.dumps(payload, default=str),
                self._now(),
            ),
        )


    def get_trades(self, limit=None):
        """Return completed trades in chronological order."""
        query = "SELECT * FROM trades ORDER BY id ASC"
        params = ()

        if limit is not None:
            query += " LIMIT ?"
            params = (int(limit),)

        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()

        return [dict(row) for row in rows]

    def get_ml_history(self):
        """Return the exact historical trade records suitable for ML."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT t.*
                FROM trades t
                INNER JOIN ml_history m
                    ON m.trade_id = t.id
                ORDER BY t.id ASC
                """
            ).fetchall()

        return [dict(row) for row in rows]

    def get_meta_training_history(self):
        """
        Return completed trades enriched with their exact entry-time
        feature payload. No outcome is altered here.
        """
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT t.*, m.payload AS ml_payload
                FROM trades t
                INNER JOIN ml_history m
                    ON m.trade_id = t.id
                ORDER BY t.id ASC
                """
            ).fetchall()

        result = []
        for row in rows:
            item = dict(row)
            payload = item.pop("ml_payload", None)
            try:
                payload = json.loads(payload) if payload else {}
            except json.JSONDecodeError:
                payload = {}

            ml_features = payload.get("ml_features", {})
            if isinstance(ml_features, dict):
                item["ml_features"] = ml_features

            for key in (
                "win_probability_at_entry",
                "expected_value_at_entry",
                "expected_value_per_risk",
                "risk_pct_at_entry",
                "risk_reward",
                "regime",
            ):
                if item.get(key) is None and key in payload:
                    item[key] = payload[key]

            result.append(item)

        return result

    def get_trade_count(self):
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS count FROM trades"
            ).fetchone()

        return int(row["count"])

    def get_win_loss(self):
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT
                    COALESCE(SUM(CASE WHEN win = 1 THEN 1 ELSE 0 END), 0)
                        AS wins,
                    COALESCE(SUM(CASE WHEN win = 0 THEN 1 ELSE 0 END), 0)
                        AS losses
                FROM trades
                """
            ).fetchone()

        return {
            "wins": int(row["wins"]),
            "losses": int(row["losses"]),
        }

    def get_pnl(self):
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT COALESCE(SUM(pnl), 0) AS pnl
                FROM trades
                """
            ).fetchone()

        return float(row["pnl"])

    # ========================================================
    # ACTIVE TRADE
    # ========================================================

    def save_active_trade(self, trade):
        """Persist the current open trade, replacing the prior state."""
        with self._lock:
            with self._connect() as conn:
                if trade is None:
                    conn.execute(
                        "DELETE FROM active_trade WHERE id = 1"
                    )
                    return

                conn.execute(
                    """
                    INSERT INTO active_trade (
                        id, payload, updated_at
                    )
                    VALUES (1, ?, ?)
                    ON CONFLICT(id) DO UPDATE SET
                        payload = excluded.payload,
                        updated_at = excluded.updated_at
                    """,
                    (
                        json.dumps(trade, default=str),
                        self._now(),
                    ),
                )

    def get_active_trade(self):
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT payload
                FROM active_trade
                WHERE id = 1
                """
            ).fetchone()

        if not row:
            return None

        try:
            return json.loads(row["payload"])
        except json.JSONDecodeError:
            log("Database: invalid active trade JSON.")
            return None

    def clear_active_trade(self):
        self.save_active_trade(None)

    # ========================================================
    # GENERIC APPLICATION STATE
    # ========================================================

    def set_state(self, key, value):
        """Persist a JSON-serializable application state value."""
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    """
                    INSERT INTO app_state (
                        key, value, updated_at
                    )
                    VALUES (?, ?, ?)
                    ON CONFLICT(key) DO UPDATE SET
                        value = excluded.value,
                        updated_at = excluded.updated_at
                    """,
                    (
                        str(key),
                        json.dumps(value, default=str),
                        self._now(),
                    ),
                )

    def get_state(self, key, default=None):
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT value
                FROM app_state
                WHERE key = ?
                """,
                (str(key),),
            ).fetchone()

        if not row:
            return default

        try:
            return json.loads(row["value"])
        except json.JSONDecodeError:
            return default

    def delete_state(self, key):
        with self._lock:
            with self._connect() as conn:
                conn.execute(
                    "DELETE FROM app_state WHERE key = ?",
                    (str(key),),
                )


# Singleton used by Athena.
db = AthenaDatabase()
