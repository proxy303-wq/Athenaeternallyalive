"""
Athena-X real trade-history -> meta-label dataset.

A completed trade becomes a meta-label example only when its exit reason
is one of Athena's explicit Triple-Barrier events:
    TAKE_PROFIT -> 1
    STOP_LOSS   -> 0
    TIME_EXIT   -> 0

Other exits are retained in the database but are NOT silently converted
into training labels.
"""

from __future__ import annotations

from pathlib import Path
import csv
import json

FEATURE_NAMES = [
    "primary_probability",
    "delta_abs",
    "atr_percent",
    "volume_ratio",
    "market_score",
    "distance_from_high",
    "distance_from_low",
    "expected_value_per_risk",
    "risk_pct_at_entry",
    "regime_score",
    "risk_reward",
]


def _f(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def regime_score(regime):
    return {
        "TRENDING": 1.00,
        "NORMAL": 0.75,
        "RANGING": 0.50,
        "LOW_VOL_RANGE": 0.40,
        "HIGH_VOL": 0.30,
    }.get(str(regime or "").upper(), 0.0)


def from_trade_record(trade: dict):
    """
    Convert one completed trade into a meta row.

    Returns None when the trade did not end through a supported
    triple-barrier event.
    """
    event = str(trade.get("exit_reason", "")).upper()

    if event == "TAKE_PROFIT":
        label = 1
    elif event in {"STOP_LOSS", "TIME_EXIT"}:
        label = 0
    else:
        return None

    ml = trade.get("ml_features")
    if isinstance(ml, str):
        try:
            ml = json.loads(ml)
        except json.JSONDecodeError:
            ml = {}
    if not isinstance(ml, dict):
        ml = {}

    delta = trade.get("delta", ml.get("delta", 0.0))

    return {
        "primary_probability": _f(
            trade.get("win_probability_at_entry")
        ),
        "delta_abs": abs(_f(delta)),
        "atr_percent": _f(
            trade.get("atr_percent", ml.get("atr_percent"))
        ),
        "volume_ratio": _f(
            trade.get("volume_ratio", ml.get("volume_ratio"))
        ),
        "market_score": _f(
            trade.get("market_score", ml.get("market_score"))
        ),
        "distance_from_high": _f(
            trade.get(
                "distance_from_high",
                ml.get("distance_from_high"),
            )
        ),
        "distance_from_low": _f(
            trade.get(
                "distance_from_low",
                ml.get("distance_from_low"),
            )
        ),
        "expected_value_per_risk": _f(
            trade.get("expected_value_per_risk")
        ),
        "risk_pct_at_entry": _f(
            trade.get("risk_pct_at_entry")
        ),
        "regime_score": regime_score(
            trade.get("regime")
        ),
        "risk_reward": _f(
            trade.get("risk_reward")
        ),
        "label": label,
        "barrier_event": event,
    }


def build_from_history(history):
    rows = []
    skipped = 0

    for trade in history:
        row = from_trade_record(trade)
        if row is None:
            skipped += 1
            continue
        rows.append(row)

    return rows, skipped


def write_csv(rows, path="data/meta_training.csv"):
    rows = list(rows)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fields = FEATURE_NAMES + ["label", "barrier_event"]

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                field: row.get(field, "")
                for field in fields
            })

    return path
