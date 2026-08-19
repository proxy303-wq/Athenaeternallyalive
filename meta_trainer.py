"""
Athena-X Meta Model Training Pipeline

Trains the secondary XGBoost only from completed, real trade records.

Important:
- chronological split, never random shuffle
- training rows must precede validation rows
- no synthetic labels
- minimum class-balance checks
- evaluation is separate from activation
"""

from __future__ import annotations

from pathlib import Path
import json
import math
from typing import Optional

import numpy as np

from meta_dataset import FEATURE_NAMES, build_from_history
from meta_label_engine import MetaLabelEngine


class MetaTrainingError(RuntimeError):
    pass


def _metrics(y_true, probabilities, threshold=0.60):
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    predictions = (probabilities >= threshold).astype(int)

    tp = int(((predictions == 1) & (y_true == 1)).sum())
    tn = int(((predictions == 0) & (y_true == 0)).sum())
    fp = int(((predictions == 1) & (y_true == 0)).sum())
    fn = int(((predictions == 0) & (y_true == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    accuracy = (tp + tn) / len(y_true) if len(y_true) else 0.0

    return {
        "rows": int(len(y_true)),
        "positive": int(y_true.sum()),
        "negative": int(len(y_true) - y_true.sum()),
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "true_positive": tp,
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
    }


def _roc_auc(y_true, probabilities):
    """Small dependency-light AUC implementation."""
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)

    positives = probabilities[y_true == 1]
    negatives = probabilities[y_true == 0]

    if len(positives) == 0 or len(negatives) == 0:
        return None

    wins = 0.0
    ties = 0.0

    for pos in positives:
        wins += float((pos > negatives).sum())
        ties += float((pos == negatives).sum())

    return (wins + 0.5 * ties) / (len(positives) * len(negatives))


def prepare_rows(history):
    rows, skipped = build_from_history(history)

    if not rows:
        raise MetaTrainingError(
            "No qualifying Triple-Barrier trades are available."
        )

    return rows, skipped


def chronological_split(rows, validation_fraction=0.25):
    rows = list(rows)

    if len(rows) < 20:
        raise MetaTrainingError(
            f"Need at least 20 qualifying trades; found {len(rows)}."
        )

    split = int(len(rows) * (1.0 - validation_fraction))
    split = max(1, min(split, len(rows) - 1))

    train = rows[:split]
    validation = rows[split:]

    if len({row["label"] for row in train}) < 2:
        raise MetaTrainingError(
            "Training portion contains only one class."
        )

    if len({row["label"] for row in validation}) < 2:
        raise MetaTrainingError(
            "Validation portion contains only one class."
        )

    return train, validation


def train_from_history(
    history,
    model_path="data/meta_label_model.joblib",
    validation_fraction=0.25,
    threshold=0.60,
):
    rows, skipped = prepare_rows(history)
    train_rows, validation_rows = chronological_split(
        rows,
        validation_fraction,
    )

    engine = MetaLabelEngine(
        model_path=model_path,
        threshold=threshold,
    )

    if not engine.available():
        raise MetaTrainingError(
            "XGBoost is unavailable. Install xgboost before training."
        )

    train_labels = [row["label"] for row in train_rows]

    engine.fit(
        train_rows,
        train_labels,
    )

    validation_labels = [
        row["label"] for row in validation_rows
    ]
    validation_probabilities = [
        engine.predict_probability(row)
        for row in validation_rows
    ]

    metrics = _metrics(
        validation_labels,
        validation_probabilities,
        threshold,
    )
    metrics["roc_auc"] = _roc_auc(
        validation_labels,
        validation_probabilities,
    )

    metrics["activation_eligible"] = bool(
        metrics["roc_auc"] is not None
        and metrics["roc_auc"] >= 0.55
        and metrics["precision"] >= 0.55
    )

    return {
        "engine": engine,
        "metrics": metrics,
        "total_rows": len(rows),
        "skipped_rows": skipped,
        "train_rows": len(train_rows),
        "validation_rows": len(validation_rows),
        "validation_start": validation_rows[0].get("timestamp"),
    }


def save_training_report(result, path="data/meta_training_report.json"):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        key: value
        for key, value in result.items()
        if key != "engine"
    }

    path.write_text(
        json.dumps(report, indent=2, default=str),
        encoding="utf-8",
    )

    return path
