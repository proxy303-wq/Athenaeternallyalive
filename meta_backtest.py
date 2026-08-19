"""
Athena-X Meta Filter Evaluation

Given already completed validation candidates with:
    primary_probability
    label

compare:
    primary signal
vs
    primary signal + meta approval

This is deliberately separate from model training.
"""

from __future__ import annotations


def evaluate_primary_vs_meta(
    rows,
    meta_engine,
    primary_threshold=0.65,
    meta_threshold=0.60,
):
    primary = []
    filtered = []

    for row in rows:
        probability = float(row.get("primary_probability", 0.0))
        label = int(row["label"])

        if probability < primary_threshold:
            continue

        primary.append(label)

        prediction = meta_engine.predict_probability(row)

        if prediction is not None and prediction >= meta_threshold:
            filtered.append(label)

    def summary(labels):
        if not labels:
            return {
                "trades": 0,
                "wins": 0,
                "losses": 0,
                "win_rate": 0.0,
            }

        wins = sum(labels)
        return {
            "trades": len(labels),
            "wins": wins,
            "losses": len(labels) - wins,
            "win_rate": wins / len(labels),
        }

    return {
        "primary": summary(primary),
        "meta_filtered": summary(filtered),
    }
