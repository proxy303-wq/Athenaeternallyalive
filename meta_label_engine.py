"""
Athena-X Meta Labeling Engine

Primary model:
    Existing Athena XGBoost -> trade candidate / probability

Meta model:
    Learns whether an already-approved candidate should actually be taken.

Training labels should come from triple-barrier outcomes:
    1 = take-profit barrier reached first
    0 = stop-loss / time barrier reached first
"""

from __future__ import annotations

from pathlib import Path
import json
from typing import Iterable, Optional

import joblib
import numpy as np

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None


class MetaLabelEngine:
    def __init__(
        self,
        model_path: str = "data/meta_label_model.joblib",
        threshold: float = 0.60,
    ):
        self.model_path = Path(model_path)
        self.activation_path = self.model_path.with_suffix(".active.json")
        self.threshold = float(threshold)
        self.model = None
        # Entry-time features only. Do NOT add exit/holding-time fields here;
        # those are only known after the outcome and would leak the label.
        self.feature_names = [
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

    def available(self) -> bool:
        return XGBClassifier is not None

    def _matrix(self, rows):
        matrix = []
        for row in rows:
            matrix.append([
                float(row.get(name, 0.0) or 0.0)
                for name in self.feature_names
            ])
        return np.asarray(matrix, dtype=float)

    def fit(
        self,
        rows: Iterable[dict],
        labels: Iterable[int],
        n_estimators: int = 250,
        max_depth: int = 4,
        learning_rate: float = 0.04,
        subsample: float = 0.85,
        colsample_bytree: float = 0.85,
        random_state: int = 42,
    ):
        if XGBClassifier is None:
            raise RuntimeError(
                "xgboost is not installed; cannot train meta-label model."
            )

        rows = list(rows)
        labels = np.asarray(list(labels), dtype=int)

        if len(rows) != len(labels):
            raise ValueError("rows and labels must have equal length.")

        if len(rows) < 20:
            raise ValueError(
                "At least 20 labeled candidates are required before training."
            )

        unique = np.unique(labels)
        if len(unique) < 2:
            raise ValueError(
                "Meta-label training requires both positive and negative labels."
            )

        X = self._matrix(rows)

        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=random_state,
        )

        self.model.fit(X, labels)

        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "feature_names": self.feature_names,
                "threshold": self.threshold,
            },
            self.model_path,
        )

        return self.model

    def load(self) -> bool:
        # A trained artifact alone is NOT sufficient for live use.
        # It must have an explicit activation record created after
        # validation has been reviewed.
        if not self.model_path.exists():
            return False

        if not self.activation_path.exists():
            return False

        try:
            activation = json.loads(
                self.activation_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError):
            return False

        if activation.get("active") is not True:
            return False

        payload = joblib.load(self.model_path)
        self.model = payload["model"]
        self.feature_names = payload.get(
            "feature_names",
            self.feature_names,
        )
        self.threshold = float(
            payload.get("threshold", self.threshold)
        )
        return True

    def predict_probability(self, row: dict) -> Optional[float]:
        if self.model is None and not self.load():
            return None

        X = self._matrix([row])
        return float(self.model.predict_proba(X)[0][1])

    def approve(self, row: dict) -> dict:
        probability = self.predict_probability(row)

        if probability is None:
            # No trained meta-model means no automatic veto.
            # This preserves the current Athena behavior until enough
            # real labeled history exists.
            return {
                "available": False,
                "probability": None,
                "approved": True,
                "reason": "META_MODEL_NOT_TRAINED",
            }

        approved = probability >= self.threshold

        return {
            "available": True,
            "probability": probability,
            "approved": approved,
            "reason": (
                "META_APPROVED"
                if approved
                else "META_REJECTED"
            ),
        }

    def activate(self, validation_metrics: dict, min_auc: float = 0.55):
        auc = validation_metrics.get("roc_auc")
        if auc is None or float(auc) < float(min_auc):
            raise ValueError(
                f"Meta model failed activation threshold: ROC-AUC={auc}"
            )

        self.activation_path.parent.mkdir(parents=True, exist_ok=True)
        self.activation_path.write_text(
            json.dumps({
                "active": True,
                "roc_auc": float(auc),
                "precision": float(
                    validation_metrics.get("precision", 0.0)
                ),
                "activated_at": __import__("datetime").datetime.now().isoformat(),
            }, indent=2),
            encoding="utf-8",
        )

    def deactivate(self):
        if self.activation_path.exists():
            self.activation_path.unlink()

    def status(self):
        return {
            "model_loaded": self.model is not None,
            "threshold": self.threshold,
            "model_path": str(self.model_path),
            "feature_count": len(self.feature_names),
        }

    def feature_importance(self):
        if self.model is None:
            return {}

        values = self.model.feature_importances_

        return {
            name: float(value)
            for name, value in zip(
                self.feature_names,
                values,
            )
        }
