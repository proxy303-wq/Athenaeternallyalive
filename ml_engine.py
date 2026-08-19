"""
ATHENA-X XGBoost ML Engine

The ML layer is a filter/decision-support component.
It never places orders and it never overrides hard risk controls.
"""

import json
import os
import numpy as np

from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import StandardScaler
from logger import log

try:
    from database import db
except Exception:
    db = None

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    xgb = None
    XGBOOST_AVAILABLE = False


FEATURE_NAMES = [
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

DISPLAY_NAMES = {
    "rsi_at_entry": "RSI",
    "ema_cross": "EMA_Cross",
    "vwap_ratio": "VWAP_Ratio",
    "atr_percent": "ATR%",
    "volume_ratio": "Volume_Ratio",
    "delta": "Delta",
    "oi_at_entry": "OI",
    "market_score": "Market_Score",
    "distance_from_high": "Dist_High",
    "distance_from_low": "Dist_Low",
}

try:
    from config import (
        ML_ENABLED,
        ML_MIN_TRADES_TO_TRAIN,
        ML_RETRAIN_INTERVAL,
        ML_YES_THRESHOLD,
        ML_MAYBE_THRESHOLD,
        XGB_N_ESTIMATORS,
        XGB_MAX_DEPTH,
        XGB_LEARNING_RATE,
        XGB_SUBSAMPLE,
        XGB_COLSAMPLE_BYTREE,
        XGB_RANDOM_STATE,
        MODEL_PATH,
        SCALER_PATH,
    )
except ImportError:
    ML_ENABLED = True
    ML_MIN_TRADES_TO_TRAIN = 10
    ML_RETRAIN_INTERVAL = 10
    ML_YES_THRESHOLD = 0.65
    ML_MAYBE_THRESHOLD = 0.50
    XGB_N_ESTIMATORS = 100
    XGB_MAX_DEPTH = 6
    XGB_LEARNING_RATE = 0.1
    XGB_SUBSAMPLE = 0.8
    XGB_COLSAMPLE_BYTREE = 0.8
    XGB_RANDOM_STATE = 42
    MODEL_PATH = "athena_xgb_model.json"
    SCALER_PATH = "athena_scaler.pkl"


from triple_barrier import label_path

class AthenaML:
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.is_trained = False
        self.trade_history = []
        self.trades_learned = 0
        self.model_type = "XGBoost"
        self.feature_importance = {}
        self.training_stats = {}
        self.last_train_trade_count = 0

        if ML_ENABLED and XGBOOST_AVAILABLE:
            self.load_trade_history_from_database()
            self.load_model()
        elif ML_ENABLED and not XGBOOST_AVAILABLE:
            log("ML: XGBoost is not installed. ML remains inactive.")

    # --------------------------------------------------------
    # Persistent history
    # --------------------------------------------------------

    def load_trade_history_from_database(self):
        """
        Restore completed trades from SQLite into the ML history.

        Database failure never blocks Athena startup; the ML layer simply
        remains unhydrated until new trades arrive.
        """
        if db is None:
            log("ML: Database unavailable; history not restored.")
            return 0

        try:
            history = db.get_ml_history()

            if not history:
                return 0

            self.trade_history = [dict(trade) for trade in history]
            self.trades_learned = len(self.trade_history)

            log(
                f"ML: Restored {self.trades_learned} "
                "completed trades from database."
            )

            return self.trades_learned

        except Exception as exc:
            log(
                "ML: Failed to restore database history - "
                + str(exc)
            )
            return 0

    # --------------------------------------------------------
    # Feature handling
    # --------------------------------------------------------

    def _feature_value(self, data, name):
        """Read a feature with backward-compatible aliases."""
        aliases = {
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
        }

        for key in aliases[name]:
            value = data.get(key)
            if value is not None:
                try:
                    value = float(value)
                    if np.isfinite(value):
                        return value
                except (TypeError, ValueError):
                    pass

        defaults = {
            "rsi_at_entry": 50.0,
            "ema_cross": 0.0,
            "vwap_ratio": 1.0,
            "atr_percent": 1.0,
            "volume_ratio": 1.0,
            "delta": 0.5,
            "oi_at_entry": 10000.0,
            "market_score": 70.0,
            "distance_from_high": 0.0,
            "distance_from_low": 0.0,
        }
        return defaults[name]

    def _to_vector(self, data):
        if not isinstance(data, dict):
            data = {}

        return np.asarray(
            [[self._feature_value(data, name) for name in FEATURE_NAMES]],
            dtype=float,
        )

    # --------------------------------------------------------
    # Training data
    # --------------------------------------------------------

    def add_trade_result(self, trade_data):
        """Add a CLOSED trade result; train only when enough data exists."""
        if not isinstance(trade_data, dict):
            log("ML: Ignoring invalid trade result.")
            return False

        if "win" not in trade_data:
            log("ML: Ignoring trade without a win/loss label.")
            return False

        self.trade_history.append(dict(trade_data))
        self.trades_learned = len(self.trade_history)

        log(
            f"ML: Added trade to history "
            f"({self.trades_learned} total)"
        )

        if self.trades_learned < ML_MIN_TRADES_TO_TRAIN:
            remaining = ML_MIN_TRADES_TO_TRAIN - self.trades_learned
            log(f"ML: Need {remaining} more trades to train")
            return False

        # Avoid retraining after every single trade.
        if (
            self.is_trained
            and self.trades_learned - self.last_train_trade_count
            < ML_RETRAIN_INTERVAL
        ):
            return True

        return self.train()

    def _build_training_data(self):
        X = []
        y = []

        for trade in self.trade_history:
            X.append([
                self._feature_value(trade, feature)
                for feature in FEATURE_NAMES
            ])
            y.append(1 if bool(trade.get("win", False)) else 0)

        return np.asarray(X, dtype=float), np.asarray(y, dtype=int)

    def train(self):
        """Train XGBoost only. No Random Forest fallback."""
        if not ML_ENABLED:
            log("ML: Disabled in configuration.")
            return False

        if not XGBOOST_AVAILABLE:
            log("ML: XGBoost unavailable; refusing to train another model.")
            return False

        if self.trades_learned < ML_MIN_TRADES_TO_TRAIN:
            log(
                f"ML: Need at least {ML_MIN_TRADES_TO_TRAIN} trades "
                f"(have {self.trades_learned})"
            )
            return False

        try:
            X, y = self._build_training_data()

            # XGBoost requires both classes to learn a classifier.
            unique_classes = np.unique(y)
            if len(unique_classes) < 2:
                log("ML: Training skipped — history contains only one class.")
                return False

            # Chronological validation: older trades train the model and
            # the newest trades are held out as genuinely unseen data.
            # This avoids randomly mixing future market regimes into training.
            split_index = int(len(X) * 0.80)
            split_index = max(1, min(split_index, len(X) - 1))

            X_train = X[:split_index]
            X_val = X[split_index:]
            y_train = y[:split_index]
            y_val = y[split_index:]

            if len(np.unique(y_train)) < 2:
                log(
                    "ML: Chronological training segment contains "
                    "only one class."
                )
                return False

            if len(np.unique(y_val)) < 2:
                log(
                    "ML: Chronological validation segment contains "
                    "only one class."
                )
                return False

            # Scaling is retained for compatibility with the existing
            # architecture. XGBoost itself does not require scaling.
            self.scaler.fit(X_train)
            X_train_scaled = self.scaler.transform(X_train)
            X_val_scaled = self.scaler.transform(X_val)

            log("ML: Training XGBoost...")

            model = xgb.XGBClassifier(
                n_estimators=XGB_N_ESTIMATORS,
                max_depth=XGB_MAX_DEPTH,
                learning_rate=XGB_LEARNING_RATE,
                subsample=XGB_SUBSAMPLE,
                colsample_bytree=XGB_COLSAMPLE_BYTREE,
                random_state=XGB_RANDOM_STATE,
                eval_metric="logloss",
                objective="binary:logistic",
                n_jobs=1,
            )

            model.fit(X_train_scaled, y_train)

            probabilities = model.predict_proba(X_val_scaled)[:, 1]
            predictions = (probabilities >= 0.50).astype(int)

            accuracy = float(accuracy_score(y_val, predictions))

            try:
                validation_logloss = float(
                    log_loss(y_val, probabilities, labels=[0, 1])
                )
            except ValueError:
                validation_logloss = None

            self.model = model
            self.is_trained = True
            self.model_type = "XGBoost"
            self.last_train_trade_count = self.trades_learned

            self.feature_importance = {
                DISPLAY_NAMES[name]: float(importance)
                for name, importance in zip(
                    FEATURE_NAMES,
                    model.feature_importances_,
                )
            }

            self.training_stats = {
                "trades_used": self.trades_learned,
                "train_samples": len(y_train),
                "validation_samples": len(y_val),
                "validation_method": "chronological_80_20",
                "accuracy": accuracy,
                "log_loss": validation_logloss,
                "model": "XGBoost",
                "success": True,
            }

            sorted_features = sorted(
                self.feature_importance.items(),
                key=lambda item: item[1],
                reverse=True,
            )

            log(f"ML: XGBoost Accuracy: {accuracy:.2%}")

            if validation_logloss is not None:
                log(f"ML: XGBoost Log Loss: {validation_logloss:.4f}")

            log(f"ML: Top 3 features: {sorted_features[:3]}")
            log(
                f"ML: Trained on {self.trades_learned} "
                "trades using XGBoost"
            )

            self.save_model()
            return True

        except Exception as exc:
            log(f"ML: XGBoost training failed - {exc}")
            return False

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    def predict(self, market_data):
        """
        Return a probability/recommendation.

        If the model is unavailable, return MAYBE with zero confidence.
        This is deliberately conservative: ML failure must not create
        a YES signal.
        """
        if not self.is_trained or self.model is None:
            return {
                "win_probability": 0.5,
                "confidence": 0.0,
                "recommendation": "MAYBE",
                "model_type": "XGBoost",
                "is_trained": False,
                "probability_source": "fallback",
            }

        try:
            features = self._to_vector(market_data)
            scaled = self.scaler.transform(features)

            probabilities = self.model.predict_proba(scaled)

            # Binary classifier should expose two classes [P(0), P(1)].
            if probabilities.shape[1] != 2:
                raise ValueError(
                    "XGBoost model did not return binary probabilities."
                )

            probability = float(probabilities[0, 1])
            probability = min(1.0, max(0.0, probability))

            confidence = max(probability, 1.0 - probability)

            if probability >= ML_YES_THRESHOLD:
                recommendation = "YES"
            elif probability >= ML_MAYBE_THRESHOLD:
                recommendation = "MAYBE"
            else:
                recommendation = "NO"

            return {
                "win_probability": probability,
                "confidence": confidence,
                "recommendation": recommendation,
                "model_type": "XGBoost",
                "is_trained": True,
                "probability_source": "xgboost_raw",
            }

        except Exception as exc:
            log(f"ML: Prediction failed - {exc}")

            # Never turn an ML error into a trade approval.
            return {
                "win_probability": 0.5,
                "confidence": 0.0,
                "recommendation": "MAYBE",
                "model_type": "XGBoost",
                "is_trained": False,
                "error": str(exc),
            }

    def get_decision_probability(self, market_data):
        """
        Return the probability intended for the quantitative risk layer.

        This keeps the ML engine responsible for prediction while the risk
        engine remains responsible for whether the trade is economically
        acceptable.
        """
        prediction = self.predict(market_data)

        return {
            "probability": float(prediction.get("win_probability", 0.5)),
            "confidence": float(prediction.get("confidence", 0.0)),
            "is_trained": bool(prediction.get("is_trained", False)),
            "recommendation": prediction.get(
                "recommendation",
                "MAYBE",
            ),
            "source": prediction.get(
                "probability_source",
                "fallback",
            ),
        }

    def create_triple_barrier_label(
        self, future_prices, entry, volatility,
        reward_multiple=1.5, risk_multiple=1.0,
        max_bars=30, direction="LONG"
    ):
        """Create a path-aware label for future supervised training."""
        return label_path(
            future_prices, entry, volatility,
            reward_multiple, risk_multiple, max_bars, direction
        )

    # --------------------------------------------------------
    # Persistence
    # --------------------------------------------------------

    def save_model(self):
        """Persist the XGBoost model and scaler."""
        if not self.is_trained or self.model is None:
            return False

        try:
            model_dir = os.path.dirname(os.path.abspath(MODEL_PATH))
            scaler_dir = os.path.dirname(os.path.abspath(SCALER_PATH))

            os.makedirs(model_dir, exist_ok=True)
            os.makedirs(scaler_dir, exist_ok=True)

            self.model.save_model(MODEL_PATH)

            # Store scaler parameters as JSON rather than relying on
            # pickle for the core ML artifact.
            scaler_data = {
                "mean": self.scaler.mean_.tolist(),
                "scale": self.scaler.scale_.tolist(),
                "var": self.scaler.var_.tolist(),
                "n_features_in": int(self.scaler.n_features_in_),
                "feature_names": FEATURE_NAMES,
            }

            with open(SCALER_PATH, "w", encoding="utf-8") as handle:
                json.dump(scaler_data, handle)

            log("ML: Model and scaler saved.")
            return True

        except Exception as exc:
            log(f"ML: Model save failed - {exc}")
            return False

    def load_model(self):
        """Load persisted XGBoost model if compatible artifacts exist."""
        if not ML_ENABLED or not XGBOOST_AVAILABLE:
            return False

        if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
            return False

        try:
            with open(SCALER_PATH, "r", encoding="utf-8") as handle:
                scaler_data = json.load(handle)

            if scaler_data.get("feature_names") != FEATURE_NAMES:
                log("ML: Saved scaler feature set does not match current model.")
                return False

            self.scaler.mean_ = np.asarray(
                scaler_data["mean"],
                dtype=float,
            )
            self.scaler.scale_ = np.asarray(
                scaler_data["scale"],
                dtype=float,
            )
            self.scaler.var_ = np.asarray(
                scaler_data["var"],
                dtype=float,
            )
            self.scaler.n_features_in_ = int(
                scaler_data["n_features_in"]
            )

            model = xgb.XGBClassifier()
            model.load_model(MODEL_PATH)

            self.model = model
            self.is_trained = True
            self.model_type = "XGBoost"

            self.feature_importance = {
                DISPLAY_NAMES[name]: float(importance)
                for name, importance in zip(
                    FEATURE_NAMES,
                    model.feature_importances_,
                )
            }

            self.last_train_trade_count = self.trades_learned

            log("ML: Existing XGBoost model loaded.")
            return True

        except Exception as exc:
            log(f"ML: Model load failed - {exc}")
            self.model = None
            self.is_trained = False
            return False

    def bootstrap_from_database(self):
        """
        Rehydrate ML history and train when a persisted model is unavailable
        but enough historical trades exist.
        """
        restored = self.load_trade_history_from_database()

        if not XGBOOST_AVAILABLE or not ML_ENABLED:
            return False

        if self.is_trained:
            return True

        if restored >= ML_MIN_TRADES_TO_TRAIN:
            log(
                "ML: No active model; training XGBoost "
                "from persisted history."
            )
            return self.train()

        return False

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    def get_stats(self):
        if not self.is_trained:
            return {
                "is_trained": False,
                "trades_learned": self.trades_learned,
                "model_type": "XGBoost",
                "message": (
                    f"Model not trained yet "
                    f"(need {ML_MIN_TRADES_TO_TRAIN} trades)"
                ),
            }

        return {
            "is_trained": True,
            "trades_learned": self.trades_learned,
            "model_type": self.model_type,
            "accuracy": self.training_stats.get("accuracy", 0),
            "log_loss": self.training_stats.get("log_loss"),
            "feature_importance": self.feature_importance,
        }


ml_engine = AthenaML()
