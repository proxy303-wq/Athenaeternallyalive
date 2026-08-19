"""
Explicit Athena meta-model activation.

Usage:
    python activate_meta_model.py

Only activates when the saved validation report explicitly says the model
is eligible. This keeps model deployment separate from model training.
"""

from pathlib import Path
import json

from meta_label_engine import MetaLabelEngine


REPORT = Path("data/meta_training_report.json")


def main():
    if not REPORT.exists():
        print("META MODEL NOT ACTIVATED")
        print("No meta_training_report.json exists.")
        return 2

    report = json.loads(REPORT.read_text(encoding="utf-8"))
    metrics = report.get("metrics", {})

    if not metrics.get("activation_eligible", False):
        print("META MODEL NOT ACTIVATED")
        print("Validation report did not pass activation criteria.")
        print(json.dumps(metrics, indent=2))
        return 2

    engine = MetaLabelEngine()
    engine.activate(metrics)

    print("META MODEL ACTIVATED")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
