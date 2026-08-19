"""
Athena-X Meta Model Training CLI.

Usage:
    python train_meta_model.py

This script trains from Athena's persisted SQLite history.
It never creates synthetic trades.
"""

from database import db
from meta_trainer import train_from_history, save_training_report, MetaTrainingError


def main():
    history = db.get_meta_training_history()

    try:
        result = train_from_history(history)
    except MetaTrainingError as exc:
        print("META MODEL NOT TRAINED")
        print(str(exc))
        return 2

    report_path = save_training_report(result)

    metrics = result["metrics"]

    print("=== ATHENA META MODEL TRAINING ===")
    print(f"Qualifying trades : {result['total_rows']}")
    print(f"Skipped exits     : {result['skipped_rows']}")
    print(f"Training rows     : {result['train_rows']}")
    print(f"Validation rows   : {result['validation_rows']}")
    print(f"Validation start  : {result['validation_start']}")
    print()
    print(f"Accuracy          : {metrics['accuracy']:.3f}")
    print(f"Precision         : {metrics['precision']:.3f}")
    print(f"Recall            : {metrics['recall']:.3f}")
    print(f"ROC-AUC           : {metrics['roc_auc']}")
    print()
    print(f"Model saved       : {result['engine'].model_path}")
    print(f"Report saved      : {report_path}")
    print()
    print("IMPORTANT: model activation remains optional.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
