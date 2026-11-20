import time
import numpy as np
import joblib
import json
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support
from typing import Dict, Optional, Tuple


def train_baseline(x_train_flat: np.ndarray, y_train: np.ndarray,
                   x_val_flat: np.ndarray, y_val: np.ndarray,
                   model_path: str = "models/baseline_logreg.pkl",
                   metrics_path: str = "reports/baseline_metrics.json",
                   cv_folds: int = 5) -> Dict:
    """Train a Logistic Regression baseline with hyperparameter tuning.

    Args:
        x_train_flat: (N, 784) float32 training features.
        y_train: (N,) uint8 training labels.
        x_val_flat: (M, 784) float32 validation features.
        y_val: (M,) uint8 validation labels.
        model_path: Path to save trained model (.pkl).
        metrics_path: Path to save metrics JSON.
        cv_folds: Number of cross-validation folds.

    Returns:
        dict with keys: model, best_params, val_accuracy, train_time, inference_time
    """
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)

    param_grid = {
        'C': [0.01, 0.1, 1.0, 10.0]
    }

    base_model = LogisticRegression(
        max_iter=1000,
        multi_class='multinomial',
        solver='lbfgs',
        random_state=42,
        n_jobs=-1
    )

    grid = GridSearchCV(
        base_model, param_grid,
        cv=cv_folds,
        scoring='accuracy',
        n_jobs=-1,
        verbose=0
    )

    train_start = time.time()
    grid.fit(x_train_flat, y_train)
    train_time = time.time() - train_start

    best_model = grid.best_estimator_

    infer_start = time.time()
    y_val_pred = best_model.predict(x_val_flat)
    inference_time = time.time() - infer_start

    val_accuracy = accuracy_score(y_val, y_val_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_val, y_val_pred, average=None
    )

    joblib.dump(best_model, model_path)

    metrics = {
        "model_type": "LogisticRegression",
        "best_params": grid.best_params_,
        "cv_mean_accuracy": float(grid.best_score_),
        "val_accuracy": float(val_accuracy),
        "train_time_seconds": round(train_time, 2),
        "inference_time_seconds": round(inference_time, 4),
        "per_class_metrics": {
            str(i): {
                "precision": float(precision[i]),
                "recall": float(recall[i]),
                "f1_score": float(f1[i])
            }
            for i in range(10)
        }
    }

    Path(metrics_path).parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"Baseline Logistic Regression trained in {train_time:.2f}s")
    print(f"Best params: {grid.best_params_}")
    print(f"Validation accuracy: {val_accuracy:.4f}")
    print(f"Model saved to: {model_path}")
    print(f"Metrics saved to: {metrics_path}")

    return {
        "model": best_model,
        "best_params": grid.best_params_,
        "val_accuracy": val_accuracy,
        "train_time": train_time,
        "inference_time": inference_time
    }


def load_baseline_model(model_path: str = "models/baseline_logreg.pkl"):
    """Load a saved baseline model from disk."""
    return joblib.load(model_path)
