"""Tests for baseline model module."""
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.baseline_model import train_baseline, load_baseline_model


class TestBaselineModel:
    def test_train_baseline(self, tmp_path):
        model_path = str(tmp_path / "baseline_logreg.pkl")
        metrics_path = str(tmp_path / "baseline_metrics.json")

        x_train = np.random.rand(500, 784).astype(np.float32)
        y_train = np.random.randint(0, 10, 500)
        x_val = np.random.rand(100, 784).astype(np.float32)
        y_val = np.random.randint(0, 10, 100)

        result = train_baseline(
            x_train, y_train, x_val, y_val,
            model_path=model_path,
            metrics_path=metrics_path,
            cv_folds=2
        )

        assert "model" in result
        assert "best_params" in result
        assert "val_accuracy" in result
        assert result["val_accuracy"] >= 0.0
        assert result["train_time"] > 0
        assert result["inference_time"] > 0

    def test_save_and_load(self, tmp_path):
        model_path = str(tmp_path / "baseline_logreg.pkl")
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression(max_iter=100, random_state=42)
        x = np.random.rand(100, 784)
        y = np.random.randint(0, 10, 100)
        model.fit(x, y)

        import joblib
        joblib.dump(model, model_path)

        loaded = load_baseline_model(model_path)
        assert loaded is not None
        preds = loaded.predict(x[:5])
        assert len(preds) == 5
