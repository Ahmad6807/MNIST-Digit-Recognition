"""Tests for evaluation module."""
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.evaluate import evaluate_model, _plot_confusion_matrix, _plot_error_examples, compare_with_baseline


class TestEvaluate:
    def test_evaluate_model_basic(self, tmp_path):
        class DummyModel:
            def predict(self, x, verbose=0):
                n = x.shape[0]
                probs = np.zeros((n, 10))
                # Predict class 5 for everything
                probs[:, 5] = 1.0
                return probs
            def count_params(self):
                return 1000

        model = DummyModel()
        x_test = np.zeros((100, 28, 28, 1), dtype=np.float32)
        y_test_oh = np.zeros((100, 10), dtype=np.float32)
        y_test_oh[:, :] = 0.1  # uniform

        save_dir = str(tmp_path)
        result = evaluate_model(model, x_test, y_test_oh, save_dir=save_dir)

        assert "test_accuracy" in result
        assert "classification_report" in result
        assert "confusion_matrix" in result
        assert "misclassified_indices" in result
        assert result["confusion_matrix"].shape == (10, 10)
        assert (tmp_path / "confusion_matrix.png").exists()
        assert (tmp_path / "evaluation_metrics.json").exists()

    def test_confusion_matrix_plot(self, tmp_path):
        cm = np.zeros((10, 10), dtype=int)
        np.fill_diagonal(cm, 100)
        save_path = str(tmp_path / "test_cm.png")
        _plot_confusion_matrix(cm, [str(i) for i in range(10)], save_path)
        assert Path(save_path).exists()

    def test_error_examples_plot(self, tmp_path):
        x_test = np.random.rand(100, 28, 28, 1).astype(np.float32)
        y_true = np.random.randint(0, 10, 100)
        y_pred = np.random.randint(0, 10, 100)
        misclassified = list(range(20))
        save_path = str(tmp_path / "test_errors.png")
        _plot_error_examples(x_test, y_true, y_pred, misclassified, save_path)
        assert Path(save_path).exists()

    def test_compare_with_baseline(self, tmp_path):
        save_path = str(tmp_path / "comparison.md")
        result = compare_with_baseline(
            cnn_accuracy=0.992,
            cnn_train_time=720,
            save_path=save_path
        )
        assert Path(save_path).exists()
        assert "CNN" in result
        assert "0.992" in result
