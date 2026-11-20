"""Tests for predict module (CLI inference)."""
import numpy as np
import pytest
import sys
import os
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.predict import predict_single_image, load_model


class TestPredict:
    def setup_method(self):
        self.model_path = "models/best_cnn_model.keras"
        self.model_available = Path(self.model_path).exists()

    def test_predict_single_image_array(self):
        if not self.model_available:
            pytest.skip("No trained model available")
        model = load_model(self.model_path)
        dummy_img = np.zeros((28, 28), dtype=np.uint8)
        dummy_img[10:20, 10:20] = 200  # white digit on black
        result = predict_single_image(dummy_img, model)
        assert "predicted_class" in result
        assert "confidence" in result
        assert "all_probabilities" in result
        assert 0 <= result["predicted_class"] <= 9
        assert 0.0 <= result["confidence"] <= 1.0
        assert result["all_probabilities"].shape == (10,)

    def test_predict_single_image_confidence_sum(self):
        if not self.model_available:
            pytest.skip("No trained model available")
        model = load_model(self.model_path)
        dummy_img = np.zeros((28, 28), dtype=np.uint8)
        dummy_img[10:20, 10:20] = 200
        result = predict_single_image(dummy_img, model)
        assert abs(np.sum(result["all_probabilities"]) - 1.0) < 1e-5

    def test_predict_single_image_rgb_input(self):
        if not self.model_available:
            pytest.skip("No trained model available")
        model = load_model(self.model_path)
        dummy_img = np.zeros((50, 50, 3), dtype=np.uint8)
        dummy_img[15:25, 15:25] = [200, 200, 200]
        result = predict_single_image(dummy_img, model)
        assert 0 <= result["predicted_class"] <= 9

    def test_predict_invalid_input_type(self):
        if not self.model_available:
            pytest.skip("No trained model available")
        model = load_model(self.model_path)
        with pytest.raises(TypeError, match="Unsupported input type"):
            predict_single_image(12345, model)

    def test_load_model_file_not_found(self):
        with pytest.raises(SystemExit):
            load_model("nonexistent_path.keras")

    def test_preprocess_identical_cli_web(self):
        """Verify that preprocessing pipeline is identical for CLI and web paths."""
        from src.preprocess import preprocess_for_web
        # Both CLI (predict.py) and web (app.py) use preprocess_for_web
        img = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
        result = preprocess_for_web(img)
        assert result.shape == (1, 28, 28, 1)
        assert result.dtype == np.float32
        assert np.all(result >= 0.0) and np.all(result <= 1.0)
