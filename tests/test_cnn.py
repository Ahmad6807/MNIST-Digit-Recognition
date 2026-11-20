"""Tests for CNN model module."""
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cnn_model import build_mnist_cnn, compile_model, get_callbacks


class TestCnnModel:
    def test_build_model_output_shape(self):
        model = build_mnist_cnn()
        dummy = np.zeros((1, 28, 28, 1), dtype=np.float32)
        output = model(dummy, training=False)
        assert output.shape == (1, 10)

    def test_build_model_parameter_count(self):
        model = build_mnist_cnn()
        params = model.count_params()
        assert params == 1625866, f"Expected 1,625,866 params, got {params}"

    def test_build_model_layer_count(self):
        model = build_mnist_cnn()
        layer_names = [layer.name for layer in model.layers]
        assert "conv1" in layer_names
        assert "conv2" in layer_names
        assert "pool1" in layer_names
        assert "dropout_conv" in layer_names
        assert "flatten" in layer_names
        assert "dense1" in layer_names
        assert "dropout_dense" in layer_names
        assert "output" in layer_names

    def test_compile_model(self):
        model = build_mnist_cnn()
        compiled = compile_model(model, learning_rate=0.001)
        assert compiled.optimizer is not None
        assert compiled.loss is not None
        assert "accuracy" in compiled.metrics_names

    def test_get_callbacks(self):
        config = {
            "training": {
                "callbacks": {
                    "early_stopping": {
                        "monitor": "val_loss",
                        "patience": 5,
                        "restore_best_weights": True
                    },
                    "reduce_lr": {
                        "monitor": "val_loss",
                        "factor": 0.5,
                        "patience": 3,
                        "min_lr": 1e-6
                    },
                    "model_checkpoint": {
                        "filepath": "models/best_cnn_model.keras",
                        "monitor": "val_accuracy",
                        "mode": "max",
                        "save_best_only": True
                    }
                }
            }
        }
        callbacks = get_callbacks(config)
        assert len(callbacks) >= 3
        callback_types = [type(c).__name__ for c in callbacks]
        assert "EarlyStopping" in callback_types
        assert "ReduceLROnPlateau" in callback_types
        assert "ModelCheckpoint" in callback_types
