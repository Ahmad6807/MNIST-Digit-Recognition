"""Tests for training pipeline."""
import numpy as np
import pytest
import yaml
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.train import train, set_seeds
from src.cnn_model import build_mnist_cnn, compile_model, get_callbacks


class TestTrain:
    def test_set_seeds(self):
        set_seeds(42)
        import random
        import tensorflow as tf
        assert random.random() is not None
        assert np.random.rand() is not None
        assert tf.random.uniform((1,)) is not None

    def test_dry_run(self, tmp_path):
        config_path = str(tmp_path / "config.yaml")
        config = {
            "data": {"source": "tf", "cache_dir": str(tmp_path / "data"), "val_size": 1000, "random_state": 42},
            "preprocessing": {"normalize": True, "reshape_for_cnn": True,
                              "augmentation": {"enabled": False}},
            "model": {"cnn": {"learning_rate": 0.001}},
            "training": {"epochs": 50, "batch_size": 128,
                         "callbacks": {
                             "early_stopping": {"monitor": "val_loss", "patience": 5, "restore_best_weights": True},
                             "reduce_lr": {"monitor": "val_loss", "factor": 0.5, "patience": 3, "min_lr": 1e-6},
                             "model_checkpoint": {"filepath": str(tmp_path / "best_cnn_model.keras"),
                                                  "monitor": "val_accuracy", "mode": "max", "save_best_only": True}
                         }},
            "inference": {"invert_threshold": 127}
        }
        with open(config_path, 'w') as f:
            yaml.dump(config, f)

        model, history = train(config_path=config_path, dry_run=True)

        assert model is not None
        assert history is not None
        assert len(history.history['loss']) == 3  # 3 epochs in dry run
        assert 'val_accuracy' in history.history

    def test_model_checkpoint_callback(self, tmp_path):
        config = {
            "training": {
                "callbacks": {
                    "model_checkpoint": {
                        "filepath": str(tmp_path / "best_cnn_model.keras"),
                        "monitor": "val_accuracy",
                        "mode": "max",
                        "save_best_only": True
                    }
                }
            }
        }
        callbacks = get_callbacks(config)
        assert len(callbacks) >= 1
        assert any("ModelCheckpoint" in str(type(c)) for c in callbacks)
