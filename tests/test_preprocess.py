"""Tests for preprocess module."""
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocess import (
    normalize, reshape_for_cnn, one_hot_encode,
    split_train_val, preprocess_for_web,
    generate_synthetic_digit, load_accessibility_templates,
    preprocess_pipeline
)


class TestNormalize:
    def test_normalize_range(self):
        x = np.array([[0, 128, 255]], dtype=np.uint8)
        result = normalize(x)
        assert result.dtype == np.float32
        assert np.all(result >= 0.0) and np.all(result <= 1.0)
        assert result[0, 0] == 0.0
        assert result[0, 2] == 1.0

    def test_normalize_shape_preserved(self):
        x = np.random.randint(0, 256, (100, 28, 28), dtype=np.uint8)
        result = normalize(x)
        assert result.shape == x.shape


class TestReshapeForCnn:
    def test_reshape_3d(self):
        x = np.zeros((100, 28, 28))
        result = reshape_for_cnn(x)
        assert result.shape == (100, 28, 28, 1)

    def test_reshape_2d_flat(self):
        x = np.zeros((100, 784))
        result = reshape_for_cnn(x)
        assert result.shape == (100, 28, 28, 1)

    def test_reshape_invalid_ndim(self):
        x = np.zeros((100, 28, 28, 1, 1))
        with pytest.raises(ValueError, match="Unexpected image dimensions"):
            reshape_for_cnn(x)


class TestOneHotEncode:
    def test_one_hot_encode(self):
        labels = np.array([0, 1, 9, 5])
        result = one_hot_encode(labels)
        assert result.shape == (4, 10)
        assert result[0, 0] == 1.0
        assert result[1, 1] == 1.0
        assert result[2, 9] == 1.0
        assert result[3, 5] == 1.0

    def test_one_hot_dtype(self):
        labels = np.array([3])
        result = one_hot_encode(labels)
        assert result.dtype == np.float32


class TestSplitTrainVal:
    def test_split_shape(self):
        x = np.random.rand(6000, 28, 28)
        y = np.random.randint(0, 10, 6000)
        x_tr, x_val, y_tr, y_val = split_train_val(x, y, val_size=1000)
        assert x_tr.shape == (5000, 28, 28)
        assert x_val.shape == (1000, 28, 28)
        assert len(y_tr) == 5000
        assert len(y_val) == 1000

    def test_split_stratified(self):
        x = np.random.rand(6000, 28, 28)
        y = np.random.randint(0, 10, 6000)
        _, x_val, _, y_val = split_train_val(x, y, val_size=1000)
        # Check all classes present
        assert len(np.unique(y_val)) == 10


class TestPreprocessForWeb:
    def test_preprocess_grayscale(self):
        img = np.ones((100, 100), dtype=np.uint8) * 200
        result = preprocess_for_web(img)
        assert result.shape == (1, 28, 28, 1)
        assert result.dtype == np.float32
        assert np.all(result >= 0.0) and np.all(result <= 1.0)

    def test_preprocess_rgb(self):
        img = np.ones((50, 50, 3), dtype=np.uint8) * 200
        result = preprocess_for_web(img)
        assert result.shape == (1, 28, 28, 1)

    def test_preprocess_rgba(self):
        img = np.ones((50, 50, 4), dtype=np.uint8) * 200
        result = preprocess_for_web(img)
        assert result.shape == (1, 28, 28, 1)

    def test_preprocess_auto_invert(self):
        # Light background (dark-on-light) should be inverted to white-on-black
        img = np.ones((28, 28), dtype=np.uint8) * 200
        img[10:20, 10:20] = 30  # dark digit on light background
        result = preprocess_for_web(img, invert_threshold=127)
        # After inversion, the digit area should have higher values
        assert result[0, 14, 14, 0] > 0.5

    def test_preprocess_no_invert_needed(self):
        # Already white-on-black (MNIST style)
        img = np.zeros((28, 28), dtype=np.uint8)
        img[10:20, 10:20] = 200  # light digit on dark background
        result = preprocess_for_web(img, invert_threshold=127)
        assert result[0, 14, 14, 0] > 0.5

    def test_preprocess_none_input(self):
        with pytest.raises(ValueError, match="No image provided"):
            preprocess_for_web(None)


class TestAccessibilityTemplates:
    def test_generate_synthetic_digit(self):
        templates = load_accessibility_templates(assets_dir="assets/accessibility")
        result = generate_synthetic_digit(3, templates)
        assert result.shape == (1, 28, 28, 1)
        assert result.dtype == np.float32
        assert np.all(result >= 0.0) and np.all(result <= 1.0)

    def test_generate_synthetic_invalid(self):
        templates = load_accessibility_templates(assets_dir="assets/accessibility")
        with pytest.raises(ValueError, match="No template for digit"):
            generate_synthetic_digit(10, templates)

    def test_load_templates_all_digits(self):
        templates = load_accessibility_templates(assets_dir="assets/accessibility")
        assert len(templates) == 10
        for i in range(10):
            assert i in templates
            assert templates[i].shape == (28, 28)


class TestPreprocessPipeline:
    def test_pipeline_outputs(self):
        x_train = np.random.randint(0, 256, (1000, 28, 28), dtype=np.uint8)
        y_train = np.random.randint(0, 10, 1000)
        x_test = np.random.randint(0, 256, (200, 28, 28), dtype=np.uint8)
        y_test = np.random.randint(0, 10, 200)

        result = preprocess_pipeline(x_train, y_train, x_test, y_test, val_size=200)

        assert "x_train_proc" in result
        assert "y_train_oh" in result
        assert "x_val_proc" in result
        assert "y_val_oh" in result
        assert "x_test_proc" in result
        assert "y_test_oh" in result
        assert result["x_train_proc"].shape[0] == 800
        assert result["x_val_proc"].shape[0] == 200
        assert result["x_test_proc"].shape[0] == 200
        assert result["x_train_proc"].shape[-1] == 1
        assert result["y_train_oh"].shape[-1] == 10
