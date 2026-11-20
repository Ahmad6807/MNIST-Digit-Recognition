"""Tests for data_loader module."""
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_loader import load_mnist, _validate_data


class TestDataLoader:
    def test_load_mnist_tf_shape(self):
        data = load_mnist(source="tf")
        assert data["x_train"].shape == (60000, 28, 28)
        assert data["y_train"].shape == (60000,)
        assert data["x_test"].shape == (10000, 28, 28)
        assert data["y_test"].shape == (10000,)

    def test_load_mnist_tf_dtype(self):
        data = load_mnist(source="tf")
        assert data["x_train"].dtype == np.uint8
        assert data["y_train"].dtype == np.uint8

    def test_load_mnist_tf_label_range(self):
        data = load_mnist(source="tf")
        y_all = np.concatenate([data["y_train"], data["y_test"]])
        assert np.all((0 <= y_all) & (y_all <= 9))
        assert len(np.unique(y_all)) == 10

    def test_load_mnist_tf_pixel_range(self):
        data = load_mnist(source="tf")
        assert np.all(data["x_train"] >= 0)
        assert np.all(data["x_train"] <= 255)
        assert np.all(data["x_test"] >= 0)
        assert np.all(data["x_test"] <= 255)

    def test_load_mnist_tf_no_nan(self):
        data = load_mnist(source="tf")
        assert not np.any(np.isnan(data["x_train"]))
        assert not np.any(np.isnan(data["y_train"]))

    def test_validate_data_valid(self):
        data = {
            "x_train": np.zeros((60000, 28, 28), dtype=np.uint8),
            "y_train": np.zeros((60000,), dtype=np.uint8),
            "x_test": np.zeros((10000, 28, 28), dtype=np.uint8),
            "y_test": np.zeros((10000,), dtype=np.uint8)
        }
        data["y_train"][::6000] = np.arange(10)
        data["y_test"][::1000] = np.arange(10)
        _validate_data(data)

    def test_validate_data_wrong_shape(self):
        data = {
            "x_train": np.zeros((50000, 28, 28)),
            "y_train": np.zeros((60000,)),
            "x_test": np.zeros((10000, 28, 28)),
            "y_test": np.zeros((10000,))
        }
        with pytest.raises(ValueError, match="Expected x_train shape"):
            _validate_data(data)

    def test_validate_data_invalid_label_range(self):
        data = {
            "x_train": np.zeros((60000, 28, 28), dtype=np.uint8),
            "y_train": np.ones((60000,), dtype=np.uint8) * 10,
            "x_test": np.zeros((10000, 28, 28), dtype=np.uint8),
            "y_test": np.zeros((10000,), dtype=np.uint8)
        }
        with pytest.raises(ValueError, match="contains labels outside range"):
            _validate_data(data)

    def test_caching(self, tmp_path):
        cache_dir = str(tmp_path)
        data1 = load_mnist(source="tf", cache_dir=cache_dir)
        cache_file = Path(cache_dir) / "mnist_raw.npz"
        assert cache_file.exists()
        data2 = load_mnist(source="tf", cache_dir=cache_dir)
        np.testing.assert_array_equal(data1["x_train"], data2["x_train"])
        np.testing.assert_array_equal(data1["y_test"], data2["y_test"])

    def test_invalid_source(self):
        with pytest.raises(ValueError, match="Unknown source"):
            load_mnist(source="invalid")
