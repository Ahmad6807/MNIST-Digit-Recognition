import os
import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from pathlib import Path


def load_mnist(source: str = "tf",
               csv_train_path: Optional[str] = None,
               csv_test_path: Optional[str] = None,
               cache_dir: Optional[str] = None) -> Dict[str, np.ndarray]:
    """Load MNIST dataset from TensorFlow datasets or Kaggle CSV.

    Args:
        source: "tf" for TensorFlow Keras datasets, "csv" for Kaggle CSV files.
        csv_train_path: Path to mnist_train.csv (required if source="csv").
        csv_test_path: Path to mnist_test.csv (required if source="csv").
        cache_dir: Directory to save/load cached .npz files.

    Returns:
        dict with keys: x_train (60000, 28, 28) uint8, y_train (60000,) uint8,
                        x_test (10000, 28, 28) uint8, y_test (10000,) uint8
    """
    cache_path = None
    if cache_dir:
        cache_dir_path = Path(cache_dir)
        cache_dir_path.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir_path / "mnist_raw.npz"
        if cache_path.exists():
            loaded = np.load(str(cache_path))
            return {
                "x_train": loaded["x_train"],
                "y_train": loaded["y_train"],
                "x_test": loaded["x_test"],
                "y_test": loaded["y_test"]
            }

    if source == "tf":
        data = _load_from_tf()
    elif source == "csv":
        if csv_train_path is None or csv_test_path is None:
            raise ValueError("csv_train_path and csv_test_path required when source='csv'")
        data = _load_from_csv(csv_train_path, csv_test_path)
    else:
        raise ValueError(f"Unknown source: {source}. Use 'tf' or 'csv'.")

    _validate_data(data)

    if cache_path is not None:
        np.savez_compressed(str(cache_path),
                            x_train=data["x_train"],
                            y_train=data["y_train"],
                            x_test=data["x_test"],
                            y_test=data["y_test"])

    return data


def _load_from_tf() -> Dict[str, np.ndarray]:
    import tensorflow as tf
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    return {
        "x_train": x_train.astype(np.uint8),
        "y_train": y_train.astype(np.uint8),
        "x_test": x_test.astype(np.uint8),
        "y_test": y_test.astype(np.uint8)
    }


def _load_from_csv(train_path: str, test_path: str) -> Dict[str, np.ndarray]:
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training CSV not found: {train_path}. "
                                "Download from https://www.kaggle.com/competitions/digit-recognizer")
    if not os.path.exists(test_path):
        raise FileNotFoundError(f"Test CSV not found: {test_path}")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    y_train = train_df.iloc[:, 0].values.astype(np.uint8)
    x_train = train_df.iloc[:, 1:].values.astype(np.uint8).reshape(-1, 28, 28)
    y_test = test_df.iloc[:, 0].values.astype(np.uint8)
    x_test = test_df.iloc[:, 1:].values.astype(np.uint8).reshape(-1, 28, 28)

    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test
    }


def _validate_data(data: Dict[str, np.ndarray]) -> None:
    expected = {
        "x_train": (60000, 28, 28),
        "y_train": (60000,),
        "x_test": (10000, 28, 28),
        "y_test": (10000,)
    }
    for key, expected_shape in expected.items():
        if key not in data:
            raise ValueError(f"Missing key: {key}")
        actual = data[key].shape
        if actual != expected_shape:
            raise ValueError(f"Expected {key} shape {expected_shape}, got {actual}")

    for key in ["y_train", "y_test"]:
        labels = data[key]
        if not np.all((0 <= labels) & (labels <= 9)):
            raise ValueError(f"{key} contains labels outside range 0-9")
        unique = np.unique(labels)
        if len(unique) < 10:
            raise ValueError(f"{key} missing some classes. Found: {sorted(unique)}")

    for key in ["x_train", "x_test"]:
        if np.any(np.isnan(data[key])):
            raise ValueError(f"{key} contains NaN values")
