"""Generate accessibility templates from MNIST test set.

Extracts one clean example per digit (0-9) from the MNIST test set
and saves them as 28x28 grayscale PNG files in assets/accessibility/.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import tensorflow as tf


def generate_templates(output_dir: str = "assets/accessibility"):
    """Extract one example per digit from MNIST test set."""
    _, (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for digit in range(10):
        indices = np.where(y_test == digit)[0]
        if len(indices) > 0:
            img = x_test[indices[0]]
            save_path = output_path / f"digit_{digit}.png"
            cv2.imwrite(str(save_path), img)
            print(f"Saved {save_path} - shape: {img.shape}, dtype: {img.dtype}")
        else:
            print(f"Warning: No test sample found for digit {digit}")

    print(f"Generated {10} accessibility templates in {output_path}")


if __name__ == "__main__":
    generate_templates()
