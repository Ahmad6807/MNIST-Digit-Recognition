import sys
import argparse
import numpy as np
import cv2
import tensorflow as tf
from pathlib import Path
from typing import Dict, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocess import preprocess_for_web


def load_model(model_path: str = "models/best_cnn_model.keras"):
    """Load a saved Keras model from disk."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model not found at {path}. Train the CNN first with: python src/train.py"
        )
    return tf.keras.models.load_model(str(path))


def predict_single_image(image_input, model,
                         invert_threshold: float = 127.0) -> Dict:
    """Predict digit from an image.

    Args:
        image_input: Either a file path (str) or a numpy array (H,W) or (H,W,C).
        model: Loaded Keras model.
        invert_threshold: Mean pixel threshold for auto-inversion.

    Returns:
        dict with keys: predicted_class, confidence, all_probabilities
    """
    if isinstance(image_input, str):
        img = cv2.imread(image_input, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError(f"Could not read image: {image_input}")
        processed = preprocess_for_web(img, invert_threshold)
    elif isinstance(image_input, np.ndarray):
        processed = preprocess_for_web(image_input, invert_threshold)
    else:
        raise TypeError(f"Unsupported input type: {type(image_input)}")

    predictions = model.predict(processed, verbose=0)[0]
    predicted_class = int(np.argmax(predictions))
    confidence = float(predictions[predicted_class])

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "all_probabilities": predictions
    }


def main():
    parser = argparse.ArgumentParser(description="MNIST Digit Recognition - CLI Inference")
    parser.add_argument("image_path", help="Path to the digit image (PNG/JPG)")
    parser.add_argument("--model", default="models/best_cnn_model.keras",
                        help="Path to saved model")
    parser.add_argument("--all", dest="show_all", action="store_true",
                        help="Show probabilities for all classes")
    args = parser.parse_args()

    img_path = Path(args.image_path)
    if not img_path.exists():
        print(f"Error: File not found at {args.image_path}", file=sys.stderr)
        sys.exit(1)

    try:
        model = load_model(args.model)

        result = predict_single_image(str(img_path), model)

        print(f"Predicted Digit: {result['predicted_class']} "
              f"(Confidence: {result['confidence']:.4f})")

        if args.show_all:
            print("\nAll class probabilities:")
            for i, prob in enumerate(result["all_probabilities"]):
                print(f"  {i}: {prob:.4f}")

    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error: Something went wrong during inference: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
