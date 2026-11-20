import numpy as np
from typing import Dict, Optional, Tuple
from pathlib import Path
from sklearn.model_selection import train_test_split


def split_train_val(x_train: np.ndarray, y_train: np.ndarray,
                    val_size: int = 10000, stratify: bool = True,
                    random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified split of training data into train and validation sets."""
    return train_test_split(
        x_train, y_train,
        test_size=val_size,
        stratify=y_train if stratify else None,
        random_state=random_state
    )


def normalize(images: np.ndarray) -> np.ndarray:
    """Convert uint8 [0, 255] to float32 [0, 1]."""
    return images.astype(np.float32) / 255.0


def reshape_for_cnn(images: np.ndarray) -> np.ndarray:
    """Add channel dimension: (N, 28, 28) -> (N, 28, 28, 1)."""
    if images.ndim == 3:
        return np.expand_dims(images, axis=-1)
    elif images.ndim == 2:
        return images.reshape(-1, 28, 28, 1)
    else:
        raise ValueError(f"Unexpected image dimensions: {images.ndim}. Expected 2 or 3.")


def one_hot_encode(labels: np.ndarray, num_classes: int = 10) -> np.ndarray:
    """Convert integer labels to one-hot encoded vectors."""
    import tensorflow as tf
    return tf.keras.utils.to_categorical(labels, num_classes)


def preprocess_pipeline(x_train: np.ndarray, y_train: np.ndarray,
                        x_test: np.ndarray, y_test: np.ndarray,
                        val_size: int = 10000,
                        augmentation: bool = False) -> Dict:
    """Full preprocessing pipeline: split, normalize, reshape, one-hot.

    Returns:
        dict with keys: x_train_proc, y_train_oh, x_val_proc, y_val_oh,
                       x_test_proc, y_test_oh, augment_gen
    """
    x_tr, x_val, y_tr, y_val = split_train_val(x_train, y_train, val_size=val_size)

    x_tr_norm = normalize(x_tr)
    x_val_norm = normalize(x_val)
    x_test_norm = normalize(x_test)

    x_tr_cnn = reshape_for_cnn(x_tr_norm)
    x_val_cnn = reshape_for_cnn(x_val_norm)
    x_test_cnn = reshape_for_cnn(x_test_norm)

    y_tr_oh = one_hot_encode(y_tr)
    y_val_oh = one_hot_encode(y_val)
    y_test_oh = one_hot_encode(y_test)

    augment_gen = None
    if augmentation:
        import tensorflow as tf
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1
        )
        datagen.fit(x_tr_cnn)
        augment_gen = datagen

    return {
        "x_train_proc": x_tr_cnn,
        "y_train_oh": y_tr_oh,
        "x_val_proc": x_val_cnn,
        "y_val_oh": y_val_oh,
        "x_test_proc": x_test_cnn,
        "y_test_oh": y_test_oh,
        "augment_gen": augment_gen,
        "x_train_flat": x_tr_norm.reshape(x_tr_norm.shape[0], -1),
        "x_val_flat": x_val_norm.reshape(x_val_norm.shape[0], -1),
        "x_test_flat": x_test_norm.reshape(x_test_norm.shape[0], -1),
        "y_train": y_tr,
        "y_val": y_val,
        "y_test": y_test
    }


def preprocess_for_web(image_array: np.ndarray,
                       invert_threshold: float = 127.0) -> np.ndarray:
    """Unified preprocessing for web (canvas + upload) and CLI inference.

    Input: (H, W) or (H, W, 3) or (H, W, 4) uint8 image.
    Output: (1, 28, 28, 1) float32 normalized [0, 1].

    Steps:
        1. Convert to grayscale if needed.
        2. Resize to 28x28 using INTER_AREA.
        3. Auto-invert if mean pixel > threshold (assumes dark-on-light input).
        4. Normalize to [0, 1].
        5. Add batch and channel dimensions.
    """
    if image_array is None:
        raise ValueError("No image provided.")

    import cv2

    gray = np.asarray(image_array)
    if gray.size == 0:
        raise ValueError("The image is empty.")
    if gray.ndim not in (2, 3):
        raise ValueError("Expected a grayscale, RGB, or RGBA image.")

    if gray.ndim == 3:
        if gray.shape[2] == 1:
            gray = gray[:, :, 0]
        elif gray.shape[2] == 3:
            gray = cv2.cvtColor(gray, cv2.COLOR_RGB2GRAY)
        elif gray.shape[2] == 4:
            # Composite transparent pixels over white before grayscale conversion.
            rgb = gray[:, :, :3].astype(np.float32)
            alpha = gray[:, :, 3:4].astype(np.float32)
            if np.issubdtype(gray.dtype, np.floating) and float(np.nanmax(alpha)) <= 1.0:
                alpha *= 255.0
            gray = cv2.cvtColor((rgb * (alpha / 255.0) + 255.0 * (1.0 - alpha / 255.0)).clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        else:
            raise ValueError("Expected an image with 1, 3, or 4 channels.")

    if gray.dtype != np.uint8:
        if not np.issubdtype(gray.dtype, np.number) or not np.all(np.isfinite(gray)):
            raise ValueError("Image pixels must contain finite numeric values.")
        if float(np.max(gray)) <= 1.0 and float(np.min(gray)) >= 0.0:
            gray = np.rint(gray * 255).astype(np.uint8)
        else:
            gray = np.clip(gray, 0, 255).astype(np.uint8)

    resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)

    if np.mean(resized) > invert_threshold:
        resized = 255 - resized

    normalized = resized.astype(np.float32) / 255.0
    result = np.expand_dims(normalized, axis=(0, -1))

    return result


def generate_synthetic_digit(digit: int,
                             templates: Dict[int, np.ndarray]) -> np.ndarray:
    """Return a pre-stored MNIST-style template for the given digit.

    Args:
        digit: Integer 0-9.
        templates: dict mapping digit -> 28x28 uint8 array (white digit on black).

    Returns:
        (1, 28, 28, 1) float32 normalized array ready for prediction.
    """
    if digit not in templates:
        raise ValueError(f"No template for digit {digit}. Must be 0-9.")
    img = templates[digit]
    normalized = img.astype(np.float32) / 255.0
    result = np.expand_dims(normalized, axis=(0, -1))
    return result


def load_accessibility_templates(assets_dir: str = "assets/accessibility") -> Dict[int, np.ndarray]:
    """Load 28x28 grayscale PNG templates (digit_0.png .. digit_9.png).

    Creates placeholder white-on-black templates if no real templates exist.
    """
    templates = {}
    import cv2
    assets_path = Path(assets_dir)
    assets_path.mkdir(parents=True, exist_ok=True)
    all_exist = all((assets_path / f"digit_{i}.png").exists() for i in range(10))

    if all_exist:
        for i in range(10):
            img_path = str(assets_path / f"digit_{i}.png")
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
                templates[i] = img
            else:
                templates[i] = _create_placeholder_template(i)
    else:
        for i in range(10):
            templates[i] = _create_placeholder_template(i)
            placeholder_path = assets_path / f"digit_{i}.png"
            cv2.imwrite(str(placeholder_path), _create_placeholder_template(i))

    return templates


def _create_placeholder_template(digit: int) -> np.ndarray:
    """Create a local white-on-black digit template without downloading MNIST."""
    img = np.zeros((28, 28), dtype=np.uint8)
    import cv2
    cv2.putText(img, str(digit), (3, 24), cv2.FONT_HERSHEY_SIMPLEX,
                0.85, 255, thickness=2, lineType=cv2.LINE_AA)
    return img
