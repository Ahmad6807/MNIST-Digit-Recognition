import os
import json
import random
import sys
import numpy as np
import tensorflow as tf
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_loader import load_config
from src.data_loader import load_mnist
from src.preprocess import preprocess_pipeline
from src.cnn_model import build_mnist_cnn, compile_model, get_callbacks


def set_seeds(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"


def train(config_path: str = "config.yaml",
          dry_run: bool = False,
          epochs: Optional[int] = None,
          batch_size: Optional[int] = None):
    """Main training pipeline.

    Args:
        config_path: Path to YAML configuration file.
        dry_run: If True, train on 2000 samples for 3 epochs.
        epochs: Override config epochs.
        batch_size: Override config batch size.
    """
    config = load_config(config_path)

    set_seeds(config.get("data", {}).get("random_state", 42))

    print("Loading MNIST dataset...")
    data = load_mnist(
        source=config["data"]["source"],
        csv_train_path=config["data"].get("csv_train_path"),
        csv_test_path=config["data"].get("csv_test_path"),
        cache_dir=config["data"].get("cache_dir")
    )

    print("Preprocessing data...")
    processed = preprocess_pipeline(
        data["x_train"], data["y_train"],
        data["x_test"], data["y_test"],
        val_size=config["data"].get("val_size", 10000),
        augmentation=config.get("preprocessing", {}).get("augmentation", {}).get("enabled", False)
    )

    x_train = processed["x_train_proc"]
    y_train = processed["y_train_oh"]
    x_val = processed["x_val_proc"]
    y_val = processed["y_val_oh"]

    if dry_run:
        print("DRY RUN: Using 2000 training samples, 3 epochs")
        x_train = x_train[:2000]
        y_train = y_train[:2000]
        x_val = x_val[:500]
        y_val = y_val[:500]
        train_epochs = 3
        train_batch_size = 128
    else:
        train_epochs = epochs or config["training"]["epochs"]
        train_batch_size = batch_size or config["training"]["batch_size"]

    print("Building CNN model...")
    model = build_mnist_cnn()
    model = compile_model(
        model,
        learning_rate=config.get("model", {}).get("cnn", {}).get("learning_rate", 0.001)
    )

    model.summary()

    print("Configuring callbacks...")
    callbacks = get_callbacks(config)

    print(f"Training for up to {train_epochs} epochs with batch size {train_batch_size}...")
    history = model.fit(
        x_train, y_train,
        batch_size=train_batch_size,
        epochs=train_epochs,
        validation_data=(x_val, y_val),
        callbacks=callbacks,
        verbose=1
    )

    Path("reports").mkdir(parents=True, exist_ok=True)
    history_path = "reports/history.json"
    history_dict = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(history_path, 'w') as f:
        json.dump(history_dict, f, indent=2)
    print(f"Training history saved to {history_path}")

    if not dry_run:
        final_path = "models/final_cnn_model.keras"
        model.save(final_path)
        print(f"Final model saved to {final_path}")

    print("Training complete!")
    final_val_acc = history.history['val_accuracy'][-1]
    print(f"Final validation accuracy: {final_val_acc:.4f}")

    return model, history


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train MNIST CNN classifier")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--dry-run", action="store_true", help="Train on small subset for testing")
    parser.add_argument("--epochs", type=int, default=None, help="Override training epochs")
    parser.add_argument("--batch-size", type=int, default=None, help="Override batch size")
    args = parser.parse_args()

    train(
        config_path=args.config,
        dry_run=args.dry_run,
        epochs=args.epochs,
        batch_size=args.batch_size
    )
