import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_mnist
from src.preprocess import preprocess_pipeline


def evaluate_model(model, x_test: np.ndarray, y_test_oh: np.ndarray,
                   class_names: Optional[List[str]] = None,
                   save_dir: str = "reports") -> Dict:
    """Evaluate a trained model on the test set.

    Args:
        model: Loaded Keras model.
        x_test: (N, 28, 28, 1) float32 test features.
        y_test_oh: (N, 10) one-hot test labels.
        class_names: List of class names (default: "0".."9").
        save_dir: Directory to save report files.

    Returns:
        dict with keys: test_accuracy, classification_report, confusion_matrix,
                       misclassified_indices, per_class_metrics
    """
    if class_names is None:
        class_names = [str(i) for i in range(10)]

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    y_pred_probs = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = np.argmax(y_test_oh, axis=1)

    test_acc = accuracy_score(y_true, y_pred)
    report_dict = classification_report(y_true, y_pred, target_names=class_names, output_dict=True)
    cm = confusion_matrix(y_true, y_pred)

    misclassified = np.where(y_pred != y_true)[0].tolist()

    print(f"Test accuracy: {test_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    _plot_confusion_matrix(cm, class_names, str(save_path / "confusion_matrix.png"))
    _plot_error_examples(x_test, y_true, y_pred, misclassified,
                         str(save_path / "error_examples.png"))

    metrics = {
        "test_accuracy": float(test_acc),
        "per_class_metrics": {
            cls: {
                "precision": float(report_dict[cls]["precision"]),
                "recall": float(report_dict[cls]["recall"]),
                "f1_score": float(report_dict[cls]["f1-score"]),
                "support": int(report_dict[cls]["support"])
            }
            for cls in class_names
        },
        "confusion_matrix": cm.tolist(),
        "total_misclassified": len(misclassified),
        "model_info": {
            "type": "cnn",
            "parameters": int(model.count_params())
        }
    }

    with open(str(save_path / "evaluation_metrics.json"), 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved to {save_path / 'evaluation_metrics.json'}")

    return {
        "test_accuracy": test_acc,
        "classification_report": report_dict,
        "confusion_matrix": cm,
        "misclassified_indices": misclassified,
        "per_class_metrics": metrics["per_class_metrics"]
    }


def _plot_confusion_matrix(cm: np.ndarray, class_names: List[str], save_path: str):
    """Plot and save confusion matrix heatmap."""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix - MNIST CNN')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def _plot_error_examples(x_test: np.ndarray, y_true: np.ndarray,
                         y_pred: np.ndarray, misclassified: List[int],
                         save_path: str, max_examples: int = 20):
    """Plot a grid of misclassified examples."""
    if len(misclassified) == 0:
        print("No misclassified examples to plot.")
        return

    examples = misclassified[:max_examples]
    n_examples = len(examples)
    cols = 5
    rows = (n_examples + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten() if rows > 1 else [axes] if cols == 1 else axes

    for i, idx in enumerate(examples):
        img = x_test[idx].squeeze()
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f"True: {y_true[idx]}, Pred: {y_pred[idx]}", fontsize=10)
        axes[i].axis('off')

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle(f"Misclassified Examples ({n_examples} shown)", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Error examples saved to {save_path}")


def compare_with_baseline(cnn_accuracy: float, cnn_train_time: float,
                          baseline_metrics_path: str = "reports/baseline_metrics.json",
                          save_path: str = "reports/comparison_table.md") -> str:
    """Generate comparison table between CNN and baseline model."""
    baseline = {}
    try:
        with open(baseline_metrics_path) as f:
            baseline = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        baseline = {
            "model_type": "N/A",
            "val_accuracy": 0.0,
            "train_time_seconds": 0
        }

    comparison = f"""| Model | Test Accuracy | Train Time | Inference Time |
|-------|--------------|------------|----------------|
| Baseline {baseline.get('model_type', 'LR')} | {baseline.get('val_accuracy', 0):.4f} | {baseline.get('train_time_seconds', 0):.1f}s | {baseline.get('inference_time_seconds', 0):.4f}s |
| CNN | {cnn_accuracy:.4f} | {cnn_train_time:.1f}s | < 50ms |
"""

    with open(save_path, 'w') as f:
        f.write("# Model Comparison\n\n")
        f.write(comparison)

    print(f"Comparison table saved to {save_path}")
    return comparison


if __name__ == "__main__":
    import argparse
    import tensorflow as tf

    parser = argparse.ArgumentParser(description="Evaluate trained MNIST model")
    parser.add_argument("--model", default="models/best_cnn_model.keras",
                        help="Path to saved model")
    parser.add_argument("--config", default="config.yaml",
                        help="Path to config YAML")
    args = parser.parse_args()

    from src.config_loader import load_config
    config = load_config(args.config)

    print("Loading data...")
    data = load_mnist(source="tf", cache_dir=config["data"].get("cache_dir"))

    print("Preprocessing...")
    processed = preprocess_pipeline(
        data["x_train"], data["y_train"],
        data["x_test"], data["y_test"],
        val_size=config["data"].get("val_size", 10000)
    )

    print(f"Loading model from {args.model}...")
    model = tf.keras.models.load_model(args.model)
    model.summary()

    print("Evaluating...")
    evaluate_model(model, processed["x_test_proc"], processed["y_test_oh"])
