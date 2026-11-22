# MNIST Digit Recognition Classifier

A machine learning pipeline for classifying handwritten digits (0-9) using the MNIST dataset, with both CLI and Gradio web inference interfaces.

## Project Goal

Achieve **≥ 99.0% test accuracy** on the MNIST test set using a Convolutional Neural Network, with a reproducible pipeline for inference on new handwritten digit images.

## Dataset

- **Source:** [MNIST](http://yann.lecun.com/exdb/mnist/) (Modified National Institute of Standards and Technology)
- **Size:** 70,000 grayscale images (28×28 pixels), 60K training + 10K test
- **Loading:** TensorFlow `tf.keras.datasets.mnist` (default) or Kaggle CSV
- **License:** [Yann LeCun / NIST](http://yann.lecun.com/exdb/mnist/)

## Setup

### Prerequisites

- Python 3.10–3.12 (Python 3.12 recommended; TensorFlow does not currently provide a stable package for Python 3.14)
- 8 GB RAM (minimum)
- GPU recommended for training (NVIDIA T4 or equivalent), CPU supported (~3h)

### Installation

```bash
git clone <repository-url>
cd mnist-digit-recognition

python -m venv venv
source venv/bin/activate          # Linux/Mac
# venv\Scripts\activate           # Windows

python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -r requirements-dev.txt  # Training, evaluation, and tests
```

If installation reports that no TensorFlow distribution is available, check
`python --version` and use Python 3.10, 3.11, or 3.12 to create the environment.
The environment's default Python may be newer than TensorFlow supports.

## Usage

### 1. Train the CNN

Full training (50 epochs, batch size 128):

```bash
python src/train.py
```

Options:
```bash
python src/train.py --dry-run           # Quick test: 2000 samples, 3 epochs
python src/train.py --epochs 30         # Override epochs
python src/train.py --batch-size 64     # Override batch size
```

**Expected training time:** ~1hr on GPU (T4), ~3hr on CPU.

### 2. Train Baseline Model (for comparison)

```bash
python -c "from src.baseline_model import train_baseline; import numpy as np; from src.data_loader import load_mnist; from src.preprocess import normalize; d=load_mnist(source='tf'); x_n=normalize(d['x_train'].reshape(-1,784)); x_t=normalize(d['x_test'].reshape(-1,784)); r=train_baseline(x_n,d['y_train'],x_t,d['y_test'])"
```

### 3. Evaluate

```bash
python src/evaluate.py --model models/best_cnn_model.keras
```

Generates:
- `reports/evaluation_metrics.json` — per-class metrics
- `reports/confusion_matrix.png` — heatmap
- `reports/error_examples.png` — 20 misclassified samples

### 4. CLI Inference

```bash
python src/predict.py path/to/digit.png
python src/predict.py path/to/digit.png --model models/best_cnn_model.keras
python src/predict.py path/to/digit.png --all   # Show all class probabilities
```

### 5. Web App (Gradio)

```bash
python src/app.py
```

Opens at `http://localhost:7860` with:
- **Draw tab:** Sketchpad to draw digits with mouse/touch
- **Upload tab:** Upload PNG/JPG images
- **Results:** Predicted digit, confidence bar (color-coded), all probabilities
- **Accessibility:** Type a digit 0-9 for synthetic prediction
- **Error handling:** Empty input, invalid files, model failures

## Project Structure

```
mnist-digit-recognition/
├── assets/
│   └── accessibility/       # Pre-stored digit templates for accessibility
├── data/                    # Raw & processed data (gitignored)
├── logs/                    # TensorBoard logs, training metrics CSV
├── models/                  # Best CNN checkpoint is included for Vercel inference
├── notebooks/
│   └── EDA.ipynb            # Exploratory Data Analysis
├── reports/                 # Generated figures, metrics, comparisons
├── scripts/
│   └── generate_accessibility_templates.py
├── src/
│   ├── __init__.py
│   ├── app.py               # Gradio web application
│   ├── baseline_model.py    # Logistic Regression classifier
│   ├── cnn_model.py         # CNN architecture definition
│   ├── config_loader.py     # YAML configuration loader
│   ├── data_loader.py       # MNIST dataset loading & validation
│   ├── evaluate.py          # Test evaluation & metrics
│   ├── predict.py            # CLI inference tool
│   ├── preprocess.py        # Preprocessing pipeline
│   └── train.py             # Training orchestrator
├── tests/
│   ├── test_data_loader.py
│   ├── test_preprocess.py
│   ├── test_baseline.py
│   ├── test_cnn.py
│   ├── test_train.py
│   ├── test_evaluate.py
│   ├── test_predict.py
│   └── test_app.py
├── config.yaml              # Centralized configuration
├── conftest.py              # Pytest configuration
├── requirements.txt         # Pinned dependencies
└── README.md
```

### Deploy on Vercel

The Gradio interface is exposed as an ASGI application from `src/app.py` for
Vercel's Python runtime. Include `models/best_cnn_model.keras` in the Git
repository so deployed predictions have a trained model. Other generated model
files and non-runtime assets are excluded from the deployed function.

## Architecture

### CNN Architecture (1.6M params)

| Layer | Type | Output Shape | Params |
|-------|------|-------------|--------|
| input | Input | (28, 28, 1) | 0 |
| conv1 | Conv2D (32, 3×3, ReLU) | (28, 28, 32) | 320 |
| conv2 | Conv2D (64, 3×3, ReLU) | (28, 28, 64) | 18,496 |
| pool1 | MaxPooling2D (2×2) | (14, 14, 64) | 0 |
| dropout_conv | Dropout (0.25) | (14, 14, 64) | 0 |
| flatten | Flatten | (12,544) | 0 |
| dense1 | Dense (128, ReLU) | (128) | 1,605,760 |
| dropout_dense | Dropout (0.5) | (128) | 0 |
| output | Dense (10, Softmax) | (10) | 1,290 |

**Total:** 1,625,866 trainable parameters

### Data Flow

```
Raw (uint8) → Normalize [0,1] → Reshape (28,28,1) → One-hot encode
→ Model.fit() → ModelCheckpoint (.keras) → Evaluate
```

### Inference Flow (CLI & Web share same preprocessing)

```
Image → Grayscale → Resize 28×28 → Auto-invert → Normalize → Predict → Result
```

## Results

| Model | Test Accuracy | Train Time | Inference Time | Model Size |
|-------|--------------|------------|----------------|------------|
| Logistic Regression | ~92.5% | ~45s | < 10ms | ~6 MB |
| CNN (this project) | **≥ 99.0%** | ~12min (GPU) | < 50ms CPU / < 5ms GPU | ~6.5 MB |

## Performance

- **Inference:** < 50ms on CPU, < 5ms on GPU (single image)
- **Training:** < 1hr on NVIDIA T4, < 3hr on CPU
- **Model size:** < 10 MB

## Reproducibility

- All random seeds fixed (Python, NumPy, TensorFlow)
- Data splits are deterministic (stratified, seed=42)
- Pinned versions in `requirements.txt`
- TensorFlow deterministic ops enabled (`TF_DETERMINISTIC_OPS=1`)

## Configuration

All training and inference parameters are in `config.yaml`. Key sections:

- `data.source`: "tf" or "csv"
- `preprocessing.augmentation.enabled`: Toggle data augmentation
- `training`: Epochs, batch size, callbacks
- `inference`: Model path, invert threshold
- `web`: Gradio host, port, file size limits

## Testing

```bash
pytest tests/ -v
```

Runs unit tests for data loading, preprocessing, model architecture, training, evaluation, and inference.

## License

This project uses the MNIST dataset which is available under the terms of the [Creative Commons Attribution-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-sa/4.0/) (CC BY-SA 4.0) by Yann LeCun and Corinna Cortes.

## Future Work

- ONNX model export
- Batch inference for multiple images
- Model switching (CNN vs baseline) in web UI
- Docker containerization
- CI/CD pipeline with automated testing
