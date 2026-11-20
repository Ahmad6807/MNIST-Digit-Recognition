"""Quick import validation script."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

# Test pure-Python modules
from src.config_loader import load_config
cfg = load_config('config.yaml')
print(f'config_loader OK: framework={cfg["web"]["framework"]}')

from src.data_loader import _validate_data
d = {
    "x_train": np.zeros((60000, 28, 28), dtype=np.uint8),
    "y_train": np.zeros((60000,), dtype=np.uint8),
    "x_test": np.zeros((10000, 28, 28), dtype=np.uint8),
    "y_test": np.zeros((10000,), dtype=np.uint8)
}
d["y_train"][::6000] = np.arange(10)
d["y_test"][::1000] = np.arange(10)
_validate_data(d)
print('data_loader._validate_data OK')

from src.preprocess import (
    normalize, reshape_for_cnn, split_train_val,
)
x = np.random.randint(0, 256, (100, 28, 28), dtype=np.uint8)
y = np.random.randint(0, 10, 100)
x_n = normalize(x)
assert x_n.dtype == np.float32 and x_n.max() <= 1.0
x_r = reshape_for_cnn(x_n)
assert x_r.shape == (100, 28, 28, 1)
x_tr, x_val, y_tr, y_val = split_train_val(x, y, val_size=20)
assert x_tr.shape[0] == 80 and x_val.shape[0] == 20

try:
    from src.preprocess import preprocess_for_web as pf
    p = pf(np.ones((100, 100, 3), dtype=np.uint8))
    assert p.shape == (1, 28, 28, 1)
    print('preprocess_for_web OK (opencv available)')
except ImportError:
    print('preprocess_for_web SKIPPED (opencv not installed)')

print('preprocess basic functions OK')

from src.baseline_model import train_baseline, load_baseline_model
print('baseline_model import OK')

from src.evaluate import (
    _plot_confusion_matrix, _plot_error_examples, compare_with_baseline
)
print('evaluate import OK')

# Test modules that require TensorFlow
for module_name, import_path, label in [
    ('train.set_seeds', 'from src.train import set_seeds', 'train.set_seeds'),
    ('cnn_model', 'from src.cnn_model import build_mnist_cnn, get_callbacks', 'cnn_model'),
    ('data_loader.load_mnist', 'from src.data_loader import load_mnist', 'data_loader.load_mnist'),
    ('predict', 'from src.predict import predict_single_image, load_model', 'predict'),
    ('train', 'from src.train import train', 'train'),
    ('app', 'from src.app import initialize, build_app', 'app'),
]:
    try:
        exec(import_path)
        if label == 'train.set_seeds':
            set_seeds(42)
        print(f'{label} import OK (TensorFlow available)')
    except ImportError as e:
        print(f'{label} import SKIPPED (TensorFlow not installed): {e}')
    except ModuleNotFoundError as e:
        print(f'{label} import SKIPPED (TensorFlow not installed)')

print()
print('=== IMPORT VALIDATION COMPLETE ===')
print(f'Python: {sys.version}')
print(f'numpy: {np.__version__}')
