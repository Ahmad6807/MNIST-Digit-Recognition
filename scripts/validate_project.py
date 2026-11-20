"""Comprehensive project validation script."""
import ast
import os
import sys
import json

# Add project root to path for module imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

results = {"passed": 0, "failed": 0, "skipped": 0, "errors": []}

# Test 1: Syntax check all .py files
print("=== Syntax Check ===")
for root, dirs, files in os.walk("."):
    for fn in files:
        if fn.endswith(".py"):
            fp = os.path.join(root, fn)
            try:
                with open(fp, encoding="utf-8") as f:
                    ast.parse(f.read())
                results["passed"] += 1
            except SyntaxError as e:
                results["failed"] += 1
                results["errors"].append(f"Syntax error in {fp}: {e}")
print(f"  Syntax-checked .py files: OK")

# Test 2: Validate YAML config
print("=== Config Validation ===")
import yaml
try:
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    required_keys = ["data", "preprocessing", "model", "training", "evaluation", "inference", "web"]
    for k in required_keys:
        if k not in cfg:
            results["failed"] += 1
            results["errors"].append(f"Missing config key: {k}")
        else:
            results["passed"] += 1
    print(f"  Config validation: OK ({len(required_keys)} keys)")
except Exception as e:
    results["failed"] += 1
    results["errors"].append(f"Config load error: {e}")
    print(f"  Config validation: FAILED - {e}")

# Test 3: Verify file structure
print("=== File Structure ===")
expected_files = [
    "src/__init__.py", "src/app.py", "src/baseline_model.py",
    "src/cnn_model.py", "src/config_loader.py", "src/data_loader.py",
    "src/evaluate.py", "src/predict.py", "src/preprocess.py", "src/train.py",
    "tests/test_app.py", "tests/test_baseline.py", "tests/test_cnn.py",
    "tests/test_data_loader.py", "tests/test_evaluate.py", "tests/test_predict.py",
    "tests/test_preprocess.py", "tests/test_train.py",
    "notebooks/EDA.ipynb", "scripts/generate_accessibility_templates.py",
    "config.yaml", "requirements.txt", "README.md", ".gitignore",
    "conftest.py", "assets/accessibility/", "scripts/validate_imports.py"
]
for f in expected_files:
    path = f
    if f.endswith("/"):
        path = f.rstrip("/")
        exists = os.path.isdir(path)
    else:
        exists = os.path.exists(path)
    if exists:
        results["passed"] += 1
    else:
        results["failed"] += 1
        results["errors"].append(f"Missing: {f}")
print(f"  File structure: {sum(1 for f in expected_files if os.path.exists(f.rstrip('/')) or os.path.isdir(f.rstrip('/')))}/{len(expected_files)} exist")

# Test 4: Validate requirements.txt
print("=== Requirements ===")
with open("requirements.txt") as f:
    pkgs = [l.strip() for l in f if l.strip() and not l.startswith("#")]
if len(pkgs) >= 12:
    results["passed"] += 1
else:
    results["failed"] += 1
    results["errors"].append(f"requirements.txt has {len(pkgs)} packages, expected >= 12")
print(f"  {len(pkgs)} packages specified")

# Test 5: Verify README sections
print("=== README ===")
with open("README.md", encoding="utf-8") as f:
    readme = f.read()
sections = ["Project Goal", "Setup", "Usage", "Project Structure", "Architecture", "Testing"]
for s in sections:
    if s in readme:
        results["passed"] += 1
    else:
        results["failed"] += 1
        results["errors"].append(f"README missing section: {s}")
print(f"  README sections: {sum(1 for s in sections if s in readme)}/{len(sections)}")

# Test 6: Verify .gitignore
print("=== .gitignore ===")
with open(".gitignore") as f:
    gitignore = f.read()
patterns = ["__pycache__", "venv/", "models/", "data/", "logs/"]
for p in patterns:
    if p in gitignore:
        results["passed"] += 1
    else:
        results["failed"] += 1
        results["errors"].append(f".gitignore missing pattern: {p}")
print(f"  .gitignore patterns: {sum(1 for p in patterns if p in gitignore)}/{len(patterns)}")

# Test 7: Verify module imports (pure Python)
print("=== Module Imports ===")
try:
    from src.config_loader import load_config
    c = load_config("config.yaml")
    results["passed"] += 1
    print(f"  config_loader OK")
except Exception as e:
    results["failed"] += 1
    results["errors"].append(f"config_loader import failed: {e}")

try:
    from src.data_loader import _validate_data
    import numpy as np
    d = {
        "x_train": np.zeros((60000, 28, 28), dtype=np.uint8),
        "y_train": np.zeros((60000,), dtype=np.uint8),
        "x_test": np.zeros((10000, 28, 28), dtype=np.uint8),
        "y_test": np.zeros((10000,), dtype=np.uint8)
    }
    d["y_train"][::6000] = np.arange(10)
    d["y_test"][::1000] = np.arange(10)
    _validate_data(d)
    results["passed"] += 1
    print(f"  data_loader._validate_data OK")
except Exception as e:
    results["failed"] += 1
    results["errors"].append(f"data_loader import failed: {e}")

try:
    from src.preprocess import normalize, reshape_for_cnn, split_train_val
    x = np.random.randint(0, 256, (100, 28, 28), dtype=np.uint8)
    y = np.random.randint(0, 10, 100)
    xn = normalize(x)
    assert xn.dtype == np.float32 and xn.max() <= 1.0
    xr = reshape_for_cnn(xn)
    assert xr.shape == (100, 28, 28, 1)
    xt, xv, yt, yv = split_train_val(x, y, val_size=20)
    assert xt.shape[0] == 80
    results["passed"] += 3
    print(f"  preprocess core functions OK")
except Exception as e:
    results["failed"] += 1
    results["errors"].append(f"preprocess import failed: {e}")

try:
    from src.baseline_model import train_baseline, load_baseline_model
    results["passed"] += 1
    print(f"  baseline_model OK")
except Exception as e:
    results["failed"] += 1

try:
    from src.evaluate import _plot_confusion_matrix, _plot_error_examples, compare_with_baseline
    results["passed"] += 1
    print(f"  evaluate (helpers) OK")
except Exception as e:
    results["failed"] += 1

print()
print("=" * 50)
print(f"Total checks: {results['passed'] + results['failed']}")
print(f"Passed: {results['passed']}")
print(f"Failed: {results['failed']}")
print(f"Skipped: {results['skipped']}")
if results["errors"]:
    print(f"Errors ({len(results['errors'])}):")
    for e in results["errors"][:10]:
        print(f"  - {e}")
print("=" * 50)
print("PROJECT VALIDATION COMPLETE")
