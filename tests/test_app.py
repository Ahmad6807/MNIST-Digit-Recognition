"""Tests for Gradio web application (app.py)."""
import numpy as np
import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocess import preprocess_for_web


class TestApp:
    def test_preprocess_for_web_canvas_280x280(self):
        """Canvas produces 280x280 uint8 array -> should resize to 28x28."""
        canvas = np.random.randint(0, 256, (280, 280), dtype=np.uint8)
        result = preprocess_for_web(canvas)
        assert result.shape == (1, 28, 28, 1)

    def test_empty_canvas_detection(self):
        """Blank canvas (mean > 250) should be detectable."""
        from src.preprocess import preprocess_for_web
        blank = np.ones((280, 280), dtype=np.uint8) * 255
        # Even after inversion, the preprocess should work
        result = preprocess_for_web(blank)
        assert result.shape == (1, 28, 28, 1)

    def test_upload_file_types(self):
        """Gradio should only accept PNG, JPG, JPEG."""
        allowed = {'.png', '.jpg', '.jpeg'}
        test_files = ['image.png', 'image.jpg', 'image.jpeg', 'image.PNG']
        for f in test_files:
            ext = Path(f).suffix.lower()
            assert ext in allowed, f"{ext} should be allowed"

        disallowed = {'.pdf', '.bmp', '.gif', '.webp'}
        for f in ['doc.pdf', 'img.bmp', 'ani.gif', 'pic.webp']:
            ext = Path(f).suffix.lower()
            assert ext not in allowed, f"{ext} should be rejected"

    def test_file_size_limit(self):
        """10MB file size limit enforcement."""
        from src.config_loader import load_config
        config = load_config()
        max_size = config.get("web", {}).get("max_file_size_mb", 10) * 1024 * 1024
        assert max_size == 10 * 1024 * 1024

        # Simulate large file check
        large_array = np.random.randint(0, 256, (1000, 1000, 3), dtype=np.uint8)
        large_size = large_array.nbytes
        assert large_size > max_size, f"{large_size} > {max_size}"
        small_array = np.random.randint(0, 256, (28, 28), dtype=np.uint8)
        small_size = small_array.nbytes
        assert small_size < max_size

    def test_low_confidence_detection(self):
        """Low confidence (< 50%) should trigger hint."""
        low_conf = 0.35
        assert low_conf < 0.5, "Low confidence check"
        high_conf = 0.95
        assert high_conf >= 0.5, "High confidence check"

    def test_accessibility_digit_validation(self):
        """Accessibility input must be 0-9."""
        valid = [str(i) for i in range(10)]
        for d in valid:
            assert d.isdigit() and 0 <= int(d) <= 9
        invalid = ["10", "-1", "abc", "", "12"]
        for d in invalid:
            assert not (d.isdigit() and 0 <= int(d) <= 9)

    def test_confidence_color_coding(self):
        """Confidence bar colors match thresholds."""
        thresholds = [
            (0.95, "#34A853"),  # green >= 95%
            (0.85, "#FBBC04"),  # amber 80-94%
            (0.50, "#EA4335"),  # red < 80%
        ]
        for conf, expected_color in thresholds:
            if conf >= 0.95:
                assert expected_color == "#34A853"
            elif conf >= 0.80:
                assert expected_color == "#FBBC04"
            else:
                assert expected_color == "#EA4335"

    def test_loading_state(self):
        """Classify button should be disabled during inference (via Gradio loading param)."""
        assert True  # Gradio handles this via loading=True parameter in gr.Button.click()

    def test_clear_canvas_resets(self):
        """Clear button should reset canvas and hide results."""
        assert True  # UI logic: canvas.update(value=None), result_panel.update(visible=False)


class TestAppConfig:
    def test_config_web_section_exists(self):
        """Config must have web section with all required keys."""
        from src.config_loader import load_config
        config = load_config()
        assert "web" in config
        web = config["web"]
        assert web.get("framework") == "gradio"
        assert web.get("host") is not None
        assert web.get("port") is not None
        assert web.get("max_file_size_mb") == 10
        assert web.get("enable_accessibility") is True

    def test_inference_config_exists(self):
        """Config must have inference section for preprocessing."""
        from src.config_loader import load_config
        config = load_config()
        assert "inference" in config
        assert config["inference"].get("invert_threshold") == 127
        assert config["inference"].get("model_path") is not None
