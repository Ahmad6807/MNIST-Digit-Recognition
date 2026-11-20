"""Gradio interface for drawing or uploading handwritten digits."""
import sys
from pathlib import Path
from typing import Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import gradio as gr
import numpy as np
import pandas as pd

from src.config_loader import load_config
from src.predict import load_model, predict_single_image
from src.preprocess import generate_synthetic_digit, load_accessibility_templates


config = None
model = None
model_load_error = None
accessibility_templates = None


def _project_path(value: str) -> Path:
    """Resolve configured relative paths against the project root."""
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def initialize() -> None:
    """Load configuration and optional runtime assets."""
    global config, model, model_load_error, accessibility_templates

    config = load_config()
    inference_cfg = config.get("inference", {})
    model_path = _project_path(inference_cfg.get("model_path", "models/best_cnn_model.keras"))
    try:
        model = load_model(str(model_path))
        model_load_error = None
        print(f"Model loaded successfully from {model_path}")
    except Exception as exc:
        model = None
        model_load_error = f"Model unavailable. Train the CNN to enable predictions. ({exc})"
        print(f"Warning: {model_load_error}", file=sys.stderr)

    web_cfg = config.get("web", {})
    accessibility_templates = None
    if web_cfg.get("enable_accessibility", True):
        assets_dir = _project_path(web_cfg.get("accessibility_assets_dir", "assets/accessibility"))
        accessibility_templates = load_accessibility_templates(str(assets_dir))


def _empty_result(message: str = "") -> Tuple:
    return (None, "", pd.DataFrame({"Digit": [], "Probability": []}),
            gr.update(visible=False), message)


def _run_prediction(image: Optional[np.ndarray]) -> Tuple:
    if model is None:
        return _empty_result(model_load_error or "The model is not ready yet.")
    if image is None:
        return _empty_result("Draw a digit or choose an image first.")

    try:
        threshold = config.get("inference", {}).get("invert_threshold", 127)
        result = predict_single_image(image, model, invert_threshold=threshold)
        return _format_result(result)
    except Exception as exc:
        print(f"Inference error: {exc}", file=sys.stderr)
        return _empty_result("Could not classify this image. Try a clearer digit.")


def classify_draw(canvas_img: Optional[np.ndarray]) -> Tuple:
    """Classify the sketchpad image, rejecting blank canvases."""
    # Gradio 6 Sketchpad (ImageEditor) sends an EditorValue dictionary.
    if isinstance(canvas_img, dict):
        canvas_img = canvas_img.get("composite")
    if canvas_img is None or canvas_img.size == 0:
        return _empty_result("Draw a digit before classifying.")
    blank_threshold = config.get("inference", {}).get("blank_threshold", 250) if config else 250
    if float(np.mean(canvas_img)) >= blank_threshold:
        return _empty_result("Draw a digit before classifying.")
    return _run_prediction(canvas_img)


def classify_upload(upload_img: Optional[np.ndarray]) -> Tuple:
    """Classify an uploaded image and enforce the configured size limit."""
    if upload_img is None:
        return _empty_result("Choose an image before classifying.")
    max_size_mb = (config or {}).get("web", {}).get("max_file_size_mb", 10)
    if upload_img.nbytes > max_size_mb * 1024 * 1024:
        return _empty_result(f"Image is too large. The limit is {max_size_mb} MB.")
    return _run_prediction(upload_img)


def accessibility_handler(digit_str: str) -> Tuple:
    """Predict from a supplied digit using local MNIST templates."""
    if not digit_str or len(digit_str) != 1 or digit_str not in "0123456789":
        return _empty_result("Enter one digit from 0 to 9.")
    if model is None:
        return _empty_result(model_load_error or "The model is not ready yet.")
    try:
        synthetic = generate_synthetic_digit(int(digit_str), accessibility_templates or {})
        probabilities = model.predict(synthetic, verbose=0)[0]
        predicted_class = int(np.argmax(probabilities))
        return _format_result({
            "predicted_class": predicted_class,
            "confidence": float(probabilities[predicted_class]),
            "all_probabilities": probabilities,
        })
    except Exception as exc:
        print(f"Accessibility inference error: {exc}", file=sys.stderr)
        return _empty_result("Could not run the example prediction. Please try again.")


def _format_result(result: dict) -> Tuple:
    digit = int(result["predicted_class"])
    confidence = float(result["confidence"])
    probabilities = np.asarray(result["all_probabilities"], dtype=float)
    if probabilities.shape != (10,):
        raise ValueError("Expected ten class probabilities from the model.")

    if confidence >= 0.95:
        color, label = "#16a36a", "High confidence"
    elif confidence >= 0.80:
        color, label = "#d58a00", "Medium confidence"
    else:
        color, label = "#d64b4b", "Low confidence"

    confidence_html = f"""
    <div class="confidence-card">
      <div class="confidence-heading"><span>{label}</span><strong>{confidence:.1%}</strong></div>
      <div class="confidence-track"><div class="confidence-fill" style="width:{confidence * 100:.1f}%;background:{color}"></div></div>
      {('<p class="confidence-hint">Try a larger, more centered digit.</p>' if confidence < 0.5 else '')}
    </div>
    """
    plot_data = pd.DataFrame({"Digit": [str(i) for i in range(10)], "Probability": probabilities})
    return digit, confidence_html, plot_data, gr.update(visible=True), ""


def clear_result() -> Tuple:
    message = f"⚠️ {model_load_error}" if model_load_error else ""
    return None, "", pd.DataFrame({"Digit": [], "Probability": []}), gr.update(visible=False), message


APP_CSS = """
.gradio-container {max-width: 1040px !important; margin: 0 auto !important;}
.app-header {text-align:center; padding: 22px 0 14px;}
.app-header h1 {margin-bottom: 6px; letter-spacing:-.03em;}
.app-subtitle {color: var(--body-text-color-subdued); font-size: 1rem;}
.confidence-card {padding: 14px 16px; border:1px solid var(--border-color-primary); border-radius:14px;}
.confidence-heading {display:flex; justify-content:space-between; margin-bottom:10px;}
.confidence-track {height:10px; background:var(--background-fill-secondary); border-radius:20px; overflow:hidden;}
.confidence-fill {height:100%; border-radius:20px; transition:width .25s ease;}
.confidence-hint {margin:10px 0 0; color:var(--body-text-color-subdued);}
#digit-result input {font-size:2.5rem !important; font-weight:700 !important; text-align:center;}
"""


def build_app() -> gr.Blocks:
    """Create the responsive digit recognition interface."""
    global config
    if config is None:
        config = load_config()
    web_cfg = config.get("web", {})
    title = web_cfg.get("title", "MNIST Digit Recognizer")

    with gr.Blocks(title=title) as demo:
        gr.Markdown(f"# {title}\n<div class='app-subtitle'>Draw or upload a handwritten digit to see what the model predicts.</div>", elem_classes="app-header")
        status = gr.Markdown(visible=bool(model_load_error), value=f"⚠️ {model_load_error}" if model_load_error else "")

        with gr.Row(equal_height=True):
            with gr.Column(scale=5):
                with gr.Tabs():
                    with gr.TabItem("Draw a digit", id="draw_tab"):
                        canvas = gr.Sketchpad(
                            height=280, width=280, canvas_size=(280, 280),
                            image_mode="RGB", layers=False,
                            brush=gr.Brush(default_size=12, colors=["#000000"], color_mode="fixed"),
                            label="Draw in the box", elem_id="digit-canvas",
                        )
                        with gr.Row():
                            draw_button = gr.Button("Recognize digit", variant="primary")
                            draw_clear = gr.Button("Clear", variant="secondary")
                        if web_cfg.get("enable_accessibility", True):
                            with gr.Accordion("Can't draw? Try a sample digit", open=False):
                                accessibility_input = gr.Textbox(
                                    label="Digit from 0 to 9", placeholder="e.g. 7",
                                    max_lines=1, elem_id="accessibility-digit-input",
                                )
                    with gr.TabItem("Upload an image", id="upload_tab"):
                        upload = gr.Image(
                            type="numpy", sources=["upload"], label="Choose a digit image",
                            height=330,
                        )
                        with gr.Row():
                            upload_button = gr.Button("Recognize image", variant="primary")
                            upload_clear = gr.Button("Clear", variant="secondary")

            with gr.Column(scale=4):
                with gr.Group(visible=False) as result_panel:
                    gr.Markdown("### Prediction")
                    predicted_digit = gr.Number(label="Recognized digit", precision=0, interactive=False, elem_id="digit-result")
                    confidence_display = gr.HTML()
                    with gr.Accordion("Class probabilities", open=True):
                        prob_plot = gr.BarPlot(
                            x="Digit", y="Probability", title="Probability by digit",
                            x_title="Digit", y_title="Probability",
                            height=300,
                        )

        outputs = [predicted_digit, confidence_display, prob_plot, result_panel, status]
        draw_button.click(classify_draw, inputs=canvas, outputs=outputs)
        upload_button.click(classify_upload, inputs=upload, outputs=outputs)
        if web_cfg.get("enable_accessibility", True):
            accessibility_input.submit(accessibility_handler, inputs=accessibility_input, outputs=outputs)
        draw_clear.click(lambda: (None, *clear_result()), outputs=[canvas, *outputs])
        upload_clear.click(lambda: (None, *clear_result()), outputs=[upload, *outputs])

    return demo


if __name__ == "__main__":
    initialize()
    app = build_app()
    web_cfg = config.get("web", {})
    app.launch(
        server_name=web_cfg.get("host", "127.0.0.1"),
        server_port=web_cfg.get("port", 7860),
        share=web_cfg.get("share", False),
        theme=gr.themes.Soft(),
        css=APP_CSS,
    )
