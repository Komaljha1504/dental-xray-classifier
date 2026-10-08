"""Gradio demo: upload a panoramic dental X-ray (or click a sample), get a prediction, its confidence and a Grad-CAM heatmap.

    python src/app.py            # opens http://127.0.0.1:7860

EDUCATIONAL DEMO ONLY. Not a medical device and not for diagnosis.
"""
import os
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch
from PIL import Image

import config as C
from dataset import preprocess_gray
from gradcam import GradCAM, overlay
from model import load_trained

TAG = os.environ.get("MODEL_TAG", "square")
LABELS = {0: "no caries annotated", 1: "caries present"}
EXAMPLES = Path(__file__).resolve().parents[1] / "examples"

torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))
_model = load_trained(C.MODELS_DIR / f"resnet18_{TAG}.pt", n_classes=2)
_cam = GradCAM(_model, _model.layer4[-1])


def analyse(image):
    if image is None:
        return None, None
    gray = image if image.ndim == 2 else cv2.cvtColor(image[..., :3], cv2.COLOR_RGB2GRAY)
    # Same resize as training (data_prep.py uses PIL bilinear), so demo scores match the evaluated model exactly.
    small = np.asarray(Image.fromarray(gray).resize((C.IMG_W, C.IMG_H), Image.BILINEAR), dtype=np.uint8)
    heat, probs, _ = _cam(preprocess_gray(small))
    scores = {LABELS[i]: float(probs[i]) for i in (0, 1)}
    shown = overlay(small, heat)
    shown = cv2.resize(shown, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)
    return scores, shown


THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.teal,
    secondary_hue=gr.themes.colors.cyan,
    neutral_hue=gr.themes.colors.slate,
    font=[gr.themes.GoogleFont("Plus Jakarta Sans"), "ui-sans-serif", "system-ui", "sans-serif"],
).set(
    body_background_fill="#f3f8f8",
    block_radius="16px",
    button_primary_background_fill="linear-gradient(135deg, #0f9b8e 0%, #0b6f8a 100%)",
    button_primary_background_fill_hover="linear-gradient(135deg, #0c8579 0%, #095c74 100%)",
    button_primary_text_color="white",
)

CSS = """
.gradio-container { max-width: 1100px !important; margin: 0 auto; }
.hero { background: linear-gradient(135deg, #0a2540 0%, #0b6f8a 55%, #0f9b8e 100%); color: #fff; padding: 28px 32px;
        border-radius: 20px; box-shadow: 0 18px 40px -22px rgba(10,37,64,.7); }
.hero h1 { margin: 0 0 6px; font-size: 30px; font-weight: 800; letter-spacing: -0.02em; color: #fff; }
.hero p { margin: 0; color: rgba(255,255,255,.86); font-size: 15px; max-width: 760px; line-height: 1.55; }
.hero b, .hero strong { color: #fff !important; background: none !important; font-weight: 700; }
.pills { margin-top: 16px; display: flex; flex-wrap: wrap; gap: 8px; }
.pill { background: rgba(255,255,255,.14); border: 1px solid rgba(255,255,255,.28); padding: 5px 12px; border-radius: 999px;
        font-size: 12.5px; font-weight: 600; color: #fff; }
.notice { background: #fff7e6; border: 1px solid #f5d9a3; color: #7a5200; padding: 12px 16px; border-radius: 12px; font-size: 13.5px; }
.card { background: #fff; border: 1px solid #dbe8e8; border-radius: 16px; padding: 18px 22px; font-size: 14px; line-height: 1.6; color: #264653; }
.card h3 { margin: 0 0 8px; font-size: 16px; color: #0a2540; }
.stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin: 10px 0 4px; }
.stat { background: #f1f8f8; border-radius: 12px; padding: 12px 14px; }
.stat b { display: block; font-size: 22px; color: #0b6f8a; }
.stat span { font-size: 12px; color: #4b6b73; }
footer { display: none !important; }
"""

HERO = """
<div class="hero">
  <h1>Dental X-ray Classifier</h1>
  <p>Upload a <b>panoramic dental X-ray</b>. A ResNet-18 (ImageNet transfer learning) estimates whether it shows
  <b>caries (cavities)</b>, and a <b>Grad-CAM</b> heatmap shows which regions influenced the prediction.</p>
  <div class="pills"><span class="pill">PyTorch</span><span class="pill">Transfer learning</span>
  <span class="pill">Grad-CAM</span><span class="pill">5,000 X-rays</span><span class="pill">Held-out test set</span></div>
</div>
"""

NOTICE = """<div class="notice"><b>Educational demo, not a medical device.</b> Do not use it for diagnosis. The model was trained only on
X-rays that contain at least one finding, so it has never seen a healthy mouth: it answers "caries or not", not "healthy or sick".</div>"""

CARD = """
<div class="card">
  <h3>How reliable is it? (measured on 750 unseen test X-rays)</h3>
  <div class="stats">
    <div class="stat"><b>0.74</b><span>ROC-AUC (95% CI 0.70 to 0.77)</span></div>
    <div class="stat"><b>65.9%</b><span>accuracy (95% CI 62.5 to 69.1)</span></div>
    <div class="stat"><b>57%</b><span>of caries images found (recall)</span></div>
    <div class="stat"><b>70%</b><span>of caries flags correct (precision)</span></div>
  </div>
  Better than chance but modest: it misses roughly 4 in 10 caries images, because small cavities lose detail when a large
  panoramic X-ray is shrunk to 224x224 pixels. The sample X-rays above were picked from test images the model gets right with
  high confidence, so they show the best case, not typical performance. Heatmaps are coarse and are an aid to interpretation,
  not proof of correct reasoning.
</div>
"""

with gr.Blocks(title="Dental X-ray Classifier") as demo:
    gr.HTML(HERO)
    gr.HTML(NOTICE)
    with gr.Row(equal_height=False):
        with gr.Column(scale=1):
            image_in = gr.Image(label="Panoramic dental X-ray", type="numpy", height=340)
            run = gr.Button("Analyse X-ray", variant="primary", size="lg")
            gr.Examples(
                examples=[[str(EXAMPLES / n)] for n in ("caries_1.jpg", "no_caries_1.jpg", "caries_2.jpg", "no_caries_2.jpg")],
                inputs=image_in,
                label="Or try a sample X-ray (click one)",
            )
        with gr.Column(scale=1):
            label_out = gr.Label(label="Prediction", num_top_classes=2)
            cam_out = gr.Image(label="Grad-CAM: where the model looked", height=340)
    gr.HTML(CARD)
    run.click(analyse, image_in, [label_out, cam_out])
    image_in.upload(analyse, image_in, [label_out, cam_out])

if __name__ == "__main__":
    demo.launch(theme=THEME, css=CSS)
