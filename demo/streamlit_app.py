"""Streamlit demo (hosted version): upload a panoramic dental X-ray or click a sample, get a prediction and a Grad-CAM heatmap.

    streamlit run demo/streamlit_app.py

EDUCATIONAL DEMO ONLY. Not a medical device and not for diagnosis.
"""
import sys
from pathlib import Path

import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import config as C  # noqa: E402
from dataset import preprocess_gray  # noqa: E402
from gradcam import GradCAM, overlay  # noqa: E402
from model import load_trained  # noqa: E402

LABELS = {0: "No caries annotated", 1: "Caries present"}
SAMPLES = {
    "Sample 1 (caries)": "caries_1.jpg",
    "Sample 2 (no caries)": "no_caries_1.jpg",
    "Sample 3 (caries)": "caries_2.jpg",
    "Sample 4 (no caries)": "no_caries_2.jpg",
}

st.set_page_config(page_title="Dental X-ray Classifier", page_icon="🦷", layout="wide")
st.markdown(
    """
<style>
.block-container{max-width:1100px;padding-top:1.4rem}
.hero{background:linear-gradient(135deg,#0a2540 0%,#0b6f8a 55%,#0f9b8e 100%);color:#fff;padding:26px 30px;border-radius:20px}
.hero h1{margin:0 0 6px;font-size:30px;color:#fff}
.hero p{margin:0;color:rgba(255,255,255,.88);font-size:15px;max-width:780px;line-height:1.55}
.notice{background:#fff7e6;border:1px solid #f5d9a3;color:#7a5200;padding:12px 16px;border-radius:12px;font-size:13.5px;margin:14px 0}
.kpi{background:#f1f8f8;border-radius:12px;padding:12px 14px}
.kpi b{display:block;font-size:22px;color:#0b6f8a}
.kpi span{font-size:12px;color:#4b6b73}
</style>
<div class="hero"><h1>Dental X-ray Classifier</h1>
<p>Upload a <b>panoramic dental X-ray</b>. A ResNet-18 (ImageNet transfer learning) estimates whether it shows <b>caries (cavities)</b>,
and a <b>Grad-CAM</b> heatmap shows which regions influenced the prediction.</p></div>
<div class="notice"><b>Educational demo, not a medical device.</b> Do not use it for diagnosis. The model was trained only on X-rays that contain
at least one finding, so it has never seen a healthy mouth: it answers "caries or not", not "healthy or sick".</div>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load():
    torch.set_num_threads(2)
    model = load_trained(C.MODELS_DIR / "resnet18_square.pt", n_classes=2)
    return model, GradCAM(model, model.layer4[-1])


model, cam = load()


def analyse(gray: np.ndarray):
    # Same resize as training (PIL bilinear), so scores match the evaluated model.
    small = np.asarray(Image.fromarray(gray).resize((C.IMG_W, C.IMG_H), Image.BILINEAR), dtype=np.uint8)
    heat, probs, _ = cam(preprocess_gray(small))
    shown = cv2.resize(overlay(small, heat), (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_LINEAR)
    return probs, shown


left, right = st.columns(2, gap="large")
with left:
    up = st.file_uploader("Panoramic dental X-ray (JPG or PNG)", type=["jpg", "jpeg", "png"])
    pick = st.radio("Or try a sample X-ray", list(SAMPLES), horizontal=False, index=None)
    image = None
    if up is not None:
        image = Image.open(up).convert("L")
    elif pick:
        image = Image.open(ROOT / "examples" / SAMPLES[pick]).convert("L")
    if image is not None:
        st.image(image, caption="Input", use_container_width=True)

with right:
    if image is None:
        st.info("Upload an X-ray or pick a sample on the left.")
    else:
        with st.spinner("Analysing..."):
            probs, shown = analyse(np.asarray(image))
        top = int(np.argmax(probs))
        st.subheader(LABELS[top])
        for i in (1, 0):
            st.progress(float(probs[i]), text=f"{LABELS[i]}: {probs[i]*100:.1f}%")
        st.image(shown, caption="Grad-CAM: where the model looked", use_container_width=True)

st.markdown("### How reliable is it? (measured on 750 unseen test X-rays)")
k = st.columns(4)
for col, (v, t) in zip(k, [("0.74", "ROC-AUC (95% CI 0.70 to 0.77)"), ("65.9%", "accuracy (95% CI 62.5 to 69.1)"),
                           ("57%", "of caries images found (recall)"), ("70%", "of caries flags correct (precision)")]):
    col.markdown(f'<div class="kpi"><b>{v}</b><span>{t}</span></div>', unsafe_allow_html=True)
st.caption(
    "Better than chance but modest: it misses roughly 4 in 10 caries images, because small cavities lose detail when a large panoramic "
    "X-ray is shrunk to 224x224 pixels. The samples were picked from test images the model gets right with high confidence, so they show "
    "the best case, not typical performance. Heatmaps are coarse and are an aid to interpretation, not proof of correct reasoning. "
    "Code and full report: github.com/Komaljha1504/dental-xray-classifier"
)
