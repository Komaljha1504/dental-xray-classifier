"""Central configuration. Paths, image size, seeds and class names live here."""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# Raw images and cached arrays are large, so they live OUTSIDE the repository.
DATA_DIR = Path(os.environ.get("DENTAL_DATA_DIR", Path.home() / "ml-data" / "dental"))
LABEL_DIR = DATA_DIR / "labels"
IMAGE_DIR = DATA_DIR / "images"
CACHE_DIR = DATA_DIR / "cache"

RESULTS_DIR = REPO / "results"
MODELS_DIR = REPO / "models"

HF_REPO = "liodon-ai/dental-panoramic-xray-yolo"
HF_API = f"https://huggingface.co/api/datasets/{HF_REPO}?blobs=true"
HF_BASE = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main/"

# The brief asks for 224 x 224. Panoramic X-rays are about 2:1, so IMG_W can be raised to
# keep the aspect ratio (see the experiment in the README).
IMG_H = int(os.environ.get("IMG_H", 224))
IMG_W = int(os.environ.get("IMG_W", 224))

SEED = 42
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Set by data_prep.py once the task is defined; kept here so every script agrees.
CLASS_NAMES = ["no_caries", "caries"]

for _d in (LABEL_DIR, IMAGE_DIR, CACHE_DIR, RESULTS_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
