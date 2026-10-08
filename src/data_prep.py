"""Download labels, derive image-level classes, build a stratified manifest, fetch images, cache arrays.

Usage (run from the repository root):
    python src/data_prep.py labels      # fetch ~10k tiny YOLO label files and build labels.csv
    python src/data_prep.py manifest    # pick a balanced subset and stratified train/val/test split
    python src/data_prep.py images      # download the selected images
    python src/data_prep.py cache       # resize to IMG_H x IMG_W grayscale and store as .npy

Task definition (important, and repeated in the README):
  The source dataset has bounding boxes for 3 findings (caries, periapical lesion, impacted tooth) and EVERY image has
  at least one finding, so there are no healthy radiographs. The image-level label is therefore
      1 = "caries"     -> at least one caries box
      0 = "no_caries"  -> no caries box (the image still has other findings)
"""
import concurrent.futures as cf
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import config as C

N_PER_CLASS = 2500


def _get(url: str, retries: int = 5) -> bytes:
    for attempt in range(retries):
        try:
            return urllib.request.urlopen(url, timeout=60).read()
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(1.5 * (attempt + 1))


def fetch_index() -> dict:
    path = C.DATA_DIR / "file_index.json"
    if not path.exists():
        files = [f["rfilename"] for f in json.loads(_get(C.HF_API))["siblings"]]
        index = {
            "labels": [f for f in files if f.startswith("labels/train/") and f.endswith(".txt")],
            "images": [f for f in files if f.startswith("images/train/")],
        }
        path.write_text(json.dumps(index))
    return json.loads(path.read_text())


def image_rel_paths() -> dict:
    """image_id -> path inside the Hugging Face repo. Most are .jpg but the DENTEX images are .png."""
    return {Path(p).stem: p for p in fetch_index()["images"]}


def local_image_path(image_id: str) -> Path:
    rel = image_rel_paths()[image_id]
    return C.IMAGE_DIR / Path(rel).name


def source_of(image_id: str) -> str:
    """DENTEX images are named train_N, OralXrays-9 images are named oral_NNNNNN."""
    return "dentex" if image_id.startswith("train_") else "oralxrays9"


def download_labels():
    index = fetch_index()

    def one(rel):
        out = C.LABEL_DIR / rel.split("/")[-1]
        if not out.exists():
            out.write_bytes(_get(C.HF_BASE + rel))

    with cf.ThreadPoolExecutor(24) as ex:
        list(tqdm(ex.map(one, index["labels"]), total=len(index["labels"]), desc="labels"))

    rows = []
    for f in sorted(C.LABEL_DIR.glob("*.txt")):
        classes = [int(line.split()[0]) for line in f.read_text().splitlines() if line.strip()]
        rows.append(
            dict(
                image_id=f.stem,
                n_caries=classes.count(0),
                n_periapical=classes.count(1),
                n_impacted=classes.count(2),
                n_boxes=len(classes),
            )
        )
    df = pd.DataFrame(rows)
    df["caries"] = (df.n_caries > 0).astype(int)
    df.to_csv(C.DATA_DIR / "labels.csv", index=False)
    print(f"labels.csv: {len(df)} images | caries {df.caries.sum()} | no_caries {(df.caries == 0).sum()}")
    print(f"images with zero annotations: {(df.n_boxes == 0).sum()}")


def build_manifest():
    df = pd.read_csv(C.DATA_DIR / "labels.csv")
    # The 678 DENTEX images are 96% caries versus 56% in OralXrays-9. Mixing them would let a model learn "which
    # scanner/source" as a shortcut to the label, so only the single large source is used (see README, "Data").
    df["source"] = df.image_id.map(source_of)
    df = df[df.source == "oralxrays9"].reset_index(drop=True)
    parts = [g.sample(n=min(N_PER_CLASS, len(g)), random_state=C.SEED) for _, g in df.groupby("caries")]
    sub = pd.concat(parts).reset_index(drop=True)
    # Stratified 70 / 15 / 15 split so the class ratio holds in every split.
    train, rest = train_test_split(sub, test_size=0.30, stratify=sub.caries, random_state=C.SEED)
    val, test = train_test_split(rest, test_size=0.50, stratify=rest.caries, random_state=C.SEED)
    train, val, test = (d.assign(split=s) for d, s in ((train, "train"), (val, "val"), (test, "test")))
    manifest = pd.concat([train, val, test]).reset_index(drop=True)
    manifest.to_csv(C.DATA_DIR / "manifest.csv", index=False)
    print(manifest.groupby(["split", "caries"]).size().unstack())


def download_images():
    manifest = pd.read_csv(C.DATA_DIR / "manifest.csv")
    rel = image_rel_paths()

    def one(image_id):
        out = C.IMAGE_DIR / Path(rel[image_id]).name
        if out.exists() and out.stat().st_size > 0:
            return
        out.write_bytes(_get(C.HF_BASE + rel[image_id]))

    with cf.ThreadPoolExecutor(16) as ex:
        list(tqdm(ex.map(one, manifest.image_id), total=len(manifest), desc="images"))


def _load_resized(args):
    path, h, w = args
    with Image.open(path) as im:
        return im.width, im.height, np.asarray(im.convert("L").resize((w, h), Image.BILINEAR), dtype=np.uint8)


def cache_arrays():
    manifest = pd.read_csv(C.DATA_DIR / "manifest.csv")
    widths, heights = [], []
    arrays = {s: [] for s in ("train", "val", "test")}
    jobs = [(str(local_image_path(i)), C.IMG_H, C.IMG_W) for i in manifest.image_id]
    with cf.ProcessPoolExecutor(max_workers=6) as ex:  # JPEG decoding dominates; use several cores
        results = list(tqdm(ex.map(_load_resized, jobs, chunksize=16), total=len(jobs), desc=f"resize {C.IMG_H}x{C.IMG_W}"))
    for row, (w, h, arr) in zip(manifest.itertuples(), results):
        widths.append(w)
        heights.append(h)
        arrays[row.split].append(arr)
    manifest["width"], manifest["height"] = widths, heights
    manifest["source"] = manifest.image_id.map(source_of)
    manifest.to_csv(C.DATA_DIR / "manifest.csv", index=False)
    for split, items in arrays.items():
        rows = manifest[manifest.split == split]
        np.save(C.CACHE_DIR / f"{split}_{C.IMG_H}x{C.IMG_W}.npy", np.stack(items))
        np.save(C.CACHE_DIR / f"{split}_labels.npy", rows.caries.to_numpy())
        rows.image_id.to_csv(C.CACHE_DIR / f"{split}_ids.csv", index=False)
    print({s: len(v) for s, v in arrays.items()})


if __name__ == "__main__":
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    steps = {"labels": download_labels, "manifest": build_manifest, "images": download_images, "cache": cache_arrays}
    for name in steps if step == "all" else [step]:
        steps[name]()
