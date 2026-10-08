"""Exploratory analysis: class distribution, finding co-occurrence, image sizes, sample images.

Run from the repository root after `data_prep.py labels` (and `cache` for the size statistics):
    python src/eda.py
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image

import config as C
from data_prep import local_image_path

sns.set_theme(style="whitegrid", context="notebook")
COLORS = {"caries": "#d9534f", "no_caries": "#4a7fb5"}


def main():
    labels = pd.read_csv(C.DATA_DIR / "labels.csv")
    manifest = pd.read_csv(C.DATA_DIR / "manifest.csv")
    summary = {}

    # 1. Class distribution (whole dataset vs the balanced subset used for training)
    full = labels.caries.map({1: "caries", 0: "no_caries"}).value_counts()
    sub = manifest.caries.map({1: "caries", 0: "no_caries"}).value_counts()
    summary["full_dataset"] = {k: int(v) for k, v in full.items()}
    summary["balanced_subset"] = {k: int(v) for k, v in sub.items()}
    summary["images_with_zero_annotations"] = int((labels.n_boxes == 0).sum())
    summary["full_caries_share"] = round(float(full["caries"] / full.sum()), 3)
    labels["source"] = labels.image_id.map(lambda i: "dentex" if i.startswith("train_") else "oralxrays9")
    by_src = labels.groupby("source").caries.agg(images="count", caries="sum")
    by_src["caries_share"] = (by_src.caries / by_src.images).round(3)
    summary["by_source"] = by_src.reset_index().to_dict("records")
    summary["decision"] = "DENTEX images excluded: 96% caries vs 56% elsewhere would create a source shortcut."

    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    full.reindex(["caries", "no_caries"]).plot.bar(ax=ax[0], color=[COLORS["caries"], COLORS["no_caries"]], rot=0)
    ax[0].set_title(f"Whole dataset (n={len(labels):,})\ncaries = {summary['full_caries_share']:.0%}")
    ax[0].set_ylabel("images")
    sub.reindex(["caries", "no_caries"]).plot.bar(ax=ax[1], color=[COLORS["caries"], COLORS["no_caries"]], rot=0)
    ax[1].set_title(f"Balanced subset used here (n={len(manifest):,})")
    finding_counts = {
        "caries": int((labels.n_caries > 0).sum()),
        "periapical\nlesion": int((labels.n_periapical > 0).sum()),
        "impacted\ntooth": int((labels.n_impacted > 0).sum()),
    }
    summary["images_per_finding"] = {k.replace("\n", " "): v for k, v in finding_counts.items()}
    ax[2].bar(finding_counts.keys(), finding_counts.values(), color="#7a6f9b")
    ax[2].set_title("Images containing each finding\n(an image can have several)")
    for a in ax:
        for p in a.patches:
            a.annotate(f"{int(p.get_height()):,}", (p.get_x() + p.get_width() / 2, p.get_height()), ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(C.RESULTS_DIR / "eda_class_distribution.png", dpi=140)
    plt.close(fig)

    # 2. Image sizes (only available once the subset has been cached)
    if "width" in manifest:
        summary["image_width"] = manifest.width.describe().round(1).to_dict()
        summary["image_height"] = manifest.height.describe().round(1).to_dict()
        manifest["aspect"] = manifest.width / manifest.height
        summary["aspect_ratio_mean"] = round(float(manifest.aspect.mean()), 2)
        summary["distinct_sizes"] = int(manifest.groupby(["width", "height"]).ngroups)
        fig, ax = plt.subplots(1, 2, figsize=(11, 4))
        sns.scatterplot(data=manifest.sample(min(800, len(manifest)), random_state=1), x="width", y="height", alpha=0.5, ax=ax[0], color="#4a7fb5")
        ax[0].set_title(f"Original image sizes ({summary['distinct_sizes']} distinct)")
        sns.histplot(manifest.aspect, bins=30, ax=ax[1], color="#7a6f9b")
        ax[1].set_title(f"Aspect ratio (mean {summary['aspect_ratio_mean']}:1)\nsquare 224x224 resizing distorts this")
        fig.tight_layout()
        fig.savefig(C.RESULTS_DIR / "eda_image_sizes.png", dpi=140)
        plt.close(fig)

    # 3. Findings per image
    fig, ax = plt.subplots(figsize=(6, 3.6))
    sns.histplot(labels.n_boxes.clip(upper=25), bins=25, ax=ax, color="#4a7fb5")
    ax.set_title("Annotated findings per image (clipped at 25)")
    ax.set_xlabel("boxes")
    fig.tight_layout()
    fig.savefig(C.RESULTS_DIR / "eda_boxes_per_image.png", dpi=140)
    plt.close(fig)

    # 4. Sample grid: 4 caries, 4 no_caries from the training split
    train = manifest[manifest.split == "train"]
    picks = [train[train.caries == 1].sample(4, random_state=3), train[train.caries == 0].sample(4, random_state=3)]
    fig, axes = plt.subplots(2, 4, figsize=(16, 5.2))
    for r, (name, rows) in enumerate(zip(["caries", "no_caries"], picks)):
        for c, row in enumerate(rows.itertuples()):
            axes[r, c].imshow(Image.open(local_image_path(row.image_id)).convert("L"), cmap="gray")
            axes[r, c].set_title(f"{name}  ({row.n_caries} caries / {row.n_periapical} periapical / {row.n_impacted} impacted)", fontsize=8)
            axes[r, c].axis("off")
    fig.tight_layout()
    fig.savefig(C.RESULTS_DIR / "eda_samples.png", dpi=110)
    plt.close(fig)

    (C.RESULTS_DIR / "eda_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
