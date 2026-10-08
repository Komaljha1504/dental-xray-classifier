"""Grad-CAM figure: the model's most confident correct and incorrect test predictions.

    python src/explain.py --tag square
"""
import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import config as C
from dataset import preprocess_gray
from gradcam import GradCAM, overlay
from model import load_trained

NAMES = {0: "no_caries", 1: "caries"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="square")
    args = ap.parse_args()

    x = np.load(C.CACHE_DIR / f"test_{C.IMG_H}x{C.IMG_W}.npy")
    pred = pd.read_csv(C.RESULTS_DIR / f"predictions_{args.tag}.csv")
    pred["conf"] = np.where(pred.pred == 1, pred.prob_caries, 1 - pred.prob_caries)
    pred["correct"] = pred.true == pred.pred

    right = pred[pred.correct]
    wrong = pred[~pred.correct]
    chosen = pd.concat([
        right[right.true == 1].nlargest(2, "conf"),   # confident true caries
        right[right.true == 0].nlargest(1, "conf"),   # confident true no_caries
        wrong[wrong.true == 1].nlargest(2, "conf"),   # caries the model missed
        wrong[wrong.true == 0].nlargest(1, "conf"),   # false alarm
    ])

    model = load_trained(C.MODELS_DIR / f"resnet18_{args.tag}.pt", n_classes=2)
    cam = GradCAM(model, model.layer4[-1])
    fig, axes = plt.subplots(2, len(chosen), figsize=(3.4 * len(chosen), 4.6))
    for col, (idx, row) in enumerate(chosen.iterrows()):
        gray = x[idx]
        heat, _, cls = cam(preprocess_gray(gray))
        verdict = "CORRECT" if row.correct else "WRONG"
        title = f"{verdict}\ntrue: {NAMES[int(row.true)]}\npred: {NAMES[int(row.pred)]} ({row.conf:.0%})"
        axes[0, col].imshow(gray, cmap="gray")
        axes[0, col].set_title(title, fontsize=8, color="#1b7f3b" if row.correct else "#b3261e")
        axes[1, col].imshow(overlay(gray, heat))
        for r in (0, 1):
            axes[r, col].axis("off")
    axes[0, 0].text(-0.02, 0.5, "X-ray", transform=axes[0, 0].transAxes, rotation=90, va="center", ha="right", fontsize=9)
    axes[1, 0].text(-0.02, 0.5, "Grad-CAM", transform=axes[1, 0].transAxes, rotation=90, va="center", ha="right", fontsize=9)
    fig.tight_layout()
    out = C.RESULTS_DIR / f"gradcam_{args.tag}.png"
    fig.savefig(out, dpi=130)
    print("saved", out, "| shown:", [(int(r.true), int(r.pred), round(float(r.conf), 2)) for _, r in chosen.iterrows()])


if __name__ == "__main__":
    main()
