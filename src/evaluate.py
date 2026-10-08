"""Evaluate a trained model on the held-out TEST split (never used for training or model selection).

    python src/evaluate.py --tag square

Reports accuracy, per-class precision / recall / F1, confusion matrix and ROC-AUC, with 95% bootstrap confidence
intervals so the uncertainty of a ~750-image test set is visible rather than hidden.
"""
import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support, roc_auc_score, roc_curve

import config as C
from dataset import XrayDataset
from model import load_trained

CLASSES = C.CLASS_NAMES  # index 0 = no_caries, 1 = caries


def predict_proba(model, images, batch=64):
    ds = XrayDataset(images, np.zeros(len(images), dtype=int), train=False)
    out = []
    with torch.no_grad():
        for i in range(0, len(ds), batch):
            xb = torch.stack([ds[j][0] for j in range(i, min(i + batch, len(ds)))])
            out.append(torch.softmax(model(xb), 1)[:, 1].numpy())
    return np.concatenate(out)


def bootstrap_ci(y, p, fn, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        vals.append(fn(y[idx], p[idx]))
    return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="square")
    args = ap.parse_args()

    x = np.load(C.CACHE_DIR / f"test_{C.IMG_H}x{C.IMG_W}.npy")
    y = np.load(C.CACHE_DIR / "test_labels.npy")
    ids = pd.read_csv(C.CACHE_DIR / "test_ids.csv").image_id.to_numpy()
    model = load_trained(C.MODELS_DIR / f"resnet18_{args.tag}.pt", n_classes=2)
    p = predict_proba(model, x)
    pred = (p >= 0.5).astype(int)

    prec, rec, f1, support = precision_recall_fscore_support(y, pred, labels=[0, 1], zero_division=0)
    metrics = {
        "tag": args.tag,
        "input_size": f"{C.IMG_H}x{C.IMG_W}",
        "test_images": int(len(y)),
        "accuracy": round(accuracy_score(y, pred), 4),
        "accuracy_ci95": bootstrap_ci(y, pred, accuracy_score),
        "roc_auc": round(roc_auc_score(y, p), 4),
        "roc_auc_ci95": bootstrap_ci(y, p, roc_auc_score),
        "macro_f1": round(f1_score(y, pred, average="macro"), 4),
        "macro_f1_ci95": bootstrap_ci(y, pred, lambda a, b: f1_score(a, b, average="macro")),
        "per_class": {
            CLASSES[i]: {"precision": round(prec[i], 4), "recall": round(rec[i], 4), "f1": round(f1[i], 4), "support": int(support[i])}
            for i in (0, 1)
        },
        "confusion_matrix": confusion_matrix(y, pred).tolist(),
    }
    (C.RESULTS_DIR / f"metrics_{args.tag}.json").write_text(json.dumps(metrics, indent=2))
    pd.DataFrame({"image_id": ids, "true": y, "prob_caries": p.round(4), "pred": pred}).to_csv(
        C.RESULTS_DIR / f"predictions_{args.tag}.csv", index=False)

    cm = confusion_matrix(y, pred)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax[0], xticklabels=CLASSES, yticklabels=CLASSES)
    ax[0].set(xlabel="predicted", ylabel="true", title="Confusion matrix (test set)")
    fpr, tpr, _ = roc_curve(y, p)
    ax[1].plot(fpr, tpr, lw=2, label=f"ResNet-18 (AUC {metrics['roc_auc']:.3f})")
    ax[1].plot([0, 1], [0, 1], "--", c="gray", label="chance")
    ax[1].set(xlabel="false positive rate", ylabel="true positive rate", title="ROC curve (test set)")
    ax[1].legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(C.RESULTS_DIR / f"evaluation_{args.tag}.png", dpi=140)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
