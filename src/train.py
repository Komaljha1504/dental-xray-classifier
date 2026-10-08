"""Fine-tune an ImageNet-pretrained ResNet-18.

    python src/train.py --tag square          # 224 x 224 (as in the brief)
    IMG_W=448 python src/train.py --tag wide  # 224 x 448, keeps the panoramic aspect ratio

Loss: cross-entropy. Optimiser: Adam (small LR for the pretrained backbone, larger for the new head).
Scheduler: ReduceLROnPlateau on validation loss. Early stopping on validation loss. Best weights are kept.
"""
import argparse
import os
import random
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

import config as C
from dataset import XrayDataset
from model import build_model, freeze_early, set_train_mode


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_split(split):
    x = np.load(C.CACHE_DIR / f"{split}_{C.IMG_H}x{C.IMG_W}.npy")
    y = np.load(C.CACHE_DIR / f"{split}_labels.npy")
    return x, y


def run_epoch(model, loader, criterion, optimizer=None):
    train = optimizer is not None
    set_train_mode(model, train)
    total_loss, correct, n = 0.0, 0, 0
    probs, targets = [], []
    with torch.set_grad_enabled(train):
        for x, y in loader:
            logits = model(x.contiguous(memory_format=torch.channels_last))
            loss = criterion(logits, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(y)
            correct += (logits.argmax(1) == y).sum().item()
            n += len(y)
            probs.append(torch.softmax(logits.detach(), 1)[:, 1].numpy())
            targets.append(y.numpy())
    auc = roc_auc_score(np.concatenate(targets), np.concatenate(probs))
    return total_loss / n, correct / n, auc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="square")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4, help="backbone LR; the new head uses 10x")
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--freeze", default="layer2", choices=["none", "layer1", "layer2"],
                    help="freeze the ImageNet backbone up to this stage (CPU-only training is slow)")
    args = ap.parse_args()

    seed_everything(C.SEED)
    torch.set_num_threads(os.cpu_count())
    xtr, ytr = load_split("train")
    xva, yva = load_split("val")
    print(f"train {xtr.shape} val {xva.shape} | caries share train {ytr.mean():.2f}")

    train_loader = DataLoader(XrayDataset(xtr, ytr, train=True), batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, persistent_workers=args.workers > 0, drop_last=True)
    val_loader = DataLoader(XrayDataset(xva, yva, train=False), batch_size=64, num_workers=0)

    model = build_model(n_classes=2).to(memory_format=torch.channels_last)
    freeze_early(model, args.freeze)
    head = list(model.fc.parameters())
    head_ids = {id(p) for p in head}
    backbone = [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids]
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"freeze={args.freeze} | trainable parameters {n_train:,} of {sum(p.numel() for p in model.parameters()):,}")
    optimizer = torch.optim.Adam([{"params": backbone, "lr": args.lr}, {"params": head, "lr": args.lr * 10}])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)
    criterion = nn.CrossEntropyLoss()

    best_loss, best_epoch, log = float("inf"), 0, []
    out_path = C.MODELS_DIR / f"resnet18_{args.tag}.pt"
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc, tr_auc = run_epoch(model, train_loader, criterion, optimizer)
        va_loss, va_acc, va_auc = run_epoch(model, val_loader, criterion)
        scheduler.step(va_loss)
        log.append(dict(epoch=epoch, train_loss=tr_loss, train_acc=tr_acc, train_auc=tr_auc,
                        val_loss=va_loss, val_acc=va_acc, val_auc=va_auc, lr=optimizer.param_groups[0]["lr"],
                        seconds=time.time() - t0))
        improved = va_loss < best_loss - 1e-4
        if improved:
            best_loss, best_epoch = va_loss, epoch
            torch.save(model.state_dict(), out_path)
        print(f"epoch {epoch:2d} | train loss {tr_loss:.3f} acc {tr_acc:.3f} | val loss {va_loss:.3f} acc {va_acc:.3f} "
              f"auc {va_auc:.3f} | {time.time() - t0:.0f}s {'*' if improved else ''}", flush=True)
        pd.DataFrame(log).to_csv(C.RESULTS_DIR / f"train_log_{args.tag}.csv", index=False)
        if epoch - best_epoch >= args.patience:
            print(f"early stopping: no val-loss improvement for {args.patience} epochs (best epoch {best_epoch})")
            break

    df = pd.DataFrame(log)
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8))
    ax[0].plot(df.epoch, df.train_loss, label="train")
    ax[0].plot(df.epoch, df.val_loss, label="validation")
    ax[0].axvline(best_epoch, ls="--", c="gray", lw=1)
    ax[0].set_title("Cross-entropy loss")
    ax[0].legend()
    ax[1].plot(df.epoch, df.train_acc, label="train")
    ax[1].plot(df.epoch, df.val_acc, label="validation")
    ax[1].axvline(best_epoch, ls="--", c="gray", lw=1)
    ax[1].set_title("Accuracy")
    ax[1].legend()
    for a in ax:
        a.set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(C.RESULTS_DIR / f"training_curves_{args.tag}.png", dpi=140)
    print(f"best epoch {best_epoch}, val loss {best_loss:.4f}, saved {out_path}")


if __name__ == "__main__":
    main()
