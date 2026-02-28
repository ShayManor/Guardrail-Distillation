"""Shared utilities: metrics, checkpointing, schedulers."""

import os
import time
import torch
import torch.nn as nn
import numpy as np
from contextlib import contextmanager

IGNORE_INDEX = 255


# ── Metrics ──

def compute_miou(pred, target, num_classes, ignore_index=IGNORE_INDEX):
    """Compute mean IoU."""
    pred = pred.flatten()
    target = target.flatten()
    mask = target != ignore_index
    pred = pred[mask]
    target = target[mask]

    ious = []
    for c in range(num_classes):
        p = pred == c
        t = target == c
        inter = (p & t).sum().item()
        union = (p | t).sum().item()
        if union > 0:
            ious.append(inter / union)
    return np.mean(ious) if ious else 0.0


class MetricTracker:
    """Running average tracker."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.vals = {}
        self.counts = {}

    def update(self, key, val, n=1):
        if key not in self.vals:
            self.vals[key] = 0.0
            self.counts[key] = 0
        self.vals[key] += val * n
        self.counts[key] += n

    def avg(self, key):
        return self.vals[key] / max(self.counts[key], 1)

    def summary(self):
        return {k: self.avg(k) for k in self.vals}


# ── Checkpointing ──

def save_checkpoint(model, optimizer, epoch, miou, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": epoch,
        "miou": miou,
    }, path)
    print(f"  → Saved checkpoint: {path} (mIoU={miou:.4f})")


def load_checkpoint(model, path, optimizer=None, device="cuda"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    if optimizer and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    print(f"  ← Loaded checkpoint: {path} (epoch={ckpt.get('epoch', '?')}, mIoU={ckpt.get('miou', '?'):.4f})")
    return ckpt


# ── Scheduler ──

def build_scheduler(optimizer, cfg, total_steps):
    if cfg.lr_scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    elif cfg.lr_scheduler == "step":
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=total_steps // 3, gamma=0.1)
    elif cfg.lr_scheduler == "poly":
        return torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: (1 - step / total_steps) ** 0.9
        )
    else:
        raise ValueError(f"Unknown scheduler: {cfg.lr_scheduler}")


# ── Evaluation ──

@torch.no_grad()
def evaluate(model, val_loader, num_classes, device="cuda"):
    """Run evaluation, return mIoU."""
    model.eval()
    all_ious = []
    for imgs, lbls in val_loader:
        imgs, lbls = imgs.to(device), lbls.to(device)
        logits = model(imgs)
        preds = logits.argmax(dim=1)
        for i in range(preds.shape[0]):
            all_ious.append(compute_miou(preds[i], lbls[i], num_classes))
    model.train()
    return np.mean(all_ious) if all_ious else 0.0


# ── Timer ──

@contextmanager
def timer(label=""):
    t0 = time.time()
    yield
    print(f"  ⏱ {label}: {time.time() - t0:.1f}s")