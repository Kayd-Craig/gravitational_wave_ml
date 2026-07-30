"""
metrics.py
==========
Comprehensive evaluation for GW classifiers, including the domain-specific
False Alarm Rate (FAR) metric used by the LIGO Scientific Collaboration.

Functions
---------
  evaluate_model          — runs a full evaluation pass on a DataLoader
  compute_far             — False Alarm Rate at a given detection threshold
  plot_confusion_matrix   — saves confusion matrix heatmap
  plot_roc_curves         — one-vs-rest ROC curves for all classes
  plot_pr_curves          — precision-recall curves
  print_classification_report — formatted per-class metrics table
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

log = logging.getLogger(__name__)

CLASS_NAMES = ["BBH", "BNS", "Glitch"]


# ─── Model inference ──────────────────────────────────────────────────────────

@torch.no_grad()
def get_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run *model* on *loader* and return (y_true, y_pred, y_proba).

    Returns
    -------
    y_true  : (N,) int ground-truth labels
    y_pred  : (N,) int predicted labels
    y_proba : (N, C) float softmax probabilities
    """
    model.eval()
    all_true, all_pred, all_proba = [], [], []

    for batch in loader:
        x, y = batch
        x = x.to(device, non_blocking=True)
        logits = model(x)
        proba = torch.softmax(logits, dim=-1)

        all_true.extend(y.numpy())
        all_pred.extend(logits.argmax(dim=1).cpu().numpy())
        all_proba.append(proba.cpu().numpy())

    y_true  = np.array(all_true,  dtype=np.int64)
    y_pred  = np.array(all_pred,  dtype=np.int64)
    y_proba = np.concatenate(all_proba, axis=0)
    return y_true, y_pred, y_proba


# ─── FAR / detection efficiency ───────────────────────────────────────────────

def compute_far(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    signal_classes: List[int],
    noise_class: int,
    segment_duration: float = 1.0,
    thresholds: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute False Alarm Rate vs detection efficiency curve.

    FAR is the rate at which noise segments are incorrectly flagged as GW
    signals, measured in events per second.

    Parameters
    ----------
    y_true           : ground-truth labels
    y_proba          : (N, C) class probabilities
    signal_classes   : class indices considered as "GW signal" {0, 1}
    noise_class      : class index for background noise {2}
    segment_duration : duration of each segment in seconds
    thresholds       : detection thresholds to sweep (default: linspace 0→1)

    Returns
    -------
    thresholds  : (K,) array
    far_vals    : (K,) false alarm rates (events/second)
    efficiency  : (K,) fraction of true GW signals detected
    """
    if thresholds is None:
        thresholds = np.linspace(0.0, 1.0, 500)

    # Signal probability = max probability over signal classes
    p_signal = y_proba[:, signal_classes].max(axis=1)

    is_signal = np.isin(y_true, signal_classes)
    is_noise  = y_true == noise_class

    n_noise   = is_noise.sum()
    n_signal  = is_signal.sum()

    far_vals   = np.zeros(len(thresholds))
    efficiency = np.zeros(len(thresholds))

    for i, thr in enumerate(thresholds):
        flagged = p_signal >= thr
        fa = (flagged & is_noise).sum()
        tp = (flagged & is_signal).sum()

        # FAR = false alarms / total observation time (seconds)
        total_noise_time = n_noise * segment_duration
        far_vals[i] = fa / max(total_noise_time, segment_duration)
        efficiency[i] = tp / max(n_signal, 1)

    return thresholds, far_vals, efficiency


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    out_path: Path,
    normalise: bool = True,
) -> None:
    """Save a confusion matrix heatmap to *out_path*."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    if normalise:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = cm.astype(float) / np.maximum(row_sums, 1)
        fmt = ".2f"
        title = "Normalised Confusion Matrix"
    else:
        fmt = "d"
        title = "Confusion Matrix"

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        vmin=0,
        vmax=1 if normalise else None,
        linewidths=0.5,
    )
    ax.set_xlabel("Predicted label", fontsize=13)
    ax.set_ylabel("True label", fontsize=13)
    ax.set_title(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("Confusion matrix saved: %s", out_path)


def plot_roc_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_names: List[str],
    out_path: Path,
) -> None:
    """One-vs-rest ROC curves for all classes."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_auc_score, roc_curve
    from sklearn.preprocessing import label_binarize

    n_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))

    fig, ax = plt.subplots(figsize=(8, 6))
    colours = ["#2196F3", "#FF9800", "#4CAF50"]

    for i, (name, col) in enumerate(zip(class_names, colours)):
            if y_proba.ndim == 1 or y_proba.shape[1] == 1:
        yp = y_proba.ravel()
        y_proba = np.column_stack([1 - yp, yp])
    if y_bin.shape[1] == 1:
        y_bin = np.hstack([1 - y_bin, y_bin])
    if y_bin[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
        auc = roc_auc_score(y_bin[:, i], y_proba[:, i])
        ax.plot(fpr, tpr, lw=2, color=col, label=f"{name}  (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC=0.5)")
    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    ax.set_title("Receiver Operating Characteristic (One-vs-Rest)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("ROC curves saved: %s", out_path)


def plot_pr_curves(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    class_names: List[str],
    out_path: Path,
) -> None:
    """Precision-recall curves — especially meaningful for imbalanced data."""
    import matplotlib.pyplot as plt
    from sklearn.metrics import average_precision_score, precision_recall_curve
    from sklearn.preprocessing import label_binarize

    n_classes = len(class_names)
    y_bin = label_binarize(y_true, classes=list(range(n_classes)))
    colours = ["#2196F3", "#FF9800", "#4CAF50"]

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (name, col) in enumerate(zip(class_names, colours)):
            if y_proba.ndim == 1 or y_proba.shape[1] == 1:
        yp = y_proba.ravel()
        y_proba = np.column_stack([1 - yp, yp])
    if y_bin.shape[1] == 1:
        y_bin = np.hstack([1 - y_bin, y_bin])
    if y_bin[:, i].sum() == 0:
            continue
        prec, rec, _ = precision_recall_curve(y_bin[:, i], y_proba[:, i])
        ap = average_precision_score(y_bin[:, i], y_proba[:, i])
        ax.plot(rec, prec, lw=2, color=col, label=f"{name}  (AP={ap:.3f})")

    ax.set_xlabel("Recall", fontsize=13)
    ax.set_ylabel("Precision", fontsize=13)
    ax.set_title("Precision-Recall Curves", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("PR curves saved: %s", out_path)


def plot_far_curve(
    thresholds: np.ndarray,
    far_vals: np.ndarray,
    efficiency: np.ndarray,
    out_path: Path,
    far_target: float = 1e-3,
) -> None:
    """Detection efficiency vs FAR (log scale) — the key GW astronomy metric."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.semilogx(far_vals + 1e-12, efficiency, lw=2, color="#9C27B0")
    ax.axvline(far_target, color="red", ls="--", lw=1.5,
               label=f"FAR target = {far_target:.0e} Hz")
    # Mark efficiency at target FAR
    idx = np.searchsorted(far_vals[::-1], far_target)
    eff_at_target = efficiency[::-1][idx] if idx < len(efficiency) else 0.0
    ax.axhline(eff_at_target, color="orange", ls=":", lw=1.2,
               label=f"Efficiency @ target = {eff_at_target:.2%}")

    ax.set_xlabel("False Alarm Rate (events / second)", fontsize=13)
    ax.set_ylabel("Detection Efficiency", fontsize=13)
    ax.set_title("FAR vs Detection Efficiency", fontsize=14, fontweight="bold")
    ax.set_xlim([1e-6, 10])
    ax.set_ylim([0, 1.02])
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3, which="both")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    log.info("FAR curve saved: %s", out_path)


def print_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
) -> Dict[str, float]:
    """Print and return a per-class metrics dict."""
    from sklearn.metrics import (
        accuracy_score, classification_report, f1_score, roc_auc_score,
    )

    print("\n" + "=" * 60)
    print("  CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4, labels=list(range(len(class_names)))))

    acc = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    summary = {
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
    }
    print(f"  Accuracy      : {acc:.4f}")
    print(f"  F1 (macro)    : {f1_macro:.4f}")
    print(f"  F1 (weighted) : {f1_weighted:.4f}")
    print("=" * 60 + "\n")

    return summary


# ─── Full evaluation pipeline ─────────────────────────────────────────────────

def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    out_dir: Path,
    cfg: dict,
    class_names: Optional[List[str]] = None,
) -> Dict[str, float]:
    """Run full evaluation: predictions → all metrics → all plots.

    Returns
    -------
    metrics : summary dict of key scalar metrics
    """
    class_names = class_names or cfg["classes"]["names"]
    seg_duration = cfg["data"]["segment_length"]
    far_target = cfg["evaluation"]["far_threshold"]
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Running inference on test set…")
    y_true, y_pred, y_proba = get_predictions(model, loader, device)

    # Save raw predictions
    np.save(out_dir / "y_true.npy",  y_true)
    np.save(out_dir / "y_pred.npy",  y_pred)
    np.save(out_dir / "y_proba.npy", y_proba)

    # Text report
    metrics = print_classification_report(y_true, y_pred, class_names)

    # FAR analysis (signals = BBH (0) + BNS (1), noise = Glitch (2))
    n_classes = cfg["model"]["num_classes"]
    signal_cls = [i for i in range(n_classes) if class_names[i] != "Glitch"]
    noise_cls = class_names.index("Glitch") if "Glitch" in class_names else n_classes - 1

    thresholds, far_vals, eff = compute_far(
        y_true, y_proba, signal_cls, noise_cls, seg_duration
    )

    # Efficiency at FAR target
    idx = np.searchsorted(np.sort(far_vals), far_target)
    eff_at_far = float(eff[np.argsort(far_vals)][min(idx, len(eff) - 1)])
    metrics["efficiency_at_far_target"] = eff_at_far
    log.info("Detection efficiency at FAR=%.1e: %.2f%%", far_target, 100 * eff_at_far)

    # Save metrics as numpy
    np.savez(out_dir / "far_curve.npz",
             thresholds=thresholds, far=far_vals, efficiency=eff)

    # ── Plots ─────────────────────────────────────────────────────────────────
    plot_confusion_matrix(y_true, y_pred, class_names,
                          out_dir / "confusion_matrix.png")
    plot_roc_curves(y_true, y_proba, class_names,
                    out_dir / "roc_curves.png")
    plot_pr_curves(y_true, y_proba, class_names,
                   out_dir / "pr_curves.png")
    plot_far_curve(thresholds, far_vals, eff,
                   out_dir / "far_curve.png", far_target=far_target)

    # Save summary
    import json
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)
    log.info("Evaluation complete. Results saved to %s", out_dir)

    return metrics
