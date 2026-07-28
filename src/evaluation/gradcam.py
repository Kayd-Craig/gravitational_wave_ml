"""
gradcam.py
==========
Gradient-weighted Class Activation Maps (Grad-CAM) for the SpectrogramCNN.

Highlights which time-frequency regions in the Q-transform spectrogram
the model relied on when classifying a segment — providing astrophysical
insight into the learned features (e.g., the characteristic chirp sweep).

Usage
-----
from src.evaluation.gradcam import GradCAMVisualiser, visualise_batch

vis = GradCAMVisualiser(model, target_layer=model.target_layer)
vis.visualise_batch(loader, device, out_dir, n_samples=12)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)

CLASS_NAMES = {0: "BBH", 1: "BNS", 2: "Glitch"}


# ─── Grad-CAM core ────────────────────────────────────────────────────────────

class GradCAMVisualiser:
    """Compute and render Grad-CAM heatmaps for a CNN model.

    Parameters
    ----------
    model        : trained SpectrogramCNN or LightweightCNN
    target_layer : the conv layer to hook (model.target_layer)
    """

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer

        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None

        # Register hooks
        self._hook_fwd = target_layer.register_forward_hook(self._save_activations)
        self._hook_bwd = target_layer.register_full_backward_hook(self._save_gradients)

    def _save_activations(self, module, input, output) -> None:
        self._activations = output.detach()

    def _save_gradients(self, module, grad_input, grad_output) -> None:
        self._gradients = grad_output[0].detach()

    def remove_hooks(self) -> None:
        self._hook_fwd.remove()
        self._hook_bwd.remove()

    def __del__(self) -> None:
        try:
            self.remove_hooks()
        except Exception:
            pass

    def compute_cam(
        self,
        image: torch.Tensor,    # (1, 1, H, W)
        class_idx: Optional[int] = None,
    ) -> Tuple[np.ndarray, int, float]:
        """Compute Grad-CAM for a single image.

        Parameters
        ----------
        image     : (1, 1, H, W) float tensor
        class_idx : target class (None → argmax / predicted class)

        Returns
        -------
        cam       : (H, W) float32 in [0, 1] — the saliency map
        pred_cls  : predicted class index
        confidence: softmax probability of the predicted / target class
        """
        self.model.eval()
        image.requires_grad_(False)

        logits = self.model(image)                    # (1, C)
        proba = F.softmax(logits, dim=-1)

        pred_cls = int(logits.argmax(dim=-1).item())
        target = class_idx if class_idx is not None else pred_cls
        confidence = float(proba[0, target].item())

        # Backprop on target class score
        self.model.zero_grad()
        logits[0, target].backward()

        # Grad-CAM formula: α_k = GAP(∂y^c / ∂A^k)
        grads = self._gradients[0]          # (C_feat, H', W')
        acts  = self._activations[0]        # (C_feat, H', W')
        weights = grads.mean(dim=(-2, -1))  # (C_feat,)

        # Weighted sum of activation maps
        cam = (weights[:, None, None] * acts).sum(dim=0)  # (H', W')
        cam = F.relu(cam)                   # discard negative activations

        # Upsample to input size
        h, w = image.shape[-2], image.shape[-1]
        cam = cam.unsqueeze(0).unsqueeze(0)  # (1, 1, H', W')
        cam = F.interpolate(cam, size=(h, w), mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()    # (H, W)

        # Normalise to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam.astype(np.float32), pred_cls, confidence

    def visualise_single(
        self,
        image: torch.Tensor,          # (1, 1, H, W)
        true_label: int,
        out_path: Path,
        class_names: Optional[dict] = None,
    ) -> None:
        """Render and save a Grad-CAM overlay for one sample."""
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        class_names = class_names or CLASS_NAMES

        cam, pred_cls, conf = self.compute_cam(image)

        spec = image.squeeze().cpu().numpy()  # (H, W)

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        # Original spectrogram
        im0 = axes[0].imshow(spec, origin="lower", aspect="auto",
                              cmap="viridis", interpolation="nearest")
        axes[0].set_title("Q-transform spectrogram", fontsize=11)
        axes[0].set_xlabel("Time bins")
        axes[0].set_ylabel("Frequency bins (log scale)")
        plt.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

        # Grad-CAM heatmap
        axes[1].imshow(cam, origin="lower", aspect="auto",
                       cmap="hot", vmin=0, vmax=1, interpolation="bilinear")
        axes[1].set_title("Grad-CAM saliency", fontsize=11)
        axes[1].set_xlabel("Time bins")
        axes[1].axis("on")

        # Overlay
        axes[2].imshow(spec, origin="lower", aspect="auto",
                       cmap="viridis", interpolation="nearest")
        axes[2].imshow(cam, origin="lower", aspect="auto",
                       cmap="hot", alpha=0.55, vmin=0, vmax=1,
                       interpolation="bilinear")
        axes[2].set_title("Overlay", fontsize=11)
        axes[2].set_xlabel("Time bins")

        true_name = class_names.get(true_label, str(true_label))
        pred_name = class_names.get(pred_cls, str(pred_cls))
        correct = "✓" if true_label == pred_cls else "✗"
        sup = (f"True: {true_name}  |  Pred: {pred_name} "
               f"({100*conf:.1f}%)  {correct}")
        fig.suptitle(sup, fontsize=13, fontweight="bold",
                     color="green" if true_label == pred_cls else "red")

        plt.tight_layout(rect=[0, 0, 1, 0.93])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
        plt.close(fig)

    def visualise_batch(
        self,
        loader: torch.utils.data.DataLoader,
        device: torch.device,
        out_dir: Path,
        n_samples: int = 12,
        class_names: Optional[dict] = None,
    ) -> None:
        """Save Grad-CAM figures for *n_samples* examples from *loader*.

        Samples one example per class where possible.
        """
        class_names = class_names or CLASS_NAMES
        out_dir.mkdir(parents=True, exist_ok=True)

        collected: dict[int, List[torch.Tensor]] = {k: [] for k in class_names}
        per_class = max(1, n_samples // len(class_names))

        for x, y in loader:
            for i in range(len(y)):
                lbl = int(y[i].item())
                if lbl in collected and len(collected[lbl]) < per_class:
                    collected[lbl].append(x[i : i + 1].to(device))
            if all(len(v) >= per_class for v in collected.values()):
                break

        count = 0
        for lbl, imgs in collected.items():
            for k, img in enumerate(imgs):
                out_path = out_dir / f"gradcam_{class_names[lbl]}_{k:02d}.png"
                self.visualise_single(img, lbl, out_path, class_names)
                count += 1

        log.info("Grad-CAM: saved %d figures to %s", count, out_dir)


# ─── Grid summary figure ──────────────────────────────────────────────────────

def make_gradcam_grid(
    model: nn.Module,
    target_layer: nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    out_path: Path,
    n_cols: int = 3,
    class_names: Optional[dict] = None,
) -> None:
    """Create a single grid figure with n_cols×n_classes Grad-CAM overlays."""
    import matplotlib.pyplot as plt
    import torch.nn.functional as F

    class_names = class_names or CLASS_NAMES
    n_classes = len(class_names)
    vis = GradCAMVisualiser(model, target_layer)

    # Collect n_cols examples per class
    samples: dict[int, list] = {k: [] for k in class_names}
    for x, y in loader:
        for i in range(len(y)):
            lbl = int(y[i])
            if lbl in samples and len(samples[lbl]) < n_cols:
                samples[lbl].append((x[i:i+1].to(device), lbl))
        if all(len(v) >= n_cols for v in samples.values()):
            break

    fig, axes = plt.subplots(n_classes, n_cols, figsize=(4 * n_cols, 4 * n_classes))
    if n_classes == 1:
        axes = [axes]

    for row, (cls_idx, img_list) in enumerate(samples.items()):
        for col, (img, lbl) in enumerate(img_list[:n_cols]):
            cam, pred, conf = vis.compute_cam(img)
            spec = img.squeeze().cpu().numpy()
            ax = axes[row][col]
            ax.imshow(spec, origin="lower", aspect="auto",
                      cmap="viridis", interpolation="nearest")
            ax.imshow(cam, origin="lower", aspect="auto",
                      cmap="hot", alpha=0.55, interpolation="bilinear")
            pred_name = class_names.get(pred, str(pred))
            ax.set_title(f"Pred: {pred_name} ({100*conf:.0f}%)", fontsize=9)
            if col == 0:
                ax.set_ylabel(class_names.get(cls_idx, str(cls_idx)), fontsize=11,
                              fontweight="bold")
            ax.set_xticks([])
            ax.set_yticks([])

    fig.suptitle("Grad-CAM: Time-frequency saliency by class", fontsize=14)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    vis.remove_hooks()
    log.info("Grad-CAM grid saved: %s", out_path)
