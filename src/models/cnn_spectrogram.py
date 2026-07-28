"""
cnn_spectrogram.py
==================
ResNet-18 based CNN classifier for Q-transform spectrogram images.

Architecture
------------
  Input : (B, 1, 128, 128)  single-channel spectrogram
  Body  : ResNet-18, first conv adapted to 1 channel, pretrained on ImageNet
          (weights transferred by averaging the 3 RGB channels → 1)
  Head  : GlobalAvgPool → Dropout → Linear(512 → num_classes)

The model supports both:
  * Binary classification  (num_classes=2 : signal / noise)
  * Multi-class            (num_classes=3 : BBH / BNS / Glitch)
"""

from __future__ import annotations

import logging
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


# ─── Grad-CAM target layer helper ─────────────────────────────────────────────

class GradCAMTarget(nn.Module):
    """Wrapper that exposes the final ResNet layer for Grad-CAM hooks."""
    def __init__(self, model: "SpectrogramCNN") -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


# ─── Main model ───────────────────────────────────────────────────────────────

class SpectrogramCNN(nn.Module):
    """ResNet-18 fine-tuned for single-channel GW spectrogram classification.

    Parameters
    ----------
    num_classes : int
        Number of output classes (2 or 3).
    pretrained  : bool
        Initialise backbone with ImageNet weights.
    dropout     : float
        Dropout probability before the final linear layer.
    freeze_backbone : bool
        Freeze all ResNet layers except the final block and head.
    """

    def __init__(
        self,
        num_classes: int = 3,
        pretrained: bool = True,
        dropout: float = 0.3,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes

        # ── Load pretrained ResNet-18 ──────────────────────────────────────────
        try:
            from torchvision.models import resnet18, ResNet18_Weights
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            backbone = resnet18(weights=weights)
            log.info("Loaded ResNet-18 with %s weights",
                     "ImageNet" if pretrained else "random")
        except ImportError:
            raise ImportError("torchvision is required: pip install torchvision")

        # ── Adapt first conv: 3 → 1 input channel ─────────────────────────────
        orig_conv = backbone.conv1
        new_conv = nn.Conv2d(
            in_channels=1,
            out_channels=orig_conv.out_channels,
            kernel_size=orig_conv.kernel_size,
            stride=orig_conv.stride,
            padding=orig_conv.padding,
            bias=False,
        )
        if pretrained:
            # Average the 3 colour channels → 1 channel weight
            with torch.no_grad():
                new_conv.weight.data = orig_conv.weight.data.mean(dim=1, keepdim=True)
        backbone.conv1 = new_conv

        # ── Replace classification head ────────────────────────────────────────
        in_features = backbone.fc.in_features   # 512 for ResNet-18
        backbone.fc = nn.Identity()             # remove original head

        self.backbone = backbone

        # ── Custom classification head ─────────────────────────────────────────
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout / 2),
            nn.Linear(256, num_classes),
        )

        # ── Optionally freeze early layers ─────────────────────────────────────
        if freeze_backbone:
            for name, param in self.backbone.named_parameters():
                if "layer4" not in name:
                    param.requires_grad = False
            log.info("Backbone frozen (except layer4)")

    # ── target layer for Grad-CAM ──────────────────────────────────────────────
    @property
    def target_layer(self) -> nn.Module:
        """Return the last conv block — used by Grad-CAM."""
        return self.backbone.layer4[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (B, 1, H, W) float tensor
        Returns (B, num_classes) logits.
        """
        features = self.backbone(x)   # (B, 512) after avgpool
        return self.head(features)    # (B, num_classes)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return class probabilities via softmax."""
        return F.softmax(self.forward(x), dim=-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─── Lightweight CNN (no torchvision dependency) ──────────────────────────────

class LightweightCNN(nn.Module):
    """Small custom CNN when torchvision / pretrained weights are unavailable.

    Architecture: 4 × (Conv → BN → ReLU → MaxPool) → GAP → FC
    """

    def __init__(self, num_classes: int = 3, dropout: float = 0.3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            self._block(1,   32, kernel=7, stride=2, pad=3),   # 64×64
            self._block(32,  64, kernel=3, stride=1, pad=1),   # 64×64
            nn.MaxPool2d(2),                                    # 32×32
            self._block(64, 128, kernel=3, stride=1, pad=1),   # 32×32
            self._block(128, 256, kernel=3, stride=1, pad=1),  # 32×32
            nn.MaxPool2d(2),                                    # 16×16
            self._block(256, 512, kernel=3, stride=1, pad=1),  # 16×16
            nn.AdaptiveAvgPool2d((1, 1)),                       # 1×1
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    @staticmethod
    def _block(in_c: int, out_c: int,
               kernel: int = 3, stride: int = 1, pad: int = 1) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel, stride=stride, padding=pad, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    @property
    def target_layer(self) -> nn.Module:
        # The last conv block before global avg pool
        return self.features[-2]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─── Factory ──────────────────────────────────────────────────────────────────

def build_cnn(cfg: dict) -> nn.Module:
    """Build the CNN model specified in *cfg*.

    Tries SpectrogramCNN (ResNet-18) first; falls back to LightweightCNN
    if torchvision is not installed.
    """
    mcfg = cfg["model"]
    n_classes = mcfg["num_classes"]
    dropout = mcfg.get("dropout", 0.3)
    pretrained = mcfg.get("pretrained", True)

    try:
        model = SpectrogramCNN(
            num_classes=n_classes,
            pretrained=pretrained,
            dropout=dropout,
        )
        log.info(
            "SpectrogramCNN (ResNet-18)  —  %d trainable parameters",
            model.count_parameters(),
        )
    except ImportError:
        log.warning("torchvision not found; using LightweightCNN")
        model = LightweightCNN(num_classes=n_classes, dropout=dropout)
        log.info(
            "LightweightCNN  —  %d trainable parameters",
            model.count_parameters(),
        )

    return model
