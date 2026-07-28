"""
augmentation.py
===============
Data augmentation and resampling strategies for addressing the severe
class imbalance in gravitational-wave datasets.

Strategies
----------
1. SMOTE on extracted features (for the Random Forest baseline)
2. Time-series augmentation: amplitude jitter, time shift, Gaussian noise
3. Spectrogram augmentation: SpecAugment-style time/frequency masking
4. Mixup: blend two training samples and interpolate labels
"""

from __future__ import annotations

import logging
from typing import Callable, Optional, Tuple

import numpy as np
import torch

log = logging.getLogger(__name__)


# ─── SMOTE for time-series features ───────────────────────────────────────────

def apply_smote(
    X: np.ndarray,
    y: np.ndarray,
    k_neighbors: int = 5,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply SMOTE oversampling to feature matrix X.

    Parameters
    ----------
    X : (N, F) feature matrix
    y : (N,) label array
    k_neighbors : number of nearest neighbours for synthesis

    Returns
    -------
    X_res, y_res : resampled arrays
    """
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        raise ImportError("imbalanced-learn required: pip install imbalanced-learn")

    sampler = SMOTE(k_neighbors=k_neighbors, random_state=seed)
    X_res, y_res = sampler.fit_resample(X, y)
    log.info("SMOTE: %d → %d samples", len(y), len(y_res))
    return X_res, y_res.astype(np.int64)


# ─── Time-series augmentations ────────────────────────────────────────────────

def augment_timeseries(
    x: np.ndarray,
    fs: float,
    amplitude_jitter: float = 0.15,
    time_shift_frac: float = 0.1,
    noise_level: float = 0.03,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Apply random augmentations to a 1-D whitened strain segment.

    Parameters
    ----------
    x               : (T,) float32 whitened strain
    fs              : sample rate (Hz)
    amplitude_jitter: multiplicative amplitude noise std
    time_shift_frac : max time shift as fraction of segment length
    noise_level     : additive Gaussian noise std relative to RMS

    Returns
    -------
    augmented copy of x
    """
    rng = rng or np.random.default_rng()
    x = x.copy().astype(np.float32)

    # Amplitude scaling
    if amplitude_jitter > 0:
        scale = rng.normal(1.0, amplitude_jitter)
        scale = np.clip(scale, 0.5, 2.0)
        x *= scale

    # Circular time shift
    if time_shift_frac > 0:
        max_shift = int(time_shift_frac * len(x))
        shift = rng.integers(-max_shift, max_shift + 1)
        x = np.roll(x, shift)

    # Additive Gaussian noise
    if noise_level > 0:
        rms = np.sqrt(np.mean(x ** 2)) + 1e-30
        x += rng.normal(0, noise_level * rms, size=x.shape).astype(np.float32)

    return x


def augment_timeseries_batch(
    X: np.ndarray,
    fs: float,
    **kwargs,
) -> np.ndarray:
    """Apply augmentation to a batch (N, T) of segments."""
    rng = np.random.default_rng()
    return np.stack([augment_timeseries(x, fs, rng=rng, **kwargs) for x in X])


# ─── Spectrogram augmentations (SpecAugment-style) ───────────────────────────

def spec_augment(
    spec: np.ndarray,
    time_mask_param: int = 20,
    freq_mask_param: int = 15,
    n_time_masks: int = 2,
    n_freq_masks: int = 2,
    fill_value: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Apply SpecAugment masking to a single (H, W) spectrogram.

    Randomly zeroes horizontal (frequency) and vertical (time) stripes.
    Prevents the model from over-relying on any single frequency band or
    time window — particularly useful for generalisation across detector epochs.

    Reference
    ---------
    Park et al. (2019) "SpecAugment: A Simple Data Augmentation Method
    for Automatic Speech Recognition"
    """
    rng = rng or np.random.default_rng()
    s = spec.copy().astype(np.float32)
    H, W = s.shape

    # Time masking (columns)
    for _ in range(n_time_masks):
        t = rng.integers(0, max(1, time_mask_param))
        t0 = rng.integers(0, max(1, W - t))
        s[:, t0 : t0 + t] = fill_value

    # Frequency masking (rows)
    for _ in range(n_freq_masks):
        f = rng.integers(0, max(1, freq_mask_param))
        f0 = rng.integers(0, max(1, H - f))
        s[f0 : f0 + f, :] = fill_value

    return s


class SpectrogramAugmenter:
    """Callable augmenter for use in PyTorch Dataset.__getitem__."""

    def __init__(
        self,
        time_mask_param: int = 20,
        freq_mask_param: int = 15,
        n_time_masks: int = 2,
        n_freq_masks: int = 2,
        horizontal_flip_prob: float = 0.2,
        noise_std: float = 0.02,
    ) -> None:
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param
        self.n_time_masks = n_time_masks
        self.n_freq_masks = n_freq_masks
        self.hflip_prob = horizontal_flip_prob
        self.noise_std = noise_std

    def __call__(self, spec: np.ndarray) -> np.ndarray:
        """Input/output: (H, W) float32 spectrogram."""
        rng = np.random.default_rng()

        spec = spec_augment(
            spec,
            time_mask_param=self.time_mask_param,
            freq_mask_param=self.freq_mask_param,
            n_time_masks=self.n_time_masks,
            n_freq_masks=self.n_freq_masks,
            rng=rng,
        )

        if rng.random() < self.hflip_prob:
            spec = np.fliplr(spec).copy()

        if self.noise_std > 0:
            spec += rng.normal(0, self.noise_std, spec.shape).astype(np.float32)

        return spec


# ─── Mixup ────────────────────────────────────────────────────────────────────

def mixup_batch(
    x: torch.Tensor,
    y: torch.Tensor,
    alpha: float = 0.2,
    n_classes: int = 3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply Mixup regularisation to a mini-batch.

    Returns mixed inputs and soft labels (one-hot weighted).
    Mixup paper: Zhang et al. (2018) https://arxiv.org/abs/1710.09412

    Parameters
    ----------
    x       : (B, ...) input tensor
    y       : (B,) integer label tensor
    alpha   : Beta distribution parameter (higher → more mixing)
    n_classes: number of classes

    Returns
    -------
    x_mixed : (B, ...) float
    y_mixed : (B, n_classes) soft-label float
    """
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0

    B = x.size(0)
    perm = torch.randperm(B, device=x.device)

    # One-hot encode
    y_oh = torch.zeros(B, n_classes, device=x.device).scatter_(1, y.unsqueeze(1), 1.0)

    x_mixed = lam * x + (1 - lam) * x[perm]
    y_mixed = lam * y_oh + (1 - lam) * y_oh[perm]

    return x_mixed, y_mixed


def mixup_loss(
    logits: torch.Tensor,
    y_soft: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy loss for soft Mixup labels."""
    log_proba = torch.log_softmax(logits, dim=-1)
    return -(y_soft * log_proba).sum(dim=-1).mean()
