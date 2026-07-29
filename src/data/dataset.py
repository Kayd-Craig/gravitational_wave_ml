"""
dataset.py
==========
PyTorch Dataset classes for the two model architectures:

  SpectrogramDataset  – yields (1, H, W) Q-transform images for the CNN
  TimeSeriesDataset   – yields (n_samples,) raw whitened strain for the 1D CNN+LSTM
  GWDataModule        – bundles train / val / test splits with optional SMOTE
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

log = logging.getLogger(__name__)

LABEL_NAMES = {0: "BBH", 1: "BNS", 2: "Glitch"}


# ─── helpers ──────────────────────────────────────────────────────────────────

def _collect_npz(root: Path, key: str = "spectrograms") -> Tuple[np.ndarray, np.ndarray]:
    """Walk *root* recursively, load all .npz files that contain *key*."""
    all_data, all_labels = [], []
    for npz_path in sorted(root.rglob("*.npz")):
        try:
            f = np.load(npz_path)
            if key in f:
                all_data.append(f[key])
                all_labels.append(f["labels"])
        except Exception as exc:
            log.warning("Skipping %s: %s", npz_path, exc)
    if not all_data:
        raise FileNotFoundError(
            f"No NPZ files with key '{key}' found under {root}"
        )
    data = np.concatenate(all_data, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    return data, labels


def _collect_raw(root: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load raw time-series segments (saved as NPZ with key 'data')."""
    return _collect_npz(root, key="data")


# ─── Spectrogram Dataset ──────────────────────────────────────────────────────

class SpectrogramDataset(Dataset):
    """Dataset of Q-transform spectrograms.

    Each item is a tuple ``(image_tensor, label)`` where:
      * ``image_tensor`` is shape ``(1, H, W)`` float32  (single-channel)
      * ``label`` is a scalar int64 ∈ {0, 1, 2}
    """

    def __init__(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        augment: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        data    : (N, H, W) float32 spectrogram array
        labels  : (N,) int64 class labels
        augment : apply random time/frequency flips (training only)
        """
        assert data.ndim == 3, f"Expected (N, H, W), got {data.shape}"
        self.data = data.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.augment = augment

        # Per-channel normalise using training statistics
        mean = self.data.mean()
        std = self.data.std() + 1e-8
        self.data = (self.data - mean) / std

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = self.data[idx].copy()  # (H, W)

        if self.augment:
            # Random horizontal flip (time axis) with p=0.3
            if np.random.rand() < 0.3:
                img = np.fliplr(img).copy()
            # Additive Gaussian noise with p=0.2
            if np.random.rand() < 0.2:
                img += np.random.randn(*img.shape).astype(np.float32) * 0.02

        x = torch.from_numpy(img).unsqueeze(0)   # (1, H, W)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

    @classmethod
    def from_directory(cls, directory: Path, augment: bool = False) -> "SpectrogramDataset":
        data, labels = _collect_npz(directory, key="spectrograms")
        log.info("Loaded %d spectrograms from %s", len(labels), directory)
        _log_class_dist(labels)
        return cls(data, labels, augment)


# ─── Time-series Dataset ──────────────────────────────────────────────────────

class TimeSeriesDataset(Dataset):
    """Dataset of raw 1-D whitened strain segments.

    Each item is ``(strain_tensor, label)`` where:
      * ``strain_tensor`` is shape ``(n_samples,)`` float32
      * ``label`` is a scalar int64 ∈ {0, 1, 2}
    """

    def __init__(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        augment: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        data    : (N, T) float32 — whitened strain segments
        labels  : (N,) int64
        augment : apply random amplitude scaling and time-shift
        """
        assert data.ndim == 2, f"Expected (N, T), got {data.shape}"
        self.data = data.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.augment = augment

        # Standardise
        mu = self.data.mean(axis=1, keepdims=True)
        sigma = self.data.std(axis=1, keepdims=True) + 1e-8
        self.data = (self.data - mu) / sigma

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx].copy()  # (T,)

        if self.augment:
            # Amplitude jitter
            if np.random.rand() < 0.3:
                x *= np.random.uniform(0.8, 1.2)
            # Random time shift (circular)
            if np.random.rand() < 0.3:
                shift = np.random.randint(0, len(x))
                x = np.roll(x, shift)
            # Additive noise
            if np.random.rand() < 0.2:
                x += np.random.randn(*x.shape).astype(np.float32) * 0.05

        tensor = torch.from_numpy(x)   # (T,)
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return tensor, y

    @classmethod
    def from_directory(cls, directory: Path, augment: bool = False) -> "TimeSeriesDataset":
        data, labels = _collect_raw(directory)
        log.info("Loaded %d time-series from %s", len(labels), directory)
        _log_class_dist(labels)
        return cls(data, labels, augment)


# ─── Utilities ────────────────────────────────────────────────────────────────

def _log_class_dist(labels: np.ndarray) -> None:
    unique, counts = np.unique(labels, return_counts=True)
    parts = [f"{LABEL_NAMES.get(int(c), c)}={n}" for c, n in zip(unique, counts)]
    log.info("  Class distribution: %s", " | ".join(parts))


def compute_class_weights(labels: np.ndarray, n_classes: int = 3) -> torch.Tensor:
    """Inverse-frequency class weights for CrossEntropyLoss."""
    counts = np.bincount(labels, minlength=n_classes).astype(np.float32)
    counts = np.maximum(counts, 1)
    weights = counts.sum() / (n_classes * counts)
    return torch.from_numpy(weights)


def make_weighted_sampler(labels: np.ndarray) -> WeightedRandomSampler:
    """Create a WeightedRandomSampler that balances mini-batches by class."""
    counts = np.bincount(labels)
    class_weights = 1.0 / np.maximum(counts, 1).astype(np.float64)
    sample_weights = class_weights[labels]
    return WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).float(),
        num_samples=len(labels),
        replacement=True,
    )


# ─── Data Module ──────────────────────────────────────────────────────────────

class GWDataModule:
    """Manages train / val / test DataLoaders for the gravitational wave dataset.

    Parameters
    ----------
    processed_dir : root directory containing ``train/``, ``val/``, ``test/``
                    sub-directories with .npz files
    cfg           : full config dict (from config.yaml)
    mode          : ``"spectrogram"`` or ``"timeseries"``
    """

    def __init__(self, processed_dir: Path, cfg: dict, mode: str = "spectrogram") -> None:
        self.processed_dir = Path(processed_dir)
        self.cfg = cfg
        self.mode = mode

        tcfg = cfg["training"]
        self.batch_size = tcfg["batch_size"]
        self.num_workers = tcfg.get("num_workers", 4)
        self.pin_memory = tcfg.get("pin_memory", True)

        self._train_ds: Optional[Dataset] = None
        self._val_ds: Optional[Dataset] = None
        self._test_ds: Optional[Dataset] = None

    def _build(self, split: str, augment: bool = False) -> Dataset:
        d = self.processed_dir / split
        if self.mode == "spectrogram":
            return SpectrogramDataset.from_directory(d, augment=augment)
        else:
            return TimeSeriesDataset.from_directory(d, augment=augment)

    def setup(self) -> None:
        self._train_ds = self._build("train", augment=True)
        self._val_ds   = self._build("val",   augment=False)
        self._test_ds  = self._build("test",  augment=False)

    def _loader(self, ds: Dataset, shuffle: bool, sampler=None) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=self.batch_size,
            shuffle=shuffle and sampler is None,
            sampler=sampler,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=True,
        )

    def train_loader(self, weighted_sampler: bool = True) -> DataLoader:
        assert self._train_ds is not None, "Call setup() first"
        sampler = None
        if weighted_sampler:
            labels = self._train_ds.labels  # type: ignore[attr-defined]
            sampler = make_weighted_sampler(labels)
        return self._loader(self._train_ds, shuffle=not weighted_sampler, sampler=sampler)

    def val_loader(self) -> DataLoader:
        assert self._val_ds is not None, "Call setup() first"
        return self._loader(self._val_ds, shuffle=False)

    def test_loader(self) -> DataLoader:
        assert self._test_ds is not None, "Call setup() first"
        return self._loader(self._test_ds, shuffle=False)

    def class_weights(self) -> torch.Tensor:
        """Inverse-frequency weights for the training set."""
        assert self._train_ds is not None
        return compute_class_weights(
            self._train_ds.labels,  # type: ignore[attr-defined]
            n_classes=self.cfg["model"]["num_classes"],
        )


# ─── Split builder ────────────────────────────────────────────────────────────

def build_splits(
    raw_spectrograms: Path,
    raw_timeseries: Optional[Path],
    out_dir: Path,
    cfg: dict,
    seed: int = 42,
) -> None:
    """Merge all NPZ files and split into train/val/test directories."""
    from sklearn.model_selection import train_test_split

    dcfg = cfg["data"]
    val_size = dcfg["val_split"]
    test_size = dcfg["test_split"]

    def _split_and_save(data: np.ndarray, labels: np.ndarray,
                        sub_dir: str, key: str) -> None:
        X_tr, X_tmp, y_tr, y_tmp = train_test_split(
            data, labels, test_size=val_size + test_size, stratify=labels,
            random_state=seed,
        )
        val_frac = val_size / (val_size + test_size)
        X_val, X_te, y_val, y_te = train_test_split(
            X_tmp, y_tmp, test_size=1 - val_frac, stratify=y_tmp,
            random_state=seed,
        )
        for split_name, Xs, ys in [
            ("train", X_tr, y_tr), ("val", X_val, y_val), ("test", X_te, y_te)
        ]:
            p = out_dir / sub_dir / split_name
            p.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(p / "data.npz", **{key: Xs, "labels": ys})
            log.info("%s/%s: %d samples", sub_dir, split_name, len(ys))

    # Spectrograms
    spec_data, spec_labels = _collect_npz(raw_spectrograms, key="spectrograms")
    _split_and_save(spec_data, spec_labels, "spectrograms", "spectrograms")

    # Time-series (optional)
    if raw_timeseries and raw_timeseries.exists():
        ts_data, ts_labels = _collect_raw(raw_timeseries)
        _split_and_save(ts_data, ts_labels, "timeseries", "data")
