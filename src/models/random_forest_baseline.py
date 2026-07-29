"""
random_forest_baseline.py
=========================
Random Forest classifier using hand-crafted signal features as a performance
benchmark against the deep learning models.

Features extracted per segment
-------------------------------
  * Peak frequency & its spectral amplitude
  * Signal-to-noise ratio (peak / noise floor)
  * Duration above 5-sigma threshold
  * Spectral bandwidth (interquartile range of power spectrum)
  * Kurtosis of the time-domain signal
  * Skewness of the time-domain signal
  * RMS amplitude
  * Band-limited energy in 5 sub-bands (20-60, 60-120, 120-240, 240-500 Hz)
  * Crest factor (peak / RMS)
  * Zero-crossing rate
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import signal as sp_signal
from scipy.stats import kurtosis, skew

log = logging.getLogger(__name__)


# ─── Feature extraction ───────────────────────────────────────────────────────

BAND_EDGES_HZ = [20, 60, 120, 240, 500]   # 4 sub-bands


def extract_features(segment: np.ndarray, fs: float) -> np.ndarray:
    """Extract a fixed-length feature vector from a single strain segment.

    Parameters
    ----------
    segment : 1-D float array of whitened strain (length = fs * seg_len)
    fs      : sample rate (Hz)

    Returns
    -------
    features : 1-D float32 array of length ~20
    """
    # NOTE: expects pre-whitened input (whitening done upstream with reference PSD)
    n = len(segment)
    x = segment.astype(np.float64)

    # ── Time-domain features ──────────────────────────────────────────────────
    rms = np.sqrt(np.mean(x ** 2)) + 1e-30
    peak_amp = np.max(np.abs(x))
    crest_factor = peak_amp / rms

    kurt = float(kurtosis(x, fisher=True, bias=False))
    sk = float(skew(x, bias=False))

    # Zero-crossing rate
    zcr = float(np.sum(np.diff(np.sign(x)) != 0)) / n

    # Duration above 5σ threshold
    sigma = np.std(x) + 1e-30
    above_thresh = np.sum(np.abs(x) > 5 * sigma) / n

    # ── Frequency-domain features ─────────────────────────────────────────────
    freqs, psd = sp_signal.welch(x, fs=fs, nperseg=min(n, 512), scaling="density")

    # Peak frequency
    peak_idx = int(np.argmax(psd))
    peak_freq = float(freqs[peak_idx])
    peak_psd = float(psd[peak_idx])

    # Noise floor (median of PSD)
    noise_floor = float(np.median(psd)) + 1e-100
    snr_estimate = peak_psd / noise_floor

    # Spectral bandwidth (IQR of cumulative power distribution)
    cumulative = np.cumsum(psd)
    cumulative /= cumulative[-1] + 1e-100
    q25_idx = int(np.searchsorted(cumulative, 0.25))
    q75_idx = int(np.searchsorted(cumulative, 0.75))
    bandwidth = float(freqs[q75_idx] - freqs[q25_idx])

    # Mean frequency (centroid)
    mean_freq = float(np.sum(freqs * psd) / (np.sum(psd) + 1e-100))

    # ── Band-limited energies ─────────────────────────────────────────────────
    band_energies = []
    for f_lo, f_hi in zip(BAND_EDGES_HZ[:-1], BAND_EDGES_HZ[1:]):
        mask = (freqs >= f_lo) & (freqs < f_hi)
        energy = float(np.sum(psd[mask]) * (freqs[1] - freqs[0]))
        band_energies.append(energy)
    total_energy = sum(band_energies) + 1e-100
    band_ratios = [e / total_energy for e in band_energies]

    # ── Chirp-specific features ───────────────────────────────────────────────
    # Frequency sweep: compute spectrogram and measure freq at peak per time bin
    try:
        f_stft, t_stft, Sxx = sp_signal.spectrogram(
            x, fs=fs, nperseg=min(n // 8, 256), noverlap=None,
        )
        # Frequency of peak power per time bin
        peak_freq_per_bin = f_stft[np.argmax(Sxx, axis=0)]
        freq_sweep = float(np.std(peak_freq_per_bin))  # spread of peak freq over time
        freq_trend = float(np.polyfit(
            np.arange(len(peak_freq_per_bin)), peak_freq_per_bin, 1)[0]
        )  # slope → positive for inspiral chirp
    except Exception:
        freq_sweep = 0.0
        freq_trend = 0.0

    # ── Assemble feature vector ───────────────────────────────────────────────
    features = np.array([
        rms,
        peak_amp,
        crest_factor,
        kurt,
        sk,
        zcr,
        above_thresh,
        peak_freq,
        peak_psd,
        snr_estimate,
        bandwidth,
        mean_freq,
        freq_sweep,
        freq_trend,
        *band_ratios,          # 4 values
    ], dtype=np.float64)
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)
    return features


def extract_feature_matrix(
    segments: np.ndarray,
    fs: float,
    verbose: bool = True,
) -> np.ndarray:
    """Extract features for all segments in (N, T) array."""
    N = len(segments)
    feats = []
    for i, seg in enumerate(segments):
        if verbose and (i % 500 == 0):
            log.info("  Feature extraction: %d / %d", i, N)
        feats.append(extract_features(seg, fs))
    return np.stack(feats)    # (N, n_features)


# ─── Random Forest model ──────────────────────────────────────────────────────

class GWRandomForest:
    """Random Forest GW classifier wrapping scikit-learn.

    Parameters
    ----------
    cfg : config dict (from config.yaml)
    """

    FEATURE_NAMES: List[str] = [
        "rms", "peak_amp", "crest_factor", "kurtosis", "skewness",
        "zcr", "above_5sigma", "peak_freq", "peak_psd", "snr_estimate",
        "bandwidth", "mean_freq", "freq_sweep", "freq_trend",
        "band_20_60", "band_60_120", "band_120_240", "band_240_500",
    ]

    def __init__(self, cfg: dict) -> None:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler

        mcfg = cfg["model"]
        self.n_classes = mcfg["num_classes"]
        self.fs = cfg["data"]["sample_rate"]
        self.class_names = cfg["classes"]["names"]
        self.class_weights = dict(
            enumerate(cfg["classes"]["weights"])
        )

        self.scaler = StandardScaler()
        self.clf = RandomForestClassifier(
            n_estimators=500,
            max_depth=None,
            min_samples_split=4,
            min_samples_leaf=2,
            class_weight=self.class_weights,
            n_jobs=-1,
            random_state=cfg["training"]["seed"],
        )

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
    ) -> None:
        """Train on raw time-series (N, T) with auto feature extraction."""
        log.info("Extracting training features…")
        F_train = extract_feature_matrix(X_train, self.fs)
        F_train = self.scaler.fit_transform(F_train)

        log.info("Fitting Random Forest on %d samples…", len(y_train))
        self.clf.fit(F_train, y_train)

        if X_val is not None and y_val is not None:
            F_val = self.scaler.transform(extract_feature_matrix(X_val, self.fs, False))
            val_acc = self.clf.score(F_val, y_val)
            log.info("Validation accuracy: %.4f", val_acc)

    def predict(self, X: np.ndarray) -> np.ndarray:
        F = self.scaler.transform(extract_feature_matrix(X, self.fs, False))
        return self.clf.predict(F)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        F = self.scaler.transform(extract_feature_matrix(X, self.fs, False))
        return self.clf.predict_proba(F)

    def feature_importances(self) -> Dict[str, float]:
        """Return dict of feature name → importance score."""
        imps = self.clf.feature_importances_
        return dict(zip(self.FEATURE_NAMES, imps.tolist()))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"clf": self.clf, "scaler": self.scaler}, f)
        log.info("Saved Random Forest → %s", path)

    @classmethod
    def load(cls, path: Path, cfg: dict) -> "GWRandomForest":
        obj = cls(cfg)
        with open(path, "rb") as f:
            state = pickle.load(f)
        obj.clf = state["clf"]
        obj.scaler = state["scaler"]
        log.info("Loaded Random Forest from %s", path)
        return obj
