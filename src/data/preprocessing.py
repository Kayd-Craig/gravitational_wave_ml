"""
preprocessing.py
================
Full LIGO strain-data preprocessing pipeline:

  1. Load HDF5 / gwf strain files  (gwpy TimeSeries)
  2. Bandpass filter           20 – 500 Hz, 8th-order Butterworth
  3. Whiten                    divide FFT by ASD from Welch PSD
  4. Segment                   fixed-length windows (default 1 s @ 4096 Hz)
  5. Q-transform               convert each segment to a 2-D time-frequency image
  6. Normalise                 map pixel values to [0, 1]

Outputs are saved as compressed NumPy arrays ready for the PyTorch Dataset.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import signal as sp_signal

log = logging.getLogger(__name__)

# Label map
LABEL_MAP: Dict[str, int] = {"BBH": 0, "BNS": 1, "Glitch": 2, "Noise": 2}


# ─── low-level DSP helpers ───────────────────────────────────────────────────

def bandpass_filter(
    data: np.ndarray,
    fs: float,
    f_low: float = 20.0,
    f_high: float = 500.0,
    order: int = 8,
) -> np.ndarray:
    """Apply a zero-phase Butterworth bandpass filter.

    Parameters
    ----------
    data : 1-D float array of strain samples
    fs   : sample rate (Hz)
    f_low, f_high : passband edges (Hz)
    order : filter order (applied twice due to sosfiltfilt → effective 2×order)
    """
    nyq = fs / 2.0
    sos = sp_signal.butter(
        order, [f_low / nyq, f_high / nyq], btype="bandpass", output="sos"
    )
    return sp_signal.sosfiltfilt(sos, data).astype(np.float32)


def estimate_psd(
    data: np.ndarray,
    fs: float,
    nperseg: int = 4096,
) -> Tuple[np.ndarray, np.ndarray]:
    """Welch PSD estimate.

    Returns
    -------
    freqs : frequency array (Hz)
    psd   : one-sided PSD (strain²/Hz)
    """
    freqs, psd = sp_signal.welch(
        data.astype(np.float64),
        fs=fs,
        nperseg=nperseg,
        noverlap=nperseg // 2,
        window="hann",
        scaling="density",
    )
    return freqs.astype(np.float32), psd.astype(np.float64)


def whiten(
    data: np.ndarray,
    fs: float,
    psd: Optional[np.ndarray] = None,
    nperseg: int = 4096,
    fduration: float = 2.0,
) -> np.ndarray:
    """Whiten strain data by dividing by the amplitude spectral density.

    Parameters
    ----------
    data     : 1-D float array
    fs       : sample rate
    psd      : pre-computed PSD (or None → estimated from *data*)
    nperseg  : Welch segment length
    fduration: roll-off duration (seconds) applied at edges to reduce ringing
    """
    n = len(data)
    # rfft frequency axis for THIS signal
    rfft_freqs = np.fft.rfftfreq(n, d=1.0 / fs)   # shape (n//2+1,)

    if psd is None:
        # Estimate directly on the rfft grid so no interpolation is needed
        freqs_psd, psd_est = estimate_psd(data, fs, nperseg)
        # Interpolate Welch PSD (defined on nperseg/2+1 points) onto rfft grid
        psd_interp = np.interp(
            rfft_freqs.astype(np.float64),
            freqs_psd.astype(np.float64),
            psd_est.astype(np.float64),
        )
    else:
        # psd was computed externally; we need a matching frequency axis.
        # Build a linear grid matching psd length (Welch convention: df = fs/nperseg)
        psd_arr = np.asarray(psd, dtype=np.float64)
        n_psd = len(psd_arr)
        # Reconstruct freq axis: assume it spans [0, fs/2] with n_psd points
        freqs_ext = np.linspace(0.0, fs / 2.0, n_psd)
        psd_interp = np.interp(
            rfft_freqs.astype(np.float64),
            freqs_ext,
            psd_arr,
        )
    asd_interp = np.sqrt(np.maximum(psd_interp, 1e-100))

    # Whiten in frequency domain
    data_fft = np.fft.rfft(data.astype(np.float64))
    white_fft = data_fft / asd_interp

    # Inverse FFT
    white = np.fft.irfft(white_fft, n=n).astype(np.float32)

    # Apply Tukey window to reduce edge effects
    roll = int(fduration * fs / 2)
    tukey = sp_signal.windows.tukey(n, alpha=max(0.0, min(1.0, 2 * roll / n)))
    white *= tukey.astype(np.float32)

    return white


def q_transform_numpy(
    data: np.ndarray,
    fs: float,
    frange: Tuple[float, float] = (20.0, 500.0),
    qrange: Tuple[float, float] = (4.0, 64.0),
    n_freq_bins: int = 128,
    n_time_bins: int = 128,
    out_shape: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Compute a Q-transform spectrogram using scipy CWT as a fallback
    when gwpy is not available.

    Returns a (n_freq_bins, n_time_bins) float32 array of normalised energy.
    """
    # Allow out_shape=(H,W) as alternative to n_freq_bins / n_time_bins
    if out_shape is not None:
        n_freq_bins, n_time_bins = out_shape

    # Log-spaced frequency axis
    f_lo, f_hi = frange
    log_freqs = np.logspace(np.log10(f_lo), np.log10(f_hi), n_freq_bins)

    # Pure-numpy manual STFT — works on all scipy/numpy versions,
    # avoids scipy.signal.stft which changed behaviour in 1.17.
    n_data = len(data)
    d = data.astype(np.float64)

    # Window and hop size: aim for ~n_time_bins frames
    win  = max(32, min(512, n_data // 4))   # FFT window (power of 2 not required)
    hop  = max(1,  n_data // n_time_bins)    # step between frames
    window = np.hanning(win)
    freqs_fft = np.fft.rfftfreq(win, d=1.0 / fs)  # (win//2+1,)

    # Build frames manually
    starts = list(range(0, n_data - win + 1, hop))
    if not starts:               # signal shorter than window
        starts = [0]
        win = n_data
        window = np.hanning(win)
        freqs_fft = np.fft.rfftfreq(win, d=1.0 / fs)

    frames = []
    for s in starts:
        seg = d[s : s + win]
        if len(seg) < win:
            seg = np.pad(seg, (0, win - len(seg)))
        frames.append(np.abs(np.fft.rfft(seg * window)) ** 2)

    Sxx = np.array(frames).T   # (n_fft_bins, n_frames)

    # Interpolate onto log-spaced frequency axis
    f_lo, f_hi = frange
    log_freqs = np.logspace(np.log10(f_lo), np.log10(f_hi), n_freq_bins)
    energy_log = np.zeros((n_freq_bins, Sxx.shape[1]), dtype=np.float64)
    for i, freq in enumerate(log_freqs):
        idx = int(np.clip(np.searchsorted(freqs_fft, freq), 0, len(freqs_fft) - 1))
        energy_log[i] = Sxx[idx]

    # Downsample time axis to n_time_bins
    t_idx = np.round(np.linspace(0, energy_log.shape[1] - 1, n_time_bins)).astype(int)
    energy_ds = energy_log[:, t_idx]

    # Median normalise per frequency bin
    median = np.median(energy_ds, axis=1, keepdims=True)
    median = np.where(median == 0.0, 1.0, median)
    normalised = energy_ds / median

    return normalised.astype(np.float32)


def q_transform_gwpy(
    data: np.ndarray,
    fs: float,
    t0: float = 0.0,
    frange: Tuple[float, float] = (20.0, 500.0),
    qrange: Tuple[float, float] = (4.0, 64.0),
    out_shape: Tuple[int, int] = (128, 128),
) -> np.ndarray:
    """Compute Q-transform via gwpy (preferred when available).

    Returns a (H, W) float32 array.
    """
    from gwpy.timeseries import TimeSeries

    ts = TimeSeries(data.astype(np.float64), t0=t0, sample_rate=fs, unit="")
    qgram = ts.q_transform(
        qrange=qrange,
        frange=frange,
        logf=True,
        norm="median",
    )
    # qgram is a Spectrogram; resample to fixed size
    from PIL import Image
    arr = np.array(qgram).astype(np.float32)
    img = Image.fromarray(arr)
    img_resized = img.resize((out_shape[1], out_shape[0]), Image.BILINEAR)
    return np.array(img_resized, dtype=np.float32)


def compute_spectrogram(
    segment: np.ndarray,
    fs: float,
    frange: Tuple[float, float] = (20.0, 500.0),
    qrange: Tuple[float, float] = (4.0, 64.0),
    out_shape: Tuple[int, int] = (128, 128),
    t0: float = 0.0,
) -> np.ndarray:
    """Compute Q-transform spectrogram, using gwpy if available.

    Returns (H, W) float32.
    """
    try:
        return q_transform_gwpy(segment, fs, t0=t0, frange=frange,
                                qrange=qrange, out_shape=out_shape)
    except Exception:
        return q_transform_numpy(segment, fs, frange=frange, qrange=qrange,
                                 n_freq_bins=out_shape[0], n_time_bins=out_shape[1])


# ─── segmentation ─────────────────────────────────────────────────────────────

def segment_strain(
    data: np.ndarray,
    fs: float,
    seg_len: float = 1.0,
    overlap: float = 0.5,
) -> List[np.ndarray]:
    """Split a long strain array into fixed-length overlapping segments.

    Parameters
    ----------
    data    : 1-D strain array
    fs      : sample rate (Hz)
    seg_len : segment duration (seconds)
    overlap : fractional overlap between consecutive segments [0, 1)
    """
    n_seg = int(seg_len * fs)
    step = int(n_seg * (1 - overlap))
    segments: List[np.ndarray] = []
    for start in range(0, len(data) - n_seg + 1, step):
        segments.append(data[start : start + n_seg].astype(np.float32))
    return segments


# ─── high-level pipeline ──────────────────────────────────────────────────────

def preprocess_strain(
    raw: np.ndarray,
    fs: float,
    seg_len: float = 1.0,
    f_low: float = 20.0,
    f_high: float = 500.0,
    filter_order: int = 8,
    overlap: float = 0.5,
    out_shape: Tuple[int, int] = (128, 128),
) -> np.ndarray:
    """End-to-end preprocessing: filter → whiten → segment → Q-transform.

    Parameters
    ----------
    raw      : raw strain time-series (1-D float64)
    fs       : sample rate (Hz)

    Returns
    -------
    spectrograms : (N, H, W) float32 array of Q-transform images
    """
    log.debug("Input length: %d samples (%.2f s)", len(raw), len(raw) / fs)

    # 1. Bandpass filter
    filtered = bandpass_filter(raw, fs, f_low, f_high, filter_order)

    # 2. Estimate PSD from full filtered stream, then whiten
    freqs, psd = estimate_psd(filtered, fs)
    whitened = whiten(filtered, fs, psd=psd)

    # 3. Segment
    segments = segment_strain(whitened, fs, seg_len, overlap)
    log.debug("Segments: %d", len(segments))

    # 4. Q-transform each segment
    specs = []
    for i, seg in enumerate(segments):
        spec = compute_spectrogram(
            seg, fs,
            frange=(f_low, f_high),
            qrange=(4.0, 64.0),
            out_shape=out_shape,
            t0=float(i * seg_len * (1 - overlap)),
        )
        specs.append(spec)

    return np.stack(specs, axis=0)   # (N, H, W)


# ─── file-level helpers ───────────────────────────────────────────────────────

def load_hdf5_strain(path: Path, detector: str = "H1") -> Tuple[np.ndarray, float]:
    """Load strain and sample-rate from a GWOSC-format HDF5 file.

    Returns (data, sample_rate).
    """
    import h5py

    with h5py.File(path, "r") as f:
        # GWOSC HDF5 layout: /strain/Strain
        strain = f["strain"]["Strain"][:]
        dt = f["strain"]["Strain"].attrs.get("Xspacing", None)
        if dt is not None:
            fs = 1.0 / float(dt)
        else:
            # Fall back to metadata
            fs = float(f["meta"]["SampleRate"][()])
    return strain.astype(np.float64), fs


def process_file(
    path: Path,
    label: int,
    out_dir: Path,
    cfg: dict,
) -> int:
    """Process a single HDF5 strain file and save spectrogram batch.

    Returns number of segments saved.
    """
    try:
        strain, fs = load_hdf5_strain(path)
    except Exception as exc:
        log.warning("Failed to load %s: %s", path, exc)
        return 0

    specs = preprocess_strain(
        strain,
        fs,
        seg_len=cfg["data"]["segment_length"],
        f_low=cfg["data"]["bandpass_low"],
        f_high=cfg["data"]["bandpass_high"],
        filter_order=cfg["data"].get("bandpass_order", 8),
        overlap=0.5,
        out_shape=(128, 128),
    )

    out_dir.mkdir(parents=True, exist_ok=True)
    stem = path.stem
    out_path = out_dir / f"{stem}_specs.npz"
    labels = np.full(len(specs), label, dtype=np.int64)
    np.savez_compressed(out_path, spectrograms=specs, labels=labels)
    log.info("Saved %d spectrograms → %s", len(specs), out_path)
    return len(specs)


def build_dataset_from_synthetic(
    synthetic_dir: Path,
    out_dir: Path,
    cfg: dict,
) -> None:
    """Convert synthetic NPZ files (from synthetic.py) into spectrogram NPZ files.

    Reads bbh_injections.npz, bns_injections.npz, noise_backgrounds.npz
    from *synthetic_dir* and writes spectrogram batches to *out_dir* in the
    same format that build_dataset / process_file produces.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    fs = float(cfg["data"]["sample_rate"])
    total = 0

    file_map = {
        "bbh_injections.npz":   0,   # BBH
        "bns_injections.npz":   1,   # BNS
        "noise_backgrounds.npz": 2,  # Noise
    }

    for fname, label in file_map.items():
        npz_path = synthetic_dir / fname
        if not npz_path.exists():
            log.warning("Synthetic file not found, skipping: %s", npz_path)
            continue

        npz = np.load(npz_path)
        strains = npz["data"]        # shape (N, samples)
        log.info("Processing %s  (%d segments, label=%d)", fname, len(strains), label)

        specs = []
        # Synthetic data is already pre-segmented: process each strain directly
        f_low  = float(cfg["data"]["bandpass_low"])
        f_high = float(cfg["data"]["bandpass_high"])
        order  = int(cfg["data"].get("bandpass_order", 8))

        n_errors = 0
        for strain in strains:
            try:
                s = strain.astype(np.float64)
                s = bandpass_filter(s, fs, f_low, f_high, order)
                _, psd = estimate_psd(s, fs)
                s = whiten(s, fs, psd=psd)
                spec = compute_spectrogram(s, fs, frange=(f_low, f_high), out_shape=(128, 128))
                specs.append(spec)
            except Exception as exc:
                n_errors += 1
                if n_errors <= 3:
                    log.warning("  Segment error: %s", exc)
        if n_errors:
            log.warning("  Skipped %d / %d segments due to errors", n_errors, len(strains))

        if specs:
            specs_arr = np.stack(specs)                           # (M, 128, 128)
            labels_arr = np.full(len(specs_arr), label, dtype=np.int64)
            stem = fname.replace(".npz", "")
            out_path = out_dir / f"{stem}_specs.npz"
            np.savez_compressed(out_path, spectrograms=specs_arr, labels=labels_arr)
            log.info("  Saved %d spectrograms → %s", len(specs_arr), out_path)
            total += len(specs_arr)

    log.info("Synthetic preprocessing complete. Total spectrograms: %d", total)


def build_dataset(
    raw_dir: Path,
    out_dir: Path,
    cfg: dict,
    label_map: Optional[Dict[str, int]] = None,
) -> None:
    """Walk *raw_dir*, process every HDF5 file, save to *out_dir*.

    Directory structure expected::

        raw_dir/
          O1/events/GW150914_H1.hdf5   → label BBH (0)
          O1/noise/*.hdf5               → label Glitch/Noise (2)
    """
    label_map = label_map or LABEL_MAP
    total = 0

    for hdf5 in sorted(raw_dir.rglob("*.hdf5")):
        # Infer label from directory name
        parts = {p.lower() for p in hdf5.parts}
        if "noise" in parts or "glitch" in parts:
            label = label_map.get("Noise", 2)
        elif "bns" in hdf5.stem.lower() or "gw170817" in hdf5.stem:
            label = label_map.get("BNS", 1)
        else:
            label = label_map.get("BBH", 0)

        count = process_file(hdf5, label, out_dir, cfg)
        total += count

    log.info("Dataset build complete. Total segments: %d", total)
