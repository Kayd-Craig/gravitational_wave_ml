"""
synthetic.py
============
Generate synthetic BBH and BNS waveforms via PyCBC and inject them into
LIGO-like Gaussian noise coloured by a real detector PSD.

This module addresses the class-imbalance problem by augmenting the positive
class (GW signals) with thousands of simulated events.

Usage
-----
python -m src.data.synthetic \\
    --n-bbh 5000 --n-bns 5000 \\
    --noise-psd data/raw/O3/psd_H1.txt \\
    --out data/synthetic
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)


# ─── PyCBC waveform generation ────────────────────────────────────────────────

def _generate_bbh_waveform(
    mass1: float,
    mass2: float,
    spin1z: float,
    spin2z: float,
    distance: float,
    fs: float,
    f_lower: float = 20.0,
    approximant: str = "IMRPhenomD",
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (hp, hc) polarisations as numpy arrays for a BBH merger."""
    from pycbc.waveform import get_td_waveform

    hp, hc = get_td_waveform(
        approximant=approximant,
        mass1=mass1,
        mass2=mass2,
        spin1z=spin1z,
        spin2z=spin2z,
        distance=distance,
        delta_t=1.0 / fs,
        f_lower=f_lower,
    )
    return np.array(hp), np.array(hc)


def _generate_bns_waveform(
    mass1: float,
    mass2: float,
    spin1z: float,
    spin2z: float,
    distance: float,
    fs: float,
    f_lower: float = 20.0,
    approximant: str = "TaylorF2",
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (hp, hc) polarisations for a BNS merger.

    TaylorF2 is in the frequency domain; we use get_fd_waveform and IFFT.
    """
    from pycbc.waveform import get_fd_waveform
    from pycbc.types import FrequencySeries

    df = 1.0 / 8.0   # frequency resolution → 8-second waveform
    hp_fd, hc_fd = get_fd_waveform(
        approximant=approximant,
        mass1=mass1,
        mass2=mass2,
        spin1z=spin1z,
        spin2z=spin2z,
        distance=distance,
        delta_f=df,
        f_lower=f_lower,
    )
    hp = hp_fd.to_timeseries(delta_t=1.0 / fs)
    hc = hc_fd.to_timeseries(delta_t=1.0 / fs)
    return np.array(hp), np.array(hc)


# ─── Coloured noise ───────────────────────────────────────────────────────────

def _load_psd(psd_path: Optional[Path], fs: float, n: int) -> np.ndarray:
    """Load a two-column (freq, PSD) text file and interpolate onto rfft grid."""
    from scipy.interpolate import interp1d

    rfft_freqs = np.fft.rfftfreq(n, d=1.0 / fs)

    if psd_path and psd_path.exists():
        table = np.loadtxt(psd_path)
        f_tab, psd_tab = table[:, 0], table[:, 1]
        interp = interp1d(f_tab, psd_tab, bounds_error=False,
                          fill_value=(psd_tab[0], psd_tab[-1]))
        psd = interp(rfft_freqs)
    else:
        # Design sensitivity (LIGO O3 approximate) using aLIGO analytical curve
        try:
            from pycbc.psd import analytical
            psd_series = analytical.aLIGODesignSensitivityP1200087(
                len(rfft_freqs), delta_f=rfft_freqs[1], low_freq_cutoff=10.0
            )
            psd = np.array(psd_series)
        except Exception:
            log.warning("PyCBC PSD unavailable; using flat noise PSD")
            psd = np.ones(len(rfft_freqs)) * 1e-46

    # Avoid divide-by-zero at DC
    psd[0] = psd[1] if len(psd) > 1 else 1e-46
    return psd.astype(np.float64)


def make_coloured_noise(
    n_samples: int,
    fs: float,
    psd: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate Gaussian noise coloured by a given one-sided PSD."""
    white = rng.standard_normal(n_samples)
    white_fft = np.fft.rfft(white)
    asd = np.sqrt(psd * fs / 2.0)
    coloured_fft = white_fft * asd
    coloured = np.fft.irfft(coloured_fft, n=n_samples)
    return coloured.astype(np.float32)


# ─── injection ────────────────────────────────────────────────────────────────

def inject_signal(
    noise: np.ndarray,
    waveform: np.ndarray,
    target_snr: float,
    psd: np.ndarray,
    fs: float,
    merger_offset: float = 0.5,
) -> np.ndarray:
    """Inject *waveform* into *noise* at a desired network SNR.

    The merger peak is placed at *merger_offset* seconds from the start.
    """
    n = len(noise)
    merger_sample = int(merger_offset * fs)

    # Trim or zero-pad waveform so it fits
    w = waveform.copy()
    if len(w) > n:
        w = w[-n:]   # keep the end (contains the merger)
    wf = np.zeros(n, dtype=np.float64)
    start = max(0, merger_sample - len(w))
    end = start + len(w)
    if end > n:
        w = w[: n - start]
        end = n
    wf[start:end] = w

    # Compute optimal SNR
    rfft_freqs = np.fft.rfftfreq(n, d=1.0 / fs)
    wf_fft = np.fft.rfft(wf)
    integrand = np.abs(wf_fft) ** 2 / np.maximum(psd, 1e-100)
    df = rfft_freqs[1] - rfft_freqs[0] if len(rfft_freqs) > 1 else 1.0
    optimal_snr = np.sqrt(4 * df * np.sum(integrand))

    if optimal_snr < 1e-30:
        log.warning("Waveform has negligible power; skipping scaling")
        return (noise + wf).astype(np.float32)

    scale = target_snr / optimal_snr
    return (noise + scale * wf).astype(np.float32)


# ─── high-level generators ────────────────────────────────────────────────────

def generate_bbh_samples(
    n: int,
    fs: float,
    seg_len: float,
    psd: np.ndarray,
    cfg: dict,
    rng: np.random.Generator,
    out_dir: Path,
) -> None:
    """Generate *n* BBH injection segments and save as NPZ."""
    n_samples = int(seg_len * fs)
    bbh_cfg = cfg.get("pycbc_waveforms", {}).get("bbh", {})
    snr_lo, snr_hi = cfg.get("pycbc_waveforms", {}).get("snr_range", [8, 30])

    out_dir.mkdir(parents=True, exist_ok=True)
    signals, labels = [], []

    for i in range(n):
        m1 = rng.uniform(*bbh_cfg.get("mass1_range", [10, 80]))
        m2 = rng.uniform(*bbh_cfg.get("mass2_range", [10, 80]))
        if m2 > m1:
            m1, m2 = m2, m1
        s1z = rng.uniform(*bbh_cfg.get("spin1z_range", [-0.9, 0.9]))
        s2z = rng.uniform(*bbh_cfg.get("spin2z_range", [-0.9, 0.9]))
        dist = rng.uniform(100, 2000)   # Mpc
        snr = rng.uniform(snr_lo, snr_hi)

        try:
            hp, _ = _generate_bbh_waveform(m1, m2, s1z, s2z, dist, fs)
            noise = make_coloured_noise(n_samples, fs, psd, rng)
            seg = inject_signal(noise, hp, snr, psd, fs)
            signals.append(seg)
            labels.append(0)  # BBH
        except Exception as exc:
            log.debug("BBH gen %d failed: %s", i, exc)

        if (i + 1) % 100 == 0:
            log.info("  BBH: %d / %d", i + 1, n)

    if signals:
        np.savez_compressed(
            out_dir / "bbh_injections.npz",
            data=np.stack(signals),
            labels=np.array(labels, dtype=np.int64),
        )
        log.info("Saved %d BBH injections", len(signals))


def generate_bns_samples(
    n: int,
    fs: float,
    seg_len: float,
    psd: np.ndarray,
    cfg: dict,
    rng: np.random.Generator,
    out_dir: Path,
) -> None:
    """Generate *n* BNS injection segments and save as NPZ."""
    n_samples = int(seg_len * fs)
    bns_cfg = cfg.get("pycbc_waveforms", {}).get("bns", {})
    snr_lo, snr_hi = cfg.get("pycbc_waveforms", {}).get("snr_range", [8, 30])

    out_dir.mkdir(parents=True, exist_ok=True)
    signals, labels = [], []

    for i in range(n):
        m1 = rng.uniform(*bns_cfg.get("mass1_range", [1.0, 2.5]))
        m2 = rng.uniform(*bns_cfg.get("mass2_range", [1.0, 2.5]))
        if m2 > m1:
            m1, m2 = m2, m1
        s1z = rng.uniform(*bns_cfg.get("spin1z_range", [-0.05, 0.05]))
        s2z = rng.uniform(*bns_cfg.get("spin2z_range", [-0.05, 0.05]))
        dist = rng.uniform(10, 500)   # Mpc (BNS range ~200 Mpc for O3)
        snr = rng.uniform(snr_lo, snr_hi)

        try:
            hp, _ = _generate_bns_waveform(m1, m2, s1z, s2z, dist, fs)
            noise = make_coloured_noise(n_samples, fs, psd, rng)
            seg = inject_signal(noise, hp, snr, psd, fs)
            signals.append(seg)
            labels.append(1)  # BNS
        except Exception as exc:
            log.debug("BNS gen %d failed: %s", i, exc)

        if (i + 1) % 100 == 0:
            log.info("  BNS: %d / %d", i + 1, n)

    if signals:
        np.savez_compressed(
            out_dir / "bns_injections.npz",
            data=np.stack(signals),
            labels=np.array(labels, dtype=np.int64),
        )
        log.info("Saved %d BNS injections", len(signals))


def generate_noise_samples(
    n: int,
    fs: float,
    seg_len: float,
    psd: np.ndarray,
    rng: np.random.Generator,
    out_dir: Path,
) -> None:
    """Generate *n* pure coloured-noise background segments."""
    n_samples = int(seg_len * fs)
    out_dir.mkdir(parents=True, exist_ok=True)
    signals = np.stack([
        make_coloured_noise(n_samples, fs, psd, rng) for _ in range(n)
    ])
    labels = np.full(n, 2, dtype=np.int64)  # Glitch / Noise
    np.savez_compressed(
        out_dir / "noise_backgrounds.npz",
        data=signals, labels=labels,
    )
    log.info("Saved %d noise backgrounds", n)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate synthetic GW injection dataset")
    p.add_argument("--n-bbh",   type=int, default=3000, help="Number of BBH injections")
    p.add_argument("--n-bns",   type=int, default=3000, help="Number of BNS injections")
    p.add_argument("--n-noise", type=int, default=6000, help="Number of noise segments")
    p.add_argument("--fs",      type=float, default=4096.0)
    p.add_argument("--seg-len", type=float, default=1.0, help="Segment length (s)")
    p.add_argument("--psd",     default=None, help="Path to two-column PSD text file")
    p.add_argument("--out",     default="data/synthetic")
    p.add_argument("--seed",    type=int, default=42)
    p.add_argument("--config",  default="config.yaml")
    return p.parse_args()


def main() -> None:
    import yaml
    args = _parse_args()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s | %(levelname)-8s | %(message)s")

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out)
    fs = args.fs
    seg_len = args.seg_len
    n_samples = int(seg_len * fs)

    psd_path = Path(args.psd) if args.psd else None
    psd = _load_psd(psd_path, fs, n_samples)

    log.info("Generating %d BBH injections…", args.n_bbh)
    generate_bbh_samples(args.n_bbh, fs, seg_len, psd, cfg, rng, out_dir)

    log.info("Generating %d BNS injections…", args.n_bns)
    generate_bns_samples(args.n_bns, fs, seg_len, psd, cfg, rng, out_dir)

    log.info("Generating %d noise backgrounds…", args.n_noise)
    generate_noise_samples(args.n_noise, fs, seg_len, psd, rng, out_dir)

    log.info("Synthetic dataset saved to %s", out_dir.resolve())


if __name__ == "__main__":
    main()
