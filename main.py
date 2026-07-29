#!/usr/bin/env python3
"""
main.py
=======
Command-line entry point for the GW detection pipeline.

Sub-commands
------------
  download    Fetch LIGO strain data from GWOSC
  generate    Create synthetic BBH/BNS injection dataset via PyCBC
  preprocess  Run the full signal-processing pipeline on raw HDF5 files
  split       Divide processed data into train/val/test splits
  train       Train a deep learning model (cnn_spectrogram | cnn_lstm)
  train-rf    Train the Random Forest baseline
  evaluate    Run full evaluation on the test set
  gradcam     Generate Grad-CAM visualisations for the CNN model

Examples
--------
  # 1. Fetch real event data + noise
  python main.py download --runs O1 O2 O3

  # 2. Generate synthetic training data (BBH + BNS injections)
  python main.py generate --n-bbh 5000 --n-bns 5000 --n-noise 10000

  # 3. Preprocess raw data into Q-transform spectrograms
  python main.py preprocess

  # 4. Create train/val/test splits
  python main.py split

  # 5. Train the ResNet-18 spectrogram CNN
  python main.py train --model cnn_spectrogram

  # 6. Evaluate on test set
  python main.py evaluate --model cnn_spectrogram

  # 7. Generate Grad-CAM explanations
  python main.py gradcam

  # 8. Train & evaluate Random Forest baseline
  python main.py train-rf
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import yaml

# ─── Logging setup ────────────────────────────────────────────────────────────

def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


# ─── Config loading ───────────────────────────────────────────────────────────

def load_config(path: str = "config.yaml") -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


# ─── Sub-commands ─────────────────────────────────────────────────────────────

def cmd_download(args: argparse.Namespace, cfg: dict) -> None:
    from src.data.download import main as download_main
    sys.argv = ["download",
                "--runs", *args.runs,
                "--out", cfg["data"]["data_dir"],
                "--detectors", *args.detectors]
    download_main()


def cmd_generate(args: argparse.Namespace, cfg: dict) -> None:
    from src.data.synthetic import (
        _load_psd, generate_bbh_samples, generate_bns_samples,
        generate_noise_samples,
    )
    import numpy as np
    rng = np.random.default_rng(cfg["training"]["seed"])
    fs = cfg["data"]["sample_rate"]
    seg_len = cfg["data"]["segment_length"]
    n_samples = int(fs * seg_len)
    psd = _load_psd(None, fs, n_samples)
    out_dir = Path(cfg["data"]["synthetic_dir"])

    logging.getLogger().info("Generating synthetic waveforms…")
    generate_bbh_samples(args.n_bbh, fs, seg_len, psd, cfg, rng, out_dir)
    generate_bns_samples(args.n_bns, fs, seg_len, psd, cfg, rng, out_dir)
    generate_noise_samples(args.n_noise, fs, seg_len, psd, rng, out_dir)


def cmd_preprocess(args: argparse.Namespace, cfg: dict) -> None:
    from src.data.preprocessing import build_dataset, build_dataset_from_synthetic
    out_dir = Path(cfg["data"]["processed_dir"]) / "spectrograms"

    # Prefer synthetic data if it exists, otherwise fall back to raw HDF5
    synthetic_dir = Path(cfg["data"]["synthetic_dir"])
    synthetic_files = list(synthetic_dir.glob("*.npz")) if synthetic_dir.exists() else []

    if synthetic_files:
        logging.getLogger().info(
            "Found %d synthetic NPZ files in %s — preprocessing those.",
            len(synthetic_files), synthetic_dir,
        )
        build_dataset_from_synthetic(synthetic_dir, out_dir, cfg)
    else:
        raw_dir = Path(cfg["data"]["data_dir"])
        logging.getLogger().info(
            "No synthetic data found; processing raw HDF5 files in %s.", raw_dir
        )
        build_dataset(raw_dir, out_dir, cfg)


def cmd_split(args: argparse.Namespace, cfg: dict) -> None:
    from src.data.dataset import build_splits
    spec_dir = Path(cfg["data"]["processed_dir"]) / "spectrograms"
    ts_dir = Path(cfg["data"]["processed_dir"]) / "timeseries"
    out_dir = Path(cfg["data"]["processed_dir"]) / "splits"
    build_splits(spec_dir, ts_dir if ts_dir.exists() else None,
                 out_dir, cfg, seed=cfg["training"]["seed"])


def cmd_train(args: argparse.Namespace, cfg: dict) -> None:
    import torch
    from src.data.dataset import GWDataModule
    from src.training.trainer import Trainer, get_device

    model_name = args.model or cfg["model"]["name"]
    cfg["model"]["name"] = model_name

    # Override config with CLI flags if provided
    if args.epochs:
        cfg["training"]["num_epochs"] = args.epochs
    if args.lr:
        cfg["training"]["learning_rate"] = args.lr
    if args.batch_size:
        cfg["training"]["batch_size"] = args.batch_size

    device = get_device()

    # ── Data ──────────────────────────────────────────────────────────────────
    mode = "spectrogram" if "cnn_spectrogram" in model_name else "timeseries"
    splits_dir = Path(cfg["data"]["processed_dir"]) / "splits" / ("spectrograms" if mode == "spectrogram" else "timeseries")
    dm = GWDataModule(splits_dir, cfg, mode=mode)
    dm.setup()

    train_loader = dm.train_loader()
    val_loader = dm.val_loader()
    class_weights = dm.class_weights()

    # ── Model ─────────────────────────────────────────────────────────────────
    if model_name == "cnn_spectrogram":
        from src.models.cnn_spectrogram import build_cnn
        model = build_cnn(cfg)
    elif model_name == "cnn_lstm":
        from src.models.cnn_lstm import build_cnn_lstm
        model = build_cnn_lstm(cfg)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # ── Train ─────────────────────────────────────────────────────────────────
    trainer = Trainer(model, cfg, device, class_weights=class_weights)
    history = trainer.train(train_loader, val_loader)

    logging.getLogger().info("Training finished. Loading best checkpoint for final eval…")
    trainer.load_best()

    from src.evaluation.metrics import evaluate_model
    out_dir = Path(cfg["evaluation"]["output_dir"]) / model_name / "val"
    evaluate_model(model, val_loader, device, out_dir, cfg)


def cmd_evaluate(args: argparse.Namespace, cfg: dict) -> None:
    import torch
    from src.data.dataset import GWDataModule
    from src.evaluation.metrics import evaluate_model
    from src.training.trainer import get_device

    model_name = args.model or cfg["model"]["name"]
    cfg["model"]["name"] = model_name

    device = get_device()

    # Load model
    if model_name == "cnn_spectrogram":
        from src.models.cnn_spectrogram import build_cnn
        model = build_cnn(cfg)
    elif model_name == "cnn_lstm":
        from src.models.cnn_lstm import build_cnn_lstm
        model = build_cnn_lstm(cfg)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    ckpt_path = args.checkpoint or Path(cfg["training"]["checkpoint_dir"]) / "best.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    logging.getLogger().info("Loaded checkpoint: %s", ckpt_path)

    mode = "spectrogram" if "cnn_spectrogram" in model_name else "timeseries"
    splits_dir = Path(cfg["data"]["processed_dir"]) / "splits" / ("spectrograms" if mode == "spectrogram" else "timeseries")
    dm = GWDataModule(splits_dir, cfg, mode=mode)
    dm.setup()
    test_loader = dm.test_loader()

    out_dir = Path(cfg["evaluation"]["output_dir"]) / model_name / "test"
    evaluate_model(model, test_loader, device, out_dir, cfg)


def cmd_gradcam(args: argparse.Namespace, cfg: dict) -> None:
    import torch
    from src.data.dataset import GWDataModule
    from src.evaluation.gradcam import GradCAMVisualiser, make_gradcam_grid
    from src.models.cnn_spectrogram import build_cnn
    from src.training.trainer import get_device

    cfg["model"]["name"] = "cnn_spectrogram"
    device = get_device()

    model = build_cnn(cfg)
    ckpt_path = args.checkpoint or Path(cfg["training"]["checkpoint_dir"]) / "best.pt"
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    splits_dir = Path(cfg["data"]["processed_dir"]) / "splits" / "spectrograms"
    dm = GWDataModule(splits_dir, cfg, mode="spectrogram")
    dm.setup()
    test_loader = dm.test_loader()

    out_dir = Path(cfg["evaluation"]["output_dir"]) / "gradcam"

    vis = GradCAMVisualiser(model, model.target_layer)
    vis.visualise_batch(test_loader, device, out_dir, n_samples=args.n_samples)

    make_gradcam_grid(
        model, model.target_layer, test_loader, device,
        out_dir / "gradcam_grid.png",
    )
    vis.remove_hooks()


def cmd_train_rf(args: argparse.Namespace, cfg: dict) -> None:
    import numpy as np
    from sklearn.model_selection import train_test_split
    from src.evaluation.metrics import print_classification_report
    from src.models.random_forest_baseline import GWRandomForest

    # Load raw strain directly from synthetic NPZ files
    syn_dir = Path(cfg["data"]["synthetic_dir"])
    file_label = [
        ("bbh_injections.npz",    0),
        ("bns_injections.npz",    1),
        ("noise_backgrounds.npz", 2),
    ]
    # Build a reference PSD from the noise backgrounds (pure noise, no signal)
    # This is how real LIGO analysis works: estimate noise floor from off-source data
    from scipy.signal import welch
    from src.data.preprocessing import bandpass_filter
    noise_npz = syn_dir / "noise_backgrounds.npz"
    ref_psd = None
    ref_freqs = None
    if noise_npz.exists():
        noise_data = np.load(noise_npz)["data"].astype(np.float64)
        fs_val = float(cfg["data"]["sample_rate"])
        # Average Welch PSD over all noise segments for a stable estimate
        psds = []
        for seg in noise_data[:200]:  # 200 segments is plenty
            seg_bp = bandpass_filter(seg, fs_val, 20.0, 500.0)
            f, p = welch(seg_bp, fs=fs_val, nperseg=min(len(seg_bp), 512))
            psds.append(p)
        ref_freqs = f
        ref_psd = np.mean(psds, axis=0)
        logging.getLogger().info("Reference PSD estimated from %d noise segments", len(psds))

    def whiten_with_ref(seg, fs_val, ref_freqs, ref_psd):
        """Whiten using a pre-computed reference PSD (not the segment's own PSD)."""
        seg = bandpass_filter(seg.astype(np.float64), fs_val, 20.0, 500.0)
        n = len(seg)
        rfft_freqs = np.fft.rfftfreq(n, d=1.0 / fs_val)
        psd_interp = np.interp(rfft_freqs, ref_freqs.astype(np.float64),
                               ref_psd.astype(np.float64))
        asd = np.sqrt(np.maximum(psd_interp, 1e-100))
        asd = np.maximum(asd, np.percentile(asd[asd > 0], 5))
        Xf = np.fft.rfft(seg)
        white = np.real(np.fft.irfft(Xf / asd, n=n))
        return white

    all_X, all_y = [], []
    for fname, label in file_label:
        p = syn_dir / fname
        if not p.exists():
            continue
        npz = np.load(p)
        strains = npz["data"].astype(np.float64)
        fs_val = float(cfg["data"]["sample_rate"])
        logging.getLogger().info("Whitening %s (%d samples)…", fname, len(strains))
        if ref_psd is not None:
            whitened = np.stack([whiten_with_ref(s, fs_val, ref_freqs, ref_psd)
                                 for s in strains])
        else:
            whitened = strains
        all_X.append(whitened)
        all_y.append(np.full(len(strains), label, dtype=np.int64))

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    # 70 / 15 / 15 split
    seed = cfg["training"]["seed"]
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.30, stratify=y, random_state=seed)
    X_val, X_test, y_val, y_test = train_test_split(X_tmp, y_tmp, test_size=0.50, stratify=y_tmp, random_state=seed)
    logging.getLogger().info("Train=%d  Val=%d  Test=%d", len(y_tr), len(y_val), len(y_test))

    rf = GWRandomForest(cfg)
    rf.fit(X_tr, y_tr, X_val, y_val)

    y_pred = rf.predict(X_test)
    print_classification_report(y_test, y_pred, cfg["classes"]["names"])

    # Feature importances
    imps = rf.feature_importances()
    print("\nTop-10 feature importances:")
    for name, imp in sorted(imps.items(), key=lambda x: -x[1])[:10]:
        print(f"  {name:25s}: {imp:.4f}")

    ckpt_dir = Path(cfg["training"]["checkpoint_dir"])
    rf.save(ckpt_dir / "random_forest.pkl")


# ─── CLI parser ───────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="gw_ml",
        description="Gravitational-Wave ML Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", default="config.yaml", help="Path to config YAML")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="command", required=True)

    # download
    dl = sub.add_parser("download", help="Download LIGO open data from GWOSC")
    dl.add_argument("--runs", nargs="+", default=["O1", "O2", "O3"])
    dl.add_argument("--detectors", nargs="+", default=["H1", "L1"])

    # generate
    gen = sub.add_parser("generate", help="Generate synthetic injection dataset")
    gen.add_argument("--n-bbh",   type=int, default=5000)
    gen.add_argument("--n-bns",   type=int, default=5000)
    gen.add_argument("--n-noise", type=int, default=10000)

    # preprocess
    sub.add_parser("preprocess", help="Run DSP pipeline on raw HDF5 files")

    # split
    sub.add_parser("split", help="Create train/val/test splits")

    # train
    tr = sub.add_parser("train", help="Train a deep learning model")
    tr.add_argument("--model", choices=["cnn_spectrogram", "cnn_lstm"],
                    default=None)
    tr.add_argument("--epochs",     type=int,   default=None)
    tr.add_argument("--lr",         type=float, default=None)
    tr.add_argument("--batch-size", type=int,   dest="batch_size", default=None)

    # train-rf
    sub.add_parser("train-rf", help="Train Random Forest baseline")

    # evaluate
    ev = sub.add_parser("evaluate", help="Evaluate model on test set")
    ev.add_argument("--model", choices=["cnn_spectrogram", "cnn_lstm"], default=None)
    ev.add_argument("--checkpoint", default=None, help="Path to .pt checkpoint")

    # gradcam
    gc = sub.add_parser("gradcam", help="Generate Grad-CAM visualisations")
    gc.add_argument("--checkpoint", default=None)
    gc.add_argument("--n-samples",  type=int, default=18, dest="n_samples")

    return p


# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    _setup_logging(args.verbose)

    cfg = load_config(args.config)

    # Fix random seed
    import random
    import numpy as np
    import torch
    seed = cfg["training"]["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    dispatch = {
        "download":   cmd_download,
        "generate":   cmd_generate,
        "preprocess": cmd_preprocess,
        "split":      cmd_split,
        "train":      cmd_train,
        "train-rf":   cmd_train_rf,
        "evaluate":   cmd_evaluate,
        "gradcam":    cmd_gradcam,
    }

    fn = dispatch.get(args.command)
    if fn is None:
        parser.print_help()
        sys.exit(1)

    fn(args, cfg)


if __name__ == "__main__":
    main()
