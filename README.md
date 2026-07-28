# Gravitational Wave Detection with Deep Learning

Detecting and classifying gravitational-wave signals from LIGO interferometer data
using deep learning — a complete ML pipeline from raw strain download to Grad-CAM
interpretability analysis.

---

## Overview

| Task | Model | Key metric |
|---|---|---|
| Signal vs. noise (binary) | CNN / BiLSTM | ROC-AUC, FAR |
| BBH / BNS / Glitch (multi-class) | CNN / BiLSTM + RF | Macro-F1 |

**Classes**
- `BBH` (0) — Binary Black Hole merger
- `BNS` (1) — Binary Neutron Star merger
- `Glitch` (2) — Instrumental / environmental artefact

---

## Architecture

```
Raw LIGO HDF5
      │
      ▼
 ┌────────────────────────────────────┐
 │  DSP Pipeline (preprocessing.py)  │
 │  ① Bandpass filter 20–500 Hz      │
 │  ② Welch PSD → Whiten             │
 │  ③ Segment (1 s @ 4096 Hz)        │
 │  ④ Q-transform → (128×128) image  │
 └────────────────┬───────────────────┘
                  │
        ┌─────────┴──────────┐
        │                    │
        ▼                    ▼
┌──────────────┐    ┌──────────────────┐
│ SpectrogramCNN│    │  CNNLSTM         │
│ ResNet-18     │    │  1D Conv ×3      │
│ (1-ch input)  │    │  BiLSTM ×2       │
│ Dropout head  │    │  Attention pool  │
└──────┬────────┘    └──────┬───────────┘
       │                    │
       └─────────┬──────────┘
                 ▼
        ┌────────────────┐
        │ Evaluation     │
        │ Accuracy, F1   │
        │ ROC-AUC, FAR   │
        │ Grad-CAM       │
        └────────────────┘
```

### SpectrogramCNN
- **Backbone**: ResNet-18 pre-trained on ImageNet; first conv adapted to 1 input channel
- **Head**: GlobalAvgPool → Dropout(0.3) → Linear(512→256) → ReLU → Linear(256→3)
- **Input**: (B, 1, 128, 128) Q-transform spectrogram

### CNNLSTM
- **1-D CNN**: 3 blocks `[64, 128, 256]` channels with kernel 15/7 and pool-4
- **BiLSTM**: 2 layers, 256 hidden units per direction
- **Attention**: Additive temporal attention over LSTM outputs
- **Input**: (B, 4096) raw whitened strain

### Random Forest Baseline
- 500 trees on 18 hand-crafted features (peak freq, SNR, kurtosis, freq sweep, etc.)
- Inverse-frequency class weighting

---

## Project Structure

```
gravitational_wave_ml/
├── config.yaml               # All hyperparameters & paths
├── main.py                   # CLI entry point
├── requirements.txt
├── src/
│   ├── data/
│   │   ├── download.py       # GWOSC strain download
│   │   ├── preprocessing.py  # Bandpass → whiten → Q-transform
│   │   ├── synthetic.py      # PyCBC waveform injection
│   │   └── dataset.py        # PyTorch Datasets + DataModule
│   ├── models/
│   │   ├── cnn_spectrogram.py  # ResNet-18 CNN
│   │   ├── cnn_lstm.py         # 1D CNN + BiLSTM
│   │   └── random_forest_baseline.py
│   ├── training/
│   │   └── trainer.py        # Training loop, early stopping, checkpointing
│   ├── evaluation/
│   │   ├── metrics.py        # FAR, ROC-AUC, F1, confusion matrix
│   │   └── gradcam.py        # Grad-CAM visualisation
│   └── utils/
│       └── augmentation.py   # SMOTE, SpecAugment, Mixup
├── notebooks/
│   └── 01_exploratory_analysis.ipynb
├── data/
│   ├── raw/                  # Downloaded HDF5 files
│   ├── processed/            # Spectrograms + time-series NPZ
│   └── synthetic/            # PyCBC injection NPZ files
├── checkpoints/              # Saved .pt model weights
├── results/                  # Evaluation plots + metrics.json
└── logs/                     # training_log.csv
```

---

## Setup

```bash
# Clone and create environment
git clone <repo_url>
cd gravitational_wave_ml
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

> **GPU**: PyTorch will automatically use CUDA if available. For Apple Silicon, MPS is used.

---

## Quickstart

### Option A — Real LIGO data

```bash
# 1. Download event strain + noise from GWOSC
python main.py download --runs O1 O2 O3

# 2. Preprocess: bandpass → whiten → Q-transform spectrograms
python main.py preprocess

# 3. Split into train/val/test
python main.py split
```

### Option B — Synthetic data only (no download needed)

```bash
# Generate 5000 BBH + 5000 BNS + 10000 noise segments via PyCBC
python main.py generate --n-bbh 5000 --n-bns 5000 --n-noise 10000
python main.py split
```

### Train

```bash
# ResNet-18 CNN on spectrograms (recommended)
python main.py train --model cnn_spectrogram

# 1D CNN + BiLSTM on raw strain
python main.py train --model cnn_lstm

# Random Forest baseline
python main.py train-rf
```

### Evaluate

```bash
# Full evaluation: metrics.json + ROC + PR + FAR + confusion matrix
python main.py evaluate --model cnn_spectrogram

# Grad-CAM interpretability (CNN only)
python main.py gradcam --n-samples 24
```

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| **Accuracy / F1** | Standard multi-class performance |
| **ROC-AUC** | One-vs-rest ranking quality per class |
| **FAR** | False Alarm Rate (events/second) — key GW metric |
| **Detection efficiency** | Fraction of true signals recovered at FAR ≤ threshold |
| **Precision-Recall** | Prioritised over accuracy given class imbalance |
| **Confusion matrix** | Per-class confusion (BBH vs BNS is the hardest boundary) |

The FAR target is configurable in `config.yaml` (default `1e-3 Hz`).

---

## Addressing Class Imbalance

Three complementary strategies:

1. **Synthetic injection** via PyCBC — generates thousands of realistic BBH/BNS waveforms
2. **Inverse-frequency class weights** in `CrossEntropyLoss`
3. **Weighted random sampler** — ensures each mini-batch has balanced class representation
4. **SMOTE** (optional, for RF features) — over-samples minority classes in feature space

---

## Interpretability

Grad-CAM overlays highlight *which time-frequency regions* drove the classification:

- **BBH signals**: bright diagonal streak (chirp sweeping 20→200 Hz in ~0.2 s)
- **BNS signals**: longer, lower-frequency chirp spending hundreds of seconds below 100 Hz
- **Glitches**: concentrated blobs at fixed frequencies (e.g., power-line harmonics at 60 Hz)

```
python main.py gradcam --n-samples 18
# Outputs: results/gradcam/gradcam_BBH_00.png, gradcam_grid.png, …
```

---

## Data Sources

- **LIGO Open Science Center**: [gw-openscience.org](https://gw-openscience.org)
- **Gravity Spy glitch catalogue**: [Zooniverse](https://www.zooniverse.org/projects/zooniverse/gravity-spy)
- **PyCBC waveform library**: [pycbc.org](https://pycbc.org)

---

## References

- Abbott et al. (2016) — GW150914: First direct detection of gravitational waves
- George & Huerta (2018) — Deep Learning for Real-time Gravitational Wave Detection
- Gabbard et al. (2018) — Matching Matched Filtering with Deep Networks
- Park et al. (2019) — SpecAugment: Data Augmentation for Speech Recognition
- Zhang et al. (2018) — Mixup: Beyond Empirical Risk Minimisation
