"""
cnn_lstm.py
===========
Hybrid 1-D CNN + Bidirectional LSTM model for gravitational-wave classification
directly on raw whitened strain time-series.

Architecture
------------
  Input  : (B, T)          raw whitened strain (T = 4096 at 4096 Hz / 1 s)
  Stage 1: 1-D Conv blocks  local feature extraction (chirp shape, frequency)
  Stage 2: BiLSTM layers    temporal sequence modelling
  Stage 3: Attention        soft attention over LSTM outputs
  Stage 4: FC head          → num_classes logits

Inspired by work from the LIGO Scientific Collaboration on real-time detection
(Gabbard et al. 2018, George & Huerta 2018).
"""

from __future__ import annotations

import logging
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

log = logging.getLogger(__name__)


# ─── Attention ────────────────────────────────────────────────────────────────

class TemporalAttention(nn.Module):
    """Additive (Bahdanau-style) attention over a sequence of LSTM outputs.

    Input : (B, T, H)
    Output: (B, H) — weighted sum of steps
    """

    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.attn = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # h : (B, T, H)
        scores = self.attn(h).squeeze(-1)       # (B, T)
        weights = F.softmax(scores, dim=-1)      # (B, T)
        context = (weights.unsqueeze(-1) * h).sum(dim=1)  # (B, H)
        return context, weights


# ─── 1-D Conv block ───────────────────────────────────────────────────────────

class Conv1dBlock(nn.Module):
    """Conv1d → BN → ReLU → optional MaxPool."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int = 7,
        stride: int = 1,
        pool: int = 2,
    ) -> None:
        super().__init__()
        pad = kernel // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, stride=stride,
                              padding=pad, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.pool = nn.MaxPool1d(pool) if pool > 1 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(F.relu(self.bn(self.conv(x)), inplace=True))


# ─── Main model ───────────────────────────────────────────────────────────────

class CNNLSTM(nn.Module):
    """1-D CNN + Bidirectional LSTM classifier for raw GW time-series.

    Parameters
    ----------
    input_len    : number of samples per segment (default 4096)
    num_classes  : 2 (binary) or 3 (multi-class)
    cnn_channels : list of conv channel widths per block
    lstm_hidden  : hidden units per LSTM direction
    lstm_layers  : number of stacked LSTM layers
    dropout      : dropout probability
    """

    def __init__(
        self,
        input_len: int = 4096,
        num_classes: int = 3,
        cnn_channels: Optional[List[int]] = None,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        cnn_channels = cnn_channels or [64, 128, 256]

        # ── 1-D CNN encoder ────────────────────────────────────────────────────
        blocks: List[nn.Module] = []
        in_ch = 1
        for i, out_ch in enumerate(cnn_channels):
            # Larger kernel at the first block to capture long-wavelength chirp
            kernel = 15 if i == 0 else 7
            blocks.append(Conv1dBlock(in_ch, out_ch, kernel=kernel, pool=4))
            in_ch = out_ch
        self.cnn = nn.Sequential(*blocks)

        # Compute the sequence length after pooling
        # Each Conv1dBlock with pool=4 reduces length by 4×
        self._seq_len = input_len
        for _ in cnn_channels:
            self._seq_len = self._seq_len // 4

        log.debug("After CNN: seq_len=%d, channels=%d", self._seq_len, cnn_channels[-1])

        # ── Bidirectional LSTM ─────────────────────────────────────────────────
        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        lstm_out_size = lstm_hidden * 2   # bidirectional → 2×

        # ── Temporal attention ─────────────────────────────────────────────────
        self.attention = TemporalAttention(lstm_out_size)

        # ── Classification head ────────────────────────────────────────────────
        self.head = nn.Sequential(
            nn.LayerNorm(lstm_out_size),
            nn.Dropout(dropout),
            nn.Linear(lstm_out_size, 128),
            nn.GELU(),
            nn.Dropout(dropout / 2),
            nn.Linear(128, num_classes),
        )

        # Initialise weights
        self._init_weights()

    def _init_weights(self) -> None:
        for name, param in self.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "weight_ih" in name:
                nn.init.kaiming_normal_(param, mode="fan_in")
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        ----------
        x : (B, T) raw whitened strain
        return_attention : if True, also return attention weights (B, seq_len)

        Returns
        -------
        logits : (B, num_classes)
        [attention_weights : (B, seq_len)]   only if return_attention=True
        """
        B = x.size(0)

        # CNN expects (B, C, T)
        out = x.unsqueeze(1)            # (B, 1, T)
        out = self.cnn(out)             # (B, C_last, seq_len)
        out = out.permute(0, 2, 1)      # (B, seq_len, C_last)

        # LSTM
        lstm_out, _ = self.lstm(out)    # (B, seq_len, 2*H)

        # Attention pooling
        context, attn_weights = self.attention(lstm_out)   # (B, 2*H)

        logits = self.head(context)     # (B, num_classes)

        if return_attention:
            return logits, attn_weights
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        return F.softmax(self.forward(x), dim=-1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ─── Factory ──────────────────────────────────────────────────────────────────

def build_cnn_lstm(cfg: dict) -> CNNLSTM:
    """Build CNNLSTM from config dict."""
    mcfg = cfg["model"]
    dcfg = cfg["data"]

    fs = dcfg["sample_rate"]
    seg_len = dcfg["segment_length"]
    input_len = int(fs * seg_len)

    model = CNNLSTM(
        input_len=input_len,
        num_classes=mcfg["num_classes"],
        cnn_channels=mcfg.get("cnn1d_channels", [64, 128, 256]),
        lstm_hidden=mcfg.get("lstm_hidden", 256),
        lstm_layers=mcfg.get("lstm_layers", 2),
        dropout=mcfg.get("dropout", 0.3),
    )
    log.info(
        "CNNLSTM model  —  %d trainable parameters  (input_len=%d)",
        model.count_parameters(), input_len,
    )
    return model
