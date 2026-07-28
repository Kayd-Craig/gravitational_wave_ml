"""
trainer.py
==========
Training loop for the deep learning models (SpectrogramCNN and CNNLSTM).

Features
--------
  * Class-weighted CrossEntropyLoss
  * AdamW optimiser with cosine / step / constant LR schedule
  * Early stopping on validation loss
  * Best-checkpoint saving (val F1)
  * TensorBoard-compatible CSV logging
  * Mixed-precision training via torch.amp (optional)
"""

from __future__ import annotations

import csv
import logging
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

log = logging.getLogger(__name__)


# ─── Helpers ──────────────────────────────────────────────────────────────────

class EarlyStopping:
    """Stop training when monitored metric stops improving."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best: Optional[float] = None
        self.wait = 0
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        if self.best is None or val_loss < self.best - self.min_delta:
            self.best = val_loss
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                self.should_stop = True
        return self.should_stop


class CSVLogger:
    """Append epoch metrics to a CSV file."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._init = False

    def log(self, metrics: Dict[str, float]) -> None:
        mode = "a" if self._init else "w"
        with open(self.path, mode, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
            if not self._init:
                writer.writeheader()
                self._init = True
            writer.writerow(metrics)


# ─── Core trainer ─────────────────────────────────────────────────────────────

class Trainer:
    """Orchestrates training and validation for a PyTorch model.

    Parameters
    ----------
    model       : nn.Module with a forward(x) → logits interface
    cfg         : full config dict
    device      : torch device (cpu / cuda / mps)
    class_weights : (num_classes,) tensor for loss weighting
    """

    def __init__(
        self,
        model: nn.Module,
        cfg: dict,
        device: torch.device,
        class_weights: Optional[torch.Tensor] = None,
    ) -> None:
        self.model = model.to(device)
        self.cfg = cfg
        self.device = device

        tcfg = cfg["training"]
        self.n_epochs = tcfg["num_epochs"]
        self.lr = tcfg["learning_rate"]
        self.weight_decay = tcfg["weight_decay"]
        self.patience = tcfg["patience"]
        self.min_delta = tcfg.get("min_delta", 1e-4)
        self.ckpt_dir = Path(tcfg["checkpoint_dir"])
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        # ── Loss ──────────────────────────────────────────────────────────────
        if class_weights is not None:
            cw = class_weights.to(device)
        else:
            cw = None
        self.criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=0.05)

        # ── Optimiser ─────────────────────────────────────────────────────────
        self.optimiser = AdamW(
            model.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay,
        )

        # ── Scheduler ─────────────────────────────────────────────────────────
        sched_type = tcfg.get("scheduler", "cosine")
        if sched_type == "cosine":
            self.scheduler = CosineAnnealingLR(
                self.optimiser, T_max=self.n_epochs, eta_min=self.lr * 0.01
            )
        elif sched_type == "step":
            self.scheduler = StepLR(self.optimiser, step_size=15, gamma=0.5)
        else:
            self.scheduler = None

        # ── Misc ──────────────────────────────────────────────────────────────
        self.early_stopping = EarlyStopping(self.patience, self.min_delta)
        self.scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

        log_dir = Path(tcfg.get("log_dir", "logs"))
        log_dir.mkdir(parents=True, exist_ok=True)
        self.csv_logger = CSVLogger(log_dir / "training_log.csv")

        self.best_val_f1 = 0.0
        self.history: list[dict] = []

    # ── Single epoch ──────────────────────────────────────────────────────────

    def _train_epoch(self, loader: DataLoader) -> Tuple[float, float]:
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0

        for x, y in tqdm(loader, desc="  Train", leave=False, ncols=80):
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            self.optimiser.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
                logits = self.model(x)
                loss = self.criterion(logits, y)

            self.scaler.scale(loss).backward()
            # Gradient clipping to stabilise LSTM training
            self.scaler.unscale_(self.optimiser)
            nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.scaler.step(self.optimiser)
            self.scaler.update()

            total_loss += loss.item() * y.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

        return total_loss / max(total, 1), correct / max(total, 1)

    @torch.no_grad()
    def _val_epoch(self, loader: DataLoader) -> Tuple[float, float, float]:
        """Returns (loss, accuracy, macro-F1)."""
        from sklearn.metrics import f1_score

        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0
        all_preds, all_targets = [], []

        for x, y in tqdm(loader, desc="  Val  ", leave=False, ncols=80):
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)

            with torch.cuda.amp.autocast(enabled=self.device.type == "cuda"):
                logits = self.model(x)
                loss = self.criterion(logits, y)

            total_loss += loss.item() * y.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(y.cpu().numpy())

        f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
        return total_loss / max(total, 1), correct / max(total, 1), float(f1)

    # ── Checkpoint ────────────────────────────────────────────────────────────

    def _save_checkpoint(self, epoch: int, val_f1: float, tag: str = "best") -> None:
        ckpt = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimiser_state_dict": self.optimiser.state_dict(),
            "val_f1": val_f1,
            "cfg": self.cfg,
        }
        path = self.ckpt_dir / f"{tag}.pt"
        torch.save(ckpt, path)
        log.info("  Checkpoint saved: %s  (val_F1=%.4f)", path, val_f1)

    def load_best(self) -> None:
        path = self.ckpt_dir / "best.pt"
        if path.exists():
            ckpt = torch.load(path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            log.info("Loaded best checkpoint (epoch %d, val_F1=%.4f)",
                     ckpt["epoch"], ckpt["val_f1"])

    # ── Main train loop ───────────────────────────────────────────────────────

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> list[dict]:
        """Run the full training loop.

        Returns
        -------
        history : list of per-epoch metric dicts
        """
        log.info("Starting training — device: %s", self.device)
        log.info("  Epochs: %d  |  LR: %.1e  |  WD: %.1e",
                 self.n_epochs, self.lr, self.weight_decay)

        for epoch in range(1, self.n_epochs + 1):
            t0 = time.time()

            train_loss, train_acc = self._train_epoch(train_loader)
            val_loss, val_acc, val_f1 = self._val_epoch(val_loader)

            if self.scheduler is not None:
                self.scheduler.step()

            elapsed = time.time() - t0
            current_lr = self.optimiser.param_groups[0]["lr"]

            metrics = {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "train_acc": round(train_acc, 6),
                "val_loss": round(val_loss, 6),
                "val_acc": round(val_acc, 6),
                "val_f1": round(val_f1, 6),
                "lr": current_lr,
                "elapsed_s": round(elapsed, 1),
            }
            self.history.append(metrics)
            self.csv_logger.log(metrics)

            log.info(
                "Epoch %3d/%d | "
                "train_loss=%.4f acc=%.4f | "
                "val_loss=%.4f acc=%.4f F1=%.4f | "
                "lr=%.2e | %.1fs",
                epoch, self.n_epochs,
                train_loss, train_acc,
                val_loss, val_acc, val_f1,
                current_lr, elapsed,
            )

            # Save best
            if val_f1 > self.best_val_f1:
                self.best_val_f1 = val_f1
                self._save_checkpoint(epoch, val_f1, tag="best")

            # Save periodic checkpoint every 10 epochs
            if epoch % 10 == 0:
                self._save_checkpoint(epoch, val_f1, tag=f"epoch_{epoch:03d}")

            # Early stopping
            if self.early_stopping(val_loss):
                log.info("Early stopping triggered at epoch %d", epoch)
                break

        log.info("Training complete. Best val F1: %.4f", self.best_val_f1)
        return self.history


# ─── Convenience function ─────────────────────────────────────────────────────

def get_device() -> torch.device:
    """Return the best available device (CUDA > MPS > CPU)."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        log.info("Using CUDA: %s", torch.cuda.get_device_name(0))
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        log.info("Using Apple MPS")
    else:
        device = torch.device("cpu")
        log.info("Using CPU")
    return device
