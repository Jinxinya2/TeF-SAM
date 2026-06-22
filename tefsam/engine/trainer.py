from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
from tqdm import tqdm

from .checkpoint import save_training_checkpoint


class BinaryMetricAccumulator:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.tp = self.fp = self.fn = self.correct = self.total = 0.0

    @torch.no_grad()
    def update(self, probabilities: torch.Tensor, target: torch.Tensor) -> None:
        prediction = probabilities >= 0.5
        truth = target > 0.5
        self.tp += float((prediction & truth).sum().item())
        self.fp += float((prediction & ~truth).sum().item())
        self.fn += float((~prediction & truth).sum().item())
        self.correct += float((prediction == truth).sum().item())
        self.total += float(truth.numel())

    def compute(self) -> Dict[str, float]:
        eps = 1e-8
        return {
            "accuracy": self.correct / max(self.total, eps),
            "dice": 2.0 * self.tp / max(2.0 * self.tp + self.fp + self.fn, eps),
            "miou": self.tp / max(self.tp + self.fp + self.fn, eps),
        }


class Trainer:
    def __init__(
        self,
        *,
        model,
        optimizer,
        scheduler,
        train_loader,
        val_loader,
        device: str,
        epochs: int,
        patience: int,
        min_epochs: int,
        checkpoint_path: str,
        repository_path: str,
        cfg,
        use_amp: bool = False,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.epochs = epochs
        self.patience = patience
        self.min_epochs = min_epochs
        self.checkpoint_path = checkpoint_path
        self.repository_path = repository_path
        self.cfg = cfg
        self.use_amp = bool(use_amp and torch.cuda.is_available())
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

    def _move(self, value):
        if isinstance(value, torch.Tensor):
            return value.to(self.device)
        if isinstance(value, dict):
            return {key: self._move(item) for key, item in value.items()}
        return value

    def train_epoch(self) -> float:
        self.model.train()
        total = 0.0
        for batch in tqdm(self.train_loader, desc="Training", leave=False):
            batch = self._move(batch)
            self.optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                output = self.model(batch)
                loss = output["loss"]
            if self.use_amp:
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                self.optimizer.step()
            total += float(loss.detach().item())
        return total / max(len(self.train_loader), 1)

    @torch.no_grad()
    def evaluate(self, loader=None) -> Dict[str, float]:
        loader = loader or self.val_loader
        self.model.eval()
        accumulator = BinaryMetricAccumulator()
        loss_total = 0.0
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            batch = self._move(batch)
            with torch.cuda.amp.autocast(enabled=False):
                output = self.model(batch)
            loss_total += float(output["loss"].item())
            accumulator.update(output["logits"].sigmoid(), output["label"])
        metrics = accumulator.compute()
        metrics["loss"] = loss_total / max(len(loader), 1)
        return metrics

    def fit(self) -> Dict[str, float]:
        best_dice = float("-inf")
        best_metrics: Dict[str, float] = {}
        stale_epochs = 0
        for epoch in range(self.epochs):
            train_loss = self.train_epoch()
            metrics = self.evaluate()
            print(
                f"Epoch {epoch + 1}/{self.epochs}: train_loss={train_loss:.4f}, "
                f"val_loss={metrics['loss']:.4f}, dice={metrics['dice']:.4f}, "
                f"miou={metrics['miou']:.4f}"
            )
            if metrics["dice"] > best_dice:
                best_dice = metrics["dice"]
                best_metrics = dict(metrics)
                stale_epochs = 0
                save_training_checkpoint(
                    self.checkpoint_path,
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    metrics=metrics,
                    cfg=self.cfg,
                    repository_path=self.repository_path,
                )
            else:
                stale_epochs += 1
            if self.scheduler is not None:
                self.scheduler.step()
            if epoch + 1 >= self.min_epochs and stale_epochs >= self.patience:
                break
        return best_metrics
