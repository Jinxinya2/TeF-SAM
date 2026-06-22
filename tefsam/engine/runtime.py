"""Shared command-line runtime for public training and evaluation."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from tefsam.config import Config, apply_overrides, load_config
from tefsam.data import MedicalDataLoader
from tefsam.data.dataset import dataset_from_config, deterministic_subset
from tefsam.engine.checkpoint import (
    load_segmentation_checkpoint,
    mark_only_lora_trainable,
)
from tefsam.engine.trainer import Trainer
from tefsam.models import FrozenPRISMRuntime, TeFSAM
from tefsam.utils import set_seed


def _parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", required=True, help="Experiment YAML file")
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Override a flattened configuration value",
    )
    return parser


def _config(arguments: argparse.Namespace) -> Config:
    return apply_overrides(load_config(arguments.config), arguments.set)


def _loader(
    cfg: Config,
    *,
    split: str,
    augment: bool,
    subset_ratio: Optional[float] = None,
    shuffle: bool = False,
) -> MedicalDataLoader:
    dataset = dataset_from_config(
        cfg,
        split=split,
        augment=augment,
    )
    if subset_ratio is not None:
        dataset = deterministic_subset(dataset, subset_ratio, int(cfg.seed))
    return MedicalDataLoader(
        dataset,
        batch_size=int(cfg.batch_size),
        shuffle=shuffle,
        num_workers=int(getattr(cfg, "num_workers", 4)),
        pin_memory=torch.cuda.is_available(),
    )


def _load_repository(cfg: Config) -> FrozenPRISMRuntime:
    checkpoint = getattr(cfg, "biomedclip_checkpoint", None)
    if checkpoint in (None, "", "null", "None"):
        checkpoint = None
    return FrozenPRISMRuntime.load(
        cfg.repository_artifact,
        manifest_path=cfg.repository_manifest,
        image_encoder_checkpoint=checkpoint,
        device=cfg.device,
        local_files_only=bool(
            getattr(cfg, "biomedclip_local_files_only", False)
        ),
    )


def _selection_split(cfg: Config) -> str:
    validation_split = str(getattr(cfg, "validation_split", "val"))
    test_split = str(getattr(cfg, "test_split", "test"))
    if validation_split == test_split:
        raise ValueError(
            "validation_split and test_split must differ; the held-out test "
            "split cannot select checkpoints"
        )
    return validation_split


def _build_model(cfg: Config) -> TeFSAM:
    repository = _load_repository(cfg)
    model = TeFSAM(cfg, repository).to(cfg.device)
    initialization = getattr(cfg, "checkpoint_path", None)
    if initialization not in (None, "", "None", "null"):
        load_segmentation_checkpoint(
            model,
            initialization,
            map_location="cpu",
            allowed_unexpected_prefixes=(
                "model.decoder1.",
                "model.out.",
                "model.decoder16.",
                "model.decoder8.",
                "model.decoder4.",
            ),
        )
        print(f"Loaded segmentation initialization: {initialization}")
    return model


def train_main() -> None:
    arguments = _parser(
        "Train TeF-SAM with a released frozen PRISM artifact"
    ).parse_args()
    cfg = _config(arguments)
    set_seed(int(cfg.seed))
    _selection_split(cfg)
    model = _build_model(cfg)

    if bool(getattr(cfg, "sam_lora_only", False)):
        mark_only_lora_trainable(model)
    parameters = [
        parameter for parameter in model.parameters() if parameter.requires_grad
    ]
    if not parameters:
        raise RuntimeError("No trainable segmentation parameters were selected")
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameters = sum(parameter.numel() for parameter in parameters)
    print(
        f"Parameters: total={total_parameters:,}, "
        f"trainable={trainable_parameters:,}"
    )

    train_loader = _loader(
        cfg,
        split="train",
        augment=True,
        subset_ratio=float(cfg.training_ratio),
        shuffle=True,
    )
    validation_ratio = getattr(cfg, "validation_ratio", None)
    val_loader = _loader(
        cfg,
        split=_selection_split(cfg),
        augment=False,
        subset_ratio=(
            float(validation_ratio) if validation_ratio is not None else None
        ),
    )
    optimizer = AdamW(
        parameters,
        lr=float(cfg.lr),
        weight_decay=float(getattr(cfg, "segmentation_weight_decay", 0.0)),
    )
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=int(getattr(cfg, "scheduler_t_max", cfg.max_epochs)),
        eta_min=float(getattr(cfg, "scheduler_min_lr", 1e-6)),
    )
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        val_loader=val_loader,
        device=str(cfg.device),
        epochs=int(cfg.max_epochs),
        patience=int(cfg.patience),
        min_epochs=int(getattr(cfg, "min_epochs", 0)),
        checkpoint_path=str(cfg.output_checkpoint),
        repository_path=str(model.model.prototype.artifact_path),
        cfg=cfg,
        use_amp=bool(getattr(cfg, "mixed_precision", False)),
    )
    trainer.fit()


def evaluate_main() -> None:
    parser = _parser("Evaluate a trained TeF-SAM segmenter")
    parser.add_argument("--checkpoint", required=True)
    arguments = parser.parse_args()
    cfg = _config(arguments)
    set_seed(int(cfg.seed))
    model = _build_model(cfg)
    load_segmentation_checkpoint(
        model, arguments.checkpoint, map_location="cpu"
    )
    loader = _loader(
        cfg,
        split=str(getattr(cfg, "test_split", "test")),
        augment=False,
    )
    trainer = Trainer(
        model=model,
        optimizer=None,
        scheduler=None,
        train_loader=None,
        val_loader=loader,
        device=str(cfg.device),
        epochs=0,
        patience=0,
        min_epochs=0,
        checkpoint_path="",
        repository_path=str(model.model.prototype.artifact_path),
        cfg=cfg,
        use_amp=False,
    )
    metrics = trainer.evaluate(loader)
    print(
        f"Test: loss={metrics['loss']:.4f}, accuracy={metrics['accuracy']:.4f}, "
        f"dice={metrics['dice']:.4f}, miou={metrics['miou']:.4f}"
    )
