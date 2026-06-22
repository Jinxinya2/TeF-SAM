from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import torch
import torch.nn as nn

from tefsam.models.lora import LoRALinear


REPOSITORY_PREFIX = "model.prototype."
CHECKPOINT_FORMAT = "tefsam-segmentation-v1"
PUBLIC_REPOSITORY_FIELDS = (
    "artifact_id",
    "runtime_format",
    "query_schema",
)


def _repository_manifest(model: nn.Module) -> Mapping:
    network = getattr(model, "model", None)
    repository = getattr(network, "prototype", None)
    manifest = getattr(repository, "manifest", None)
    return manifest if isinstance(manifest, Mapping) else {}


def _public_repository_summary(model: nn.Module) -> Dict[str, object]:
    manifest = _repository_manifest(model)
    return {
        field: manifest[field]
        for field in PUBLIC_REPOSITORY_FIELDS
        if manifest.get(field) is not None
    }


def segmentation_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {
        key: value
        for key, value in model.state_dict().items()
        if not key.startswith(REPOSITORY_PREFIX)
    }


def save_training_checkpoint(
    path: str | Path,
    *,
    model: nn.Module,
    optimizer=None,
    scheduler=None,
    epoch: int,
    metrics: Dict[str, float],
    cfg=None,
    repository_path: Optional[str] = None,
) -> Path:
    del cfg, repository_path
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "checkpoint_format": CHECKPOINT_FORMAT,
        "state_dict": segmentation_state_dict(model),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": int(epoch),
        "metrics": dict(metrics),
        "repository": _public_repository_summary(model),
    }
    temporary = output.with_suffix(output.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(output)
    return output


def _remap_base_linears_for_lora(
    model: nn.Module, state_dict: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    remapped = dict(state_dict)
    for module_name, module in model.named_modules():
        if not isinstance(module, LoRALinear):
            continue
        for parameter_name in ("weight", "bias"):
            if getattr(module.base, parameter_name) is None:
                continue
            old_key = f"{module_name}.{parameter_name}"
            new_key = f"{module_name}.base.{parameter_name}"
            if old_key in remapped and new_key not in remapped:
                remapped[new_key] = remapped.pop(old_key)
    return remapped


def load_segmentation_checkpoint(
    model: nn.Module,
    path: str | Path,
    *,
    map_location="cpu",
    allowed_missing_prefixes: Sequence[str] = (REPOSITORY_PREFIX,),
    allowed_unexpected_prefixes: Sequence[str] = (),
):
    checkpoint = torch.load(Path(path), map_location=map_location)
    state_dict = checkpoint.get("state_dict", checkpoint)
    state_dict = {
        key: value
        for key, value in state_dict.items()
        if not key.startswith(REPOSITORY_PREFIX)
    }
    state_dict = _remap_base_linears_for_lora(model, state_dict)
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = [
        key
        for key in incompatible.missing_keys
        if not any(key.startswith(prefix) for prefix in allowed_missing_prefixes)
        and not key.endswith(("lora_A.weight", "lora_B.weight"))
    ]
    unexpected = [
        key
        for key in incompatible.unexpected_keys
        if not any(key.startswith(prefix) for prefix in allowed_unexpected_prefixes)
    ]
    if missing or unexpected:
        raise RuntimeError(
            f"Incompatible segmentation checkpoint: missing={missing}, "
            f"unexpected={unexpected}"
        )
    return checkpoint


def mark_only_lora_trainable(model: nn.Module) -> Iterable[str]:
    for parameter in model.parameters():
        parameter.requires_grad = False
    names = []
    for module_name, module in model.named_modules():
        if not isinstance(module, LoRALinear):
            continue
        module.lora_A.weight.requires_grad = True
        module.lora_B.weight.requires_grad = True
        names.extend(
            [f"{module_name}.lora_A.weight", f"{module_name}.lora_B.weight"]
        )
    if not names:
        raise ValueError("LoRA-only training requested but no LoRA modules exist")
    return names
