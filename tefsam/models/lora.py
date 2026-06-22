"""LoRA adapters for the final SAM TwoWayTransformer attention block."""

from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Add a trainable low-rank residual branch to a frozen linear layer."""

    def __init__(
        self,
        base: nn.Linear,
        rank: int,
        alpha: float,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if not isinstance(base, nn.Linear):
            raise TypeError("LoRALinear can only wrap nn.Linear")
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("LoRA dropout must be in [0, 1)")

        self.base = base
        self.rank = int(rank)
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.lora_A = nn.Linear(base.in_features, self.rank, bias=False)
        self.lora_B = nn.Linear(self.rank, base.out_features, bias=False)

        for parameter in self.base.parameters():
            parameter.requires_grad = False
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = self.lora_B(self.lora_A(self.dropout(inputs)))
        return self.base(inputs) + residual * self.scaling


def _wrap_linear(
    module: nn.Module,
    attribute: str,
    rank: int,
    alpha: float,
    dropout: float,
) -> None:
    linear = getattr(module, attribute)
    if isinstance(linear, LoRALinear):
        raise ValueError(f"{attribute} already has a LoRA adapter")
    setattr(module, attribute, LoRALinear(linear, rank, alpha, dropout))


def inject_lora_into_last_two_way_block(
    transformer: nn.Module,
    rank: int,
    alpha: float,
    dropout: float = 0.0,
) -> List[str]:
    """Inject LoRA into q/v projections and the two MLP linears of the last block."""

    layers = getattr(transformer, "layers", None)
    if layers is None or len(layers) == 0:
        raise ValueError("TwoWayTransformer must contain at least one attention block")

    block_index = len(layers) - 1
    block = layers[block_index]
    targets: List[str] = []
    attention_names = (
        "self_attn",
        "cross_attn_token_to_image",
        "cross_attn_image_to_token",
    )
    for attention_name in attention_names:
        attention = getattr(block, attention_name)
        for projection_name in ("q_proj", "v_proj"):
            _wrap_linear(attention, projection_name, rank, alpha, dropout)
            targets.append(
                f"layers.{block_index}.{attention_name}.{projection_name}"
            )

    for linear_name in ("lin1", "lin2"):
        _wrap_linear(block.mlp, linear_name, rank, alpha, dropout)
        targets.append(f"layers.{block_index}.mlp.{linear_name}")

    return targets


def count_lora_parameters(module: nn.Module) -> int:
    """Return the number of trainable parameters introduced by LoRA branches."""

    return sum(
        parameter.numel()
        for submodule in module.modules()
        if isinstance(submodule, LoRALinear)
        for parameter in (submodule.lora_A.weight, submodule.lora_B.weight)
    )
