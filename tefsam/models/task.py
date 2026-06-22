from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
from monai.losses import DiceLoss

from .segmenter import TeFSAMSegmenter


class TeFSAM(nn.Module):
    """Training task wrapper around the released TeF-SAM network."""

    def __init__(self, cfg, prototype: nn.Module) -> None:
        super().__init__()
        self.model = TeFSAMSegmenter(cfg, prototype)
        self.mask_prompt_loss_weight = float(
            getattr(cfg, "mask_prompt_loss_weight", 0.2)
        )
        # Eq. (21): Dice + CE for the final mask, CE only for the coarse mask.
        self.dice_loss = DiceLoss(sigmoid=True)
        self.segmentation_ce_loss = nn.BCEWithLogitsLoss()
        self.auxiliary_loss = nn.BCEWithLogitsLoss()

    def forward(self, batch: Dict[str, object]) -> Dict[str, torch.Tensor]:
        outputs = self.model(
            batch["image"],
            image_emb=batch.get("image_emb"),
            image_feature=batch.get("image_feature"),
            target=batch.get("label"),
        )
        result = {
            "logits": outputs["logits"],
            "aux_logits": outputs["aux_logits"],
        }
        if "label" in batch:
            result["label"] = batch["label"]
            loss_dice = self.dice_loss(outputs["logits"], batch["label"])
            loss_ce = self.segmentation_ce_loss(
                outputs["logits"], batch["label"]
            )
            loss_seg = loss_dice + loss_ce
            loss_aux = self.auxiliary_loss(outputs["aux_logits"], batch["label"])
            result.update(
                {
                    "loss": loss_seg + self.mask_prompt_loss_weight * loss_aux,
                    "loss_seg": loss_seg.detach(),
                    "loss_dice": loss_dice.detach(),
                    "loss_ce": loss_ce.detach(),
                    "loss_aux": loss_aux.detach(),
                }
            )
        return result
