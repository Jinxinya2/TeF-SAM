from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from .config import TeFSAMConfig
from .encoders import VisionPyramidEncoder
from .fusion import MultiScaleFeatureFusion
from .sam_adapter import SAMPromptDecoder
from .semantic_memory import SemanticMemory


class TeFSAM(nn.Module):
    """
    Prototype-guided SAM segmentation model.

    Forward input:
        batch["image"]: [B, C, H, W]
        batch["image_emb"] optional: precomputed BiomedCLIP image embedding
        batch["label"] optional during training for iterative point correction
    """

    def __init__(self, cfg: TeFSAMConfig, semantic_memory: SemanticMemory) -> None:
        super().__init__()
        self.cfg = cfg
        self.vision_encoder = VisionPyramidEncoder(cfg.vision_type)
        self.semantic_memory = semantic_memory
        self.fusion = MultiScaleFeatureFusion(
            in_channels=[96, 192, 384, 768],
            prompt_dim=cfg.sam_prompt_dim,
            token_grid=56,
        )
        self.sam_decoder = SAMPromptDecoder(cfg)

    def _prepare_image(self, image: torch.Tensor) -> torch.Tensor:
        if image.shape[1] == 1:
            return repeat(image, "b 1 h w -> b 3 h w")
        return image[:, :3]

    def _extract_features(self, image: torch.Tensor):
        maps = self.vision_encoder(image)
        maps = sorted(maps, key=lambda x: x.shape[-1], reverse=True)
        if len(maps) != 4:
            raise RuntimeError(f"Expected four feature maps, got {len(maps)}.")
        _, visual_tokens = self.fusion(maps)
        deepest_tokens = rearrange(maps[-1], "b c h w -> b (h w) c")
        return visual_tokens, deepest_tokens

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        image = batch["image"]
        sam_image = batch.get("sam_image", image)
        image_rgb = self._prepare_image(image)
        target_size = tuple(image.shape[-2:])

        visual_tokens, deepest_tokens = self._extract_features(image_rgb)
        semantic_tokens = self.semantic_memory.query(
            image_rgb,
            image_emb=batch.get("image_emb"),
            image_tokens=deepest_tokens,
        )

        raw_logits, mask_prompt_logits, point_coords, point_labels = self.sam_decoder(
            image=sam_image,
            visual_tokens=visual_tokens,
            semantic_tokens=semantic_tokens,
            target_size=target_size,
            mask_gt=batch.get("label"),
        )

        prob = torch.sigmoid(raw_logits)
        mask_prompt = torch.sigmoid(F.interpolate(mask_prompt_logits, size=target_size, mode="bilinear", align_corners=False))
        return {
            "logits": prob,
            "raw_logits": raw_logits,
            "mask_prompt_logits": mask_prompt,
            "point_coords": point_coords,
            "point_labels": point_labels,
            "semantic_tokens": semantic_tokens,
        }

    def trainable_parameter_names(self):
        return [name for name, param in self.named_parameters() if param.requires_grad]
