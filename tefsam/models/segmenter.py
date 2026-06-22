from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
from einops import rearrange, repeat

from .layers import GuidedApproximation
from .frozen_prism import validate_repository_contract
from .multiscale_fusion import MultiScaleFeatureFusion
from .networks import VisionModel
from .sam_adapter import MedicalSAMAdapter


class TeFSAMSegmenter(nn.Module):
    """Image-only segmenter guided by a frozen PRISM semantic repository."""

    def __init__(self, cfg, prototype: nn.Module) -> None:
        super().__init__()
        validate_repository_contract(prototype, cfg)

        self.prototype = prototype
        self.agg = str(getattr(cfg, "agg", "memory"))
        self.use_approx1 = bool(getattr(cfg, "use_approx1", False))
        self.vision_encoder = VisionModel(
            cfg.vision_type,
            local_files_only=bool(getattr(cfg, "vision_local_files_only", False)),
            revision=getattr(cfg, "vision_revision", None),
        )

        feature_dims = list(
            getattr(cfg, "vision_feature_dims", [768, 384, 192, 96])
        )
        if len(feature_dims) != 4:
            raise ValueError("vision_feature_dims must contain four channel values")
        self.spatial_dims = list(
            getattr(cfg, "vision_spatial_dims", [7, 14, 28, 56])
        )

        self.approx1: Optional[nn.Module] = None
        if self.agg == "attention":
            self.approx1 = GuidedApproximation(
                cfg.text_dim,
                cfg.num_candidate,
                feature_dims[0],
                feature_dims[1],
            )
        elif self.agg != "memory":
            raise ValueError(
                "The released TeF-SAM segmenter supports PRISM memory or "
                "attention aggregation"
            )

        self.fusion = MultiScaleFeatureFusion(
            in_channels=list(reversed(feature_dims)),
            prompt_dim=cfg.sam_prompt_dim,
            token_grid=int(getattr(cfg, "fusion_token_grid", self.spatial_dims[-1])),
        )
        self.sam_adapter = MedicalSAMAdapter(
            image_dim=cfg.sam_prompt_dim,
            text_dim=cfg.text_dim,
            prompt_dim=cfg.sam_prompt_dim,
            image_embedding_size=cfg.sam_image_embedding,
            input_image_size=cfg.image_size,
            point_topk=int(getattr(cfg, "point_topk", 8)),
            point_threshold=float(getattr(cfg, "point_threshold", 0.9)),
            sam_checkpoint=getattr(cfg, "sam_checkpoint", None),
            sam_model_type=getattr(cfg, "sam_model_type", "vit_b"),
            lora_enabled=bool(getattr(cfg, "sam_lora_enabled", False)),
            lora_rank=int(getattr(cfg, "sam_lora_rank", 4)),
            lora_alpha=float(getattr(cfg, "sam_lora_alpha", 8.0)),
            lora_dropout=float(getattr(cfg, "sam_lora_dropout", 0.0)),
            correction_points_per_class=int(
                getattr(cfg, "correction_points_per_class", 2)
            ),
        )

    def _feature_maps(self, image: torch.Tensor):
        hidden_states = self.vision_encoder(image)
        if hidden_states[0].dim() == 4:
            feature_maps = list(hidden_states[1:])
            if len(feature_maps) != 4:
                feature_maps = feature_maps[-4:]
            feature_tokens = [
                rearrange(feature, "b c h w -> b (h w) c")
                for feature in feature_maps
            ]
            return feature_maps, feature_tokens

        feature_maps = []
        spatial_sizes = list(reversed(self.spatial_dims))
        for feature, spatial_size in zip(hidden_states[:4], spatial_sizes):
            feature_maps.append(
                rearrange(
                    feature,
                    "b (h w) c -> b c h w",
                    h=spatial_size,
                    w=spatial_size,
                )
            )
        return feature_maps, list(hidden_states[:4])

    def _refine_semantics(
        self,
        semantic_tokens: torch.Tensor,
        feature_tokens,
    ) -> torch.Tensor:
        if self.use_approx1 and self.approx1 is not None:
            return self.approx1(
                semantic_tokens,
                feature_tokens[3],
                feature_tokens[2],
            )
        return semantic_tokens

    def forward(
        self,
        image: torch.Tensor,
        *,
        image_emb: Optional[torch.Tensor] = None,
        image_feature: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if image.shape[1] == 1:
            image = repeat(image, "b 1 h w -> b 3 h w")

        feature_maps, feature_tokens = self._feature_maps(image)
        _, visual_tokens = self.fusion(feature_maps)
        semantic_tokens = self.prototype.query(
            image,
            image_emb=image_emb,
            image_feature=image_feature,
        )
        semantic_tokens = self._refine_semantics(semantic_tokens, feature_tokens)

        segmentation_logits, auxiliary_logits = self.sam_adapter(
            image_tokens=visual_tokens,
            text_tokens=semantic_tokens,
            target_size=tuple(image.shape[-2:]),
            target=target,
        )
        return {
            "logits": segmentation_logits,
            "aux_logits": auxiliary_logits,
        }
