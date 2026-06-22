from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from segment_anything import sam_model_registry
from segment_anything.modeling import MaskDecoder, PromptEncoder, TwoWayTransformer

from .layers import CrossAttentionLayer, SelfAttentionLayer
from .lora import inject_lora_into_last_two_way_block


class MaskPromptGenerator(nn.Module):
    """
    Generates mask prompts with self-attention and cross-attention to text features.
    """

    def __init__(self, image_dim: int, text_dim: int, prompt_dim: int):
        super().__init__()
        self.image_proj = nn.Linear(image_dim, prompt_dim)
        self.text_proj = nn.Linear(text_dim, prompt_dim)
        self.self_attn = SelfAttentionLayer(prompt_dim)
        self.cross_attn = CrossAttentionLayer(prompt_dim, output_text_len=1)
        self.mask_head = nn.Conv2d(prompt_dim, 1, kernel_size=1)

    def forward(
        self, image_map: torch.Tensor, text_feat: torch.Tensor, image_size
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            image_map (Tensor): [B, C, H, W] fused image features.
            text_feat (Tensor): [B, L, D] text features.
            image_size (tuple): (H, W) output mask size.
        Returns:
            Tuple[Tensor, Tensor]:
                - [B, C, H, W] prompt-aligned image features.
                - [B, 1, H, W] mask prompt logits.
        """
        b, c, h, w = image_map.shape
        image_tokens = rearrange(image_map, "b c h w -> b (h w) c")
        image_tokens = self.image_proj(image_tokens)

        if text_feat.dim() == 2:
            text_feat = text_feat.unsqueeze(1)
        text_tokens = self.text_proj(text_feat)

        image_tokens = self.self_attn(image_tokens)
        image_tokens = self.cross_attn(image_tokens, text_tokens)

        image_map = rearrange(image_tokens, "b (h w) c -> b c h w", h=h, w=w)

        mask_logits = self.mask_head(image_map)
        mask_logits = F.interpolate(
            mask_logits,
            size=image_size,
            mode="bilinear",
            align_corners=False,
        )
        return image_map, mask_logits


class MedicalSAMAdapter(nn.Module):
    """
    Medical SAM Adapter inspired by
    \"Medical SAM Adapter: Adapting Segment Anything Model for Medical Image
    Segmentation\".

    Key ideas:
    - Keep the heavy SAM prompt encoder and mask decoder frozen.
    - Learn lightweight adapters that fuse image tokens with text-guided cues to form
      dense mask prompts and sparse point prompts.
    - Inject text-aware sparse prompts into the SAM decoder for
      language-conditioned segmentation.
    """

    def __init__(
        self,
        image_dim: int,
        text_dim: int,
        prompt_dim: int,
        image_embedding_size: int,
        input_image_size,
        point_topk: int = 3,
        point_threshold: float = 0.9,
        num_heads: int = 4,
        sam_checkpoint: Optional[str] = None,
        sam_model_type: str = "vit_b",
        lora_enabled: bool = False,
        lora_rank: int = 4,
        lora_alpha: float = 8.0,
        lora_dropout: float = 0.0,
        correction_points_per_class: int = 2,
    ):
        super().__init__()
        self.image_embedding_size = image_embedding_size
        self.prompt_dim = prompt_dim
        self.input_image_size = input_image_size
        self.point_topk = point_topk
        self.point_threshold = point_threshold
        self.correction_points_per_class = int(correction_points_per_class)
        if self.point_topk <= 0:
            raise ValueError("point_topk must be positive")
        if self.correction_points_per_class < 0:
            raise ValueError("correction_points_per_class must be non-negative")

        # Initialize or load SAM modules and keep them frozen
        if sam_checkpoint:
            sam_model = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint)
            self.prompt_encoder = sam_model.prompt_encoder
            self.mask_decoder = sam_model.mask_decoder
        else:
            self.prompt_encoder = PromptEncoder(
                embed_dim=prompt_dim,
                image_embedding_size=(image_embedding_size, image_embedding_size),
                input_image_size=input_image_size,
                mask_in_chans=16,
            )
            self.mask_decoder = MaskDecoder(
                transformer_dim=prompt_dim,
                transformer=TwoWayTransformer(
                    depth=2,
                    embedding_dim=prompt_dim,
                    mlp_dim=2048,
                    num_heads=num_heads,
                ),
                num_multimask_outputs=3,
            )

        for module in [self.prompt_encoder, self.mask_decoder]:
            for param in module.parameters():
                param.requires_grad = False

        self.lora_target_names = []
        if lora_enabled:
            self.lora_target_names = inject_lora_into_last_two_way_block(
                self.mask_decoder.transformer,
                rank=lora_rank,
                alpha=lora_alpha,
                dropout=lora_dropout,
            )

        # Lightweight convolutional adapter used by Medical SAM Adapter.
        self.image_adapter = nn.Sequential(
            nn.Conv2d(image_dim, prompt_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(
                prompt_dim,
                prompt_dim,
                kernel_size=3,
                padding=1,
                groups=prompt_dim,
            ),
            nn.GELU(),
            nn.Conv2d(prompt_dim, prompt_dim, kernel_size=1),
        )

        # Text-guided fusion to derive prompts
        self.mask_prompt_generator = MaskPromptGenerator(
            prompt_dim, text_dim, prompt_dim
        )

    @staticmethod
    def _diverse_indices(
        scores: torch.Tensor,
        candidate_mask: torch.Tensor,
        count: int,
        *,
        repeat_last: bool,
    ) -> torch.Tensor:
        candidates = candidate_mask.flatten().nonzero(as_tuple=False).flatten()
        if candidates.numel() == 0 or count <= 0:
            return torch.empty(0, dtype=torch.long, device=scores.device)

        height, width = scores.shape
        candidate_scores = scores.flatten()[candidates]
        first = int(candidate_scores.argmax().item())
        selected_positions = [first]
        limit = min(count, int(candidates.numel()))
        while len(selected_positions) < limit:
            selected_indices = candidates[
                torch.as_tensor(selected_positions, device=scores.device)
            ]
            selected_coords = torch.stack(
                [
                    torch.div(selected_indices, width, rounding_mode="floor"),
                    torch.remainder(selected_indices, width),
                ],
                dim=1,
            ).float()
            candidate_coords = torch.stack(
                [
                    torch.div(candidates, width, rounding_mode="floor"),
                    torch.remainder(candidates, width),
                ],
                dim=1,
            ).float()
            min_distance = torch.cdist(candidate_coords, selected_coords).min(
                dim=1
            ).values
            min_distance[
                torch.as_tensor(selected_positions, device=scores.device)
            ] = -torch.inf
            selected_positions.append(int(min_distance.argmax().item()))

        selected = candidates[
            torch.as_tensor(selected_positions, device=scores.device)
        ]
        if repeat_last and selected.numel() < count:
            selected = torch.cat(
                [selected, selected[-1:].expand(count - selected.numel())]
            )
        return selected

    def _indices_to_input_coords(
        self,
        indices: torch.Tensor,
        height: int,
        width: int,
    ) -> torch.Tensor:
        if indices.numel() == 0:
            return torch.empty(0, 2, device=indices.device, dtype=torch.float32)
        ys = torch.div(indices, width, rounding_mode="floor").float()
        xs = torch.remainder(indices, width).float()
        prompt_encoder_size = getattr(
            getattr(self, "prompt_encoder", None),
            "input_image_size",
            self.input_image_size,
        )
        input_height, input_width = prompt_encoder_size
        xs = (xs + 0.5) * (float(input_width) / width) - 0.5
        ys = (ys + 0.5) * (float(input_height) / height) - 0.5
        return torch.stack([xs, ys], dim=-1)

    def _initial_points_from_coarse_mask(
        self, coarse_mask_logits: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        probabilities = coarse_mask_logits[:, 0].sigmoid()
        batch_coords = []
        for scores in probabilities:
            candidates = scores >= self.point_threshold
            required = min(self.point_topk, scores.numel())
            if int(candidates.sum().item()) < required:
                top_indices = scores.flatten().topk(required).indices
                candidates = torch.zeros_like(candidates, dtype=torch.bool)
                candidates.flatten()[top_indices] = True
            indices = self._diverse_indices(
                scores,
                candidates,
                self.point_topk,
                repeat_last=False,
            )
            batch_coords.append(
                self._indices_to_input_coords(indices, *scores.shape)
            )
        coords = torch.stack(batch_coords, dim=0)
        labels = torch.ones(
            coords.shape[:2], device=coords.device, dtype=torch.long
        )
        return coords, labels

    def _correction_points(
        self,
        prediction_logits: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if target.ndim != 4 or target.size(1) != 1:
            raise ValueError("target must have shape (B, 1, H, W)")
        if prediction_logits.shape[-2:] != target.shape[-2:]:
            prediction_logits = F.interpolate(
                prediction_logits,
                size=target.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )

        probabilities = prediction_logits[:, 0].sigmoid()
        truth = target[:, 0] > 0.5
        count = self.correction_points_per_class
        all_coords = []
        all_labels = []
        for scores, sample_truth in zip(probabilities, truth):
            false_negative = sample_truth & (scores < 0.5)
            false_positive = ~sample_truth & (scores >= 0.5)
            positive_indices = self._diverse_indices(
                1.0 - scores,
                false_negative,
                count,
                repeat_last=False,
            )
            negative_indices = self._diverse_indices(
                scores,
                false_positive,
                count,
                repeat_last=False,
            )
            positive_coords = self._indices_to_input_coords(
                positive_indices, *scores.shape
            )
            negative_coords = self._indices_to_input_coords(
                negative_indices, *scores.shape
            )

            sample_coords = []
            sample_labels = []
            padding = scores.new_zeros(2)
            for index in range(count):
                if index < positive_coords.size(0):
                    sample_coords.append(positive_coords[index])
                    sample_labels.append(1)
                else:
                    sample_coords.append(padding)
                    sample_labels.append(-1)
                if index < negative_coords.size(0):
                    sample_coords.append(negative_coords[index])
                    sample_labels.append(0)
                else:
                    sample_coords.append(padding)
                    sample_labels.append(-1)
            all_coords.append(torch.stack(sample_coords, dim=0))
            all_labels.append(
                torch.tensor(sample_labels, device=scores.device, dtype=torch.long)
            )
        return torch.stack(all_coords, dim=0), torch.stack(all_labels, dim=0)

    def _generate_coarse_prompt(
        self, image_tokens: torch.Tensor, text_tokens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        spatial_size = int(math.sqrt(image_tokens.shape[1]))
        if spatial_size * spatial_size != image_tokens.shape[1]:
            raise ValueError("Image tokens must form a square grid for SAM adapter.")

        image_tokens = rearrange(
            image_tokens,
            "B (H W) C -> B C H W",
            H=spatial_size,
            W=spatial_size,
        )

        image_tokens = self.image_adapter(image_tokens)

        dense_prompt, coarse_mask_logits = self.mask_prompt_generator(
            image_tokens,
            text_tokens,
            image_size=(self.image_embedding_size, self.image_embedding_size),
        )
        dense_prompt = F.interpolate(
            dense_prompt,
            size=(self.image_embedding_size, self.image_embedding_size),
            mode="bilinear",
            align_corners=False,
        )
        return dense_prompt, coarse_mask_logits

    def _encode_prompts(
        self,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
        coarse_mask_logits: torch.Tensor,
        text_tokens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mask_input = F.interpolate(
            coarse_mask_logits.sigmoid(),
            size=self.prompt_encoder.mask_input_size,
            mode="bilinear",
            align_corners=False,
        )
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=(point_coords, point_labels),
            boxes=None,
            masks=mask_input,
        )
        semantic_prompt = self.mask_prompt_generator.text_proj(
            text_tokens.mean(dim=1)
        ).unsqueeze(1)
        return torch.cat([sparse_embeddings, semantic_prompt], dim=1), dense_embeddings

    def _decode_mask(
        self,
        dense_prompt: torch.Tensor,
        sparse_embeddings: torch.Tensor,
        dense_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        image_pe = self.prompt_encoder.get_dense_pe().to(dense_prompt.device)
        if image_pe.shape[-2:] != dense_prompt.shape[-2:]:
            image_pe = F.interpolate(
                image_pe,
                size=dense_prompt.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        if dense_embeddings.shape[-2:] != dense_prompt.shape[-2:]:
            dense_embeddings = F.interpolate(
                dense_embeddings,
                size=dense_prompt.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
        masks, _ = self.mask_decoder(
            image_embeddings=dense_prompt,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        return masks

    def forward(
        self,
        image_tokens: torch.Tensor,
        text_tokens: torch.Tensor,
        target_size,
        target: Optional[torch.Tensor] = None,
    ):
        dense_prompt, coarse_mask_logits = self._generate_coarse_prompt(
            image_tokens, text_tokens
        )
        point_coords, point_labels = self._initial_points_from_coarse_mask(
            coarse_mask_logits
        )
        sparse_embeddings, dense_embeddings = self._encode_prompts(
            point_coords,
            point_labels,
            coarse_mask_logits,
            text_tokens,
        )
        low_res_masks = self._decode_mask(
            dense_prompt, sparse_embeddings, dense_embeddings
        )

        if (
            self.training
            and target is not None
            and self.correction_points_per_class > 0
        ):
            correction_logits = F.interpolate(
                low_res_masks.detach(),
                size=target.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            correction_coords, correction_labels = self._correction_points(
                correction_logits, target
            )
            if (correction_labels >= 0).any():
                refined_coords = torch.cat(
                    [point_coords, correction_coords], dim=1
                )
                refined_labels = torch.cat(
                    [point_labels, correction_labels], dim=1
                )
                sparse_embeddings, dense_embeddings = self._encode_prompts(
                    refined_coords,
                    refined_labels,
                    coarse_mask_logits,
                    text_tokens,
                )
                low_res_masks = self._decode_mask(
                    dense_prompt, sparse_embeddings, dense_embeddings
                )

        segmentation_logits = F.interpolate(
            low_res_masks,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        coarse_mask_logits = F.interpolate(
            coarse_mask_logits,
            size=target_size,
            mode="bilinear",
            align_corners=False,
        )
        return segmentation_logits, coarse_mask_logits
