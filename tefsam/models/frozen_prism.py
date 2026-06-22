"""Frozen, image-query-only runtime for released PRISM artifacts.

This module intentionally contains no repository construction, text encoding,
clustering, assignment, or update code. It only loads a sanitized artifact and
retrieves persistent semantic memories with image features.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, Mapping, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file
from transformers import AutoModel


RUNTIME_FORMAT = "tefsam-frozen-prism-v1"
QUERY_SCHEMA = "hierarchical-regional-retrieval-v1"


def load_manifest(path: Union[str, Path]) -> Dict[str, object]:
    manifest_path = Path(path)
    with manifest_path.open("r", encoding="utf-8") as stream:
        manifest = json.load(stream)
    if not isinstance(manifest, dict):
        raise TypeError("A PRISM manifest must contain a JSON object")
    if manifest.get("runtime_format") != RUNTIME_FORMAT:
        raise ValueError(
            f"Unsupported PRISM runtime format: {manifest.get('runtime_format')!r}"
        )
    if manifest.get("query_schema") != QUERY_SCHEMA:
        raise ValueError(
            f"Unsupported PRISM query schema: {manifest.get('query_schema')!r}"
        )
    return manifest


class _BiomedCLIPVisionEncoder(nn.Module):
    """Expose only the BiomedCLIP vision tower needed for image queries."""

    def __init__(
        self,
        checkpoint: str,
        *,
        revision: Optional[str] = None,
        local_files_only: bool = False,
    ) -> None:
        super().__init__()
        full_model = AutoModel.from_pretrained(
            checkpoint,
            trust_remote_code=True,
            output_hidden_states=True,
            low_cpu_mem_usage=True,
            local_files_only=local_files_only,
            revision=revision,
        )
        if not hasattr(full_model, "vision_model"):
            raise AttributeError("The configured BiomedCLIP model has no vision tower")
        if not hasattr(full_model, "visual_projection"):
            raise AttributeError(
                "The configured BiomedCLIP model has no visual projection"
            )
        self.vision_model = full_model.vision_model
        self.visual_projection = full_model.visual_projection

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    @staticmethod
    def _ensure_rgb(image: torch.Tensor) -> torch.Tensor:
        if image.ndim == 3:
            image = image.unsqueeze(0)
        if image.ndim != 4:
            raise ValueError("image must have shape (B, C, H, W)")
        if image.size(1) == 1:
            image = image.expand(-1, 3, -1, -1)
        if image.size(1) != 3:
            raise ValueError("BiomedCLIP image queries require one or three channels")
        return image

    def encode_image_with_tokens(
        self, image: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        image = self._ensure_rgb(image).to(self.device)
        output = self.vision_model(image)
        sequence = output[0]
        if sequence.ndim != 3 or sequence.size(1) < 2:
            raise ValueError("BiomedCLIP must return CLS and patch tokens")
        post_layernorm = getattr(self.vision_model, "post_layernorm", None)
        if post_layernorm is not None:
            patch_tokens = post_layernorm(sequence[:, 1:])
            pooled = output[1] if len(output) > 1 else post_layernorm(sequence[:, 0])
        else:
            patch_tokens = sequence[:, 1:]
            pooled = output[1] if len(output) > 1 else sequence[:, 0]
        return self.visual_projection(pooled), self.visual_projection(patch_tokens)


class FrozenPRISMRuntime(nn.Module):
    """Load and query a released PRISM memory without construction support."""

    exclude_from_segmentation_checkpoint = True

    def __init__(
        self,
        manifest: Mapping[str, object],
        *,
        image_encoder_checkpoint: Optional[str] = None,
        local_files_only: bool = False,
    ) -> None:
        super().__init__()
        self.manifest = dict(manifest)
        artifact_id = manifest.get("artifact_id")
        self.artifact_id = str(artifact_id) if artifact_id is not None else None
        self.num_regions = int(manifest["num_regions"])
        self.num_prototypes = int(manifest["num_prototypes"])
        self.num_candidate = int(manifest["num_candidates"])
        self.embedding_dim = int(manifest["embedding_dim"])
        self.retrieval_dim = int(manifest["retrieval_dim"])
        self.feature_dim = int(manifest["feature_dim"])
        self.memory_tokens = int(manifest["memory_tokens"])
        self.grid_rows = int(manifest["grid_rows"])
        self.grid_cols = int(manifest["grid_cols"])
        if self.grid_rows * self.grid_cols != self.num_regions:
            raise ValueError("grid_rows * grid_cols must equal num_regions")
        if self.num_candidate <= 0:
            raise ValueError("num_candidates must be positive")

        self.retrieval_temperature = float(manifest["retrieval_temperature"])
        self.response_temperature = float(manifest["response_temperature"])
        self.patch_mil_temperature = float(manifest["patch_mil_temperature"])
        self.region_gate_weight = float(manifest["region_gate_weight"])
        self.region_position_scale = float(manifest["region_position_scale"])
        if min(
            self.retrieval_temperature,
            self.response_temperature,
            self.patch_mil_temperature,
        ) <= 0:
            raise ValueError("PRISM temperatures must be positive")

        encoder_spec = manifest.get("image_encoder")
        if not isinstance(encoder_spec, Mapping):
            raise ValueError("manifest.image_encoder must be an object")
        checkpoint = image_encoder_checkpoint or str(encoder_spec["checkpoint"])
        revision = None
        if image_encoder_checkpoint is None:
            configured_revision = encoder_spec.get("revision")
            revision = (
                str(configured_revision) if configured_revision is not None else None
            )
        self.image_encoder = _BiomedCLIPVisionEncoder(
            checkpoint,
            revision=revision,
            local_files_only=local_files_only,
        )
        self.image_projector = nn.Linear(
            self.embedding_dim, self.retrieval_dim, bias=False
        )
        self.region_head = nn.Linear(self.retrieval_dim, self.num_regions)

        self.register_buffer(
            "prototype_keys",
            torch.empty(
                self.num_regions, self.num_prototypes, self.retrieval_dim
            ),
        )
        self.register_buffer(
            "memory_text",
            torch.empty(
                self.num_regions,
                self.num_prototypes,
                self.memory_tokens,
                self.feature_dim,
            ),
        )
        self.register_buffer(
            "prototype_support",
            torch.empty(
                self.num_regions, self.num_prototypes, dtype=torch.bool
            ),
        )
        self.register_buffer(
            "global_image_center", torch.empty(self.embedding_dim)
        )
        self.register_buffer(
            "regional_query_center", torch.empty(self.embedding_dim)
        )
        self.register_buffer(
            "memory_anchor", torch.empty(self.memory_tokens, self.feature_dim)
        )
        self.register_buffer("memory_residual_scale", torch.empty(()))
        self.register_buffer(
            "repository_ready", torch.tensor(False, dtype=torch.bool)
        )
        self.register_buffer(
            "region_position_code",
            self._build_region_position_code(),
            persistent=False,
        )
        self._coverage_cache: Dict[Tuple[object, ...], torch.Tensor] = {}
        self._artifact_path: Optional[Path] = None

    @classmethod
    def load(
        cls,
        artifact_path: Union[str, Path],
        *,
        manifest_path: Union[str, Path],
        image_encoder_checkpoint: Optional[str] = None,
        device: Union[str, torch.device] = "cpu",
        local_files_only: bool = False,
    ) -> "FrozenPRISMRuntime":
        artifact = Path(artifact_path)
        if not artifact.is_file():
            raise FileNotFoundError(f"Frozen PRISM artifact not found: {artifact}")
        manifest = load_manifest(manifest_path)

        runtime = cls(
            manifest,
            image_encoder_checkpoint=image_encoder_checkpoint,
            local_files_only=local_files_only,
        )
        runtime._load_runtime_tensors(load_file(str(artifact), device="cpu"))
        runtime._artifact_path = artifact.resolve()
        runtime.to(device)
        runtime.freeze()
        return runtime

    @property
    def artifact_path(self) -> Path:
        if self._artifact_path is None:
            raise RuntimeError("Frozen PRISM has not been loaded from an artifact")
        return self._artifact_path

    @property
    def matrix(self) -> torch.Tensor:
        return self.prototype_keys

    def _load_runtime_tensors(self, tensors: Mapping[str, torch.Tensor]) -> None:
        buffer_names = (
            "prototype_keys",
            "memory_text",
            "prototype_support",
            "global_image_center",
            "regional_query_center",
            "memory_anchor",
            "memory_residual_scale",
        )
        required = set(buffer_names)
        required.update(
            {
                "image_projector.weight",
                "region_head.weight",
                "region_head.bias",
            }
        )
        missing = sorted(required.difference(tensors))
        if missing:
            raise ValueError(
                "Frozen PRISM artifact is missing tensors: " + ", ".join(missing)
            )
        unknown = sorted(
            name
            for name in tensors
            if name not in required and not name.startswith("vision_delta.")
        )
        if unknown:
            raise ValueError(
                "Frozen PRISM artifact contains unsupported tensors: "
                + ", ".join(unknown)
            )

        with torch.no_grad():
            for name in buffer_names:
                destination = getattr(self, name)
                source = tensors[name]
                if tuple(source.shape) != tuple(destination.shape):
                    raise ValueError(
                        f"Tensor shape mismatch for {name}: expected "
                        f"{tuple(destination.shape)}, got {tuple(source.shape)}"
                    )
                destination.copy_(source.to(dtype=destination.dtype))
            self.image_projector.weight.copy_(tensors["image_projector.weight"])
            self.region_head.weight.copy_(tensors["region_head.weight"])
            self.region_head.bias.copy_(tensors["region_head.bias"])

            encoder_parameters = dict(self.image_encoder.named_parameters())
            prefix = "vision_delta."
            delta_names = {
                name[len(prefix) :]: value
                for name, value in tensors.items()
                if name.startswith(prefix)
            }
            if not delta_names:
                raise ValueError("Frozen PRISM artifact contains no vision delta")
            missing_parameters = sorted(set(delta_names).difference(encoder_parameters))
            if missing_parameters:
                raise ValueError(
                    "Vision delta is incompatible with the configured BiomedCLIP: "
                    + ", ".join(missing_parameters[:5])
                )
            for name, value in delta_names.items():
                parameter = encoder_parameters[name]
                if tuple(value.shape) != tuple(parameter.shape):
                    raise ValueError(f"Vision delta shape mismatch for {name}")
                parameter.copy_(value.to(dtype=parameter.dtype))
            self.repository_ready.fill_(True)

    def freeze(self) -> None:
        for parameter in self.parameters():
            parameter.requires_grad = False
        super().train(False)

    def train(self, mode: bool = True) -> "FrozenPRISMRuntime":
        del mode
        super().train(False)
        return self

    def _build_region_position_code(self) -> torch.Tensor:
        frequency_count = max(math.ceil(self.feature_dim / 4), 1)
        frequencies = math.pi * torch.exp(
            torch.linspace(0.0, math.log(16.0), frequency_count)
        )
        codes = []
        for region_index in range(self.num_regions):
            row = region_index // self.grid_cols
            col = region_index % self.grid_cols
            y = 2.0 * ((row + 0.5) / float(self.grid_rows)) - 1.0
            x = 2.0 * ((col + 0.5) / float(self.grid_cols)) - 1.0
            components = torch.stack(
                [
                    torch.sin(x * frequencies),
                    torch.sin(y * frequencies),
                    torch.cos(x * frequencies),
                    torch.cos(y * frequencies),
                ],
                dim=1,
            ).flatten()[: self.feature_dim]
            components = components - components.mean()
            components = components / components.square().mean().sqrt().clamp_min(
                1e-6
            )
            codes.append(components)
        return torch.stack(codes, dim=0)[:, None, :]

    @staticmethod
    def _patch_grid_shape(
        num_patches: int, image_hw: Optional[Tuple[int, int]] = None
    ) -> Tuple[int, int]:
        target_ratio = 1.0
        if image_hw is not None and image_hw[0] > 0 and image_hw[1] > 0:
            target_ratio = image_hw[1] / float(image_hw[0])
        candidates = [
            (rows, num_patches // rows)
            for rows in range(1, int(math.sqrt(num_patches)) + 1)
            if num_patches % rows == 0
        ]
        if not candidates:
            raise ValueError(f"Cannot infer a 2D grid from {num_patches} patches")
        return min(
            candidates,
            key=lambda shape: abs(
                math.log((shape[1] / float(shape[0])) / target_ratio)
            ),
        )

    def _normalize_patch_tokens(
        self, tokens: torch.Tensor, image_hw: Tuple[int, int]
    ) -> Tuple[torch.Tensor, int, int]:
        if tokens.ndim != 3:
            raise ValueError("Patch features must have shape (B, N, D)")
        token_count = tokens.size(1)
        target_ratio = image_hw[1] / float(image_hw[0])

        def score(count: int) -> Tuple[float, Tuple[int, int]]:
            shape = self._patch_grid_shape(count, image_hw)
            ratio = shape[1] / float(shape[0])
            return abs(math.log(ratio / target_ratio)), shape

        full_score, full_shape = score(token_count)
        cls_score, cls_shape = float("inf"), (0, 0)
        if token_count > 1:
            cls_score, cls_shape = score(token_count - 1)
        if cls_score + 1e-9 < full_score:
            tokens = tokens[:, 1:]
            patch_h, patch_w = cls_shape
        else:
            patch_h, patch_w = full_shape
        if tokens.size(-1) != self.embedding_dim:
            raise ValueError(
                f"Patch dimension must be {self.embedding_dim}, got {tokens.size(-1)}"
            )
        return tokens, patch_h, patch_w

    def _patch_region_coverage(
        self,
        patch_h: int,
        patch_w: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        key = (
            patch_h,
            patch_w,
            device.type,
            device.index,
            dtype,
            self.grid_rows,
            self.grid_cols,
        )
        cached = self._coverage_cache.get(key)
        if cached is not None:
            return cached

        patch_y0 = torch.arange(patch_h, device=device, dtype=dtype) / patch_h
        patch_y1 = patch_y0 + 1.0 / patch_h
        region_y0 = (
            torch.arange(self.grid_rows, device=device, dtype=dtype) / self.grid_rows
        )
        region_y1 = region_y0 + 1.0 / self.grid_rows
        coverage_y = (
            torch.minimum(patch_y1[:, None], region_y1[None])
            - torch.maximum(patch_y0[:, None], region_y0[None])
        ).clamp_min(0) * patch_h

        patch_x0 = torch.arange(patch_w, device=device, dtype=dtype) / patch_w
        patch_x1 = patch_x0 + 1.0 / patch_w
        region_x0 = (
            torch.arange(self.grid_cols, device=device, dtype=dtype) / self.grid_cols
        )
        region_x1 = region_x0 + 1.0 / self.grid_cols
        coverage_x = (
            torch.minimum(patch_x1[:, None], region_x1[None])
            - torch.maximum(patch_x0[:, None], region_x0[None])
        ).clamp_min(0) * patch_w

        coverage = torch.einsum("hr,wc->hwrc", coverage_y, coverage_x).reshape(
            patch_h * patch_w, self.num_regions
        )
        coverage = coverage / coverage.sum(dim=1, keepdim=True).clamp_min(1e-12)
        self._coverage_cache[key] = coverage
        return coverage

    @staticmethod
    def _center_and_normalize(
        features: torch.Tensor, center: torch.Tensor
    ) -> torch.Tensor:
        centered = features - center.to(features.dtype)
        centered = torch.where(
            centered.norm(dim=-1, keepdim=True) <= 1e-8,
            features,
            centered,
        )
        return F.normalize(centered, dim=-1)

    def _global_query(self, image_embedding: torch.Tensor) -> torch.Tensor:
        centered = self._center_and_normalize(
            image_embedding, self.global_image_center
        )
        return F.normalize(self.image_projector(centered), dim=-1)

    def _local_queries(self, features: torch.Tensor) -> torch.Tensor:
        centered = self._center_and_normalize(
            features, self.regional_query_center
        )
        return F.normalize(self.image_projector(centered), dim=-1)

    def _patch_mil_similarities(
        self,
        patch_queries: torch.Tensor,
        patch_coverage: torch.Tensor,
        region_index: int,
        prototypes: torch.Tensor,
    ) -> torch.Tensor:
        similarities = torch.einsum("bnd,kd->bnk", patch_queries, prototypes)
        weights = patch_coverage[:, region_index]
        weights = weights / weights.sum().clamp_min(1e-8)
        log_weights = torch.log(weights.clamp_min(1e-12))[None, :, None]
        return self.patch_mil_temperature * torch.logsumexp(
            similarities / self.patch_mil_temperature + log_weights,
            dim=1,
        )

    def _retrieval_scores(
        self,
        global_query: torch.Tensor,
        patch_queries: torch.Tensor,
        patch_coverage: torch.Tensor,
    ) -> torch.Tensor:
        if not bool(self.repository_ready.item()):
            raise RuntimeError("Frozen PRISM artifact is not ready")
        valid_regions = self.prototype_support.any(dim=1)
        if not valid_regions.any():
            raise RuntimeError("Frozen PRISM contains no supported prototypes")
        region_logits = self.region_head(global_query)
        region_logits = region_logits.masked_fill(~valid_regions[None], -torch.inf)
        log_region = F.log_softmax(region_logits, dim=-1)
        scores = global_query.new_full(
            (global_query.size(0), self.num_regions, self.num_prototypes),
            -torch.inf,
        )
        prototypes = F.normalize(self.prototype_keys, dim=-1)
        for region_index in range(self.num_regions):
            support = self.prototype_support[region_index]
            if not support.any():
                continue
            active_prototypes = prototypes[region_index, support]
            local_similarity = self._patch_mil_similarities(
                patch_queries,
                patch_coverage,
                region_index,
                active_prototypes,
            )
            local_log_probability = F.log_softmax(
                local_similarity / self.retrieval_temperature, dim=-1
            )
            joint = local_log_probability + self.region_gate_weight * log_region[
                :, region_index, None
            ]
            scores[:, region_index, support] = joint
        return scores

    def _respond(self, scores: torch.Tensor) -> torch.Tensor:
        flat_scores = scores.flatten(1)
        candidate_count = min(
            self.num_candidate, int(self.prototype_support.sum().item())
        )
        tie_break = torch.arange(
            flat_scores.size(1), device=flat_scores.device, dtype=torch.float32
        ) * torch.finfo(torch.float32).eps
        _, top_indices = torch.topk(
            flat_scores.float() - tie_break[None],
            k=candidate_count,
            dim=1,
        )
        top_scores = flat_scores.gather(1, top_indices)

        responses = []
        for sample_indices, sample_scores in zip(top_indices, top_scores):
            regions = torch.div(
                sample_indices, self.num_prototypes, rounding_mode="floor"
            )
            prototypes = torch.remainder(sample_indices, self.num_prototypes)
            raw_memories = self.memory_text[regions, prototypes]
            region_codes = self.region_position_code[regions]
            memories = (
                raw_memories - self.memory_anchor[None]
            ) / self.memory_residual_scale.clamp_min(1e-6)
            memories = memories + self.region_position_scale * region_codes
            weights = F.softmax(
                sample_scores / self.response_temperature, dim=0
            )
            responses.append((memories * weights[:, None, None]).sum(dim=0))
        return torch.stack(responses, dim=0)

    @torch.no_grad()
    def query(
        self,
        images: torch.Tensor,
        image_emb: Optional[torch.Tensor] = None,
        image_feature: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        output_device = images.device
        with torch.cuda.amp.autocast(enabled=False):
            if isinstance(image_emb, torch.Tensor) and isinstance(
                image_feature, torch.Tensor
            ):
                global_image = image_emb.to(self.prototype_keys.device).float()
                patch_tokens = image_feature.to(self.prototype_keys.device).float()
            else:
                global_image, patch_tokens = (
                    self.image_encoder.encode_image_with_tokens(images.float())
                )
                global_image = global_image.float()
                patch_tokens = patch_tokens.float()
            patch_tokens, patch_h, patch_w = self._normalize_patch_tokens(
                patch_tokens, tuple(images.shape[-2:])
            )
            global_query = self._global_query(global_image)
            patch_queries = self._local_queries(patch_tokens)
            patch_coverage = self._patch_region_coverage(
                patch_h,
                patch_w,
                device=patch_queries.device,
                dtype=patch_queries.dtype,
            )
            scores = self._retrieval_scores(
                global_query,
                patch_queries,
                patch_coverage,
            )
            response = self._respond(scores)
        return response.to(output_device)


def validate_repository_contract(
    repository: FrozenPRISMRuntime, model_cfg: object
) -> None:
    expected_dim = int(getattr(model_cfg, "text_dim", 768))
    if repository.feature_dim != expected_dim:
        raise ValueError(
            f"PRISM feature dimension {repository.feature_dim} does not match "
            f"model text_dim {expected_dim}"
        )
    if not bool(repository.repository_ready.item()):
        raise RuntimeError("Frozen PRISM must be loaded before creating TeF-SAM")
