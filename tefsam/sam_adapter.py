import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from segment_anything import sam_model_registry

from .layers import MaskPromptGenerator


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 1.0) -> None:
        super().__init__()
        self.base = base
        self.rank = rank
        self.scaling = alpha / max(rank, 1)
        self.lora_down = nn.Linear(base.in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, base.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_up(self.lora_down(x)) * self.scaling


def inject_lora(module: nn.Module, rank: int, alpha: float) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, rank, alpha))
        else:
            inject_lora(child, rank, alpha)


class SAMPromptDecoder(nn.Module):
    """Text/prototype-conditioned SAM decoder with dense mask and point prompts."""

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.input_image_size = tuple(cfg.image_size)
        self.image_embedding_size = cfg.sam_image_embedding
        self.point_topk = cfg.point_topk
        self.point_threshold = cfg.point_threshold
        self.point_quantile = cfg.point_quantile
        self.use_local_max = cfg.use_local_max
        self.use_quantile_threshold = cfg.use_quantile_threshold
        self.max_candidates = cfg.max_candidates
        self.initial_point_topk = cfg.initial_point_topk
        self.correction_iters = cfg.correction_iters
        self.correction_num_points = cfg.correction_num_points
        self.decoder_image_source = cfg.decoder_image_source
        self.sam_encoder_batch_size = max(1, int(cfg.sam_encoder_batch_size))

        self.image_adapter = nn.Sequential(
            nn.Conv2d(cfg.sam_prompt_dim, cfg.sam_prompt_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(cfg.sam_prompt_dim, cfg.sam_prompt_dim, kernel_size=3, padding=1, groups=cfg.sam_prompt_dim),
            nn.GELU(),
            nn.Conv2d(cfg.sam_prompt_dim, cfg.sam_prompt_dim, kernel_size=1),
        )
        self.mask_prompt_generator = MaskPromptGenerator(cfg.sam_prompt_dim, cfg.text_dim, cfg.sam_prompt_dim)
        self.text_prompt = nn.Linear(cfg.text_dim, cfg.sam_prompt_dim)

        sam = sam_model_registry[cfg.sam_model_type](checkpoint=cfg.sam_checkpoint)
        self.prompt_encoder = sam.prompt_encoder
        self.mask_decoder = sam.mask_decoder
        self.sam_image_encoder = sam.image_encoder
        self.register_buffer("pixel_mean", sam.pixel_mean.detach().clone(), persistent=False)
        self.register_buffer("pixel_std", sam.pixel_std.detach().clone(), persistent=False)

        if cfg.lora_rank > 0:
            inject_lora(self.mask_decoder.transformer, cfg.lora_rank, cfg.lora_alpha)

        for p in self.prompt_encoder.parameters():
            p.requires_grad = False
        for p in self.sam_image_encoder.parameters():
            p.requires_grad = False
        for name, p in self.mask_decoder.named_parameters():
            p.requires_grad = "lora_" in name

    def _preprocess_sam_image(self, image: torch.Tensor) -> torch.Tensor:
        x = image.detach().float()
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        elif x.shape[1] > 3:
            x = x[:, :3]
        x_min = x.amin(dim=(2, 3), keepdim=True)
        x_max = x.amax(dim=(2, 3), keepdim=True)
        if (x_min < 0).any() or (x_max <= 5.0).any() or (x_max > 255.0).any():
            x = (x - x_min) / (x_max - x_min).clamp_min(1e-6) * 255.0
        size = int(self.sam_image_encoder.img_size)
        if x.shape[-2:] != (size, size):
            x = F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
        return (x - self.pixel_mean.to(x)) / self.pixel_std.to(x)

    @torch.no_grad()
    def encode_sam_image(self, image: torch.Tensor) -> torch.Tensor:
        self.sam_image_encoder.eval()
        chunks = self._preprocess_sam_image(image).split(self.sam_encoder_batch_size, dim=0)
        return torch.cat([self.sam_image_encoder(chunk) for chunk in chunks], dim=0)

    def _scale_points(self, coords: torch.Tensor, target_size: Tuple[int, int]) -> torch.Tensor:
        h, w = target_size
        sam_h, sam_w = self.prompt_encoder.input_image_size
        scale = torch.tensor([sam_w / max(w, 1), sam_h / max(h, 1)], device=coords.device, dtype=coords.dtype)
        return coords * scale

    def _select_points(self, heatmap: torch.Tensor, k: int) -> torch.Tensor:
        b, h, w = heatmap.shape
        coords_all = []
        for i in range(b):
            scores = heatmap[i]
            flat = scores.flatten()
            threshold = self.point_threshold
            if self.use_quantile_threshold:
                threshold = float(torch.quantile(flat, min(max(self.point_quantile, 0.0), 1.0)).item())
            if self.use_local_max:
                pooled = F.max_pool2d(scores[None, None], kernel_size=3, stride=1, padding=1)[0, 0]
                mask = scores.eq(pooled) & (scores > threshold)
            else:
                mask = scores > threshold
            idx = mask.flatten().nonzero(as_tuple=False).squeeze(1)
            if idx.numel() == 0:
                idx = torch.topk(flat, k=min(self.max_candidates, flat.numel())).indices
            elif idx.numel() > self.max_candidates:
                idx = idx[torch.topk(flat[idx], k=self.max_candidates).indices]

            xy = torch.stack([(idx % w).float(), (idx // w).float()], dim=1)
            score = flat[idx]
            first = int(torch.argmax(score).item())
            selected = [first]
            min_dist = ((xy - xy[first]) ** 2).sum(dim=1)
            while len(selected) < min(k, xy.size(0)):
                far = int(torch.argmax(min_dist).item())
                selected.append(far)
                min_dist = torch.minimum(min_dist, ((xy - xy[far]) ** 2).sum(dim=1))
            while len(selected) < k:
                selected.append(selected[-1])
            chosen = xy[torch.tensor(selected, device=xy.device)]
            scale = torch.tensor([self.input_image_size[1] / w, self.input_image_size[0] / h], device=xy.device)
            coords_all.append(chosen * scale)
        return torch.stack(coords_all, dim=0)

    def _generate_prompts(self, visual_tokens: torch.Tensor, semantic_tokens: torch.Tensor, num_points: Optional[int] = None):
        grid = int(math.sqrt(visual_tokens.size(1)))
        visual_map = rearrange(visual_tokens, "b (h w) c -> b c h w", h=grid, w=grid)
        visual_map = self.image_adapter(visual_map)
        mask_input_size = tuple(int(x) for x in self.prompt_encoder.mask_input_size)
        mask_prob = self.mask_prompt_generator(visual_map, semantic_tokens, mask_input_size)
        mask_logits = torch.logit(mask_prob.clamp(1e-4, 1 - 1e-4))
        k = self.point_topk if num_points is None else num_points
        point_coords = self._select_points(mask_prob[:, 0], k)
        point_labels = torch.ones(point_coords.shape[:2], dtype=torch.int64, device=point_coords.device)
        return visual_map, mask_logits, point_coords, point_labels

    def _decoder_image_embeddings(self, image: torch.Tensor, visual_map: torch.Tensor) -> torch.Tensor:
        if self.decoder_image_source == "dense_prompt":
            return F.interpolate(
                visual_map,
                size=(self.image_embedding_size, self.image_embedding_size),
                mode="bilinear",
                align_corners=False,
            )
        if self.decoder_image_source == "sam_encoder":
            return self.encode_sam_image(image)
        raise ValueError(f"Unknown decoder_image_source: {self.decoder_image_source}")

    def _decode(
        self,
        image_embeddings: torch.Tensor,
        semantic_tokens: torch.Tensor,
        mask_logits: torch.Tensor,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
        target_size: Tuple[int, int],
    ) -> torch.Tensor:
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points=(self._scale_points(point_coords, target_size), point_labels),
            boxes=None,
            masks=mask_logits,
        )
        text_token = self.text_prompt(semantic_tokens.mean(dim=1)).unsqueeze(1).to(sparse_embeddings.dtype)
        sparse_embeddings = torch.cat([sparse_embeddings, text_token], dim=1).to(image_embeddings)
        dense_embeddings = dense_embeddings.to(image_embeddings)
        if dense_embeddings.shape[-2:] != image_embeddings.shape[-2:]:
            dense_embeddings = F.interpolate(dense_embeddings, size=image_embeddings.shape[-2:], mode="bilinear", align_corners=False)
        image_pe = self.prompt_encoder.get_dense_pe().to(image_embeddings)
        if image_pe.shape[-2:] != image_embeddings.shape[-2:]:
            image_pe = F.interpolate(image_pe, size=image_embeddings.shape[-2:], mode="bilinear", align_corners=False)
        low_res, _ = self.mask_decoder(
            image_embeddings=image_embeddings,
            image_pe=image_pe,
            sparse_prompt_embeddings=sparse_embeddings,
            dense_prompt_embeddings=dense_embeddings,
            multimask_output=False,
        )
        return F.interpolate(low_res, size=target_size, mode="bilinear", align_corners=False)

    def add_correction_clicks(self, pred: torch.Tensor, gt: torch.Tensor, n: int, prev: torch.Tensor):
        prob = torch.sigmoid(pred) if pred.min() < 0 or pred.max() > 1 else pred
        if gt.dim() == 3:
            gt = gt.unsqueeze(1)
        if gt.shape[-2:] != prob.shape[-2:]:
            gt = F.interpolate(gt.float(), size=prob.shape[-2:], mode="nearest")
        fn = (gt > 0.5) & ~(prob > 0.5)
        fp = (prob > 0.5) & ~(gt > 0.5)
        b, _, h, w = prob.shape
        coords = torch.zeros(b, n, 2, device=prob.device)
        labels = torch.zeros(b, n, dtype=torch.int64, device=prob.device)
        for i in range(b):
            for j in range(n):
                region = fn[i, 0] if j % 2 == 0 else fp[i, 0]
                if not region.any():
                    region = fp[i, 0] if j % 2 == 0 else fn[i, 0]
                if region.any():
                    y, x = region.nonzero(as_tuple=True)
                    pick = torch.randint(0, y.numel(), (1,), device=prob.device)
                    coords[i, j] = torch.stack([x[pick].float(), y[pick].float()]).flatten()
                    labels[i, j] = 1 if j % 2 == 0 else 0
                else:
                    coords[i, j] = prev[i, -1] if prev.numel() else torch.tensor([w / 2, h / 2], device=prob.device)
                    labels[i, j] = 1
        return coords, labels

    def forward(self, image: torch.Tensor, visual_tokens: torch.Tensor, semantic_tokens: torch.Tensor, target_size: Tuple[int, int], mask_gt: Optional[torch.Tensor] = None):
        n0 = min(self.initial_point_topk, self.point_topk) if self.training and mask_gt is not None else self.point_topk
        visual_map, mask_logits, point_coords, point_labels = self._generate_prompts(visual_tokens, semantic_tokens, n0)
        image_embeddings = self._decoder_image_embeddings(image, visual_map)

        if self.training and mask_gt is not None:
            for _ in range(self.correction_iters):
                pred = self._decode(image_embeddings, semantic_tokens, mask_logits, point_coords, point_labels, target_size)
                new_coords, new_labels = self.add_correction_clicks(pred, mask_gt, self.correction_num_points, point_coords)
                point_coords = torch.cat([point_coords, new_coords], dim=1)
                point_labels = torch.cat([point_labels, new_labels], dim=1)

        logits = self._decode(image_embeddings, semantic_tokens, mask_logits, point_coords, point_labels, target_size)
        return logits, mask_logits, point_coords, point_labels
