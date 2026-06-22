import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 5000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)].detach()


class SelfAttentionLayer(nn.Module):
    def __init__(self, dim: int, heads: int = 1) -> None:
        super().__init__()
        self.pos = PositionalEncoding(dim)
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.out_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.norm(self.pos(x))
        y, _ = self.attn(q, q, q)
        return x + self.out_norm(y)


class CrossAttentionLayer(nn.Module):
    def __init__(self, dim: int, heads: int = 4) -> None:
        super().__init__()
        self.image_pos = PositionalEncoding(dim)
        self.text_pos = PositionalEncoding(dim)
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.out_norm = nn.LayerNorm(dim)
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, image: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        q = self.norm(self.image_pos(image))
        k = self.text_pos(text)
        y, _ = self.attn(q, k, text)
        return image + self.scale * self.out_norm(y)


class MaskPromptGenerator(nn.Module):
    """Cross-attends visual tokens to prototype semantic tokens and predicts a coarse mask."""

    def __init__(self, image_dim: int, text_dim: int, prompt_dim: int) -> None:
        super().__init__()
        self.image_proj = nn.Linear(image_dim, prompt_dim)
        self.text_proj = nn.Linear(text_dim, prompt_dim)
        self.self_attn = SelfAttentionLayer(prompt_dim)
        self.cross_attn = CrossAttentionLayer(prompt_dim)
        self.mask_head = nn.Conv2d(prompt_dim, 1, kernel_size=1)

    def forward(self, image_map: torch.Tensor, text_feat: torch.Tensor, image_size: Tuple[int, int]) -> torch.Tensor:
        b, _, h, w = image_map.shape
        image_tokens = rearrange(image_map, "b c h w -> b (h w) c")
        image_tokens = self.image_proj(image_tokens)
        if text_feat.dim() == 2:
            text_feat = text_feat.unsqueeze(1)
        text_tokens = self.text_proj(text_feat)
        image_tokens = self.self_attn(image_tokens)
        image_tokens = self.cross_attn(image_tokens, text_tokens)
        fused = rearrange(image_tokens, "b (h w) c -> b c h w", h=h, w=w)
        logits = self.mask_head(fused)
        logits = F.interpolate(logits, size=image_size, mode="bilinear", align_corners=False)
        return torch.sigmoid(logits)


class AttentionApproximation(nn.Module):
    def __init__(self, text_dim: int, num_candidates: int, image_dim: int) -> None:
        super().__init__()
        self.image_proj = nn.Linear(image_dim, text_dim)
        self.self_attn = SelfAttentionLayer(text_dim)
        self.cross_attn = CrossAttentionLayer(text_dim)
        self.num_candidates = num_candidates

    def forward(self, text: torch.Tensor, image_tokens: torch.Tensor) -> torch.Tensor:
        image_tokens = self.image_proj(image_tokens)
        text = self.self_attn(text)
        return self.cross_attn(text, image_tokens)
