from __future__ import annotations

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        position = torch.arange(max_len).unsqueeze(1)
        frequency = torch.exp(
            torch.arange(0, dim, 2) * -(math.log(10000.0) / dim)
        )
        encoding = torch.zeros(max_len, dim)
        encoding[:, 0::2] = torch.sin(position * frequency)
        encoding[:, 1::2] = torch.cos(position * frequency)
        self.register_buffer("pe", encoding.unsqueeze(0))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.dropout(value + self.pe[:, : value.size(1)].detach())


class SelfAttentionLayer(nn.Module):
    def __init__(self, channels: int, num_heads: int = 1) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.vis_pos = PositionalEncoding(channels)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True,
        )
        self.self_attn_norm = nn.LayerNorm(channels)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        value = value + self.vis_pos(value)
        normalized = self.norm(value)
        attended, _ = self.self_attn(normalized, normalized, normalized)
        return value + self.self_attn_norm(attended)


class CrossAttentionLayer(nn.Module):
    def __init__(
        self,
        channels: int,
        output_text_len: int = 1,
        num_heads: int = 4,
    ) -> None:
        super().__init__()
        del output_text_len
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=channels,
            num_heads=num_heads,
            batch_first=True,
        )
        self.vis_pos = PositionalEncoding(channels)
        self.txt_pos = PositionalEncoding(channels)
        self.norm = nn.LayerNorm(channels)
        self.cross_attn_norm = nn.LayerNorm(channels)
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, image: torch.Tensor, text: torch.Tensor) -> torch.Tensor:
        image = image + self.vis_pos(image)
        text = text + self.txt_pos(text)
        attended, _ = self.cross_attn(self.norm(image), text, text)
        return image + self.scale * self.cross_attn_norm(attended)


class GuidedApproximation(nn.Module):
    def __init__(
        self,
        text_dim: int,
        num_candidates: int,
        visual_dim: int,
        skip_dim: int,
    ) -> None:
        super().__init__()
        self.projection1 = nn.Linear(visual_dim, text_dim)
        self.projection2 = nn.Linear(skip_dim, text_dim)
        self.self_attn1 = SelfAttentionLayer(text_dim)
        self.self_attn2 = SelfAttentionLayer(text_dim)
        self.cross_attn1 = CrossAttentionLayer(text_dim, num_candidates)
        self.cross_attn2 = CrossAttentionLayer(text_dim, num_candidates)
        self.norm = nn.LayerNorm(text_dim)

    def forward(
        self,
        text: torch.Tensor,
        visual: torch.Tensor,
        skip: torch.Tensor,
    ) -> torch.Tensor:
        visual = self.projection1(visual)
        skip = self.projection2(skip)
        visual_response = self.cross_attn1(self.self_attn1(text), visual)
        skip_response = self.cross_attn2(self.self_attn2(text), skip)
        return self.norm(text + visual_response + skip_response)
