from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureAdapter(nn.Module):
    def __init__(self, in_channels: int, prompt_dim: int) -> None:
        super().__init__()
        self.adapter = nn.Sequential(
            nn.Conv2d(in_channels, prompt_dim, kernel_size=1),
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.adapter(x)


class LightASPP(nn.Module):
    def __init__(
        self,
        prompt_dim: int,
        rates: Tuple[int, int, int, int] = (1, 3, 6, 9),
    ) -> None:
        super().__init__()
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        prompt_dim,
                        prompt_dim,
                        kernel_size=3,
                        padding=rate,
                        dilation=rate,
                        groups=prompt_dim,
                    ),
                    nn.GELU(),
                )
                for rate in rates
            ]
        )
        self.project = nn.Conv2d(
            prompt_dim * len(rates), prompt_dim, kernel_size=1
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = [branch(x) for branch in self.branches]
        return self.project(torch.cat(feats, dim=1))


class MultiScaleFeatureFusion(nn.Module):
    def __init__(
        self,
        in_channels: List[int],
        prompt_dim: int,
        token_grid: int = 8,
    ) -> None:
        super().__init__()
        if len(in_channels) != 4:
            raise ValueError("in_channels must have 4 elements for F0-F3.")
        self.prompt_dim = prompt_dim
        self.token_grid = token_grid

        self.adapters = nn.ModuleList(
            [FeatureAdapter(c, prompt_dim) for c in in_channels]
        )

        self.w_td = nn.ParameterList(
            [nn.Parameter(torch.ones(2)) for _ in range(3)]
        )
        self.w_bu = nn.ParameterList(
            [nn.Parameter(torch.ones(2)) for _ in range(3)]
        )
        self.eps = 1e-6

        self.asff_weights = nn.Conv2d(prompt_dim * 4, 4, kernel_size=1)
        self.aspp = LightASPP(prompt_dim)

    def _normalize_weights(self, w: torch.Tensor) -> torch.Tensor:
        w = F.relu(w)
        return w / (w.sum() + self.eps)

    def forward(
        self, image_features: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(image_features) != 4:
            raise ValueError("image_features must be a list of 4 tensors.")

        f0, f1, f2, f3 = [
            adapter(feature)
            for adapter, feature in zip(self.adapters, image_features)
        ]

        w2 = self._normalize_weights(self.w_td[0])
        p2 = w2[0] * f2 + w2[1] * F.interpolate(
            f3, size=f2.shape[-2:], mode="bilinear", align_corners=False
        )
        w1 = self._normalize_weights(self.w_td[1])
        p1 = w1[0] * f1 + w1[1] * F.interpolate(
            p2, size=f1.shape[-2:], mode="bilinear", align_corners=False
        )
        w0 = self._normalize_weights(self.w_td[2])
        p0 = w0[0] * f0 + w0[1] * F.interpolate(
            p1, size=f0.shape[-2:], mode="bilinear", align_corners=False
        )

        b0 = p0
        wb1 = self._normalize_weights(self.w_bu[0])
        b1 = wb1[0] * p1 + wb1[1] * F.avg_pool2d(b0, kernel_size=2, stride=2)
        wb2 = self._normalize_weights(self.w_bu[1])
        b2 = wb2[0] * p2 + wb2[1] * F.avg_pool2d(b1, kernel_size=2, stride=2)
        wb3 = self._normalize_weights(self.w_bu[2])
        b3 = wb3[0] * f3 + wb3[1] * F.avg_pool2d(b2, kernel_size=2, stride=2)

        b1_up = F.interpolate(
            b1, size=b0.shape[-2:], mode="bilinear", align_corners=False
        )
        b2_up = F.interpolate(
            b2, size=b0.shape[-2:], mode="bilinear", align_corners=False
        )
        b3_up = F.interpolate(
            b3, size=b0.shape[-2:], mode="bilinear", align_corners=False
        )

        asff_in = torch.cat([b0, b1_up, b2_up, b3_up], dim=1)
        weight_maps = F.softmax(self.asff_weights(asff_in), dim=1)
        fused_map = (
            weight_maps[:, 0:1] * b0
            + weight_maps[:, 1:2] * b1_up
            + weight_maps[:, 2:3] * b2_up
            + weight_maps[:, 3:4] * b3_up
        )

        fused_map = self.aspp(fused_map)

        pooled = F.adaptive_avg_pool2d(fused_map, (self.token_grid, self.token_grid))
        visual_tokens = pooled.flatten(2).transpose(1, 2)

        return fused_map, visual_tokens
