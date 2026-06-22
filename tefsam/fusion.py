from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class FeatureAdapter(nn.Module):
    def __init__(self, in_channels: int, prompt_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, prompt_dim, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(prompt_dim, prompt_dim, kernel_size=3, padding=1, groups=prompt_dim),
            nn.GELU(),
            nn.Conv2d(prompt_dim, prompt_dim, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiScaleFeatureFusion(nn.Module):
    """BiFPN-like fusion that converts a four-level visual pyramid to SAM prompt tokens."""

    def __init__(self, in_channels: List[int], prompt_dim: int, token_grid: int = 56) -> None:
        super().__init__()
        if len(in_channels) != 4:
            raise ValueError("in_channels must contain four pyramid channel sizes.")
        self.adapters = nn.ModuleList([FeatureAdapter(c, prompt_dim) for c in in_channels])
        self.w_td = nn.ParameterList([nn.Parameter(torch.ones(2)) for _ in range(3)])
        self.w_bu = nn.ParameterList([nn.Parameter(torch.ones(2)) for _ in range(3)])
        self.asff_weights = nn.Conv2d(prompt_dim * 4, 4, kernel_size=1)
        self.aspp = nn.Sequential(
            nn.Conv2d(prompt_dim, prompt_dim, kernel_size=3, padding=1, groups=prompt_dim),
            nn.GELU(),
            nn.Conv2d(prompt_dim, prompt_dim, kernel_size=1),
        )
        self.token_grid = token_grid
        self.eps = 1e-6

    def _norm_weight(self, w: torch.Tensor) -> torch.Tensor:
        w = F.relu(w)
        return w / (w.sum() + self.eps)

    def forward(self, features: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        f0, f1, f2, f3 = [adapter(feat) for adapter, feat in zip(self.adapters, features)]

        w = self._norm_weight(self.w_td[0])
        p2 = w[0] * f2 + w[1] * F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        w = self._norm_weight(self.w_td[1])
        p1 = w[0] * f1 + w[1] * F.interpolate(p2, size=f1.shape[-2:], mode="bilinear", align_corners=False)
        w = self._norm_weight(self.w_td[2])
        p0 = w[0] * f0 + w[1] * F.interpolate(p1, size=f0.shape[-2:], mode="bilinear", align_corners=False)

        w = self._norm_weight(self.w_bu[0])
        b1 = w[0] * p1 + w[1] * F.avg_pool2d(p0, kernel_size=2, stride=2)
        w = self._norm_weight(self.w_bu[1])
        b2 = w[0] * p2 + w[1] * F.avg_pool2d(b1, kernel_size=2, stride=2)
        w = self._norm_weight(self.w_bu[2])
        b3 = w[0] * f3 + w[1] * F.avg_pool2d(b2, kernel_size=2, stride=2)

        b1 = F.interpolate(b1, size=p0.shape[-2:], mode="bilinear", align_corners=False)
        b2 = F.interpolate(b2, size=p0.shape[-2:], mode="bilinear", align_corners=False)
        b3 = F.interpolate(b3, size=p0.shape[-2:], mode="bilinear", align_corners=False)
        stacked = torch.cat([p0, b1, b2, b3], dim=1)
        weights = F.softmax(self.asff_weights(stacked), dim=1)
        fused = weights[:, 0:1] * p0 + weights[:, 1:2] * b1 + weights[:, 2:3] * b2 + weights[:, 3:4] * b3
        fused = self.aspp(fused)
        tokens = F.adaptive_avg_pool2d(fused, (self.token_grid, self.token_grid)).flatten(2).transpose(1, 2)
        return fused, tokens
