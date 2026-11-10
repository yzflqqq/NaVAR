"""Exposure, spatial mixing, color transforms and aligned feature sampling."""

import math
import torch
from torch import nn
from torch.nn import functional as F


class LayerNorm2d(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class GatedResidualBlock(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.norm1, self.norm2 = (LayerNorm2d(channels), LayerNorm2d(channels))
        self.expand = nn.Conv2d(channels, channels * 2, 1)
        self.depthwise = nn.Conv2d(
            channels * 2, channels * 2, 3, padding=1, groups=channels * 2
        )
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), nn.Conv2d(channels, channels, 1)
        )
        self.project = nn.Conv2d(channels, channels, 1)
        self.ff1 = nn.Conv2d(channels, channels * 2, 1)
        self.ff2 = nn.Conv2d(channels, channels, 1)
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def _drop(self, x):
        return x

    def forward(self, x):
        a, b = self.depthwise(self.expand(self.norm1(x))).chunk(2, dim=1)
        mixed = a * b
        x = x + self._drop(self.beta * self.project(mixed * self.attention(mixed)))
        a, b = self.ff1(self.norm2(x)).chunk(2, dim=1)
        return x + self._drop(self.gamma * self.ff2(a * b))


class MultiPriorExposure(nn.Module):

    def __init__(self, width=32):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(12, width, 3, padding=1),
            GatedResidualBlock(width),
            GatedResidualBlock(width),
        )
        self.global_gain = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(width, width, 1),
            nn.GELU(),
            nn.Conv2d(width, 3, 1),
        )
        self.local_gain = nn.Conv2d(width, 3, 3, padding=1)
        nn.init.zeros_(self.global_gain[-1].weight)
        nn.init.constant_(self.global_gain[-1].bias, math.log(5.0))
        nn.init.zeros_(self.local_gain.weight)
        nn.init.zeros_(self.local_gain.bias)

    @staticmethod
    def priors(x):
        r, g, b = x.split(1, dim=1)
        maximum, minimum = (x.amax(1, keepdim=True), x.amin(1, keepdim=True))
        luminance = 0.2126 * r + 0.7152 * g + 0.0722 * b
        denom = (r + g + b).clamp_min(0.001)
        illumination = torch.cat(
            (
                luminance,
                maximum,
                (maximum + minimum) * 0.5,
                x.square().sum(1, keepdim=True).add(1e-06).sqrt(),
            ),
            dim=1,
        )
        chroma = torch.cat(
            (
                -0.168736 * r - 0.331264 * g + 0.5 * b,
                0.5 * r - 0.418688 * g - 0.081312 * b,
                r / denom,
                g / denom,
                (maximum - minimum) / maximum.clamp_min(0.001),
            ),
            dim=1,
        )
        return (torch.cat((x, illumination, chroma), dim=1), luminance)

    def forward(self, x, global_gain=None):
        priors, luminance = self.priors(x)
        features = self.features(priors)
        if global_gain is None:
            global_gain = self.global_gain(features)
        log_gain = (global_gain + 2 * self.local_gain(features).tanh()).clamp(-2, 4)
        enhanced = -torch.expm1(-x * log_gain.exp())
        return (enhanced, features, global_gain, luminance)


def sample_region(features, roi, output_size):
    batch = features.shape[0]
    if roi is None:
        return F.interpolate(
            features, size=output_size, mode="bilinear", align_corners=False
        )
    height, width = output_size
    top, left, bottom, right = roi.float().unbind(1)
    yy = (torch.arange(height, device=features.device).float() + 0.5) / height
    xx = (torch.arange(width, device=features.device).float() + 0.5) / width
    yy = top[:, None] + yy[None] * (bottom - top)[:, None]
    xx = left[:, None] + xx[None] * (right - left)[:, None]
    grid = (
        torch.stack(
            (
                xx[:, None].expand(batch, height, width),
                yy[:, :, None].expand(batch, height, width),
            ),
            dim=-1,
        )
        * 2
        - 1
    )
    return F.grid_sample(
        features.float(),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    ).to(features.dtype)


def rgb_to_yuv(rgb):
    r, g, b = rgb.float().split(1, dim=1)
    return torch.cat(
        (
            0.299 * r + 0.587 * g + 0.114 * b,
            -0.168736 * r - 0.331264 * g + 0.5 * b,
            0.5 * r - 0.418688 * g - 0.081312 * b,
        ),
        dim=1,
    )


def haar_high(luminance):
    h, w = luminance.shape[-2:]
    x = F.pad(luminance, (0, w % 2, 0, h % 2), mode="replicate")
    a, b, c, d = (
        x[..., 0::2, 0::2],
        x[..., 0::2, 1::2],
        x[..., 1::2, 0::2],
        x[..., 1::2, 1::2],
    )
    return torch.cat(
        ((a - b + c - d) * 0.5, (a + b - c - d) * 0.5, (a - b - c + d) * 0.5), dim=1
    )
