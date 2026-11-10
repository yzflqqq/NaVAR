"""Global restoration and luminance/chroma restoration with pyramid guidance."""

import torch
from torch import nn
from torch.nn import functional as F

from .blocks import GatedResidualBlock, LayerNorm2d, rgb_to_yuv, haar_high


class NativeRestorer(nn.Module):

    def __init__(self, width=32):
        super().__init__()
        self.intro = nn.Conv2d(12, width, 3, padding=1)
        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        for level, count in enumerate((2, 2, 3)):
            channels = width * 2**level
            self.encoders.append(
                nn.Sequential(*[GatedResidualBlock(channels) for _ in range(count)])
            )
            self.downs.append(nn.Conv2d(channels, channels * 2, 2, stride=2))
        self.middle = nn.Sequential(*[GatedResidualBlock(width * 8) for _ in range(4)])
        self.prior_proj = nn.Conv2d(16, width * 8, 1)
        nn.init.normal_(self.prior_proj.weight, std=0.001)
        nn.init.zeros_(self.prior_proj.bias)
        self.ups, self.decoders = (nn.ModuleList(), nn.ModuleList())
        for level, count in zip((2, 1, 0), (3, 2, 2)):
            channels = width * 2**level
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(channels * 2, channels * 4, 1), nn.PixelShuffle(2)
                )
            )
            self.decoders.append(
                nn.Sequential(*[GatedResidualBlock(channels) for _ in range(count)])
            )
        self.out = nn.Conv2d(width, 3, 3, padding=1)
        nn.init.normal_(self.out.weight, std=0.001)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, base, guidance=None):
        h, w = x.shape[-2:]
        yuv = rgb_to_yuv(base)
        high = F.interpolate(haar_high(yuv[:, :1]), (h, w), mode="nearest")
        inputs = torch.cat((x, base, yuv, high), dim=1)
        inputs = F.pad(inputs, (0, -w % 8, 0, -h % 8), mode="replicate")
        feature, skips = (self.intro(inputs), [])
        for encoder, down in zip(self.encoders, self.downs):
            feature = encoder(feature)
            skips.append(feature)
            feature = down(feature)
        if guidance is not None:
            feature = feature + self.prior_proj(
                F.interpolate(
                    guidance, feature.shape[-2:], mode="bilinear", align_corners=False
                )
            )
        feature = self.middle(feature)
        for up, decoder, skip in zip(self.ups, self.decoders, reversed(skips)):
            feature = decoder(up(feature) + skip)
        return base + self.out(feature)[..., :h, :w]


class ChannelAttention(nn.Module):

    def __init__(self, channels, heads=8):
        super().__init__()
        self.heads = heads
        self.norm = LayerNorm2d(channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.depthwise = nn.Conv2d(
            channels * 3, channels * 3, 3, padding=1, groups=channels * 3
        )
        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))
        self.project = nn.Conv2d(channels, channels, 1)
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        b, c, h, w = x.shape
        q, k, v = self.depthwise(self.qkv(self.norm(x))).chunk(3, dim=1)
        q, k, v = (t.reshape(b, self.heads, c // self.heads, h * w) for t in (q, k, v))
        with torch.autocast(x.device.type, enabled=False):
            q, k = (F.normalize(q.float(), dim=-1), F.normalize(k.float(), dim=-1))
            attention = (q @ k.transpose(-2, -1) * self.temperature).softmax(-1)
            out = (attention @ v.float()).reshape(b, c, h, w)
        out = self.project(out.to(x.dtype))
        return x + self.gamma * out


class PyramidRestorer(NativeRestorer):

    def __init__(self, width=48, prior_channels=64, film_dim=256, attn_blocks=0):
        super().__init__(width)
        del self.intro, self.prior_proj
        self.luma = nn.Sequential(
            nn.Conv2d(5, width, 3, padding=1), GatedResidualBlock(width)
        )
        self.chroma = nn.Sequential(
            nn.Conv2d(4, width, 3, padding=1), GatedResidualBlock(width)
        )
        self.fuse = nn.Conv2d(width * 2, width, 1)
        blocks = list(self.middle)
        if not 0 <= attn_blocks <= len(blocks):
            raise ValueError(f"attn_blocks must be in [0, {len(blocks)}]")
        if attn_blocks:
            layers = []
            for i, block in enumerate(blocks):
                layers.append(block)
                if i < attn_blocks:
                    layers.append(ChannelAttention(width * 8))
            self.middle = nn.Sequential(*layers)
        levels = (width * 8, width * 4, width * 2)
        self.prior = nn.ModuleList((nn.Conv2d(prior_channels, c, 1) for c in levels))
        self.film = nn.ModuleList((nn.Linear(film_dim, 2 * c) for c in levels))
        for layer in [*self.prior, *self.film]:
            nn.init.normal_(layer.weight, std=0.001)
            nn.init.zeros_(layer.bias)

    def _inject(self, feature, level, guidance, film):
        if guidance is not None:
            feature = feature + self.prior[level](
                F.interpolate(
                    guidance, feature.shape[-2:], mode="bilinear", align_corners=False
                )
            )
        if film is not None:
            scale, shift = self.film[level](film)[..., None, None].chunk(2, dim=1)
            feature = feature * (1 + scale) + shift
        return feature

    def forward(self, x, base, guidance=None, film=None):
        h, w = x.shape[-2:]
        x = F.pad(x, (0, -w % 8, 0, -h % 8), mode="replicate")
        base_pad = F.pad(base, (0, -w % 8, 0, -h % 8), mode="replicate")
        source, clean = (rgb_to_yuv(x), rgb_to_yuv(base_pad))
        high = F.interpolate(
            haar_high(clean[:, :1]), base_pad.shape[-2:], mode="nearest"
        )
        luma = self.luma(torch.cat((source[:, :1], clean[:, :1], high), 1))
        chroma = self.chroma(torch.cat((source[:, 1:], clean[:, 1:]), 1))
        feature, skips = (self.fuse(torch.cat((luma, chroma), 1)), [])
        for encoder, down in zip(self.encoders, self.downs):
            feature = encoder(feature)
            skips.append(feature)
            feature = down(feature)
        feature = self.middle(self._inject(feature, 0, guidance, film))
        for i, (up, decoder, skip) in enumerate(
            zip(self.ups, self.decoders, reversed(skips))
        ):
            feature = up(feature) + skip
            if i < 2:
                feature = self._inject(feature, i + 1, guidance, film)
            feature = decoder(feature)
        return base + self.out(feature)[..., :h, :w]
