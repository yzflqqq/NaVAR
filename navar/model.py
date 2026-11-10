"""NaVAR: scene exposure, autoregressive priors and native-resolution restoration."""

import torch
from torch import nn
from torch.nn import functional as F

from .blocks import GatedResidualBlock, MultiPriorExposure, sample_region
from .tokenizer import VQVAE
from .backbone import VAR
from .prior import (
    ConditionalVAR,
    PriorPyramid,
    CoarsePrior,
    rollout_prefix,
    prior_sequence,
    soft_coarse_latents,
)
from .restoration import NativeRestorer, PyramidRestorer


class SceneExposure(nn.Module):

    def __init__(self, width=32):
        super().__init__()
        self.tone = MultiPriorExposure(width)
        self.affine = nn.Sequential(
            nn.AdaptiveAvgPool2d(32),
            nn.Conv2d(width, 64, 3, stride=2, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 96, 3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(96 * 16, 128),
            nn.GELU(),
            nn.Linear(128, 12),
        )
        nn.init.zeros_(self.affine[-1].weight)
        nn.init.zeros_(self.affine[-1].bias)

    def forward(self, x, gain=None, calibration=None):
        base, features, gain, _ = self.tone(x, gain)
        if calibration is None:
            calibration = self.affine(features).float()
        identity = torch.eye(3, device=x.device, dtype=torch.float32)[None]
        matrix = identity + 0.5 * calibration[:, :9].tanh().reshape(-1, 3, 3)
        bias = 0.1 * calibration[:, 9:].tanh()[..., None, None]
        with torch.autocast(x.device.type, enabled=False):
            base = torch.einsum("bij,bjhw->bihw", matrix, base.float()) + bias
        return (base, features, gain, calibration)


class NaVAR(nn.Module):
    """Network definition with randomly initialized parameters and no external assets.

    Input: floating point RGB tensor (B, 3, H, W), in [0, 1].
    Output: restored RGB tensor of the same shape, without output clipping.
    Context is an optional full-image tensor of shape (B, 3, 256, 256).
    ROI is an optional tensor (B, 4) of normalized top/left/bottom/right coordinates.
    """

    def __init__(
        self,
        depth=16,
        width=48,
        prior_channels=64,
        adapter_rank=32,
        attn_blocks=0,
        coarse_prior=0,
        soft_temperature=1.0,
        device="cpu",
    ):
        super().__init__()
        patch_nums = (1, 2, 3, 4, 5, 6, 8, 10, 13, 16)
        if not 0 <= coarse_prior <= len(patch_nums):
            raise ValueError("coarse_prior must be between 0 and 10.")
        if soft_temperature <= 0:
            raise ValueError("soft_temperature must be positive.")
        self.coarse_prior = coarse_prior
        self.soft_temperature = soft_temperature
        self.vae = VQVAE(
            vocab_size=4096,
            z_channels=32,
            ch=160,
            share_quant_resi=4,
            v_patch_nums=patch_nums,
        )
        backbone = VAR(
            vae_local=self.vae,
            num_classes=1000,
            depth=depth,
            embed_dim=depth * 64,
            num_heads=depth,
            attn_l2_norm=True,
            patch_nums=patch_nums,
            flash_if_available=False,
            fused_if_available=False,
        )
        self.var = ConditionalVAR(
            backbone, condition_channels=96, adapter_rank=adapter_rank
        )
        self.exposure = SceneExposure(32)
        self.global_restorer = NativeRestorer(min(width, 32))
        self.global_restorer.prior_proj.requires_grad_(False)
        self.detail = PyramidRestorer(width, prior_channels, attn_blocks=attn_blocks)
        if coarse_prior:
            self.prior = CoarsePrior(
                patch_nums[:coarse_prior], backbone.C, backbone.Cvae, prior_channels
            )
        else:
            self.hidden_proj = nn.Sequential(
                nn.Conv2d(backbone.C, 32, 1), nn.GroupNorm(8, 32)
            )
            self.latent_refiner = nn.Sequential(
                nn.Conv2d(96, 64, 3, padding=1),
                GatedResidualBlock(64),
                GatedResidualBlock(64),
                nn.Conv2d(64, 32, 3, padding=1),
            )
            nn.init.normal_(self.latent_refiner[-1].weight, std=1e-3)
            nn.init.zeros_(self.latent_refiner[-1].bias)
            self.prior = PriorPyramid(patch_nums, backbone.C, prior_channels)
        self.to(device)

    def _encode(self, x):
        return self.vae.quant_conv(self.vae.encoder(x * 2 - 1))

    def forward(self, x, context=None, roi=None):
        if context is None:
            context = F.interpolate(
                x, (256, 256), mode="bilinear", align_corners=False, antialias=True
            )
        if context.shape[-2:] != (256, 256):
            raise ValueError("VAE/VAR context must be 256x256.")
        global_base, features, gain, calibration = self.exposure(context)
        base, _, _, _ = self.exposure(x, gain, calibration)
        global_clean = self.global_restorer(context, global_base)
        delta = F.adaptive_avg_pool2d(global_clean - global_base, (16, 16))
        anchor = base + sample_region(delta, roi, x.shape[-2:])
        with torch.no_grad():
            original = self._encode(context)
            clean = self._encode(global_clean.detach().clamp(0, 1))
            condition_features = torch.cat(
                (original, clean, F.adaptive_avg_pool2d(features.detach(), (16, 16))), 1
            )
        grid, tokens, condition = self.var.condition(condition_features)
        generated, _, prefix = rollout_prefix(
            self.var,
            grid,
            tokens,
            condition,
            soft_scales=self.coarse_prior,
            temperature=self.soft_temperature,
            max_scales=self.coarse_prior,
        )
        sequence = prior_sequence(
            self.var, grid, tokens, condition, prefix, scales=self.coarse_prior
        )
        if self.coarse_prior:
            latents = soft_coarse_latents(
                self.var.backbone,
                self.vae.quantize,
                sequence,
                condition,
                self.coarse_prior,
                self.soft_temperature,
            )
            spatial, film = self.prior(sequence, latents[-1], clean, condition_features)
        else:
            size = self.var.backbone.patch_nums[-1]
            hidden = (
                sequence[:, -size * size :]
                .transpose(1, 2)
                .reshape(-1, sequence.shape[-1], size, size)
            )
            projected = self.hidden_proj(hidden)
            refined = clean + self.latent_refiner(
                torch.cat((clean, generated, projected), 1)
            )
            spatial, film = self.prior(sequence, refined, projected, condition_features)
        guidance = sample_region(
            spatial, roi, ((x.shape[-2] + 3) // 4, (x.shape[-1] + 3) // 4)
        )
        return self.detail(x, anchor, guidance, film)
