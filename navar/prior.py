"""Image conditioning and full-pyramid or coarse-scale prior features."""

import math
import torch
from torch import nn
from torch.nn import functional as F

from .blocks import GatedResidualBlock


class ConditionCrossAttention(nn.Module):

    def __init__(self, dim, condition_dim=128, heads=4):
        super().__init__()
        self.heads = heads
        self.norm = nn.LayerNorm(dim)
        self.q = nn.Linear(dim, condition_dim)
        self.kv = nn.Linear(condition_dim, condition_dim * 2)
        self.proj = nn.Linear(condition_dim, dim)
        self.gain = nn.Parameter(torch.tensor(0.1))

    def forward(self, x, condition):
        batch, length, _ = x.shape
        q = self.q(self.norm(x)).view(batch, length, self.heads, -1).transpose(1, 2)
        k, v = self.kv(condition).chunk(2, dim=-1)
        k = k.view(batch, -1, self.heads, q.shape[-1]).transpose(1, 2)
        v = v.view(batch, -1, self.heads, q.shape[-1]).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.transpose(1, 2).reshape(batch, length, -1)
        return x + self.gain.tanh() * self.proj(out)


class ConditionalVAR(nn.Module):

    def __init__(self, backbone, condition_channels=96, adapter_rank=32):
        super().__init__()
        self.backbone = backbone.requires_grad_(False)
        self.condition_proj = nn.Conv2d(condition_channels, 128, 1)
        self.condition_norm = nn.LayerNorm(128)
        self.aligned_proj = nn.Linear(128, backbone.C)
        self.global_proj = nn.Linear(128, backbone.C)
        self.cross = nn.ModuleDict(
            {
                str(i): ConditionCrossAttention(backbone.C)
                for i in range(3, backbone.depth, 4)
            }
        )
        self.adapters = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(backbone.C),
                    nn.Linear(backbone.C, adapter_rank),
                    nn.GELU(),
                    nn.Linear(adapter_rank, backbone.C),
                )
                for _ in range(backbone.depth)
            ]
        )
        for adapter in self.adapters:
            nn.init.normal_(adapter[-1].weight, std=0.001)
            nn.init.zeros_(adapter[-1].bias)
        nn.init.normal_(self.global_proj.weight, std=0.001)
        nn.init.zeros_(self.global_proj.bias)
        self.register_buffer(
            "condition_position", self._positions(16), persistent=False
        )

    @staticmethod
    def _positions(size):
        grid = torch.linspace(-1, 1, size)
        yy, xx = torch.meshgrid(grid, grid, indexing="ij")
        frequencies = torch.linspace(1, 8, 32) * math.pi
        xy = torch.stack((xx, yy), dim=-1).reshape(-1, 2, 1) * frequencies
        return torch.cat((xy.sin(), xy.cos()), dim=-1).reshape(1, size * size, 128)

    def condition(self, features):
        grid = self.condition_proj(features)
        tokens = grid.flatten(2).transpose(1, 2)
        tokens = self.condition_norm(tokens) + self.condition_position.to(tokens)
        base = self.backbone.class_emb.weight[self.backbone.num_classes]
        global_condition = base.unsqueeze(0) + self.global_proj(tokens.mean(dim=1))
        return (grid, tokens, global_condition)

    def _block(self, x, global_condition, tokens, bias, index):
        x = self.backbone.blocks[index](x, global_condition, bias)
        x = x + self.adapters[index](x)
        if str(index) in self.cross:
            x = self.cross[str(index)](x, tokens)
        return x

    def _aligned(self, grid, size):
        local = F.adaptive_avg_pool2d(grid, (size, size)).flatten(2).transpose(1, 2)
        return self.aligned_proj(local)


@torch.no_grad()
def rollout_prefix(
    cvar, grid, tokens, global_condition, soft_scales=0, temperature=1.0, max_scales=0
):
    var, indices, inputs = (cvar.backbone, [], [])
    count = max_scales or len(var.patch_nums)
    quant = var.vae_quant_proxy[0]
    batch = grid.shape[0]
    positions = var.lvl_embed(var.lvl_1L) + var.pos_1LC
    cond = var.shared_ada_lin(global_condition)
    next_tokens = global_condition[:, None] + var.pos_start
    f_hat = grid.new_zeros(
        batch, var.Cvae, var.patch_nums[-1], var.patch_nums[-1], dtype=torch.float32
    )
    offset = 0
    for block in var.blocks:
        block.attn.kv_caching(True)
    try:
        for stage, size in enumerate(var.patch_nums[:count]):
            length = size * size
            x = (
                next_tokens
                + positions[:, offset : offset + length]
                + cvar._aligned(grid, size)
            )
            for i in range(var.depth):
                x = cvar._block(x, cond, tokens, None, i)
            logits = var.get_logits(x, global_condition)
            ids = logits.argmax(dim=-1)
            if stage < soft_scales:
                probs = (logits.float() / temperature).softmax(-1)
                embeddings = probs @ quant.embedding.weight.float()
            else:
                embeddings = quant.embedding(ids)
            indices.append(ids)
            embeddings = embeddings.transpose(1, 2).reshape(batch, var.Cvae, size, size)
            f_hat, next_map = quant.get_next_autoregressive_input(
                stage, len(var.patch_nums), f_hat, embeddings
            )
            offset += length
            if stage + 1 < count:
                inputs.append(next_map.flatten(2).transpose(1, 2))
                next_tokens = var.word_embed(inputs[-1])
        prefix = (
            torch.cat(inputs, dim=1) if inputs else f_hat.new_zeros(batch, 0, var.Cvae)
        )
        return (f_hat, indices, prefix)
    finally:
        for block in var.blocks:
            block.attn.kv_caching(False)


def soft_coarse_latents(
    var, quant, sequence, global_condition, scales, temperature=1.0
):
    patch_nums = var.patch_nums
    counts = [p * p for p in patch_nums[:scales]]
    logits = var.get_logits(sequence[:, : sum(counts)], global_condition)
    with torch.autocast(sequence.device.type, enabled=False):
        size, last = (patch_nums[-1], len(patch_nums) - 1)
        f_hat = logits.new_zeros(logits.shape[0], var.Cvae, size, size)
        latents, offset = ([], 0)
        for stage, (p, n) in enumerate(zip(patch_nums[:scales], counts)):
            probs = (logits[:, offset : offset + n].float() / temperature).softmax(-1)
            embedding = (
                (probs @ quant.embedding.weight.float())
                .transpose(1, 2)
                .reshape(-1, var.Cvae, p, p)
            )
            if stage != last:
                embedding = F.interpolate(embedding, (size, size), mode="bicubic")
            f_hat = f_hat + quant.quant_resi[stage / last](embedding)
            latents.append(f_hat)
            offset += n
        return latents


class CoarsePrior(nn.Module):

    def __init__(self, patch_nums, dim=1024, latent=32, channels=64, film_dim=256):
        super().__init__()
        self.patch_nums = tuple(patch_nums)
        self.norm = nn.LayerNorm(dim)
        self.coarse_proj = nn.Linear(dim, 64)
        self.film = nn.Sequential(
            nn.Linear(64 * len(self.patch_nums) + 96, film_dim),
            nn.GELU(),
            nn.Linear(film_dim, film_dim),
        )
        self.hidden_proj = nn.Linear(dim, 32)
        self.spatial = nn.Sequential(
            nn.Conv2d(2 * latent + 32, channels, 3, padding=1),
            GatedResidualBlock(channels),
            GatedResidualBlock(channels),
        )

    def forward(self, sequence, coarse_latent, clean, condition_features):
        chunks = self.norm(sequence).split([p * p for p in self.patch_nums], dim=1)
        film = self.film(
            torch.cat(
                [self.coarse_proj(c).mean(1) for c in chunks]
                + [condition_features.mean((2, 3))],
                1,
            )
        )
        p, size = (self.patch_nums[-1], coarse_latent.shape[-1])
        hidden = self.hidden_proj(chunks[-1]).transpose(1, 2).reshape(-1, 32, p, p)
        hidden = F.interpolate(
            hidden, (size, size), mode="bilinear", align_corners=False
        )
        return (
            self.spatial(
                torch.cat(
                    (coarse_latent.to(hidden.dtype), hidden, clean.to(hidden.dtype)), 1
                )
            ),
            film,
        )


class PriorPyramid(nn.Module):

    def __init__(self, patch_nums, dim=1024, channels=64, film_dim=256, coarse_max=6):
        super().__init__()
        self.patch_nums = tuple(patch_nums)
        self.coarse = [i for (i, p) in enumerate(self.patch_nums) if p <= coarse_max]
        self.fine = [i for (i, p) in enumerate(self.patch_nums) if p > coarse_max]
        self.norm = nn.LayerNorm(dim)
        self.coarse_proj = nn.Linear(dim, 64)
        self.film = nn.Sequential(
            nn.Linear(64 * len(self.coarse) + 96, film_dim),
            nn.GELU(),
            nn.Linear(film_dim, film_dim),
        )
        self.fine_proj = nn.Linear(dim, 32)
        self.spatial = nn.Sequential(
            nn.Conv2d(32 * len(self.fine) + 64, channels, 3, padding=1),
            GatedResidualBlock(channels),
            GatedResidualBlock(channels),
        )

    def forward(self, sequence, refined, projected, condition_features):
        size = self.patch_nums[-1]
        chunks = self.norm(sequence).split([p * p for p in self.patch_nums], dim=1)
        pooled = [self.coarse_proj(chunks[i]).mean(1) for i in self.coarse]
        film = self.film(torch.cat(pooled + [condition_features.mean((2, 3))], 1))
        maps = []
        for i in self.fine:
            p = self.patch_nums[i]
            local = self.fine_proj(chunks[i]).transpose(1, 2).reshape(-1, 32, p, p)
            maps.append(
                F.interpolate(local, (size, size), mode="bilinear", align_corners=False)
            )
        return (self.spatial(torch.cat(maps + [refined, projected], 1)), film)


def prior_sequence(cvar, grid, tokens, global_condition, prefix, scales=0):
    """Read all hidden scales from the model-generated prefix."""
    var = cvar.backbone
    patch_nums = var.patch_nums[: scales or len(var.patch_nums)]
    length = sum(p * p for p in patch_nums)
    sequence = torch.cat(
        (
            global_condition[:, None] + var.pos_start,
            var.word_embed(prefix[:, : length - 1]),
        ),
        dim=1,
    )
    aligned = torch.cat([cvar._aligned(grid, p) for p in patch_nums], dim=1)
    sequence = (
        sequence
        + var.lvl_embed(var.lvl_1L[:, :length])
        + var.pos_1LC[:, :length]
        + aligned
    )
    cond = var.shared_ada_lin(global_condition)
    bias = var.attn_bias_for_masking[:, :, :length, :length].to(sequence.dtype)
    for i in range(var.depth):
        sequence = cvar._block(sequence, cond, tokens, bias, i)
    return sequence
