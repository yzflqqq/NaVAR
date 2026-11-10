"""Next-scale autoregressive transformer architecture."""

import math
from functools import partial
from typing import Optional, Tuple, Union
import torch
from torch import nn
from torch.nn import functional as F

from .attention import AdaLNBeforeHead, AdaLNSelfAttn
from .tokenizer import VQVAE
from .quantizer import VectorQuantizer2


class SharedAdaLin(nn.Linear):

    def forward(self, cond_BD):
        C = self.weight.shape[0] // 6
        return super().forward(cond_BD).view(-1, 1, 6, C)


class VAR(nn.Module):

    def __init__(
        self,
        vae_local: VQVAE,
        num_classes=1000,
        depth=16,
        embed_dim=1024,
        num_heads=16,
        mlp_ratio=4.0,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_eps=1e-06,
        shared_aln=False,
        cond_drop_rate=0.1,
        attn_l2_norm=False,
        patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
        flash_if_available=True,
        fused_if_available=True,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0
        self.Cvae, self.V = (vae_local.Cvae, vae_local.vocab_size)
        self.depth, self.C, self.D, self.num_heads = (
            depth,
            embed_dim,
            embed_dim,
            num_heads,
        )
        self.cond_drop_rate = cond_drop_rate
        self.patch_nums: Tuple[int] = patch_nums
        self.L = sum((pn**2 for pn in self.patch_nums))
        self.first_l = self.patch_nums[0] ** 2
        self.begin_ends = []
        cur = 0
        for i, pn in enumerate(self.patch_nums):
            self.begin_ends.append((cur, cur + pn**2))
            cur += pn**2
        self.num_stages_minus_1 = len(self.patch_nums) - 1
        quant: VectorQuantizer2 = vae_local.quantize
        self.vae_proxy: Tuple[VQVAE] = (vae_local,)
        self.vae_quant_proxy: Tuple[VectorQuantizer2] = (quant,)
        self.word_embed = nn.Linear(self.Cvae, self.C)
        init_std = math.sqrt(1 / self.C / 3)
        self.num_classes = num_classes
        self.uniform_prob = torch.full(
            (1, num_classes),
            fill_value=1.0 / num_classes,
            dtype=torch.float32,
            device="cpu",
        )
        self.class_emb = nn.Embedding(self.num_classes + 1, self.C)
        nn.init.trunc_normal_(self.class_emb.weight.data, mean=0, std=init_std)
        self.pos_start = nn.Parameter(torch.empty(1, self.first_l, self.C))
        nn.init.trunc_normal_(self.pos_start.data, mean=0, std=init_std)
        pos_1LC = []
        for i, pn in enumerate(self.patch_nums):
            pe = torch.empty(1, pn * pn, self.C)
            nn.init.trunc_normal_(pe, mean=0, std=init_std)
            pos_1LC.append(pe)
        pos_1LC = torch.cat(pos_1LC, dim=1)
        assert tuple(pos_1LC.shape) == (1, self.L, self.C)
        self.pos_1LC = nn.Parameter(pos_1LC)
        self.lvl_embed = nn.Embedding(len(self.patch_nums), self.C)
        nn.init.trunc_normal_(self.lvl_embed.weight.data, mean=0, std=init_std)
        self.shared_ada_lin = (
            nn.Sequential(nn.SiLU(inplace=False), SharedAdaLin(self.D, 6 * self.C))
            if shared_aln
            else nn.Identity()
        )
        norm_layer = partial(nn.LayerNorm, eps=norm_eps)
        self.drop_path_rate = drop_path_rate
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                AdaLNSelfAttn(
                    cond_dim=self.D,
                    shared_aln=shared_aln,
                    block_idx=block_idx,
                    embed_dim=self.C,
                    norm_layer=norm_layer,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[block_idx],
                    last_drop_p=0 if block_idx == 0 else dpr[block_idx - 1],
                    attn_l2_norm=attn_l2_norm,
                    flash_if_available=flash_if_available,
                    fused_if_available=fused_if_available,
                )
                for block_idx in range(depth)
            ]
        )
        fused_add_norm_fns = [b.fused_add_norm_fn is not None for b in self.blocks]
        self.using_fused_add_norm_fn = any(fused_add_norm_fns)
        d: torch.Tensor = torch.cat(
            [torch.full((pn * pn,), i) for (i, pn) in enumerate(self.patch_nums)]
        ).view(1, self.L, 1)
        dT = d.transpose(1, 2)
        lvl_1L = dT[:, 0].contiguous()
        self.register_buffer("lvl_1L", lvl_1L)
        attn_bias_for_masking = torch.where(d >= dT, 0.0, -torch.inf).reshape(
            1, 1, self.L, self.L
        )
        self.register_buffer(
            "attn_bias_for_masking", attn_bias_for_masking.contiguous()
        )
        self.head_nm = AdaLNBeforeHead(self.C, self.D, norm_layer=norm_layer)
        self.head = nn.Linear(self.C, self.V)

    def get_logits(
        self,
        h_or_h_and_residual: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
        cond_BD: Optional[torch.Tensor],
    ):
        if not isinstance(h_or_h_and_residual, torch.Tensor):
            h, resi = h_or_h_and_residual
            h = resi + self.blocks[-1].drop_path(h)
        else:
            h = h_or_h_and_residual
        return self.head(self.head_nm(h.float(), cond_BD).float()).float()

    def extra_repr(self):
        return f"drop_path_rate={self.drop_path_rate:g}"
