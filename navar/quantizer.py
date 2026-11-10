"""Multi-scale codebook embeddings and residual accumulation."""

from typing import List, Tuple, Optional
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class VectorQuantizer2(nn.Module):

    def __init__(
        self,
        vocab_size,
        Cvae,
        using_znorm,
        default_qresi_counts=0,
        v_patch_nums=None,
        quant_resi=0.5,
        share_quant_resi=4,
    ):
        super().__init__()
        self.vocab_size: int = vocab_size
        self.Cvae: int = Cvae
        self.using_znorm: bool = using_znorm
        self.v_patch_nums: Tuple[int] = v_patch_nums
        self.quant_resi_ratio = quant_resi
        if share_quant_resi == 0:
            self.quant_resi = PhiNonShared(
                [
                    Phi(Cvae, quant_resi) if abs(quant_resi) > 1e-06 else nn.Identity()
                    for _ in range(default_qresi_counts or len(self.v_patch_nums))
                ]
            )
        elif share_quant_resi == 1:
            self.quant_resi = PhiShared(
                Phi(Cvae, quant_resi) if abs(quant_resi) > 1e-06 else nn.Identity()
            )
        else:
            self.quant_resi = PhiPartiallyShared(
                nn.ModuleList(
                    [
                        (
                            Phi(Cvae, quant_resi)
                            if abs(quant_resi) > 1e-06
                            else nn.Identity()
                        )
                        for _ in range(share_quant_resi)
                    ]
                )
            )
        self.embedding = nn.Embedding(self.vocab_size, self.Cvae)

    def get_next_autoregressive_input(
        self, si: int, SN: int, f_hat: torch.Tensor, h_BChw: torch.Tensor
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        HW = self.v_patch_nums[-1]
        if si != SN - 1:
            h = self.quant_resi[si / (SN - 1)](
                F.interpolate(h_BChw, size=(HW, HW), mode="bicubic")
            )
            f_hat.add_(h)
            return (
                f_hat,
                F.interpolate(
                    f_hat,
                    size=(self.v_patch_nums[si + 1], self.v_patch_nums[si + 1]),
                    mode="area",
                ),
            )
        else:
            h = self.quant_resi[si / (SN - 1)](h_BChw)
            f_hat.add_(h)
            return (f_hat, f_hat)


class Phi(nn.Conv2d):

    def __init__(self, embed_dim, quant_resi):
        ks = 3
        super().__init__(
            in_channels=embed_dim,
            out_channels=embed_dim,
            kernel_size=ks,
            stride=1,
            padding=ks // 2,
        )
        self.resi_ratio = abs(quant_resi)

    def forward(self, h_BChw):
        return h_BChw.mul(1 - self.resi_ratio) + super().forward(h_BChw).mul_(
            self.resi_ratio
        )


class PhiShared(nn.Module):

    def __init__(self, qresi: Phi):
        super().__init__()
        self.qresi: Phi = qresi

    def __getitem__(self, _) -> Phi:
        return self.qresi


class PhiPartiallyShared(nn.Module):

    def __init__(self, qresi_ls: nn.ModuleList):
        super().__init__()
        self.qresi_ls = qresi_ls
        K = len(qresi_ls)
        self.ticks = (
            np.linspace(1 / 3 / K, 1 - 1 / 3 / K, K)
            if K == 4
            else np.linspace(1 / 2 / K, 1 - 1 / 2 / K, K)
        )

    def __getitem__(self, at_from_0_to_1: float) -> Phi:
        return self.qresi_ls[np.argmin(np.abs(self.ticks - at_from_0_to_1)).item()]

    def extra_repr(self) -> str:
        return f"ticks={self.ticks}"


class PhiNonShared(nn.ModuleList):

    def __init__(self, qresi: List):
        super().__init__(qresi)
        K = len(qresi)
        self.ticks = (
            np.linspace(1 / 3 / K, 1 - 1 / 3 / K, K)
            if K == 4
            else np.linspace(1 / 2 / K, 1 - 1 / 2 / K, K)
        )

    def __getitem__(self, at_from_0_to_1: float) -> Phi:
        return super().__getitem__(
            np.argmin(np.abs(self.ticks - at_from_0_to_1)).item()
        )

    def extra_repr(self) -> str:
        return f"ticks={self.ticks}"
