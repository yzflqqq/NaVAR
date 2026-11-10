"""Image tokenizer architecture, including the original encoder and decoder."""

import torch
from torch import nn
from torch.nn import functional as F

from .vae_blocks import Encoder, Decoder
from .quantizer import VectorQuantizer2


class VQVAE(nn.Module):

    def __init__(
        self,
        vocab_size=4096,
        z_channels=32,
        ch=128,
        dropout=0.0,
        using_znorm=False,
        quant_conv_ks=3,
        quant_resi=0.5,
        share_quant_resi=4,
        default_qresi_counts=0,
        v_patch_nums=(1, 2, 3, 4, 5, 6, 8, 10, 13, 16),
        freeze=True,
    ):
        super().__init__()
        self.freeze = freeze
        self.V, self.Cvae = (vocab_size, z_channels)
        ddconfig = dict(
            dropout=dropout,
            ch=ch,
            z_channels=z_channels,
            in_channels=3,
            ch_mult=(1, 1, 2, 2, 4),
            num_res_blocks=2,
            using_sa=True,
            using_mid_sa=True,
        )
        ddconfig.pop("double_z", None)
        self.encoder = Encoder(double_z=False, **ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.vocab_size = vocab_size
        self.downsample = 2 ** (len(ddconfig["ch_mult"]) - 1)
        self.quantize: VectorQuantizer2 = VectorQuantizer2(
            vocab_size=vocab_size,
            Cvae=self.Cvae,
            using_znorm=using_znorm,
            default_qresi_counts=default_qresi_counts,
            v_patch_nums=v_patch_nums,
            quant_resi=quant_resi,
            share_quant_resi=share_quant_resi,
        )
        self.quant_conv = torch.nn.Conv2d(
            self.Cvae, self.Cvae, quant_conv_ks, stride=1, padding=quant_conv_ks // 2
        )
        self.post_quant_conv = torch.nn.Conv2d(
            self.Cvae, self.Cvae, quant_conv_ks, stride=1, padding=quant_conv_ks // 2
        )
        if freeze:
            self.eval()
            [p.requires_grad_(False) for p in self.parameters()]

    def fhat_to_img(self, f_hat: torch.Tensor):
        return self.decoder(self.post_quant_conv(f_hat)).clamp_(-1, 1)
