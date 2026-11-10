# NaVAR

Anonymous network architecture release for low-light image restoration on LOLv1
and LOLv2.

## Architecture

- Scene exposure and shared RGB calibration.
- Global restoration and a spatially aligned RGB anchor.
- Image-conditioned next-scale autoregressive transformer with residual adapters
  and cross attention.
- Prior pyramid: coarse-scale global FiLM and fine-scale spatial guidance.
- Native-resolution restoration with luminance/chroma stems and guidance at
  three resolutions.

The default architecture uses a transformer depth of 16, restoration width of 48,
64 prior channels, adapter rank of 32, and grids
`(1, 2, 3, 4, 5, 6, 8, 10, 13, 16)`. LOLv1 and LOLv2 share this model class.
The constructor also exposes the existing coarse-scale prior (`coarse_prior`)
and bottleneck channel-attention (`attn_blocks`) architecture options. For
example, `width=64, attn_blocks=2, coarse_prior=6` selects the wider restoration
network with six continuous coarse prior scales.

## Environment

Python 3.10 or later, PyTorch 2.2 or later, and NumPy 1.24 or later.

```bash
pip install -r requirements.txt
```

## Model definition

```python
from navar import NaVAR

model = NaVAR()
```

`NaVAR.forward(x, context=None, roi=None)` accepts an RGB tensor of shape
`(B, 3, H, W)` with values in `[0, 1]` and returns an RGB tensor of the same shape.
The output is not clipped. When omitted, the full-image context is resized to
`256 x 256` internally. An explicit context must have shape `(B, 3, 256, 256)`.
The optional `roi` has shape `(B, 4)` and contains normalized
`(top, left, bottom, right)` coordinates in the full-image context.

## Layout

```text
navar/
  model.py          # NaVAR and scene exposure
  blocks.py         # shared spatial blocks and feature alignment
  restoration.py    # global and native-resolution restoration
  prior.py          # conditioning, autoregression and prior pyramid
  backbone.py       # autoregressive transformer
  attention.py      # attention and adaptive normalization
  tokenizer.py      # image tokenizer architecture
  vae_blocks.py     # convolutional encoder and decoder
  quantizer.py      # codebook and multi-scale residual accumulation
```

Third-party component attribution is provided in `THIRD_PARTY_NOTICES.md`.
