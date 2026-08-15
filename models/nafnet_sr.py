"""NAFNet-SR: NAFNet U-Net backbone + PixelShuffle x2 head.

Faithful to the NAFNet paper (Chen et al., ECCV 2022) with one addition: the
network runs entirely at LR resolution and upsamples once at the very end, so
the x2 output costs almost nothing extra over pure denoising.

Design notes worth knowing before you tune it:

* No activation functions anywhere. SimpleGate (split channels, multiply)
  replaces GELU/ReLU. Fewer ops and it exports cleanly to ONNX/TensorRT.
* LayerNorm is applied per pixel across channels. This is the reason the model
  tolerates global intensity and contrast shifts, which is exactly the failure
  mode you face on out-of-distribution inspection images.
* The global bicubic skip means the network only has to learn the residual.
  Convergence is much faster and early training never produces garbage.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    """LayerNorm over the channel dimension only, per spatial location."""

    def __init__(self, channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Computed in fp32 even under autocast. The variance and its backward
        # are the classic fp16 failure point in NAFNet-family models; forcing
        # fp32 here costs almost nothing (no matmuls) and removes the risk.
        in_dtype = x.dtype
        with torch.autocast(device_type=x.device.type, enabled=False):
            xf = x.float()
            mu = xf.mean(1, keepdim=True)
            var = (xf - mu).pow(2).mean(1, keepdim=True)
            xf = (xf - mu) * torch.rsqrt(var + self.eps)
            xf = (xf * self.weight[None, :, None, None].float()
                  + self.bias[None, :, None, None].float())
        return xf.to(in_dtype)


class SimpleGate(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = x.chunk(2, dim=1)
        return a * b


class NAFBlock(nn.Module):
    def __init__(self, c: int, dw_expand: int = 2, ffn_expand: int = 2, drop: float = 0.0):
        super().__init__()
        dw = c * dw_expand
        self.norm1 = LayerNorm2d(c)
        self.conv1 = nn.Conv2d(c, dw, 1)
        self.conv2 = nn.Conv2d(dw, dw, 3, padding=1, groups=dw)
        self.conv3 = nn.Conv2d(dw // 2, c, 1)
        # Simplified Channel Attention: global pool + 1x1, no nonlinearity.
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(dw // 2, dw // 2, 1))

        ffn = c * ffn_expand
        self.norm2 = LayerNorm2d(c)
        self.conv4 = nn.Conv2d(c, ffn, 1)
        self.conv5 = nn.Conv2d(ffn // 2, c, 1)

        self.sg = SimpleGate()
        self.drop1 = nn.Dropout2d(drop) if drop > 0 else nn.Identity()
        self.drop2 = nn.Dropout2d(drop) if drop > 0 else nn.Identity()
        # Zero-init residual scales: every block starts as an identity map.
        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        x = self.conv2(self.conv1(self.norm1(inp)))
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.drop1(self.conv3(x))
        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.drop2(self.conv5(x))
        return y + x * self.gamma


class NAFNetSR(nn.Module):
    def __init__(
        self,
        in_ch: int = 1,
        out_ch: int = 1,
        width: int = 32,
        enc_blks: tuple[int, ...] = (2, 2, 4, 8),
        middle_blks: int = 12,
        dec_blks: tuple[int, ...] = (2, 2, 2, 2),
        scale: int = 2,
        drop: float = 0.0,
        global_skip: str = "bicubic",
    ):
        super().__init__()
        assert len(enc_blks) == len(dec_blks), "encoder and decoder depth must match"
        self.scale = scale
        self.global_skip = global_skip
        self.padder = 2 ** len(enc_blks)

        self.intro = nn.Conv2d(in_ch, width, 3, padding=1)

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        chan = width
        for n in enc_blks:
            self.encoders.append(nn.Sequential(*[NAFBlock(chan, drop=drop) for _ in range(n)]))
            self.downs.append(nn.Conv2d(chan, chan * 2, 2, 2))
            chan *= 2

        self.middle = nn.Sequential(*[NAFBlock(chan, drop=drop) for _ in range(middle_blks)])

        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        for n in dec_blks:
            self.ups.append(nn.Sequential(nn.Conv2d(chan, chan * 2, 1, bias=False),
                                          nn.PixelShuffle(2)))
            chan //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(chan, drop=drop) for _ in range(n)]))

        # x2 head. PixelShuffle is used rather than transposed conv because it
        # cannot produce checkerboard artifacts, which would read as false
        # texture on an inspection image.
        self.upsampler = nn.Sequential(
            nn.Conv2d(width, width * (scale ** 2), 3, padding=1),
            nn.PixelShuffle(scale),
        )
        self.tail = nn.Conv2d(width, out_ch, 3, padding=1)
        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)

    # ------------------------------------------------------------------
    def _pad(self, x: torch.Tensor):
        _, _, h, w = x.shape
        m = self.padder
        ph, pw = (m - h % m) % m, (m - w % m) % m
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="reflect")
        return x, h, w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.global_skip == "none":
            base = 0.0
        else:
            base = F.interpolate(x, scale_factor=self.scale,
                                 mode=self.global_skip, align_corners=False)

        y, h, w = self._pad(x)
        y = self.intro(y)

        skips = []
        for enc, down in zip(self.encoders, self.downs):
            y = enc(y)
            skips.append(y)
            y = down(y)

        y = self.middle(y)

        for dec, up, skip in zip(self.decoders, self.ups, skips[::-1]):
            y = up(y)
            y = y + skip
            y = dec(y)

        y = self.tail(self.upsampler(y))
        y = y[:, :, : h * self.scale, : w * self.scale]
        return y + base


# ----------------------------------------------------------------------
# Presets. Start at "base"; drop to "small" only if the timing benchmark
# forces it, and go to "large" only if you have budget left over.
# ----------------------------------------------------------------------
PRESETS = {
    # name      params (measured)
    "small":  dict(width=16, enc_blks=(1, 1, 2, 4), middle_blks=6,  dec_blks=(1, 1, 1, 1)),  # 3.9M
    "medium": dict(width=32, enc_blks=(1, 1, 2, 4), middle_blks=8,  dec_blks=(1, 1, 1, 1)),  # 15.5M
    "base":   dict(width=32, enc_blks=(2, 2, 4, 8), middle_blks=12, dec_blks=(2, 2, 2, 2)),  # 29.2M
    "large":  dict(width=64, enc_blks=(2, 2, 4, 8), middle_blks=12, dec_blks=(2, 2, 2, 2)),  # 116.1M
}


def build_model(preset: str = "base", **overrides) -> NAFNetSR:
    cfg = dict(PRESETS[preset])
    cfg.update(overrides)
    return NAFNetSR(**cfg)
