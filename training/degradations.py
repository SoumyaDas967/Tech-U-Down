"""Synthetic degradation, applied on the fly to ground-truth images.

This is the single biggest lever on the out-of-distribution half of the test
set, and it costs zero inference time. The provided pairs teach the model one
exact degradation; this teaches it a FAMILY of degradations, so an unseen tool
or structure type lands inside the training distribution instead of outside it.

Everything here operates on torch tensors shaped (1, H, W), float, in [0, 1]
for the clean input. Outputs are deliberately NOT clipped: speckle must be
allowed to push values outside [0, 1], exactly as the real data does.
"""
from __future__ import annotations

import random

import torch
import torch.nn.functional as F


def _gaussian_kernel1d(sigma: float, device) -> torch.Tensor:
    radius = max(1, int(3 * sigma + 0.5))
    x = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
    k = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    return k / k.sum()


def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur. x: (C, H, W)."""
    if sigma <= 1e-3:
        return x
    k = _gaussian_kernel1d(sigma, x.device)
    r = (k.numel() - 1) // 2
    c = x.shape[0]
    x = x.unsqueeze(0)
    x = F.pad(x, (r, r, r, r), mode="reflect")
    x = F.conv2d(x, k.view(1, 1, 1, -1).expand(c, 1, 1, -1), groups=c)
    x = F.conv2d(x, k.view(1, 1, -1, 1).expand(c, 1, -1, 1), groups=c)
    return x.squeeze(0)


def resize_half(x: torch.Tensor, mode: str) -> torch.Tensor:
    x = x.unsqueeze(0)
    if mode == "area":
        y = F.avg_pool2d(x, 2)
    elif mode == "nearest":
        y = F.interpolate(x, scale_factor=0.5, mode="nearest")
    else:
        y = F.interpolate(x, scale_factor=0.5, mode=mode, align_corners=False)
    return y.squeeze(0)


def add_speckle(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Multiplicative speckle: y = x * (1 + n), n ~ N(0, sigma^2).

    Note this naturally pushes bright pixels further out of range than dark
    ones, which is precisely the asymmetry the audit script measures.
    """
    return x * (1.0 + torch.randn_like(x) * sigma)


def add_gaussian(x: torch.Tensor, sigma: float) -> torch.Tensor:
    return x + torch.randn_like(x) * sigma


def add_poisson(x: torch.Tensor, peak: float) -> torch.Tensor:
    """Shot noise, the physically correct model for electron-count imaging."""
    lam = torch.clamp(x, min=0.0) * peak
    return torch.poisson(lam) / peak


class SyntheticDegrader:
    """Randomised HR -> LR degradation.

    These defaults are CALIBRATED to the KLA PS01 training set, not generic.
    They were set by measuring the real degradation with preflight.py and
    matching it: real residual std 0.0792 vs synthetic 0.0837 (1.06x), real
    multiplicative ratio 0.848 vs 0.867, and 2.67% vs 2.94% of pixels outside
    [0, 1]. The ~6% margin is deliberate -- training on a slightly wider family
    than you measure is what buys out-of-distribution robustness.

    On a different dataset these numbers are wrong. Re-run preflight.py and
    rescale speckle_sigma by (target residual std / measured residual std);
    the relationship is close to linear once speckle dominates.
    """

    def __init__(
        self,
        blur_sigma=(0.0, 0.7),
        speckle_sigma=(0.0, 0.40),
        gauss_sigma=(0.0, 0.03),
        poisson_peak=(60.0, 500.0),
        p_speckle=1.0,
        p_gauss=0.25,
        p_poisson=0.15,
        p_blur=0.5,
        p_noise_first=0.3,
        gamma=(0.9, 1.12),
        p_gamma=0.25,
        contrast=(0.9, 1.1),
        p_contrast=0.25,
        modes=("bicubic", "bilinear", "area"),
        mode_weights=(0.75, 0.15, 0.10),):
        self.blur_sigma = blur_sigma
        self.speckle_sigma = speckle_sigma
        self.gauss_sigma = gauss_sigma
        self.poisson_peak = poisson_peak
        self.p_speckle = p_speckle
        self.p_gauss = p_gauss
        self.p_poisson = p_poisson
        self.p_blur = p_blur
        self.p_noise_first = p_noise_first
        self.gamma = gamma
        self.p_gamma = p_gamma
        self.contrast = contrast
        self.p_contrast = p_contrast
        self.modes = list(modes)
        self.mode_weights = list(mode_weights)

    @staticmethod
    def _u(rng: random.Random, lo_hi) -> float:
        return rng.uniform(*lo_hi)

    def _noise(self, x: torch.Tensor, rng: random.Random) -> torch.Tensor:
        if rng.random() < self.p_poisson:
            x = add_poisson(x, self._u(rng, self.poisson_peak))
        if rng.random() < self.p_speckle:
            x = add_speckle(x, self._u(rng, self.speckle_sigma))
        if rng.random() < self.p_gauss:
            x = add_gaussian(x, self._u(rng, self.gauss_sigma))
        return x

    def __call__(self, hr: torch.Tensor, rng: random.Random | None = None) -> torch.Tensor:
        """hr: (1, H, W) in [0, 1] -> lr: (1, H/2, W/2), NOT clipped."""
        rng = rng or random
        x = hr

        # Photometric jitter widens the appearance distribution across tools.
        if rng.random() < self.p_gamma:
            g = self._u(rng, self.gamma)
            x = torch.clamp(x, min=1e-6) ** g
        if rng.random() < self.p_contrast:
            c = self._u(rng, self.contrast)
            m = x.mean()
            x = (x - m) * c + m

        noise_first = rng.random() < self.p_noise_first
        if noise_first:
            # Sensor noise before optical downsampling: partially averaged away.
            x = self._noise(x, rng)

        if rng.random() < self.p_blur:
            x = gaussian_blur(x, self._u(rng, self.blur_sigma))

        mode = rng.choices(self.modes, weights=self.mode_weights, k=1)[0]
        x = resize_half(x, mode)

        if not noise_first:
            x = self._noise(x, rng)

        return x
