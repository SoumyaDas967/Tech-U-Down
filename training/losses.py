"""Losses.

Default recipe: Charbonnier + 0.2 * (1 - SSIM).

Charbonnier is a smooth L1. Use it rather than L2/MSE: L2 optimises PSNR
directly but its optimum is the conditional mean, which is a blurred image.
On inspection data that blur is exactly what you are being paid to avoid, and
in practice L1-family losses end up winning on PSNR too.

LPIPS is available but should stay at a small weight and only in a short
fine-tuning phase. It is a natural-image perceptual metric; pushed hard on
grayscale SEM-like data it will invent texture, and invented texture on an
inspection image is a false defect call.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CharbonnierLoss(nn.Module):
    def __init__(self, eps: float = 1e-3):
        super().__init__()
        self.eps2 = eps * eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.sqrt((pred - target) ** 2 + self.eps2).mean()


def _gauss_window(size: int, sigma: float, channels: int, device, dtype):
    coords = torch.arange(size, dtype=dtype, device=device) - (size - 1) / 2.0
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = (g / g.sum()).unsqueeze(0)
    w2 = (g.t() @ g).unsqueeze(0).unsqueeze(0)
    return w2.expand(channels, 1, size, size).contiguous()


def ssim(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0,
         win_size: int = 11, sigma: float = 1.5, size_average: bool = True):
    """Standard Wang et al. SSIM with a Gaussian window. Inputs (N, C, H, W)."""
    c = pred.shape[1]
    win = _gauss_window(win_size, sigma, c, pred.device, pred.dtype)
    pad = win_size // 2

    mu1 = F.conv2d(pred, win, padding=pad, groups=c)
    mu2 = F.conv2d(target, win, padding=pad, groups=c)
    mu1_sq, mu2_sq, mu1_mu2 = mu1 * mu1, mu2 * mu2, mu1 * mu2

    s11 = F.conv2d(pred * pred, win, padding=pad, groups=c) - mu1_sq
    s22 = F.conv2d(target * target, win, padding=pad, groups=c) - mu2_sq
    s12 = F.conv2d(pred * target, win, padding=pad, groups=c) - mu1_mu2

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    m = ((2 * mu1_mu2 + c1) * (2 * s12 + c2)) / ((mu1_sq + mu2_sq + c1) * (s11 + s22 + c2))
    return m.mean() if size_average else m.flatten(1).mean(1)


class SSIMLoss(nn.Module):
    def __init__(self, data_range: float = 1.0):
        super().__init__()
        self.data_range = data_range

    def forward(self, pred, target):
        return 1.0 - ssim(pred, target, self.data_range)


class GradientLoss(nn.Module):
    """L1 on first differences. Cheap, and it directly penalises edge softening."""

    def forward(self, pred, target):
        px = pred[..., :, 1:] - pred[..., :, :-1]
        tx = target[..., :, 1:] - target[..., :, :-1]
        py = pred[..., 1:, :] - pred[..., :-1, :]
        ty = target[..., 1:, :] - target[..., :-1, :]
        return (px - tx).abs().mean() + (py - ty).abs().mean()


class RestorationLoss(nn.Module):
    def __init__(self, w_char: float = 1.0, w_ssim: float = 0.2,
                 w_grad: float = 0.0, w_lpips: float = 0.0, device: str = "cuda"):
        super().__init__()
        self.char = CharbonnierLoss()
        self.ssim = SSIMLoss()
        self.grad = GradientLoss()
        self.w_char, self.w_ssim, self.w_grad, self.w_lpips = w_char, w_ssim, w_grad, w_lpips
        self.lpips = None
        if w_lpips > 0:
            import lpips  # pip install lpips

            self.lpips = lpips.LPIPS(net="alex").to(device).eval()
            for p in self.lpips.parameters():
                p.requires_grad_(False)

    def forward(self, pred, target):
        parts = {}
        total = 0.0
        if self.w_char:
            parts["char"] = self.char(pred, target)
            total = total + self.w_char * parts["char"]
        if self.w_ssim:
            parts["ssim"] = self.ssim(pred.clamp(0, 1), target)
            total = total + self.w_ssim * parts["ssim"]
        if self.w_grad:
            parts["grad"] = self.grad(pred, target)
            total = total + self.w_grad * parts["grad"]
        if self.lpips is not None:
            p3 = pred.clamp(0, 1).repeat(1, 3, 1, 1) * 2 - 1
            t3 = target.repeat(1, 3, 1, 1) * 2 - 1
            parts["lpips"] = self.lpips(p3, t3).mean()
            total = total + self.w_lpips * parts["lpips"]
        parts["total"] = total
        return total, parts
