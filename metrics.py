"""Metrics.

Fix these conventions once, write them down, and never change them mid-project.
Most "my score dropped" incidents are a convention change, not a model change.

  * Predictions are clamped to [0, 1] before scoring. Ground truth already is.
  * data_range = 1.0 (i.e. full scale), not the per-image max.
  * SSIM uses the Gaussian-window formulation with win=11, sigma=1.5, which is
    what scikit-image gives you with gaussian_weights=True. skimage's DEFAULT
    is a 7x7 uniform window and scores noticeably differently. If the
    organisers publish their scoring code, match it exactly and delete this.
  * LPIPS takes 3-channel input in [-1, 1]; grayscale is replicated.
"""
from __future__ import annotations

import torch

from losses import ssim as _ssim


@torch.no_grad()
def psnr(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> float:
    pred = pred.clamp(0, 1)
    mse = torch.mean((pred - target) ** 2)
    if mse == 0:
        return 99.0
    return float(10 * torch.log10(data_range ** 2 / mse))


@torch.no_grad()
def ssim_metric(pred: torch.Tensor, target: torch.Tensor, data_range: float = 1.0) -> float:
    return float(_ssim(pred.clamp(0, 1), target, data_range))


class LPIPSMetric:
    def __init__(self, device: str = "cuda", net: str = "alex"):
        import lpips

        self.model = lpips.LPIPS(net=net).to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        p = pred.clamp(0, 1).to(self.device)
        t = target.to(self.device)
        if p.shape[1] == 1:
            p, t = p.repeat(1, 3, 1, 1), t.repeat(1, 3, 1, 1)
        return float(self.model(p * 2 - 1, t * 2 - 1).mean())


class MetricAccumulator:
    """Tracks metrics overall and per group, so the OOD gap is always visible."""

    def __init__(self):
        self.rows: list[tuple[str, float, float, float | None]] = []

    def add(self, group: str, p: float, s: float, l: float | None = None):
        self.rows.append((group, p, s, l))

    def _agg(self, rows):
        if not rows:
            return {}
        out = {"n": len(rows),
               "psnr": sum(r[1] for r in rows) / len(rows),
               "ssim": sum(r[2] for r in rows) / len(rows)}
        lp = [r[3] for r in rows if r[3] is not None]
        if lp:
            out["lpips"] = sum(lp) / len(lp)
        return out

    def summary(self) -> dict:
        groups = sorted({r[0] for r in self.rows})
        return {"overall": self._agg(self.rows),
                "per_group": {g: self._agg([r for r in self.rows if r[0] == g]) for g in groups}}

    def pretty(self) -> str:
        s = self.summary()
        o = s["overall"]
        lines = [f"overall  n={o['n']:<5d} PSNR {o['psnr']:.3f}  SSIM {o['ssim']:.4f}"
                 + (f"  LPIPS {o['lpips']:.4f}" if "lpips" in o else "")]
        for g, a in s["per_group"].items():
            lines.append(f"  {g:<26} n={a['n']:<5d} PSNR {a['psnr']:.3f}  SSIM {a['ssim']:.4f}"
                         + (f"  LPIPS {a['lpips']:.4f}" if "lpips" in a else ""))
        return "\n".join(lines)
