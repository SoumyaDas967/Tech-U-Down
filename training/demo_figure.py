#!/usr/bin/env python3
"""Make the before/after figure for the demo video.

    python demo_figure.py --lr_dir demo/NoisyLR --gt_dir demo/GT --out demo_result.png

Three columns per row: degraded input, this model's restoration, ground truth.
Per-image PSNR and SSIM against that image's ground truth are printed on the
restored panel, and the bicubic score is printed underneath so the numbers have
a reference point.

Without --gt_dir it falls back to two columns (input, restored) and no metrics,
which is what the released test set allows since no ground truth exists for it.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
from imageio_utils import list_images, read_gray  # noqa: E402
from restore import load_model  # noqa: E402


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = float(np.mean((np.clip(a, 0, 1) - b) ** 2))
    return 99.0 if mse == 0 else 10 * np.log10(1.0 / mse)


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Gaussian-window SSIM, matching metrics.py."""
    from losses import ssim as _ssim
    ta = torch.from_numpy(np.clip(a, 0, 1).astype(np.float32))[None, None]
    tb = torch.from_numpy(b.astype(np.float32))[None, None]
    return float(_ssim(ta, tb, 1.0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lr_dir", required=True, help="degraded inputs")
    ap.add_argument("--gt_dir", default=None, help="ground truth, if available")
    ap.add_argument("--ckpt", default=str(Path(__file__).resolve().parent
                                          / "weights" / "best.pth"))
    ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--out", default="demo_result.png")
    ap.add_argument("--maxval", type=float, default=1.0)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(Path(args.ckpt), device)

    files = list_images(args.lr_dir)
    if not files:
        raise SystemExit(f"no images in {args.lr_dir}")
    sel = [files[i] for i in np.linspace(0, len(files) - 1, min(args.n, len(files))).astype(int)]

    gt_map = {}
    if args.gt_dir:
        gt_map = {p.stem: p for p in list_images(args.gt_dir)}

    ncol = 3 if gt_map else 2
    fig, axes = plt.subplots(len(sel), ncol, figsize=(5.0 * ncol, 6.0 * len(sel)))
    if len(sel) == 1:
        axes = axes[None]

    times = []
    for i, f in enumerate(sel):
        lo = read_gray(f) / args.maxval
        x = torch.from_numpy(np.ascontiguousarray(lo.astype(np.float32)))[None, None].to(device)
        with torch.no_grad():
            t0 = time.perf_counter()
            y = model(x.to(memory_format=torch.channels_last))
            if device == "cuda":
                torch.cuda.synchronize()
            times.append((time.perf_counter() - t0) * 1000)
        out = np.clip(y[0, 0].float().cpu().numpy(), 0, 1)

        panels = [(lo, f"DEGRADED INPUT  {lo.shape[0]}x{lo.shape[1]}")]
        if f.stem in gt_map:
            gt = read_gray(gt_map[f.stem]) / args.maxval
            up = F.interpolate(torch.from_numpy(lo.astype(np.float32))[None, None],
                               size=gt.shape, mode="bicubic",
                               align_corners=False)[0, 0].numpy()
            panels.append((out, f"RESTORED  {out.shape[0]}x{out.shape[1]}\n"
                                f"PSNR {psnr(out, gt):.2f} dB   SSIM {ssim(out, gt):.4f}\n"
                                f"(bicubic: {psnr(up, gt):.2f} dB   {ssim(up, gt):.4f})"))
            panels.append((gt, f"GROUND TRUTH  {gt.shape[0]}x{gt.shape[1]}"))
        else:
            panels.append((out, f"RESTORED  {out.shape[0]}x{out.shape[1]}"))

        for j, (img, t) in enumerate(panels):
            axes[i, j].imshow(img, cmap="gray", vmin=0, vmax=1)
            axes[i, j].set_title(t, fontsize=11)
            axes[i, j].axis("off")

    plt.tight_layout(h_pad=2.6)
    plt.savefig(args.out, dpi=110, bbox_inches="tight")
    print(f"\nwrote {args.out}")
    print(f"inference: {np.median(times):.1f} ms/image median on {device} "
          f"({len(times)} images, includes first-call warmup)")


if __name__ == "__main__":
    main()
