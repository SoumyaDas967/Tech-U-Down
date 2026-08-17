"""Inference and latency benchmarking.

    # restore a folder
    python infer.py --ckpt runs/nafnet_base/best.pth --in_dir test/degraded \
        --out_dir preds --ensemble 1 --half

    # measure latency the way a benchmark will
    python infer.py --ckpt runs/nafnet_base/best.pth --benchmark 256 --half

Self-ensemble (`--ensemble 8`) averages the 8 dihedral transforms. It reliably
buys 0.1-0.3 dB and a little SSIM, at exactly 8x the cost. Whether that trade
is worth it depends on the scoring weights; measure both and decide with
numbers, not taste. `--ensemble 4` (identity + 3 rotations) is a reasonable
middle point.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from imageio_utils import list_images, native_maxval, read_gray, write_gray
from models.nafnet_sr import build_model


# ----------------------------------------------------------------------
def load_model(ckpt_path: str, device: str, prefer_ema: bool = True,
               half: bool = False, compile_model: bool = False):
    ck = torch.load(ckpt_path, map_location="cpu")
    preset = ck.get("args", {}).get("preset", "base")
    model = build_model(preset)
    sd = ck["ema"] if (prefer_ema and ck.get("ema")) else ck["model"]
    model.load_state_dict(sd)
    model = model.to(device).eval().to(memory_format=torch.channels_last)
    if half and device == "cuda":
        model = model.half()
    if compile_model:
        model = torch.compile(model, mode="max-autotune")
    return model


# ----------------------------------------------------------------------
_TRANSFORMS = [
    (lambda t: t, lambda t: t),
    (lambda t: torch.rot90(t, 1, [-2, -1]), lambda t: torch.rot90(t, -1, [-2, -1])),
    (lambda t: torch.rot90(t, 2, [-2, -1]), lambda t: torch.rot90(t, -2, [-2, -1])),
    (lambda t: torch.rot90(t, 3, [-2, -1]), lambda t: torch.rot90(t, -3, [-2, -1])),
    (lambda t: torch.flip(t, [-1]), lambda t: torch.flip(t, [-1])),
    (lambda t: torch.rot90(torch.flip(t, [-1]), 1, [-2, -1]),
     lambda t: torch.flip(torch.rot90(t, -1, [-2, -1]), [-1])),
    (lambda t: torch.rot90(torch.flip(t, [-1]), 2, [-2, -1]),
     lambda t: torch.flip(torch.rot90(t, -2, [-2, -1]), [-1])),
    (lambda t: torch.rot90(torch.flip(t, [-1]), 3, [-2, -1]),
     lambda t: torch.flip(torch.rot90(t, -3, [-2, -1]), [-1])),
]


@torch.no_grad()
def forward_ensemble(model, x: torch.Tensor, n: int = 1) -> torch.Tensor:
    if n <= 1:
        return model(x).float()
    acc = None
    for fwd, inv in _TRANSFORMS[:n]:
        y = inv(model(fwd(x).contiguous()).float())
        acc = y if acc is None else acc + y
    return acc / n


@torch.no_grad()
def forward_tiled(model, x: torch.Tensor, tile: int = 0, overlap: int = 16,
                  scale: int = 2, ensemble: int = 1) -> torch.Tensor:
    """Tile only when the image will not fit in memory. Overlap avoids seams."""
    if tile <= 0 or (x.shape[-2] <= tile and x.shape[-1] <= tile):
        return forward_ensemble(model, x, ensemble)

    _, _, H, W = x.shape
    out = torch.zeros(x.shape[0], x.shape[1], H * scale, W * scale,
                      device=x.device, dtype=torch.float32)
    weight = torch.zeros_like(out)
    step = tile - overlap
    for y0 in range(0, H, step):
        for x0 in range(0, W, step):
            y1, x1 = min(y0 + tile, H), min(x0 + tile, W)
            y0a, x0a = max(0, y1 - tile), max(0, x1 - tile)
            patch = x[:, :, y0a:y1, x0a:x1]
            pred = forward_ensemble(model, patch, ensemble)
            out[:, :, y0a * scale:y1 * scale, x0a * scale:x1 * scale] += pred
            weight[:, :, y0a * scale:y1 * scale, x0a * scale:x1 * scale] += 1.0
            if x1 == W:
                break
        if y1 == H:
            break
    return out / weight.clamp(min=1.0)


# ----------------------------------------------------------------------
@torch.no_grad()
def benchmark(model, size: int, device: str, half: bool, ensemble: int,
              warmup: int = 20, runs: int = 100) -> None:
    x = torch.rand(1, 1, size, size, device=device)
    if half and device == "cuda":
        x = x.half()
    x = x.to(memory_format=torch.channels_last)

    for _ in range(warmup):
        forward_ensemble(model, x, ensemble)
    if device == "cuda":
        torch.cuda.synchronize()

    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        forward_ensemble(model, x, ensemble)
        if device == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)

    times.sort()
    print(f"input {size}x{size} -> {size*2}x{size*2} | ensemble={ensemble} | "
          f"half={half} | device={device}")
    print(f"  median {times[len(times)//2]:.2f} ms | p90 {times[int(0.9*len(times))]:.2f} ms "
          f"| min {times[0]:.2f} ms")
    print("  (report median over >=100 runs after warmup, with explicit "
          "cuda.synchronize -- anything else understates latency)")


# ----------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--in_dir", default=None)
    ap.add_argument("--out_dir", default="preds")
    ap.add_argument("--benchmark", type=int, default=0, help="input side length to time")
    ap.add_argument("--ensemble", type=int, default=1, choices=[1, 2, 4, 8])
    ap.add_argument("--tile", type=int, default=0)
    ap.add_argument("--overlap", type=int, default=16)
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--maxval", type=float, default=None)
    ap.add_argument("--out_ext", default=None,
                    help="output file extension; defaults to matching the input")
    ap.add_argument("--no_ema", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.ckpt, device, not args.no_ema, args.half, args.compile)

    if args.benchmark:
        benchmark(model, args.benchmark, device, args.half, args.ensemble)
        if not args.in_dir:
            return

    if not args.in_dir:
        raise SystemExit("nothing to do: pass --in_dir and/or --benchmark")

    files = list_images(args.in_dir)
    if not files:
        raise SystemExit(f"no images found in {args.in_dir}")
    maxval = args.maxval or native_maxval(files[0])

    # Output format defaults to the input's. This matters: with float data the
    # full-scale value is 1.0, so writing an 8-bit format would quantise the
    # whole prediction to 0 and 1 and silently destroy it.
    out_ext = (args.out_ext or files[0].suffix).lower()
    if not out_ext.startswith("."):
        out_ext = "." + out_ext
    if maxval <= 1.0 and out_ext != ".npy":
        raise SystemExit(
            f"maxval is {maxval} (float data) but --out_ext is '{out_ext}'. Writing an "
            f"integer format at this scale rounds every pixel to 0 or 1 and destroys the "
            f"prediction. Use --out_ext .npy, or pass an explicit --maxval (255/65535).")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{len(files)} images | maxval {maxval} | out_ext {out_ext} | writing to {out_dir}")

    probe = read_gray(files[0]) / maxval
    print(f"input range [{probe.min():.4f}, {probe.max():.4f}] | shape {probe.shape}")
    if probe.max() > 4.0:
        print("  [warn] inputs far exceed 1.0. Check --maxval before trusting the output.")

    total = 0.0
    for i, f in enumerate(files):
        arr = read_gray(f) / maxval          # NOT clipped: preserve out-of-range speckle
        x = torch.from_numpy(arr)[None, None].to(device)
        if args.half:
            x = x.half()
        x = x.to(memory_format=torch.channels_last)

        t0 = time.perf_counter()
        y = forward_tiled(model, x, args.tile, args.overlap, 2, args.ensemble)
        if device == "cuda":
            torch.cuda.synchronize()
        total += time.perf_counter() - t0

        pred = y[0, 0].float().cpu().numpy()
        if i == 0:
            print(f"first output: shape {pred.shape} | range "
                  f"[{pred.min():.4f}, {pred.max():.4f}] (clipped to [0,1] on write)")
        write_gray(out_dir / f"{f.stem}{out_ext}", pred, maxval)
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(files)}")

    print(f"mean latency {1000 * total / len(files):.2f} ms/image")
    print(f"wrote {len(files)} files to {out_dir}")


if __name__ == "__main__":
    main()
