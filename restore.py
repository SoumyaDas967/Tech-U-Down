#!/usr/bin/env python3
"""Restore a folder of degraded images. This is the evaluation entry point.

    python restore.py --input_dir path/to/degraded --output_dir path/to/restored

Nothing else needs setting. The checkpoint defaults to weights/best.pth, the
model configuration is read out of the checkpoint, the intensity scale is
inferred from the files, and outputs are written in the input's own format.

Accepted spellings for the two arguments, so nothing has to be looked up:
    --input_dir  / --in_dir  / --input  / --i     (or the first positional)
    --output_dir / --out_dir / --output / --o     (or the second positional)

Supported input formats: .npy, .png, .tif/.tiff, .bmp, .jpg/.jpeg, grayscale.
Any image size is accepted; the network is fully convolutional. Output is
exactly 2x the input in each dimension.

    python restore.py --input_dir test/ --output_dir preds/ --ensemble 8
        8x self-ensemble: about +0.1-0.3 dB for 8x the runtime.

    python restore.py --input_dir test/ --output_dir preds/ --tile 512
        Tile large images. Only needed if a full image will not fit in memory.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Run correctly from any working directory.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from imageio_utils import list_images, native_maxval, read_gray, write_gray  # noqa: E402
from models.nafnet_sr import build_model  # noqa: E402

DEFAULT_CKPT = Path(__file__).resolve().parent / "weights" / "best.pth"


# ----------------------------------------------------------------------
def load_model(ckpt_path: Path, device: str, prefer_ema: bool = True,
               half: bool = False):
    """Build the network from the config stored in the checkpoint and load it."""
    if not Path(ckpt_path).exists():
        raise SystemExit(
            f"\nCheckpoint not found: {ckpt_path}\n"
            f"Download the trained weights and place them at that path.\n"
            f"See weights/README.md for the download link.\n")

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    # A bare state_dict is also accepted, so a re-exported checkpoint still works.
    if isinstance(ck, dict) and any(k in ck for k in ("model", "ema", "state_dict")):
        preset = ck.get("args", {}).get("preset", "medium")
        sd = ck["ema"] if (prefer_ema and ck.get("ema")) else (
            ck.get("model") or ck.get("state_dict"))
        iters = ck.get("iter")
    else:
        preset, sd, iters = "medium", ck, None

    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model = build_model(preset)
    model.load_state_dict(sd, strict=True)
    model = model.to(device).eval().to(memory_format=torch.channels_last)
    if half and device == "cuda":
        model = model.half()

    n = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"model    : NAFNet-SR '{preset}', {n:.2f}M parameters"
          + (f", checkpoint iteration {iters}" if iters is not None else "")
          + (" (EMA weights)" if prefer_ema and isinstance(ck, dict)
             and ck.get("ema") else ""))
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
def forward_tiled(model, x: torch.Tensor, tile: int = 0, overlap: int = 32,
                  scale: int = 2, ensemble: int = 1) -> torch.Tensor:
    """Tile only when a whole image will not fit. Overlapping avoids seams."""
    if tile <= 0 or (x.shape[-2] <= tile and x.shape[-1] <= tile):
        return forward_ensemble(model, x, ensemble)

    _, _, H, W = x.shape
    out = torch.zeros(x.shape[0], x.shape[1], H * scale, W * scale,
                      device=x.device, dtype=torch.float32)
    weight = torch.zeros_like(out)
    step = max(1, tile - overlap)
    for y0 in range(0, H, step):
        for x0 in range(0, W, step):
            y1, x1 = min(y0 + tile, H), min(x0 + tile, W)
            y0a, x0a = max(0, y1 - tile), max(0, x1 - tile)
            pred = forward_ensemble(model, x[:, :, y0a:y1, x0a:x1], ensemble)
            out[:, :, y0a * scale:y1 * scale, x0a * scale:x1 * scale] += pred
            weight[:, :, y0a * scale:y1 * scale, x0a * scale:x1 * scale] += 1.0
            if x1 == W:
                break
        if y1 == H:
            break
    return out / weight.clamp(min=1.0)


# ----------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Restore degraded grayscale images (denoise + 2x super-resolution).",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pos_input", nargs="?", default=None,
                    help=argparse.SUPPRESS)
    ap.add_argument("pos_output", nargs="?", default=None,
                    help=argparse.SUPPRESS)
    ap.add_argument("--input_dir", "--in_dir", "--input", "--i", dest="input_dir",
                    default=None, help="directory of degraded input images")
    ap.add_argument("--output_dir", "--out_dir", "--output", "--o", dest="output_dir",
                    default=None, help="directory to write restored images to")
    ap.add_argument("--ckpt", "--weights", dest="ckpt", default=str(DEFAULT_CKPT),
                    help="model checkpoint (default: weights/best.pth)")
    ap.add_argument("--ensemble", type=int, default=1, choices=[1, 2, 4, 8],
                    help="self-ensemble over N dihedral transforms (default 1)")
    ap.add_argument("--tile", type=int, default=0,
                    help="tile size for very large images; 0 = whole image")
    ap.add_argument("--overlap", type=int, default=32)
    ap.add_argument("--out_ext", default=None,
                    help="output extension; default matches the input")
    ap.add_argument("--maxval", type=float, default=None,
                    help="full-scale value; default inferred from the files")
    ap.add_argument("--half", action="store_true",
                    help="fp16 inference (CUDA only)")
    ap.add_argument("--device", default=None, choices=["cuda", "cpu"])
    ap.add_argument("--no_ema", action="store_true")
    args = ap.parse_args()

    in_dir = args.input_dir or args.pos_input
    out_dir = args.output_dir or args.pos_output
    if not in_dir or not out_dir:
        ap.print_help()
        raise SystemExit("\nBoth an input and an output directory are required.\n"
                         "  python restore.py --input_dir <degraded> --output_dir <restored>")

    in_dir, out_dir = Path(in_dir), Path(out_dir)
    if not in_dir.is_dir():
        raise SystemExit(f"Input directory does not exist: {in_dir}")

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    print(f"device   : {device}"
          + (f" ({torch.cuda.get_device_name(0)})" if device == "cuda" else ""))

    model = load_model(Path(args.ckpt), device, not args.no_ema, args.half)

    files = list_images(in_dir)
    if not files:
        raise SystemExit(f"No images found in {in_dir}. "
                         f"Supported: .npy .png .tif .tiff .bmp .jpg .jpeg")

    maxval = args.maxval if args.maxval is not None else native_maxval(files[0])

    out_ext = (args.out_ext or files[0].suffix).lower()
    if not out_ext.startswith("."):
        out_ext = "." + out_ext
    # Float data has a full scale of 1.0. Writing it to an 8-bit format would
    # round every pixel to 0 or 1 and destroy the prediction silently, so refuse.
    if maxval <= 1.0 and out_ext != ".npy":
        raise SystemExit(
            f"Input is float data (maxval {maxval}) but --out_ext is '{out_ext}'.\n"
            f"Writing an integer format at this scale would quantise every pixel "
            f"to 0 or 1.\nUse --out_ext .npy, or pass an explicit --maxval "
            f"(e.g. 255 or 65535).")

    out_dir.mkdir(parents=True, exist_ok=True)
    probe = read_gray(files[0]) / maxval
    print(f"input    : {len(files)} images from {in_dir}")
    print(f"           first is {probe.shape}, range [{probe.min():.4f}, {probe.max():.4f}], "
          f"full scale {maxval}")
    print(f"output   : {out_ext} files to {out_dir}"
          + (f" | self-ensemble x{args.ensemble}" if args.ensemble > 1 else "")
          + (f" | tile {args.tile}" if args.tile else ""))
    print()

    total = 0.0
    with torch.no_grad():
        for i, f in enumerate(files):
            arr = read_gray(f) / maxval      # deliberately NOT clipped: speckle
            x = torch.from_numpy(np.ascontiguousarray(arr.astype(np.float32)))[None, None]
            x = x.to(device)
            if args.half and device == "cuda":
                x = x.half()
            x = x.to(memory_format=torch.channels_last)

            t0 = time.perf_counter()
            y = forward_tiled(model, x, args.tile, args.overlap, 2, args.ensemble)
            if device == "cuda":
                torch.cuda.synchronize()
            total += time.perf_counter() - t0

            pred = y[0, 0].float().cpu().numpy()
            if i == 0:
                print(f"first output: {pred.shape} "
                      f"(input {arr.shape}), range "
                      f"[{pred.min():.4f}, {pred.max():.4f}] before clipping to [0,1]")
            write_gray(out_dir / f"{f.stem}{out_ext}", pred, maxval)

            if (i + 1) % 50 == 0 or (i + 1) == len(files):
                print(f"  {i + 1}/{len(files)}")

    print(f"\ndone: {len(files)} images written to {out_dir}")
    print(f"mean inference time {1000 * total / len(files):.2f} ms/image on {device}")


if __name__ == "__main__":
    main()
