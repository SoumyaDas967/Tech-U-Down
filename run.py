#!/usr/bin/env python3
"""Restoration entry point.

    python run.py <input-dir> <output-dir>

Reads every .npy file in <input-dir>, restores it, and writes one .npy of the
same filename to <output-dir>. The output directory is created if it does not
exist.

Runs fully offline. The model weights ship in models/best.pth; nothing is
downloaded, no API key is used, and no interactive input is requested.

Output guarantees, verified before each file is written:
  * shape (H, W), float32, exactly 2x the input in each dimension
  * every value inside [0, 1]
  * no NaN and no Inf

The network is fully convolutional, so any input size is accepted. CUDA is used
when available and CPU otherwise.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from models.nafnet_sr import build_model  # noqa: E402

# Weights live next to the code so the solution runs without any download.
WEIGHT_CANDIDATES = [
    HERE / "models" / "best.pth",
    HERE / "weights" / "best.pth",
    HERE / "best.pth",
]


def find_weights(explicit: str | None) -> Path:
    if explicit:
        p = Path(explicit)
        if not p.is_file():
            raise SystemExit(f"[FATAL] weights not found: {p}")
        return p
    for p in WEIGHT_CANDIDATES:
        if p.is_file():
            return p
    raise SystemExit(
        "[FATAL] model weights not found. Expected one of:\n  "
        + "\n  ".join(str(p) for p in WEIGHT_CANDIDATES))


def load_model(ckpt_path: Path, device: str) -> torch.nn.Module:
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if isinstance(ck, dict) and any(k in ck for k in ("ema", "model", "state_dict")):
        preset = (ck.get("args") or {}).get("preset", "medium")
        sd = ck.get("ema") or ck.get("model") or ck.get("state_dict")
    else:                                   # a bare state_dict also loads
        preset, sd = "medium", ck

    sd = {k.replace("module.", ""): v for k, v in sd.items()}
    model = build_model(preset)
    model.load_state_dict(sd, strict=True)
    model = model.to(device).eval().to(memory_format=torch.channels_last)

    n = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"model  : NAFNet-SR '{preset}', {n:.2f}M parameters, from {ckpt_path.name}")
    return model


def read_npy(path: Path) -> tuple[np.ndarray, float]:
    """Return a 2-D float32 array and the full-scale value implied by its dtype."""
    arr = np.load(path, allow_pickle=False)
    arr = np.asarray(arr)

    if arr.ndim == 3:                       # (H, W, 1) or (H, W, C)
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"{path.name}: expected 2-D after channel reduction, got {arr.shape}")

    # Integer storage implies a full scale; float data is assumed already in [0, 1].
    maxval = float(np.iinfo(arr.dtype).max) if np.issubdtype(arr.dtype, np.integer) else 1.0
    return arr.astype(np.float32, copy=False), maxval


def sanitise(pred: np.ndarray) -> tuple[np.ndarray, int, int]:
    """Force the output into the required range. Returns (array, n_nan, n_inf)."""
    n_nan = int(np.isnan(pred).sum())
    n_inf = int(np.isinf(pred).sum())
    if n_nan or n_inf:
        pred = np.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=0.0)
    # np.clip alone does NOT remove NaN, so nan_to_num must come first.
    return np.clip(pred, 0.0, 1.0).astype(np.float32), n_nan, n_inf


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Restore degraded grayscale .npy images (denoise + 2x super-resolution).",
        usage="python run.py <input-dir> <output-dir>")
    ap.add_argument("input_dir", help="directory containing degraded .npy files")
    ap.add_argument("output_dir", help="directory to write restored .npy files to")
    ap.add_argument("--weights", default=None,
                    help="override the checkpoint path (default: models/best.pth)")
    ap.add_argument("--device", default=None, choices=["cuda", "cpu"])
    ap.add_argument("--batch_report", type=int, default=50)
    args = ap.parse_args()

    in_dir, out_dir = Path(args.input_dir), Path(args.output_dir)
    if not in_dir.is_dir():
        raise SystemExit(f"[FATAL] input directory does not exist: {in_dir}")

    files = sorted(p for p in in_dir.rglob("*.npy")
                   if not p.name.startswith("._") and "__MACOSX" not in p.parts)
    if not files:
        raise SystemExit(f"[FATAL] no .npy files found in {in_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda":
        torch.backends.cudnn.benchmark = True
        print(f"device : cuda ({torch.cuda.get_device_name(0)})")
    else:
        print("device : cpu")

    model = load_model(find_weights(args.weights), device)
    print(f"input  : {len(files)} .npy files from {in_dir}")
    print(f"output : {out_dir}\n")

    total = 0.0
    n_written = 0
    bad_values = 0

    with torch.no_grad():
        for i, f in enumerate(files):
            arr, maxval = read_npy(f)
            x = arr / maxval          # NOT clipped: speckle legitimately exceeds [0, 1]
            t = torch.from_numpy(np.ascontiguousarray(x))[None, None].to(device)
            t = t.to(memory_format=torch.channels_last)

            t0 = time.perf_counter()
            y = model(t)
            if device == "cuda":
                torch.cuda.synchronize()
            total += time.perf_counter() - t0

            pred, n_nan, n_inf = sanitise(y[0, 0].float().cpu().numpy())
            if n_nan or n_inf:
                bad_values += 1
                print(f"  [warn] {f.name}: {n_nan} NaN and {n_inf} Inf replaced")

            expected = (arr.shape[0] * 2, arr.shape[1] * 2)
            if pred.shape != expected:
                raise SystemExit(f"[FATAL] {f.name}: got {pred.shape}, expected {expected}")

            np.save(out_dir / f.name, pred)
            n_written += 1

            if i == 0:
                print(f"first  : {arr.shape} -> {pred.shape}, dtype {pred.dtype}, "
                      f"range [{pred.min():.4f}, {pred.max():.4f}]")
            if (i + 1) % args.batch_report == 0 or (i + 1) == len(files):
                print(f"  {i + 1}/{len(files)}")

    print(f"\nwrote {n_written} files to {out_dir}")
    print(f"mean inference {1000 * total / len(files):.2f} ms/image on {device}")

    # ---- final self-check against the submission requirements ----------
    print("\n--- output verification ---")
    outs = sorted(out_dir.glob("*.npy"))
    names_in = {f.name for f in files}
    names_out = {f.name for f in outs}
    checks = []
    checks.append(("one output per input", len(outs) == len(files)))
    checks.append(("filenames match inputs", names_in == names_out))

    ok_shape = ok_range = ok_finite = True
    for f in outs:
        a = np.load(f)
        if a.ndim not in (2, 3) or (a.ndim == 3 and a.shape[2] != 1):
            ok_shape = False
        if not np.isfinite(a).all():
            ok_finite = False
        if a.size and (a.min() < 0.0 or a.max() > 1.0):
            ok_range = False
    checks.append(("shape (H, W) or (H, W, 1)", ok_shape))
    checks.append(("values within [0, 1]", ok_range))
    checks.append(("no NaN or Inf", ok_finite))

    for name, passed in checks:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    if bad_values:
        print(f"  note: {bad_values} file(s) required NaN/Inf replacement")

    if not all(p for _, p in checks):
        raise SystemExit("\n[FATAL] output verification failed.")
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
