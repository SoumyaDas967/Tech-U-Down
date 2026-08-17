#!/usr/bin/env python3
"""Shrink a training checkpoint to just the weights needed for inference.

    python strip_checkpoint.py runs/run4/best.pth models/best.pth

A training checkpoint carries the raw weights, the EMA weights, and the AdamW
optimizer state (two tensors per parameter), which is roughly 4x the size of
the model. Inference needs only the EMA weights, so stripping takes the file
from ~291 MB to ~76 MB -- under GitHub's 100 MB per-file limit, which is what
lets the weights ship inside the repository instead of behind a download.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: python strip_checkpoint.py <full.pth> <slim.pth>")
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    if not src.is_file():
        raise SystemExit(f"not found: {src}")

    ck = torch.load(src, map_location="cpu", weights_only=False)
    if not isinstance(ck, dict):
        raise SystemExit("checkpoint is not a dict; nothing to strip")

    sd = ck.get("ema") or ck.get("model") or ck.get("state_dict")
    if sd is None:
        raise SystemExit(f"no weights found. Top-level keys: {list(ck)}")
    which = "ema" if ck.get("ema") else "model"

    preset = (ck.get("args") or {}).get("preset", "medium")
    slim = {"ema": sd, "args": {"preset": preset}, "iter": ck.get("iter")}

    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(slim, dst)

    n = sum(t.numel() for t in sd.values() if torch.is_tensor(t))
    print(f"kept     : {which} weights, {n / 1e6:.2f}M parameters, preset '{preset}'")
    print(f"dropped  : {[k for k in ck if k not in ('ema', 'args', 'iter')]}")
    print(f"{src.name}: {src.stat().st_size / 1e6:.1f} MB")
    print(f"{dst.name}: {dst.stat().st_size / 1e6:.1f} MB")
    if dst.stat().st_size > 100e6:
        print("\n[warn] still over GitHub's 100 MB limit.")


if __name__ == "__main__":
    main()
