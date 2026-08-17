"""Scoring, with the bicubic baseline alongside so the numbers mean something.

    python evaluate.py --ckpt runs/nafnet_base/best.pth --manifest manifest.json \
        --val_groups waferB waferC --lpips --ensemble 8

Always report the per-group table, not just the mean. A model that is excellent
on four groups and broken on one has an average that hides the problem you
actually need to fix.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from data import RestorationValSet, load_manifest, split_groups, val_collate
from infer import forward_tiled, load_model
from metrics import LPIPSMetric, MetricAccumulator, psnr, ssim_metric
from torch.utils.data import DataLoader


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--val_groups", nargs="*", default=None)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--ensemble", type=int, default=1, choices=[1, 2, 4, 8])
    ap.add_argument("--tile", type=int, default=0)
    ap.add_argument("--half", action="store_true")
    ap.add_argument("--lpips", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--save_json", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    man = load_manifest(args.manifest)
    _, val_pairs = split_groups(man, args.val_frac, args.seed, args.val_groups)
    ds = RestorationValSet(val_pairs, man["maxval"], limit=args.limit)
    loader = DataLoader(ds, batch_size=1, num_workers=2, collate_fn=val_collate)
    print(f"evaluating {len(ds)} images from groups {sorted({p['group'] for p in val_pairs})}")

    model = load_model(args.ckpt, device, half=args.half)
    lp = LPIPSMetric(device) if args.lpips else None

    model_acc, base_acc = MetricAccumulator(), MetricAccumulator()

    for lrs, hrs, groups in loader:
        for lr, hr, g in zip(lrs, hrs, groups):
            x = lr[None].to(device)
            if args.half:
                x = x.half()
            y = forward_tiled(model, x.to(memory_format=torch.channels_last),
                              args.tile, 16, 2, args.ensemble).float().cpu()
            t = hr[None]

            # Bicubic baseline: if your model is not comfortably ahead of this,
            # stop and debug the pipeline rather than the architecture.
            b = F.interpolate(lr[None].float(), scale_factor=2,
                              mode="bicubic", align_corners=False)

            model_acc.add(g, psnr(y, t), ssim_metric(y, t),
                          lp(y, t) if lp else None)
            base_acc.add(g, psnr(b, t), ssim_metric(b, t),
                         lp(b, t) if lp else None)

    print("\n--- bicubic baseline ---")
    print(base_acc.pretty())
    print("\n--- model ---")
    print(model_acc.pretty())

    mo, bo = model_acc.summary()["overall"], base_acc.summary()["overall"]
    print(f"\ndelta vs bicubic: PSNR {mo['psnr'] - bo['psnr']:+.3f} dB | "
          f"SSIM {mo['ssim'] - bo['ssim']:+.4f}"
          + (f" | LPIPS {mo['lpips'] - bo['lpips']:+.4f}" if "lpips" in mo else ""))

    if args.save_json:
        Path(args.save_json).write_text(json.dumps(
            {"model": model_acc.summary(), "bicubic": base_acc.summary()}, indent=2))


if __name__ == "__main__":
    main()
