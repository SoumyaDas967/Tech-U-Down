"""Decide --amp with measurements instead of guesses, and localise NaNs.

Two jobs:

  1. For each of off / bf16 / fp16, run real training steps on real data and
     report whether weights actually move, how many steps get skipped, and how
     fast it is. A dtype that is 2x faster is worthless if it learns nothing.
  2. If a dtype produces non-finite gradients, --locate finds the exact layer
     and the exact op whose backward is responsible.

    python check_amp.py --manifest manifest.json --val_groups c04
    python check_amp.py --manifest manifest.json --val_groups c04 --locate fp16

Takes about three minutes. Run it before the long training run.
"""
from __future__ import annotations

import argparse
import time

import torch
from torch.utils.data import DataLoader

from data import RestorationTrainSet, load_manifest, split_groups
from degradations import SyntheticDegrader
from losses import RestorationLoss
from models.nafnet_sr import build_model


def make_loader(args, man, pairs):
    ds = RestorationTrainSet(pairs, man["maxval"], patch=args.patch,
                             scale=man.get("scale", 2), synth_ratio=args.synth_ratio,
                             degrader=SyntheticDegrader(),
                             length=args.batch * (args.steps + 5), seed=1)
    return DataLoader(ds, batch_size=args.batch, num_workers=args.workers,
                      pin_memory=True, drop_last=True)


def run_mode(mode: str, args, man, pairs, device: str) -> dict:
    torch.manual_seed(0)
    torch.backends.cudnn.benchmark = True
    loader = make_loader(args, man, pairs)

    model = build_model(args.preset).to(device).to(memory_format=torch.channels_last)
    crit = RestorationLoss(1.0, args.w_ssim, 0.0, 0.0, device=device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, args.beta2))
    scaler = torch.amp.GradScaler("cuda", enabled=(mode == "fp16" and device == "cuda"))
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(mode)

    ref_name, ref_w = next((n, p.detach().clone())
                           for n, p in model.named_parameters() if p.ndim > 1)

    skipped, bad_fwd, losses = 0, 0, []
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    t_start = None

    for i, (lr_img, hr_img) in enumerate(loader):
        if i >= args.steps:
            break
        if i == args.warmup:                       # time only past cudnn autotune
            if device == "cuda":
                torch.cuda.synchronize()
            t_start = time.perf_counter()

        lr_img = lr_img.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        hr_img = hr_img.to(device, non_blocking=True).to(memory_format=torch.channels_last)

        with torch.autocast("cuda", dtype=amp_dtype,
                            enabled=(amp_dtype is not None and device == "cuda")):
            pred = model(lr_img)
            loss, _ = crit(pred.float(), hr_img)

        if not torch.isfinite(pred).all():
            bad_fwd += 1
        losses.append(float(loss.detach()))

        opt.zero_grad(set_to_none=True)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            prev = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            if scaler.get_scale() < prev:
                skipped += 1
        else:
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(gn):
                skipped += 1
                opt.zero_grad(set_to_none=True)
            else:
                opt.step()

    if device == "cuda":
        torch.cuda.synchronize()
    n_timed = max(1, min(args.steps, len(losses)) - args.warmup)
    ips = n_timed / (time.perf_counter() - t_start) if t_start else float("nan")
    moved = float((dict(model.named_parameters())[ref_name].detach() - ref_w).abs().max())
    peak = torch.cuda.max_memory_allocated() / 1e9 if device == "cuda" else 0.0

    return {"mode": mode, "ips": ips, "skipped": skipped, "bad_fwd": bad_fwd,
            "moved": moved, "loss0": losses[0] if losses else float("nan"),
            "lossN": sum(losses[-10:]) / max(1, len(losses[-10:])), "peak_gb": peak,
            "scale": scaler.get_scale() if scaler.is_enabled() else None,
            "ok": moved > 0 and skipped <= args.steps * 0.2 and bad_fwd == 0}


def locate(mode: str, args, man, pairs, device: str) -> None:
    """Find which layer, and which op, first produces a non-finite gradient."""
    print(f"\n{'=' * 70}\nLOCATING non-finite gradients under --amp {mode}\n{'=' * 70}")
    torch.manual_seed(0)
    loader = make_loader(args, man, pairs)
    model = build_model(args.preset).to(device).to(memory_format=torch.channels_last)
    crit = RestorationLoss(1.0, args.w_ssim, 0.0, 0.0, device=device)
    scaler = torch.amp.GradScaler("cuda", enabled=(mode == "fp16"))
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(mode)

    lr_img, hr_img = next(iter(loader))
    lr_img = lr_img.to(device).to(memory_format=torch.channels_last)
    hr_img = hr_img.to(device).to(memory_format=torch.channels_last)

    # Pass 1: which parameters hold non-finite grads, in forward order.
    with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
        pred = model(lr_img)
        loss, _ = crit(pred.float(), hr_img)
    print(f"forward finite: {torch.isfinite(pred).all().item()} | "
          f"pred absmax {float(pred.detach().float().abs().max()):.3e} | loss {float(loss):.5f}")
    (scaler.scale(loss) if scaler.is_enabled() else loss).backward()

    bad = [(n, float(p.grad.float().abs().max()))
           for n, p in model.named_parameters()
           if p.grad is not None and not torch.isfinite(p.grad).all()]
    if not bad:
        print("All parameter gradients are finite on this step. If training still "
              "collapses, it is intermittent -- raise --steps and rerun.")
        return
    print(f"\n{len(bad)} of {sum(1 for _ in model.parameters())} parameters have "
          f"non-finite grads. First few in forward order:")
    for n, v in bad[:12]:
        print(f"  {n}")

    # Pass 2: anomaly mode names the exact op whose backward produced it.
    print("\nRunning autograd anomaly detection (slow, one step)...")
    model.zero_grad(set_to_none=True)
    try:
        with torch.autograd.set_detect_anomaly(True):
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype is not None):
                pred = model(lr_img)
                loss, _ = crit(pred.float(), hr_img)
            (scaler.scale(loss) if scaler.is_enabled() else loss).backward()
        print("Anomaly detection did not fire on this step.")
    except RuntimeError as e:
        print(f"\n>>> ANOMALY: {str(e).splitlines()[0]}")
        print(">>> The traceback above points at the forward line whose backward "
              "produced the NaN. That is the layer to keep in fp32.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--val_groups", nargs="*", default=None)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--preset", default="medium")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--patch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--beta2", type=float, default=0.99)
    ap.add_argument("--w_ssim", type=float, default=0.2)
    ap.add_argument("--synth_ratio", type=float, default=0.25)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--modes", nargs="*", default=["off", "bf16", "fp16"])
    ap.add_argument("--locate", default=None, help="run the NaN locator for this mode")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        cap = torch.cuda.get_device_capability()
        print(f"GPU: {torch.cuda.get_device_name(0)} | compute capability {cap[0]}.{cap[1]}")
        print(f"  fp16 tensor cores: {cap >= (7, 0)}")
        print(f"  bf16 tensor cores: {cap >= (8, 0)}"
              + ("   <-- absent here; bf16 runs but is not accelerated"
                 if cap < (8, 0) else ""))

    man = load_manifest(args.manifest)
    pairs, _ = split_groups(man, args.val_frac, args.seed, args.val_groups)
    print(f"{len(pairs)} training pairs | preset {args.preset} | "
          f"batch {args.batch} | patch {args.patch}\n")

    if args.locate:
        locate(args.locate, args, man, pairs, device)
        return

    results = []
    for mode in args.modes:
        print(f"--- testing --amp {mode} ({args.steps} steps) ---")
        try:
            r = run_mode(mode, args, man, pairs, device)
        except Exception as e:                                  # noqa: BLE001
            print(f"  FAILED: {type(e).__name__}: {e}\n")
            continue
        results.append(r)
        print(f"  {r['ips']:.2f} it/s | skipped {r['skipped']}/{args.steps} | "
              f"bad forwards {r['bad_fwd']} | weights moved {r['moved']:.3e} | "
              f"peak {r['peak_gb']:.2f} GB")
        print(f"  loss {r['loss0']:.5f} -> {r['lossN']:.5f} | "
              f"{'USABLE' if r['ok'] else 'BROKEN -- do not train with this'}\n")

    usable = [r for r in results if r["ok"]]
    print("=" * 70)
    if not usable:
        print("No dtype is usable. Something is wrong beyond AMP; run with --locate off.")
        return
    best = max(usable, key=lambda r: r["ips"])
    print(f"USE:  --amp {best['mode']}      ({best['ips']:.2f} it/s)")
    for h in (6, 7, 8):
        print(f"  {h} h of training at this rate = {int(best['ips'] * 3600 * h * 0.90 / 1000) * 1000:,} iterations")
    broken = [r["mode"] for r in results if not r["ok"]]
    if broken:
        print(f"\nBroken: {', '.join(broken)}. Diagnose with: "
              f"python check_amp.py --manifest {args.manifest} --locate {broken[0]}")


if __name__ == "__main__":
    main()
