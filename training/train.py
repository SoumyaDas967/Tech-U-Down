"""Training.

    python train.py --manifest manifest.json --preset base --iters 300000 \
        --batch 16 --patch 64 --synth_ratio 0.5 --out runs/nafnet_base

Defaults follow the NAFNet recipe: AdamW, lr 1e-3 with cosine decay to 1e-7,
betas (0.9, 0.9), no weight decay on norms/biases. EMA is on by default and is
usually worth 0.1-0.2 dB for free.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data import RestorationTrainSet, RestorationValSet, load_manifest, split_groups, val_collate
from degradations import SyntheticDegrader
from losses import RestorationLoss
from metrics import MetricAccumulator, psnr, ssim_metric
from models.nafnet_sr import build_model


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for s, p in zip(self.shadow.state_dict().values(), model.state_dict().values()):
            if s.dtype.is_floating_point:
                s.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)
            else:
                s.copy_(p)


def param_groups(model: torch.nn.Module, wd: float):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or n.endswith(".bias") or "beta" in n or "gamma" in n:
            no_decay.append(p)
        else:
            decay.append(p)
    return [{"params": decay, "weight_decay": wd},
            {"params": no_decay, "weight_decay": 0.0}]


@torch.no_grad()
def validate(model, loader, device, max_side: int = 1024) -> MetricAccumulator:
    model.eval()
    acc = MetricAccumulator()
    for lrs, hrs, groups in loader:
        for lr, hr, g in zip(lrs, hrs, groups):
            x = lr[None].to(device)
            # Validation runs in fp32 regardless of --amp. It is a handful of
            # images and it keeps the score independent of the training dtype.
            y = model(x).float().cpu()
            acc.add(g, psnr(y, hr[None]), ssim_metric(y, hr[None]))
    model.train()
    return acc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default="runs/exp")
    ap.add_argument("--preset", default="base", choices=["small", "medium", "base", "large"])
    ap.add_argument("--iters", type=int, default=300_000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--patch", type=int, default=64, help="LR patch size; HR patch is 2x this")
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--beta2", type=float, default=0.99,
                    help="AdamW beta2. NAFNet's 0.9 is tuned for large batches; "
                         "0.99 is far steadier at batch 16.")
    ap.add_argument("--fit_hours", type=float, default=0.0,
                    help="measure it/s over the first 600 steps, then shrink --iters "
                         "to whatever fits this many hours. --iters becomes the cap.")
    ap.add_argument("--min_lr", type=float, default=1e-7)
    ap.add_argument("--warmup", type=int, default=2000)
    ap.add_argument("--wd", type=float, default=0.0)
    ap.add_argument("--synth_ratio", type=float, default=0.5)
    ap.add_argument("--w_ssim", type=float, default=0.2)
    ap.add_argument("--w_grad", type=float, default=0.0)
    ap.add_argument("--w_lpips", type=float, default=0.0)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--val_every", type=int, default=5000)
    ap.add_argument("--val_limit", type=int, default=64)
    ap.add_argument("--val_frac", type=float, default=0.15)
    ap.add_argument("--val_groups", nargs="*", default=None)
    ap.add_argument("--cache", action="store_true", help="cache decoded images in RAM")
    ap.add_argument("--resume", default=None)
    ap.add_argument("--amp", default="bf16", choices=["bf16", "fp16", "off"])
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    # Fixed training shapes, so autotuning the conv algorithms is free speed.
    torch.backends.cudnn.benchmark = True
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))

    man = load_manifest(args.manifest)
    train_pairs, val_pairs = split_groups(man, args.val_frac, args.seed, args.val_groups)
    print(f"train {len(train_pairs)} pairs | val {len(val_pairs)} pairs "
          f"| val groups: {sorted({p['group'] for p in val_pairs})}")

    train_set = RestorationTrainSet(
        train_pairs, man["maxval"], patch=args.patch, scale=man.get("scale", 2),
        synth_ratio=args.synth_ratio, degrader=SyntheticDegrader(),
        length=args.batch * args.iters, cache=args.cache, seed=args.seed)
    val_set = RestorationValSet(val_pairs, man["maxval"], limit=args.val_limit)

    train_loader = DataLoader(train_set, batch_size=args.batch, num_workers=args.workers,
                              pin_memory=True, drop_last=True, persistent_workers=args.workers > 0,
                              prefetch_factor=4 if args.workers > 0 else None)
    val_loader = DataLoader(val_set, batch_size=4, num_workers=2, collate_fn=val_collate)

    model = build_model(args.preset).to(device).to(memory_format=torch.channels_last)
    n_par = sum(p.numel() for p in model.parameters())
    print(f"model {args.preset}: {n_par / 1e6:.2f}M params")

    crit = RestorationLoss(1.0, args.w_ssim, args.w_grad, args.w_lpips, device=device)
    opt = torch.optim.AdamW(param_groups(model, args.wd), lr=args.lr,
                            betas=(0.9, args.beta2))
    scaler = torch.amp.GradScaler("cuda", enabled=(args.amp == "fp16" and device == "cuda"))
    amp_dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}.get(args.amp)
    ema = EMA(model, args.ema) if args.ema > 0 else None

    start_it = 0
    if args.resume:
        ck = torch.load(args.resume, map_location=device)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        if ema and ck.get("ema"):
            ema.shadow.load_state_dict(ck["ema"])
        start_it = ck["iter"] + 1
        train_set.index_offset = start_it * args.batch
        print(f"resumed from {args.resume} @ iter {start_it}")

    def lr_at(it: int) -> float:
        if it < args.warmup:
            return args.lr * (it + 1) / args.warmup
        t = (it - args.warmup) / max(1, args.iters - args.warmup)
        return args.min_lr + 0.5 * (args.lr - args.min_lr) * (1 + math.cos(math.pi * t))

    best = -1e9
    t0 = time.time()
    running = 0.0
    skipped = 0
    checked_data = False
    fit_t0, fit_done = time.time(), (args.fit_hours <= 0)
    ref_name, ref_w = next((n, q.detach().clone())
                           for n, q in model.named_parameters() if q.ndim > 1)
    model.train()

    for it, (lr_img, hr_img) in enumerate(train_loader, start=start_it):
        if it >= args.iters:
            break
        for g in opt.param_groups:
            g["lr"] = lr_at(it)

        # ---- one-time data sanity check -------------------------------
        # A wrong `maxval` in the manifest is silent everywhere else: the loss
        # still decreases, the metrics still print, and every number is wrong.
        if not checked_data:
            checked_data = True
            lo_h, hi_h = float(hr_img.min()), float(hr_img.max())
            lo_l, hi_l = float(lr_img.min()), float(lr_img.max())
            print(f"[data] HR range [{lo_h:.4f}, {hi_h:.4f}]  "
                  f"LR range [{lo_l:.4f}, {hi_l:.4f}]  maxval={man['maxval']}")
            if hi_h > 1.05 or lo_h < -0.05:
                raise SystemExit(
                    f"[FATAL] Ground truth is not in [0,1] (max {hi_h:.3f}). The manifest "
                    f"maxval={man['maxval']} is wrong. Re-run audit_data.py with an explicit "
                    f"--maxval (255 / 4095 / 65535) or every PSNR and SSIM number is invalid.")
            if hi_h < 0.5:
                print(f"  [warn] GT never exceeds {hi_h:.3f}. maxval may be too large; "
                      f"PSNR will read low because the data does not use full scale.")

        lr_img = lr_img.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        hr_img = hr_img.to(device, non_blocking=True).to(memory_format=torch.channels_last)

        with torch.autocast("cuda", dtype=amp_dtype, enabled=(amp_dtype is not None and device == "cuda")):
            pred = model(lr_img)
            loss, parts = crit(pred.float(), hr_img)

        opt.zero_grad(set_to_none=True)
        if scaler.is_enabled():
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            prev_scale = scaler.get_scale()
            scaler.step(opt)
            scaler.update()
            if scaler.get_scale() < prev_scale:
                skipped += 1
        else:
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if not torch.isfinite(gn):
                skipped += 1
                opt.zero_grad(set_to_none=True)
            else:
                opt.step()

        # ---- AMP health check -----------------------------------------
        # The failure this catches is silent: the loss keeps printing sensible
        # numbers while no weight has moved since step 0. Abort in seconds
        # rather than discovering it after an eight-hour run.
        n_done = it - start_it + 1
        if n_done == 100:
            moved = float((dict(model.named_parameters())[ref_name].detach() - ref_w).abs().max())
            print(f"[amp] {args.amp}: {skipped}/100 steps skipped | "
                  f"scale {scaler.get_scale() if scaler.is_enabled() else 'n/a'} | "
                  f"max weight change {moved:.3e}")
            if skipped > 20 or moved == 0.0:
                raise SystemExit(
                    f"[FATAL] --amp {args.amp} is producing non-finite gradients: "
                    f"{skipped}/100 steps skipped, weights moved {moved:.3e}. Training here "
                    f"would burn the whole session and learn nothing. Run check_amp.py to "
                    f"find the offending layer, and relaunch with --amp bf16 or --amp off.")

        # ---- size the schedule to the time budget ----------------------
        if not fit_done and n_done == 600:
            fit_done = True
            ips_now = 500 / (time.time() - fit_t0)   # steps 100-600, past warmup jitter
            fitted = int(ips_now * 3600 * args.fit_hours * 0.90 / 1000) * 1000
            fitted = max(20000, min(args.iters, fitted))
            print(f"[fit] {ips_now:.2f} it/s measured -> setting --iters {fitted} "
                  f"to fill {args.fit_hours:.1f} h (cap was {args.iters}). "
                  f"The cosine schedule now completes inside the budget.")
            args.iters = fitted
        if not fit_done and n_done == 100:
            fit_t0 = time.time()

        if ema:
            ema.update(model)
        running += float(loss.detach())

        if (it + 1) % 200 == 0:
            ips = 200 / (time.time() - t0)
            eta = (args.iters - it - 1) / max(ips, 1e-6) / 3600
            print(f"it {it+1:>7d} | loss {running/200:.5f} | lr {lr_at(it):.2e} | "
                  f"{ips:.1f} it/s | eta {eta:.2f} h"
                  + (f" | skipped {skipped}" if skipped else ""))
            running, t0 = 0.0, time.time()

        if (it + 1) % args.val_every == 0 or (it + 1) == args.iters:
            target = ema.shadow if ema else model
            acc = validate(target, val_loader, device)
            print("VAL @", it + 1)
            print(acc.pretty())
            score = acc.summary()["overall"]["psnr"]
            ck = {"model": model.state_dict(), "opt": opt.state_dict(),
                  "ema": ema.shadow.state_dict() if ema else None,
                  "iter": it, "args": vars(args), "val": acc.summary()}
            torch.save(ck, out / "last.pth")
            if score > best:
                best = score
                torch.save(ck, out / "best.pth")
                print(f"  new best PSNR {best:.3f} -> {out/'best.pth'}")
            t0 = time.time()

    print(f"done. best val PSNR {best:.3f}")


if __name__ == "__main__":
    main()
