"""Read the manifest and check the synthetic degradation actually matches reality.

    python preflight.py --manifest manifest.json --n 60

`synth_ratio 0.5` means half of every batch is made by SyntheticDegrader. If
that degrader is harsher, softer or noisier than the real degradation, half the
training signal is teaching the model to solve a different problem, and the
model hedges by producing something smooth and bicubic-like. Nothing in the
training log will tell you this is happening.

This prints the real and synthetic statistics side by side and suggests
degrader ranges from the measured numbers. It changes no files.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from audit_data import downsample, noise_character
from degradations import SyntheticDegrader
from imageio_utils import read_gray


def hf_energy(x: np.ndarray) -> float:
    """Mean absolute first difference: a cheap 'how sharp is this' number."""
    return float(np.abs(np.diff(x, axis=1)).mean() + np.abs(np.diff(x, axis=0)).mean()) / 2


def describe(tag: str, stats: dict) -> None:
    print(f"  {tag:<10} resid_std {stats['resid']:.4f} | mult_ratio {stats['mult']:.3f} | "
          f"hf {stats['hf']:.4f} | range [{stats['lo']:.3f}, {stats['hi']:.3f}] | "
          f"outside[0,1] {100 * stats['out']:.2f}%")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--n", type=int, default=60, help="pairs to sample")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    man = json.loads(Path(args.manifest).read_text())
    maxval = man["maxval"]
    pairs = man["pairs"]

    print("=" * 72)
    print("MANIFEST")
    print("=" * 72)
    print(f"  pairs                 {len(pairs)}")
    print(f"  maxval                {maxval}")
    print(f"  scale                 {man.get('scale')}")
    print(f"  multiplicative_ratio  {man.get('multiplicative_ratio')}")
    print(f"  residual_std_mean     {man.get('residual_std_mean')}")
    print("  kernels (top 5):")
    for k in man.get("kernels", [])[:5]:
        print(f"    {k['mode']:<9} sigma={k['sigma']:<5} count={k['count']}")
    gc = Counter(p["group"] for p in pairs)
    print(f"  groups ({len(gc)}):")
    for g, n in gc.most_common():
        print(f"    {g:<10} {n:>6}  ({100 * n / len(pairs):.1f}%)")

    if not man.get("kernels"):
        raise SystemExit("Manifest has no kernel info. Re-run audit_data.py.")
    top = man["kernels"][0]
    mode, sigma = top["mode"], float(top["sigma"])

    rng = np.random.default_rng(args.seed)
    prng = random.Random(args.seed)
    sel = [pairs[i] for i in rng.choice(len(pairs), min(args.n, len(pairs)), replace=False)]
    deg = SyntheticDegrader()

    acc = {"real": [], "synth": []}
    bicubic_psnr, hr_lo, hr_hi = [], [], []

    for rec in sel:
        hr = read_gray(rec["hr"]) / maxval
        lr = read_gray(rec["lr"]) / maxval
        if hr.shape[0] != lr.shape[0] * 2 or hr.shape[1] != lr.shape[1] * 2:
            continue
        hr_lo.append(hr.min()); hr_hi.append(hr.max())

        t_hr = torch.from_numpy(hr.astype(np.float32))[None, None]
        ref = downsample(t_hr, mode, sigma)[0, 0].numpy()

        lr_s = deg(t_hr[0], prng)[0].numpy()
        if lr_s.shape != lr.shape:
            continue

        for tag, arr in (("real", lr), ("synth", lr_s)):
            nc = noise_character(ref, arr)
            acc[tag].append({
                "resid": float((arr - ref).std()), "mult": nc["multiplicative_ratio"],
                "hf": hf_energy(arr), "lo": float(arr.min()), "hi": float(arr.max()),
                "out": float(((arr < 0) | (arr > 1)).mean())})

        up = F.interpolate(torch.from_numpy(lr.astype(np.float32))[None, None],
                           scale_factor=2, mode="bicubic", align_corners=False)
        mse = float(((up.clamp(0, 1) - t_hr) ** 2).mean())
        bicubic_psnr.append(10 * np.log10(1.0 / max(mse, 1e-12)))

    if not acc["real"]:
        raise SystemExit("No usable pairs sampled.")
    mean = lambda tag: {k: float(np.mean([d[k] for d in acc[tag]])) for k in acc["real"][0]}  # noqa: E731
    R, S = mean("real"), mean("synth")

    print("\n" + "=" * 72)
    print("GROUND TRUTH RANGE")
    print("=" * 72)
    print(f"  min {np.mean(hr_lo):.4f}   max {np.mean(hr_hi):.4f}")
    if np.mean(hr_hi) > 1.02:
        print("  [FATAL] GT exceeds 1.0. maxval is wrong; every metric is invalid.")
    elif np.mean(hr_hi) < 0.6:
        print("  [warn] GT never approaches full scale. PSNR reads low for this reason "
              "alone; it is not necessarily a model problem.")
    print(f"  bicubic PSNR on this sample: {np.mean(bicubic_psnr):.3f} dB")

    print("\n" + "=" * 72)
    print(f"REAL vs SYNTHETIC DEGRADATION  (reference kernel: {mode} sigma={sigma})")
    print("=" * 72)
    describe("REAL", R)
    describe("SYNTH", S)

    print("\n  interpretation:")
    ratio = S["resid"] / max(R["resid"], 1e-8)
    if ratio > 1.35:
        print(f"    Synthetic noise is {ratio:.2f}x the real noise. Too harsh -- the model "
              f"spends capacity denoising a problem it will never see, and hedges toward "
              f"smooth output. Lower speckle_sigma / gauss_sigma, or lower --synth_ratio.")
    elif ratio < 0.7:
        print(f"    Synthetic noise is only {ratio:.2f}x the real noise. Too mild to add "
              f"robustness. Raise speckle_sigma.")
    else:
        print(f"    Noise level matches well ({ratio:.2f}x). Good.")

    hfr = S["hf"] / max(R["hf"], 1e-8)
    if hfr < 0.75:
        print(f"    Synthetic images are {hfr:.2f}x as sharp as real ones -- over-blurred. "
              f"Reduce blur_sigma / p_blur.")
    elif hfr > 1.3:
        print(f"    Synthetic images are {hfr:.2f}x as sharp as real ones -- under-blurred. "
              f"Raise blur_sigma.")
    else:
        print(f"    Sharpness matches well ({hfr:.2f}x). Good.")

    if abs(S["mult"] - R["mult"]) > 0.25:
        print(f"    Noise CHARACTER differs (real {R['mult']:.2f} vs synth {S['mult']:.2f}, "
              f"0=additive 1=speckle). Rebalance p_speckle vs p_gauss.")

    print("\n" + "=" * 72)
    print("SUGGESTED SyntheticDegrader RANGES  (edit degradations.py)")
    print("=" * 72)
    sp = 2.5 * R["resid"]
    print(f"    speckle_sigma=(0.0, {sp:.3f}),")
    print(f"    gauss_sigma=(0.0, {0.6 * R['resid']:.3f}),")
    print(f"    blur_sigma=(0.0, {max(0.6, sigma * 1.5) if sigma > 0 else 0.8:.2f}),")
    agg: dict[str, int] = {}
    for k in man["kernels"]:
        agg[k["mode"]] = agg.get(k["mode"], 0) + k["count"]
    top3 = sorted(agg.items(), key=lambda kv: -kv[1])[:3]
    kmodes = [m for m, _ in top3]
    kcount = [c for _, c in top3]
    tot = sum(kcount) or 1
    print(f"    modes={tuple(kmodes)},")
    print(f"    mode_weights={tuple(round(c / tot, 2) for c in kcount)},")
    print("\n  These widen the observed degradation by roughly 50%, which is the point:")
    print("  the test set is half out-of-distribution, so training on a slightly wider")
    print("  family than you measured is what buys generalisation.")


if __name__ == "__main__":
    main()
