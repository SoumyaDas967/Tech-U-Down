"""Step 1. Audit the dataset BEFORE writing a single line of model code.

This script answers the five questions that decide your training pipeline:

  1. Do LR/HR pairs match, and is the scale factor exactly 2 everywhere?
  2. What is the storage dtype / full-scale value?
  3. By how much does the degraded range exceed the ground-truth range?
  4. Is the noise MULTIPLICATIVE (speckle) or ADDITIVE (Gaussian)?
     -> decided empirically, not assumed. This determines whether the
        log-domain trick is worth anything on your data.
  5. Which downsampling kernel was used, and is there extra blur on top?

It writes manifest.json, which every other script consumes.

Usage:
    python audit_data.py --hr_dir data/train/gt --lr_dir data/train/degraded \
        --out manifest.json --sample 200
"""
from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from imageio_utils import list_images, native_maxval, read_gray


# --------------------------------------------------------------------------
# pairing
# --------------------------------------------------------------------------
def normalise_stem(stem: str) -> str:
    """Strip common role suffixes so gt/degraded filenames line up."""
    s = stem.lower()
    for tag in ("_gt", "_hr", "_clean", "_high", "_target",
                "_lr", "_low", "_noisy", "_degraded", "_input", "_deg"):
        if s.endswith(tag):
            s = s[: -len(tag)]
    return s


def group_of(stem: str) -> str:
    """Heuristic 'data origin' label used to build a leakage-free OOD split.

    Strips the trailing index so `waferA_00123` -> `waferA`. Inspect the
    printed group table and override it if your dataset encodes origin
    differently (e.g. by subdirectory).

    If filenames are pure numbers (e.g. `001804.npy`) this returns "default"
    for everything, which is useless for an OOD split. In that case use
    --cluster_groups K to derive groups from image content instead.
    """
    s = re.sub(r"[_\-]?\d+$", "", normalise_stem(stem))
    return s if s else "default"


# --------------------------------------------------------------------------
# content-based grouping (fallback when filenames carry no origin info)
# --------------------------------------------------------------------------
def image_features(img: np.ndarray) -> np.ndarray:
    """Cheap descriptors of 'what kind of structure is this'.

    Brightness, contrast, edge density, texture direction and coarse spectral
    content. Images from the same tool and structure type land near each other,
    which is enough to build a validation split that actually tests transfer.
    """
    x = img.astype(np.float32)
    x = (x - x.mean()) / (x.std() + 1e-8)
    gx = np.abs(np.diff(x, axis=1)).mean()
    gy = np.abs(np.diff(x, axis=0)).mean()

    f = np.abs(np.fft.rfft2(x))
    h, w = f.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    yy = np.minimum(yy, h - yy)
    r = np.sqrt(yy ** 2 + xx ** 2)
    r_max = r.max() + 1e-8
    bands = [float(f[(r >= lo * r_max) & (r < hi * r_max)].mean())
             for lo, hi in [(0.0, 0.1), (0.1, 0.25), (0.25, 0.5), (0.5, 1.0)]]
    tot = sum(bands) + 1e-8

    return np.array([
        float(img.mean()), float(img.std()),
        gx, gy, (gx - gy) / (gx + gy + 1e-8),          # anisotropy: lines vs contacts
        float(np.percentile(x, 90) - np.percentile(x, 10)),
        *[b / tot for b in bands],
    ], dtype=np.float32)


def kmeans(X: np.ndarray, k: int, iters: int = 50, seed: int = 0) -> np.ndarray:
    """Minimal k-means++ so we don't depend on scikit-learn."""
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    k = min(k, n)
    centres = [X[rng.integers(n)]]
    for _ in range(k - 1):
        d = np.min(((X[:, None] - np.array(centres)[None]) ** 2).sum(-1), axis=1)
        p = d / (d.sum() + 1e-12)
        centres.append(X[rng.choice(n, p=p)])
    C = np.array(centres)
    labels = np.zeros(n, dtype=int)
    for _ in range(iters):
        d = ((X[:, None] - C[None]) ** 2).sum(-1)
        new = d.argmin(1)
        if (new == labels).all():
            break
        labels = new
        for j in range(k):
            m = labels == j
            if m.any():
                C[j] = X[m].mean(0)
    return labels


def cluster_groups(pairs, maxval: float, k: int, on: str = "hr") -> dict[str, str]:
    """Return {hr_path: group_label} by clustering image content."""
    print(f"\nClustering {len(pairs)} images into {k} content groups...")
    feats, keys = [], []
    for i, (lr_p, hr_p) in enumerate(pairs):
        p = hr_p if on == "hr" else lr_p
        feats.append(image_features(read_gray(p) / maxval))
        keys.append(str(hr_p))
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(pairs)}")
    X = np.stack(feats)
    X = (X - X.mean(0)) / (X.std(0) + 1e-8)
    labels = kmeans(X, k)
    return {key: f"c{lab:02d}" for key, lab in zip(keys, labels)}


def build_pairs(hr_dir: Path, lr_dir: Path) -> list[tuple[Path, Path]]:
    hr = {normalise_stem(p.stem): p for p in list_images(hr_dir)}
    lr = {normalise_stem(p.stem): p for p in list_images(lr_dir)}
    common = sorted(set(hr) & set(lr))
    missing_hr, missing_lr = sorted(set(lr) - set(hr)), sorted(set(hr) - set(lr))
    if missing_hr:
        print(f"  [warn] {len(missing_hr)} degraded files have no GT, e.g. {missing_hr[:3]}")
    if missing_lr:
        print(f"  [warn] {len(missing_lr)} GT files have no degraded, e.g. {missing_lr[:3]}")
    return [(lr[k], hr[k]) for k in common]


# --------------------------------------------------------------------------
# downsample kernel identification
# --------------------------------------------------------------------------
def gaussian_kernel1d(sigma: float, radius: int | None = None) -> torch.Tensor:
    if radius is None:
        radius = max(1, int(3 * sigma + 0.5))
    x = torch.arange(-radius, radius + 1, dtype=torch.float32)
    k = torch.exp(-(x ** 2) / (2 * sigma ** 2))
    return k / k.sum()


def gaussian_blur(x: torch.Tensor, sigma: float) -> torch.Tensor:
    if sigma <= 1e-6:
        return x
    k = gaussian_kernel1d(sigma).to(x.device)
    r = (k.numel() - 1) // 2
    x = F.pad(x, (r, r, r, r), mode="reflect")
    x = F.conv2d(x, k.view(1, 1, 1, -1))
    x = F.conv2d(x, k.view(1, 1, -1, 1))
    return x


def downsample(hr: torch.Tensor, mode: str, sigma: float = 0.0) -> torch.Tensor:
    x = gaussian_blur(hr, sigma) if sigma > 0 else hr
    if mode == "area":
        return F.avg_pool2d(x, 2)
    kw = {} if mode == "nearest" else {"align_corners": False}
    return F.interpolate(x, scale_factor=0.5, mode=mode, **kw)


CANDIDATES = [("bicubic", 0.0), ("bilinear", 0.0), ("area", 0.0), ("nearest", 0.0),
              ("bicubic", 0.5), ("bicubic", 1.0), ("area", 0.5), ("area", 1.0),
              ("bilinear", 1.5), ("bicubic", 2.0)]


def identify_kernel(hr: torch.Tensor, lr: torch.Tensor) -> tuple[str, float, float]:
    """Return the (mode, sigma) whose residual against LR has the lowest std."""
    best = (None, None, float("inf"))
    for mode, sigma in CANDIDATES:
        ref = downsample(hr, mode, sigma)
        if ref.shape != lr.shape:
            continue
        # Noise is zero-mean, so compare robust spread of the residual.
        r = (lr - ref).flatten()
        score = float(r.std())
        if score < best[2]:
            best = (mode, sigma, score)
    return best


# --------------------------------------------------------------------------
# noise character
# --------------------------------------------------------------------------
def noise_character(ref: np.ndarray, lr: np.ndarray, nbins: int = 12) -> dict:
    """Is residual std constant with intensity (additive) or growing (multiplicative)?

    Bins pixels by their clean reference intensity and measures the residual
    std per bin, then fits std = a + b * intensity. A large positive slope
    relative to the intercept means speckle dominates.
    """
    r = (lr - ref).ravel()
    v = ref.ravel()
    lo, hi = np.percentile(v, [2, 98])
    if hi - lo < 1e-8:
        return {"slope": 0.0, "intercept": float(r.std()), "multiplicative_ratio": 0.0}
    edges = np.linspace(lo, hi, nbins + 1)
    idx = np.clip(np.digitize(v, edges) - 1, 0, nbins - 1)
    centres, stds = [], []
    for b in range(nbins):
        m = idx == b
        if m.sum() < 64:
            continue
        centres.append(float(v[m].mean()))
        stds.append(float(r[m].std()))
    if len(centres) < 3:
        return {"slope": 0.0, "intercept": float(r.std()), "multiplicative_ratio": 0.0}
    b, a = np.polyfit(np.array(centres), np.array(stds), 1)
    mid = float(np.mean(centres))
    denom = abs(a) + abs(b) * mid + 1e-8
    return {"slope": float(b), "intercept": float(a),
            "multiplicative_ratio": float(abs(b) * mid / denom)}


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hr_dir", required=True)
    ap.add_argument("--lr_dir", required=True)
    ap.add_argument("--out", default="manifest.json")
    ap.add_argument("--sample", type=int, default=200,
                    help="how many pairs to deep-analyse (full set is still indexed)")
    ap.add_argument("--maxval", type=float, default=None,
                    help="override full-scale value; default inferred from dtype")
    ap.add_argument("--cluster_groups", type=int, default=0,
                    help="derive K content-based groups (use when filenames are just numbers)")
    args = ap.parse_args()

    pairs = build_pairs(Path(args.hr_dir), Path(args.lr_dir))
    if not pairs:
        raise SystemExit("No LR/HR pairs matched. Check --hr_dir/--lr_dir and filenames.")
    print(f"Matched {len(pairs)} pairs.")

    maxval = args.maxval if args.maxval else native_maxval(pairs[0][1])

    # Float data carries no implied full scale, so check the actual values.
    raw = read_gray(pairs[0][1])
    raw_lo, raw_hi = float(raw.min()), float(raw.max())
    print(f"Full-scale value: {maxval}")
    print(f"Raw GT values in first file: [{raw_lo:.4f}, {raw_hi:.4f}]  dtype float/int")
    if maxval == 1.0 and raw_hi > 1.5:
        print(f"  [ACTION] Float data is NOT in [0,1]. Re-run with --maxval "
              f"{2 ** round(np.log2(max(raw_hi, 1))) if raw_hi > 1 else 1.0:.0f} "
              f"(or 255 / 4095 / 65535 as appropriate) so SSIM and the loss "
              f"are scaled correctly.")
    elif maxval == 1.0:
        print("  Float data already in [0,1]. No scaling needed.")

    rng = np.random.default_rng(0)
    probe = [pairs[i] for i in rng.choice(len(pairs), min(args.sample, len(pairs)), replace=False)]

    shape_counter: Counter = Counter()
    bad_scale: list[str] = []
    hr_lo, hr_hi, lr_lo, lr_hi = [], [], [], []
    over, under = [], []
    kernels: Counter = Counter()
    mult_ratios, noise_sigmas = [], []
    records = []

    for lr_p, hr_p in probe:
        lr = read_gray(lr_p) / maxval
        hr = read_gray(hr_p) / maxval
        shape_counter[(hr.shape, lr.shape)] += 1
        if hr.shape[0] != lr.shape[0] * 2 or hr.shape[1] != lr.shape[1] * 2:
            bad_scale.append(hr_p.name)
            continue

        hr_lo.append(hr.min()); hr_hi.append(hr.max())
        lr_lo.append(lr.min()); lr_hi.append(lr.max())
        over.append(float((lr > hr.max()).mean()))
        under.append(float((lr < hr.min()).mean()))

        t_hr = torch.from_numpy(hr)[None, None]
        t_lr = torch.from_numpy(lr)[None, None]
        mode, sigma, resid = identify_kernel(t_hr, t_lr)
        kernels[(mode, sigma)] += 1
        noise_sigmas.append(resid)

        ref = downsample(t_hr, mode, sigma)[0, 0].numpy()
        nc = noise_character(ref, lr)
        mult_ratios.append(nc["multiplicative_ratio"])
        records.append(nc)

    # ---------------- report ----------------
    print("\n=== SHAPES ===")
    for (hs, ls), n in shape_counter.most_common():
        print(f"  GT {hs} <- LR {ls}   x{n}")
    if bad_scale:
        print(f"  [ERROR] {len(bad_scale)} pairs are NOT exactly x2, e.g. {bad_scale[:5]}")
        print("          Decide explicitly how to handle these before training.")
    else:
        print("  All sampled pairs are exactly x2. Good: one x2 model covers both regimes.")

    print("\n=== INTENSITY RANGE (normalised by full scale) ===")
    print(f"  GT : min {np.mean(hr_lo):.4f}  max {np.mean(hr_hi):.4f}")
    print(f"  LR : min {np.mean(lr_lo):.4f}  max {np.mean(lr_hi):.4f}")
    print(f"  LR pixels above GT max: {100 * np.mean(over):.2f}%")
    print(f"  LR pixels below GT min: {100 * np.mean(under):.2f}%")
    print("  -> DO NOT clip the input. Feed these values through as-is.")

    print("\n=== DOWNSAMPLE KERNEL (best match) ===")
    for (mode, sigma), n in kernels.most_common(5):
        print(f"  {mode:<9} sigma={sigma:<4} x{n}")
    print(f"  residual std (proxy for noise level): mean {np.mean(noise_sigmas):.4f}")
    print("  -> mirror the top kernels in degradations.py; randomise across them for OOD.")

    print("\n=== NOISE CHARACTER ===")
    mr = float(np.mean(mult_ratios))
    print(f"  multiplicative ratio: {mr:.3f}   (0 = pure additive, 1 = pure speckle)")
    if mr > 0.5:
        print("  -> Strongly multiplicative. The log-domain transform is worth an A/B test.")
    elif mr > 0.25:
        print("  -> Mixed. Train the plain model first; treat log-domain as an ablation.")
    else:
        print("  -> Mostly additive. SKIP the log transform; it will not help.")

    if args.cluster_groups:
        gmap = cluster_groups(pairs, maxval, args.cluster_groups)
        group_for = lambda hr_p: gmap[str(hr_p)]  # noqa: E731
    else:
        group_for = lambda hr_p: group_of(hr_p.stem)  # noqa: E731

    groups = defaultdict(int)
    for _, hr_p in pairs:
        groups[group_for(hr_p)] += 1
    print(f"\n=== GROUPS ({len(groups)}) ===")
    for g, n in sorted(groups.items(), key=lambda kv: -kv[1])[:20]:
        print(f"  {g:<28} {n}")
    print("  -> Hold out WHOLE groups for validation. Random splits leak and will lie to you.")
    if len(groups) == 1:
        print("  [ACTION] Only one group. Your filenames carry no origin information, so a "
              "group split is impossible and validation will fall back to a random split "
              "(optimistic). Re-run with --cluster_groups 6 to derive groups from image "
              "content instead.")

    manifest = {
        "maxval": maxval,
        "scale": 2,
        "multiplicative_ratio": mr,
        "kernels": [{"mode": m, "sigma": s, "count": n} for (m, s), n in kernels.most_common()],
        "residual_std_mean": float(np.mean(noise_sigmas)) if noise_sigmas else None,
        "pairs": [
            {"lr": str(lp), "hr": str(hp), "group": group_for(hp)}
            for lp, hp in pairs
        ],
    }
    Path(args.out).write_text(json.dumps(manifest, indent=2))
    print(f"\nWrote {args.out} with {len(pairs)} pairs.")


if __name__ == "__main__":
    main()
