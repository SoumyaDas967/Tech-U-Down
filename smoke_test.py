"""End-to-end smoke test on generated data. Run this FIRST, before the real
dataset lands. It builds a fake paired dataset, runs the audit, trains for a
few hundred iterations and evaluates, so every seam in the pipeline is
exercised while the stakes are zero.

    python smoke_test.py
"""
from __future__ import annotations

import random
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from degradations import SyntheticDegrader          # noqa: E402
from imageio_utils import write_gray                 # noqa: E402

ROOT = Path("smoke_data")


def synth_structure(size: int, rng: np.random.Generator) -> np.ndarray:
    """Crude line/contact patterns, loosely inspection-like."""
    img = np.full((size, size), 0.25, dtype=np.float32)
    kind = rng.integers(0, 3)
    if kind == 0:  # line/space
        period = int(rng.integers(8, 24))
        phase = int(rng.integers(0, period))
        for x in range(size):
            if ((x + phase) % period) < period // 2:
                img[:, x] = 0.8
    elif kind == 1:  # contact array
        period = int(rng.integers(12, 30))
        r = max(2, period // 4)
        for cy in range(period // 2, size, period):
            for cx in range(period // 2, size, period):
                yy, xx = np.ogrid[:size, :size]
                img[(yy - cy) ** 2 + (xx - cx) ** 2 <= r * r] = 0.85
    else:  # random polygons
        for _ in range(int(rng.integers(3, 9))):
            y0, x0 = rng.integers(0, size - 10, 2)
            h, w = rng.integers(6, size // 3, 2)
            img[y0:y0 + h, x0:x0 + w] = float(rng.uniform(0.5, 0.95))
    img += rng.normal(0, 0.01, img.shape).astype(np.float32)
    return np.clip(img, 0, 1)


def build_dataset(n_per_group: int = 24) -> None:
    if ROOT.exists():
        shutil.rmtree(ROOT)
    (ROOT / "gt").mkdir(parents=True)
    (ROOT / "degraded").mkdir(parents=True)
    rng = np.random.default_rng(0)
    prng = random.Random(0)
    deg = SyntheticDegrader()
    for gi, group in enumerate(["lines", "contacts", "polys"]):
        for i in range(n_per_group):
            hr_size = 512 if i % 2 == 0 else 256
            hr = synth_structure(hr_size, rng)
            lr = deg(torch.from_numpy(hr)[None], prng)[0].numpy()
            write_gray(ROOT / "gt" / f"{group}_{i:04d}.png", hr, 255)
            # Degraded images may exceed [0,1]; PNG storage clips them, which is
            # a limitation of the smoke fixture only, not of the pipeline.
            write_gray(ROOT / "degraded" / f"{group}_{i:04d}.png", lr, 255)
    print(f"built {3 * n_per_group} pairs in {ROOT}")


def run(cmd: list[str]) -> None:
    print("\n$", " ".join(cmd))
    r = subprocess.run(cmd, cwd=Path(__file__).parent)
    if r.returncode != 0:
        raise SystemExit(f"FAILED: {' '.join(cmd)}")


def main() -> None:
    build_dataset()
    py = sys.executable
    run([py, "audit_data.py", "--hr_dir", str(ROOT / "gt"),
         "--lr_dir", str(ROOT / "degraded"), "--out", "smoke_manifest.json", "--sample", "20"])
    run([py, "train.py", "--manifest", "smoke_manifest.json", "--preset", "small",
         "--iters", "300", "--batch", "4", "--patch", "32", "--workers", "0",
         "--val_every", "150", "--val_limit", "6", "--val_groups", "polys",
         "--out", "runs/smoke", "--amp", "off", "--cache"])
    run([py, "evaluate.py", "--ckpt", "runs/smoke/best.pth",
         "--manifest", "smoke_manifest.json", "--val_groups", "polys", "--limit", "6"])
    run([py, "infer.py", "--ckpt", "runs/smoke/best.pth", "--benchmark", "128"])
    print("\nSmoke test passed. The pipeline is wired correctly end to end.")


if __name__ == "__main__":
    main()
