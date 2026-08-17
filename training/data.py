"""Datasets and the train/val split.

Two decisions here matter more than anything in the model file:

  1. The validation split is BY GROUP, not by image. A random split leaks
     near-duplicate fields of view across the boundary and will report a
     validation score you cannot reproduce on the real test set.

  2. Training mixes real pairs with synthetically degraded ones. Real pairs
     pin down the exact degradation; synthetic ones cover the family around
     it. `synth_ratio` is the knob.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from degradations import SyntheticDegrader
from imageio_utils import read_gray


def load_manifest(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def split_groups(manifest: dict, val_frac: float = 0.15, seed: int = 0,
                 val_groups: list[str] | None = None) -> tuple[list, list]:
    """Hold out entire groups. Returns (train_pairs, val_pairs)."""
    pairs = manifest["pairs"]
    groups = sorted({p["group"] for p in pairs})
    if val_groups is None:
        rng = random.Random(seed)
        rng.shuffle(groups)
        n_val = max(1, int(round(val_frac * len(groups))))
        val_groups = set(groups[:n_val])
    else:
        val_groups = set(val_groups)
    train = [p for p in pairs if p["group"] not in val_groups]
    val = [p for p in pairs if p["group"] in val_groups]
    if not val:  # single-group dataset: fall back to a deterministic image split
        rng = random.Random(seed)
        idx = list(range(len(pairs)))
        rng.shuffle(idx)
        n_val = max(1, int(round(val_frac * len(pairs))))
        val = [pairs[i] for i in idx[:n_val]]
        train = [pairs[i] for i in idx[n_val:]]
        print("[warn] only one group found; fell back to a random split. "
              "Your validation score will be optimistic.")
    return train, val


def _augment(lr: torch.Tensor, hr: torch.Tensor, rng: random.Random):
    """Dihedral group. Applied identically to both, so alignment is preserved."""
    if rng.random() < 0.5:
        lr, hr = torch.flip(lr, [-1]), torch.flip(hr, [-1])
    if rng.random() < 0.5:
        lr, hr = torch.flip(lr, [-2]), torch.flip(hr, [-2])
    k = rng.randint(0, 3)
    if k:
        lr, hr = torch.rot90(lr, k, [-2, -1]), torch.rot90(hr, k, [-2, -1])
    return lr.contiguous(), hr.contiguous()


class RestorationTrainSet(Dataset):
    """Random LR patches of size `patch` with matching HR patches of size 2*patch."""

    def __init__(self, pairs: list[dict], maxval: float, patch: int = 64,
                 scale: int = 2, synth_ratio: float = 0.5,
                 degrader: SyntheticDegrader | None = None,
                 length: int | None = None, cache: bool = False, seed: int = 1234,
                 index_offset: int = 0):
        self.pairs = pairs
        self.maxval = maxval
        self.patch = patch
        self.scale = scale
        self.synth_ratio = synth_ratio
        self.degrader = degrader or SyntheticDegrader()
        self.length = length or len(pairs)
        self.cache = cache
        self._cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self.seed = seed
        # Shifted on resume so the sample stream continues instead of replaying
        # the first N samples of the original run.
        self.index_offset = index_offset

    def __len__(self) -> int:
        return self.length

    def _read(self, i: int):
        if self.cache and i in self._cache:
            return self._cache[i]
        rec = self.pairs[i]
        lr = read_gray(rec["lr"]) / self.maxval
        hr = read_gray(rec["hr"]) / self.maxval
        if self.cache:
            self._cache[i] = (lr, hr)
        return lr, hr

    def __getitem__(self, idx: int):
        # Per-sample RNG keeps workers decorrelated but epochs reproducible.
        rng = random.Random((self.seed * 1000003 + idx + self.index_offset) & 0x7FFFFFFF)
        i = rng.randrange(len(self.pairs))
        lr_np, hr_np = self._read(i)

        s, p = self.scale, self.patch
        H, W = lr_np.shape
        if H < p or W < p:
            raise ValueError(f"LR image {self.pairs[i]['lr']} smaller than patch {p}")

        use_synth = rng.random() < self.synth_ratio
        if use_synth:
            # Crop the HR image first, then degrade the crop. Degrading the
            # full image and cropping would waste compute on discarded pixels.
            y0 = rng.randrange(0, H - p + 1)
            x0 = rng.randrange(0, W - p + 1)
            hr_c = hr_np[y0 * s:(y0 + p) * s, x0 * s:(x0 + p) * s]
            hr_t = torch.from_numpy(np.ascontiguousarray(hr_c))[None]
            # DataLoader seeds torch's RNG per worker, so noise draws are
            # already decorrelated across workers.
            lr_t = self.degrader(hr_t, rng)
        else:
            y0 = rng.randrange(0, H - p + 1)
            x0 = rng.randrange(0, W - p + 1)
            lr_c = lr_np[y0:y0 + p, x0:x0 + p]
            hr_c = hr_np[y0 * s:(y0 + p) * s, x0 * s:(x0 + p) * s]
            lr_t = torch.from_numpy(np.ascontiguousarray(lr_c))[None]
            hr_t = torch.from_numpy(np.ascontiguousarray(hr_c))[None]

        lr_t, hr_t = _augment(lr_t, hr_t, rng)
        # NOTE: lr is intentionally NOT clamped. hr is genuine ground truth
        # and already lies in [0, 1].
        return lr_t.float(), hr_t.float()


class RestorationValSet(Dataset):
    """Full images, no augmentation, no synthesis. Deterministic."""

    def __init__(self, pairs: list[dict], maxval: float, limit: int | None = None):
        self.pairs = pairs[:limit] if limit else pairs
        self.maxval = maxval

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        rec = self.pairs[idx]
        lr = torch.from_numpy(read_gray(rec["lr"]) / self.maxval)[None].float()
        hr = torch.from_numpy(read_gray(rec["hr"]) / self.maxval)[None].float()
        return lr, hr, rec["group"]


def val_collate(batch):
    """Validation images may differ in size, so keep them as a list."""
    return [b[0] for b in batch], [b[1] for b in batch], [b[2] for b in batch]
