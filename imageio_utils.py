"""Grayscale image IO that preserves the native intensity range.

The single rule that matters here: never clip, never auto-normalise per image
inside the reader. The degraded images legitimately exceed the ground-truth
range (speckle pushes pixels beyond it) and clipping destroys real signal.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

IMG_EXTS = {".png", ".tif", ".tiff", ".bmp", ".jpg", ".jpeg", ".npy"}


def read_gray(path) -> np.ndarray:
    """Read an image as a 2-D float32 array in its ORIGINAL value scale."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".npy":
        arr = np.load(path)
    elif ext in (".tif", ".tiff"):
        import tifffile  # pip install tifffile

        arr = tifffile.imread(str(path))
    else:
        from PIL import Image

        arr = np.array(Image.open(str(path)))

    arr = np.asarray(arr)
    if arr.ndim == 3:
        # Grayscale stored as RGB/RGBA -> take the first channel.
        arr = arr[..., 0]
    if arr.ndim != 2:
        raise ValueError(f"{path} is not 2-D after channel reduction: {arr.shape}")
    return arr.astype(np.float32, copy=False)


def native_maxval(path) -> float:
    """Full-scale value implied by the file's storage dtype."""
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".npy":
        dt = np.load(path, mmap_mode="r").dtype
    elif ext in (".tif", ".tiff"):
        import tifffile

        dt = tifffile.imread(str(path)).dtype
    else:
        from PIL import Image

        mode = Image.open(str(path)).mode
        dt = np.uint16 if mode in ("I;16", "I;16B", "I", "I;16L") else np.uint8
    if np.issubdtype(dt, np.integer):
        return float(np.iinfo(dt).max)
    return 1.0


def write_gray(path, arr: np.ndarray, maxval: float = 255.0) -> None:
    """Write a [0, 1] float array back out at the requested bit depth."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    arr = np.clip(arr, 0.0, 1.0) * maxval
    if path.suffix.lower() == ".npy":
        np.save(path, arr.astype(np.float32))
        return
    dtype = np.uint16 if maxval > 255 else np.uint8
    arr = np.rint(arr).astype(dtype)
    if path.suffix.lower() in (".tif", ".tiff"):
        import tifffile

        tifffile.imwrite(str(path), arr)
    else:
        from PIL import Image

        Image.fromarray(arr).save(str(path))


def is_junk(p: Path) -> bool:
    """macOS archive metadata masquerading as data files.

    Zips made on a Mac contain a __MACOSX/ tree full of AppleDouble files named
    `._something.npy`. They carry the right extension and the wrong contents,
    so they must be filtered out or np.load will crash on them.
    """
    if p.name.startswith("._") or p.name == ".DS_Store":
        return True
    return "__MACOSX" in p.parts


def list_images(root) -> list[Path]:
    root = Path(root)
    return sorted(p for p in root.rglob("*")
                  if p.suffix.lower() in IMG_EXTS and not is_junk(p))
