# NAFNet-SR — AI-Based Restoration of Degraded Images

KLA i4C Hackathon · PS01

Removes speckle and Gaussian noise and reconstructs 2× resolution in a single
forward pass. 19.02 M parameters.

## Run

```bash
pip install -r requirements.txt
python run.py <input-dir> <output-dir>
```

Example:

```bash
python run.py test/NoisyLR results/
```

That is the whole procedure. Model weights are included in `models/best.pth`;
nothing is downloaded, no API key or network access is needed, and no file
requires editing.

## What it does

Reads every `.npy` file in `<input-dir>` (recursively), restores it, and writes
one `.npy` per input to `<output-dir>` under the **same filename**. The output
directory is created if it does not exist.

| | |
|---|---|
| input | grayscale `.npy`, shape `(H, W)` or `(H, W, 1)`, any size |
| output | grayscale `.npy`, shape `(2H, 2W)`, `float32`, values in `[0, 1]` |
| device | CUDA when available, CPU otherwise — selected automatically |
| speed | 29.5 ms per 128→256 image on an NVIDIA T4 |

Integer inputs are scaled by their dtype maximum; float inputs are taken as
already being in `[0, 1]`. Values above 1.0 in the input are **not** clipped —
speckle legitimately pushes pixels out of range, and clipping would discard real
signal.

Every output is checked before writing: correct 2× shape, no NaN, no Inf, all
values inside `[0, 1]`. `run.py` prints a PASS/FAIL summary at the end and exits
non-zero if any check fails.

## Structure

```
├── run.py              entry point:  python run.py <input-dir> <output-dir>
├── requirements.txt    torch, numpy, with versions
├── README.md
└── models/
    ├── nafnet_sr.py    NAFNet-SR architecture
    ├── __init__.py
    └── best.pth        trained weights (76 MB, included)
```

`train.py` and the supporting scripts that reproduce training are in
[`training/`](training/).

## Architecture

A 4-stage NAFNet U-Net that runs entirely at low resolution and upsamples once
at the very end, so the ×2 output costs almost nothing over pure denoising.

- **No activation functions.** SimpleGate — split the channels, multiply —
  replaces GELU/ReLU.
- **LayerNorm per pixel across channels**, which is what makes the model
  tolerate the global intensity and contrast shifts in unseen data.
- **PixelShuffle, not transposed convolution**, because PixelShuffle cannot
  produce checkerboard artifacts, and a checkerboard on an inspection image
  reads as a false defect.
- **A global bicubic ×2 skip**, so the network only learns the residual.

Trained with Charbonnier + 0.2 × (1 − SSIM). L2 is deliberately avoided: its
optimum is the conditional mean, which is a blurred image.

## Results

Validation holds out an entire content group — 420 images, full resolution,
never trained on. A random split was rejected because the dataset contains
near-duplicate fields of view, which leak across a random boundary.

| | Bicubic | NAFNet-SR | Δ |
|---|---|---|---|
| PSNR (dB) | 21.151 | **23.671** | +2.520 |
| SSIM | 0.5299 | **0.6097** | +0.0798 |
| LPIPS | 0.4621 | **0.3412** | −0.1209 |

Absolute PSNR is low because the degradation is severe — bicubic reaches only
21.15 dB on this group. The margin over bicubic is the number that transfers.

## Requirements

Python 3.10–3.13, PyTorch 2.x. PyTorch does not yet publish wheels for Python
3.14; on 3.14 pip installs a partial build that fails with
`No module named 'torch._strobelight'`.
