# NAFNet-SR — Restoration of Degraded Semiconductor Inspection Images

KLA i4C Hackathon · PS01 — AI-Based Restoration of Degraded Images

A single network that removes speckle noise and reconstructs 2× resolution in one
forward pass. 19.02 M parameters, 29.5 ms per 128→256 image on an NVIDIA T4.

| | Bicubic | **NAFNet-SR** | Δ |
|---|---|---|---|
| PSNR (dB) | 21.151 | **23.671** | **+2.520** |
| SSIM | 0.5299 | **0.6097** | **+0.0798** |
| LPIPS | 0.4621 | **0.3412** | **−0.1209** |

Measured on 420 images from a held-out content group, full images, no cropping.
LPIPS is lower-is-better. See [Results](#results) for what the split means.

---

## Quick start — inference in three commands

```bash
git clone https://github.com/
SoumyaDas967/kla-nafnet-restoration.git
cd kla-nafnet-restoration
pip install -r requirements.txt
```

Download `best.pth` (see [weights/README.md](weights/README.md)) and place it at
`weights/best.pth`. Then:

```bash
python restore.py --input_dir path/to/degraded --output_dir path/to/restored
```

That is the whole procedure. No file needs editing, no environment variable needs
setting, and the script works from any working directory.

**What it does.** Reads every image in `--input_dir`, runs the model, and writes a
restored image of exactly 2× the input dimensions to `--output_dir` under the same
filename. Grayscale `.npy`, `.png`, `.tif`, `.tiff`, `.bmp`, `.jpg` are all accepted;
output format matches the input unless `--out_ext` says otherwise. Any image size
works — the network is fully convolutional.

<details>
<summary>Optional flags</summary>

```bash
--ensemble 8      # average 8 dihedral transforms: ~+0.1-0.3 dB, 8x the runtime
--tile 512        # tile large images with overlap; only needed if memory is tight
--half            # fp16 inference (CUDA only)
--device cpu      # force CPU
--ckpt path.pth   # a different checkpoint
--out_ext .npy    # override the output format
--maxval 255      # override the full-scale value (normally inferred)
```

`--input_dir` also accepts `--in_dir`, `--input`, `--i`, or the first positional
argument; `--output_dir` likewise accepts `--out_dir`, `--output`, `--o`, or the
second positional.
</details>

---

## Repository contents

| Path | What it is |
|---|---|
| `restore.py` | **Evaluation entry point.** Input dir → output dir. Runs with no edits. |
| `train.py` | Training. Reproduces the submitted model from scratch. |
| `models/nafnet_sr.py` | NAFNet-SR architecture and the four size presets. |
| `audit_data.py` | Step 1: profiles the dataset, writes `manifest.json`. |
| `preflight.py` | Checks the synthetic degradation against the real one. |
| `check_amp.py` | Picks the mixed-precision mode by measurement. |
| `evaluate.py` | Scores a checkpoint against the bicubic baseline (PSNR/SSIM/LPIPS). |
| `infer.py` | Batch inference plus latency benchmarking. |
| `data.py` | Datasets and the group-wise train/validation split. |
| `degradations.py` | Synthetic degradation, calibrated to this dataset. |
| `losses.py` | Charbonnier, SSIM, gradient, LPIPS. |
| `metrics.py` | PSNR / SSIM / LPIPS with fixed conventions. |
| `imageio_utils.py` | Grayscale IO that preserves the native intensity range. |
| `smoke_test.py` | End-to-end pipeline test on generated data (~2 min). |
| `weights/` | Trained checkpoint (download link inside). |
| `outputs/` | Restored outputs on the released test set. |
| `notebooks/quickstart.ipynb` | The same pipeline as an annotated notebook. |

---

## The problem, and the three decisions that mattered

Inputs carry **multiplicative speckle** at σ ≈ 0.086 on a [0, 1] scale, and are
downsampled 2×. Speckle pushes pixels **outside [0, 1]** — up to 1.54 in the released
test set — because it scales with intensity.

**1. Never clip the input.** `imageio_utils.read_gray` and the dataset both pass
out-of-range values through untouched. Clipping would destroy real signal; the
out-of-range excursions are exactly where the speckle information lives.

**2. Validate on a whole held-out group.** The dataset contains near-duplicate fields
of view. A random split puts near-identical images on both sides and reports a score
that will not survive contact with the real test set. `audit_data.py --cluster_groups 6`
derives six content groups from image statistics and one is held out entirely.

**3. Calibrate the synthetic degradation, don't guess it.** 35 % of every training
batch is synthetically degraded ground truth, which is what buys out-of-distribution
robustness. Those ranges are measured, not assumed:

| | real data | synthetic |
|---|---|---|
| residual std | 0.0792 | 0.0837 (1.06×) |
| multiplicative ratio | 0.848 | 0.867 |
| pixels outside [0, 1] | 2.67 % | 2.94 % |

`preflight.py` prints this table. The ~6 % margin is deliberate — a slightly wider
family than you measure is what generalises.

---

## Architecture

NAFNet-SR: a 4-stage NAFNet U-Net that runs **entirely at low resolution** and
upsamples once at the very end, so the 2× output costs almost nothing over pure
denoising.

```
                 ┌──────────── global bicubic ×2 skip ────────────┐
                 │            (network learns the residual)       ▼
degraded ─→ conv ─→ E1 ─→ E2 ─→ E3 ─→ E4 ─→ middle ─→ D4 ─→ D3 ─→ D2 ─→ D1 ─→ PixelShuffle ×2 ─→ ⊕ ─→ restored
 128²·1     w=32   1×    1×    2×    4×      8×       1×    1×    1×    1×        + zero-init conv      256²·1
                  32·128² 64·64² 128·32² 256·16²  512·8²
```

Three properties that matter for this problem:

- **No activation functions anywhere.** SimpleGate — split the channels, multiply —
  replaces GELU/ReLU. Fewer ops, and it exports cleanly to ONNX/TensorRT.
- **LayerNorm per pixel across channels.** This is why the model tolerates global
  intensity and contrast shifts, which is the out-of-distribution failure mode here.
- **PixelShuffle, never transposed convolution.** PixelShuffle cannot produce
  checkerboard artifacts, and a checkerboard on an inspection image reads as a
  false defect.

Presets in `models/nafnet_sr.py`: `small` 3.9 M · `medium` **19.02 M (submitted)** ·
`base` 29.2 M · `large` 116.1 M.

**Loss:** Charbonnier (smooth L1) at weight 1.0 + 0.2 × (1 − SSIM). L2 is deliberately
avoided — its optimum is the conditional mean, which is a blurred image, and blur is
exactly what this task is paid to avoid.

---

## Reproducing the training

```bash
# 1. profile the dataset -> manifest.json  (~10 min)
python audit_data.py --hr_dir data/train/GT --lr_dir data/train/NoisyLR \
    --out manifest.json --sample 200 --cluster_groups 6

# 2. verify the synthetic degradation matches the real one  (~2 min)
python preflight.py --manifest manifest.json --n 80

# 3. pick the precision mode by measurement, not by guessing  (~3 min)
python check_amp.py --manifest manifest.json --val_groups c04

# 4. sanity run: synthesis off, must beat bicubic clearly  (~10 min)
python train.py --manifest manifest.json --preset medium \
    --iters 4000 --batch 16 --patch 64 --lr 3e-4 --beta2 0.99 \
    --synth_ratio 0.0 --amp off --val_groups c04 \
    --val_every 1000 --val_limit 128 --workers 2 --cache --out runs/sanity

# 5. the full run. --iters is a CEILING; --fit_hours sizes the real schedule
python train.py --manifest manifest.json --preset medium \
    --iters 250000 --fit_hours 7.0 \
    --batch 16 --patch 64 --lr 3e-4 --beta2 0.99 --warmup 2000 \
    --synth_ratio 0.35 --w_ssim 0.2 --amp off \
    --val_groups c04 --val_every 2500 --val_limit 128 \
    --workers 2 --cache --out runs/run4

# 6. score against the bicubic baseline
python evaluate.py --ckpt runs/run4/best.pth --manifest manifest.json \
    --val_groups c04 --lpips
```

Resume an interrupted run with identical flags plus `--resume runs/run4/last.pth`;
the sample stream and the cosine schedule both continue where they stopped.

**Three console markers** are worth watching, each of which aborts on a fault that is
otherwise silent:

| marker | when | what it catches |
|---|---|---|
| `[data]` | first batch | ground truth outside [0, 1] — a wrong full-scale value invalidates every metric while everything still looks normal |
| `[amp]` | iteration 100 | non-finite gradients — the loss keeps printing plausible numbers while no weight moves |
| `[fit]` | iteration 600 | the iteration count the time budget allows |

Verify the whole pipeline first with `python smoke_test.py` (~2 min on generated
data). It must end with `Smoke test passed.`

**Training configuration as submitted:** 105 000 iterations, batch 16, LR patch 64,
AdamW (β = 0.9/0.99) at lr 3e-4 with 2000-step warmup and cosine decay to 1e-7,
EMA 0.999, fp32, `synth_ratio` 0.35. About 3.5 hours on one T4 at 8.2 it/s, peak
memory 1.85 GB.

> **On mixed precision:** fp16 produces non-finite gradients on this model — 120/120
> steps skipped, weights never move — and offers no speedup here (9.12 vs 9.04 it/s),
> so the submitted model is trained in fp32. bf16 runs on a T4 but without tensor-core
> acceleration, since Turing (SM 7.5) has no bf16 units. `check_amp.py` measures all
> three and tells you which to use on your hardware.

---

## Results

Validation holds out content group `c04` — 420 images, whole group, never trained on.

| | Bicubic | NAFNet-SR | Δ |
|---|---|---|---|
| PSNR (dB) | 21.151 | 23.671 | +2.520 |
| SSIM | 0.5299 | 0.6097 | +0.0798 |
| LPIPS | 0.4621 | 0.3412 | −0.1209 |
| Inference / image | — | 29.5 ms | on a T4, fp32, over all 400 test images |

**Read the bicubic column, not just the model column.** Absolute PSNR here is low
because the degradation is severe — bicubic manages only 21.15 dB — and `c04` is the
hardest group in the dataset (bicubic averages 23.33 dB across all six). The margin
over bicubic is the number that transfers; the absolute value depends heavily on
which images you score.

`outputs/` holds the model's restorations of the 400 released test images. No ground
truth exists for those, so they are provided for inspection rather than scoring.

---

## Requirements

`requirements.txt` is the complete `pip freeze` from the Kaggle training environment
(932 packages), as required for reproducibility.

To just run inference, `requirements-minimal.txt` is a tested three-package subset:

    pip install -r requirements-minimal.txt
    # CPU-only torch, avoids ~4 GB of CUDA wheels:
    # pip install torch --index-url https://download.pytorch.org/whl/cpu

Python 3.10–3.13. PyTorch has no wheels for 3.14 yet.

```bash
pip install -r requirements.txt
```

`lpips` is only needed for `--lpips` scoring, and `tifffile` only for TIFF input;
`restore.py` runs without either.

---

## References

- L. Chen, X. Chu, X. Zhang, J. Sun. *Simple Baselines for Image Restoration.* ECCV 2022. — NAFNet backbone
- W. Shi et al. *Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel CNN.* CVPR 2016. — PixelShuffle
- Z. Wang et al. *Image Quality Assessment: From Error Visibility to Structural Similarity.* IEEE TIP 2004. — SSIM
- R. Zhang et al. *The Unreasonable Effectiveness of Deep Features as a Perceptual Metric.* CVPR 2018. — LPIPS
