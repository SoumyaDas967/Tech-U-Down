# NAFNet-SR — AI-Based Restoration of Degraded Images

**KLA i4C Hackathon · PS01** · Team **Tech-U-Down**

A single network that removes speckle noise and reconstructs 2× resolution in one
forward pass. 19.02 M parameters, 29.5 ms per 128→256 image on an NVIDIA T4.

| | Bicubic | **NAFNet-SR** | Δ |
|---|---|---|---|
| PSNR (dB) | 21.151 | **23.671** | **+2.520** |
| SSIM | 0.5299 | **0.6097** | **+0.0798** |
| LPIPS | 0.4621 | **0.3412** | **−0.1209** |

Measured on 420 images from a held-out content group, full images, no cropping.
LPIPS is lower-is-better. See [Results](#results) for what that split means and
why the absolute numbers look the way they do.

---

## Quick start

```bash
pip install -r requirements.txt
python run.py <input-dir> <output-dir>
```

Example:

```bash
python run.py test/NoisyLR results/
```

That is the whole procedure. **The trained weights are included** in
`models/best.pth` — nothing is downloaded, no network access or API key is
needed, no file requires editing, and no interactive input is requested.

### What it does

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
already lying in `[0, 1]`. Input values above 1.0 are **not** clipped — see
[the three decisions](#the-problem-and-the-three-decisions-that-mattered) below.

### Output guarantees

Every restored file is checked before it is written:

- shape is exactly 2× the input in each dimension
- `float32`, shape `(H, W)`
- every value inside `[0, 1]`
- no `NaN`, no `Inf`

`run.py` prints a PASS/FAIL summary after the run and **exits non-zero** if any
check fails, so a silently bad output is not possible.

> Note on the NaN guard: `np.clip` does **not** remove `NaN` —
> `np.clip(nan, 0, 1)` returns `nan`. `run.py` therefore calls `np.nan_to_num`
> first and clips second. The ordering matters.

### Optional flags

```bash
--weights path.pth    # a different checkpoint (default: models/best.pth)
--device cpu          # force CPU even if CUDA is present
--device cuda         # force CUDA
```

---

## Repository structure

```
Tech-U-Down/
├── run.py                    entry point:  python run.py <input-dir> <output-dir>
├── requirements.txt          torch and numpy, versions pinned
├── README.md                 this file
├── models/
│   ├── nafnet_sr.py          NAFNet-SR architecture and the four size presets
│   ├── __init__.py
│   └── best.pth              trained weights, 72.7 MB, included in the repo
├── strip_checkpoint.py       shrinks a training checkpoint to inference weights
├── SUBMISSION_CHECK.md       requirement-by-requirement verification
├── outputs/
│   └── predictions.zip       restored outputs for the 400 released test images
└── training/                 everything used to produce the model
    ├── train.py              training loop
    ├── audit_data.py         dataset profiling, writes manifest.json
    ├── preflight.py          checks synthetic degradation against the real one
    ├── check_amp.py          picks the precision mode by measurement
    ├── evaluate.py           scoring against the bicubic baseline
    ├── infer.py              batch inference and latency benchmarking
    ├── data.py               datasets and the group-wise train/val split
    ├── degradations.py       synthetic degradation, calibrated to this dataset
    ├── losses.py             Charbonnier, SSIM, gradient, LPIPS
    ├── metrics.py            PSNR / SSIM / LPIPS with fixed conventions
    ├── imageio_utils.py      grayscale IO preserving the native intensity range
    ├── smoke_test.py         end-to-end pipeline test on generated data
    └── notebooks/quickstart.ipynb
```

---

## The problem, and the three decisions that mattered

Inputs carry **multiplicative speckle** at σ ≈ 0.086 on a `[0, 1]` scale, and
are downsampled 2×. Because speckle scales with intensity, it pushes pixels
**outside `[0, 1]`** — up to 1.54 in the released test set.

### 1. Never clip the input

`run.py` and the training pipeline both pass out-of-range values through
untouched. Clipping would destroy real signal: the out-of-range excursions are
exactly where the speckle information lives, and they are what tells the network
how much noise to remove at a given brightness.

### 2. Validate on a whole held-out group

The dataset contains near-duplicate fields of view. A random split puts
near-identical images on both sides of the boundary and reports a score that
will not survive contact with the real test set.

`audit_data.py --cluster_groups 6` derives six content groups from image
statistics — brightness, contrast, edge density, texture anisotropy, and coarse
spectral content — and one entire group is held out. Every number in this README
is measured that way.

### 3. Calibrate the synthetic degradation, don't guess it

35 % of every training batch is synthetically degraded ground truth, which is
what buys out-of-distribution robustness. Those ranges are measured rather than
assumed:

| | real data | synthetic |
|---|---|---|
| residual std | 0.0792 | 0.0837 (1.06×) |
| multiplicative ratio | 0.848 | 0.867 |
| pixels outside `[0, 1]` | 2.67 % | 2.94 % |

`preflight.py` prints this comparison. The ~6 % margin is deliberate — training
on a slightly wider family than you measure is what generalises; training on a
much wider one just wastes capacity.

---

## Architecture

NAFNet-SR is a 4-stage NAFNet U-Net that runs **entirely at low resolution** and
upsamples once at the very end, so the 2× output costs almost nothing over pure
denoising.

```
                 ┌──────────── global bicubic ×2 skip ────────────┐
                 │            (network learns the residual)       ▼
degraded ─→ conv ─→ E1 ─→ E2 ─→ E3 ─→ E4 ─→ middle ─→ D4 ─→ D3 ─→ D2 ─→ D1 ─→ PixelShuffle ×2 ─→ ⊕ ─→ restored
 128²·1     w=32   1×    1×    2×    4×      8×       1×    1×    1×    1×       + zero-init conv     256²·1
                 32·128² 64·64² 128·32² 256·16²  512·8²
```

Inside one NAFBlock:

```
LayerNorm → Conv1×1 (c→2c) → DWConv3×3 → SimpleGate → SCA → Conv1×1 → ⊕β
          → LayerNorm → Conv1×1 (c→2c) → SimpleGate → Conv1×1 → ⊕γ
```

Four properties that matter for this problem:

- **No activation functions anywhere.** SimpleGate — split the channels,
  multiply — replaces GELU/ReLU. Fewer operations, and it exports cleanly to
  ONNX/TensorRT.
- **LayerNorm per pixel across channels.** This is why the model tolerates the
  global intensity and contrast shifts that characterise unseen inspection
  tools, which is the out-of-distribution failure mode here.
- **PixelShuffle, never transposed convolution.** PixelShuffle cannot produce
  checkerboard artifacts, and a checkerboard on an inspection image reads as a
  false defect.
- **Zero-initialised residual scales (β, γ) and a zero-init conv tail**, so
  every block starts as an identity map and the network begins at exactly
  bicubic. Early training never produces garbage.

Presets in `models/nafnet_sr.py`: `small` 3.9 M · **`medium` 19.02 M (submitted)**
· `base` 29.2 M · `large` 116.1 M.

### Loss

Charbonnier (smooth L1) at weight 1.0 + 0.2 × (1 − SSIM).

L2 is deliberately avoided. It optimises PSNR directly, but its optimum is the
conditional mean — which is a blurred image. On inspection data that blur is
exactly what you are being paid to avoid, and in practice L1-family losses end
up winning on PSNR anyway.

LPIPS is available but stays at zero weight. It is a natural-image perceptual
metric; pushed hard on this data it invents texture, and invented texture on an
inspection image is a false defect call.

---

## Results

Validation holds out content group `c04` — 420 images, whole group, never
trained on.

| | Bicubic | NAFNet-SR | Δ |
|---|---|---|---|
| PSNR (dB) | 21.151 | 23.671 | +2.520 |
| SSIM | 0.5299 | 0.6097 | +0.0798 |
| LPIPS | 0.4621 | 0.3412 | −0.1209 |
| Inference / image | — | 29.5 ms | T4, fp32, over all 400 test images |

**Read the bicubic column, not just the model column.** Absolute PSNR here is
low because the degradation is severe — bicubic manages only 21.15 dB — and
`c04` is the hardest group in the dataset, where bicubic averages 23.33 dB
across all six. The margin over bicubic is the number that transfers between
splits; the absolute value depends heavily on which images you score.

`outputs/predictions.zip` holds the model's restorations of the 400 released
test images. No ground truth was released for those, so they are provided for
inspection rather than scoring.

---

## Reproducing the training

All scripts are in `training/`.

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

Resume an interrupted run with identical flags plus
`--resume runs/run4/last.pth`; the sample stream and the cosine schedule both
continue where they stopped.

Verify the whole pipeline first with `python smoke_test.py` (~2 min on generated
data). It must end with `Smoke test passed.`

### Console markers

Three markers each abort on a fault that is otherwise completely silent:

| marker | when | what it catches |
|---|---|---|
| `[data]` | first batch | ground truth outside `[0, 1]` — a wrong full-scale value invalidates every metric while everything still looks normal |
| `[amp]` | iteration 100 | non-finite gradients — the loss keeps printing plausible numbers while no weight ever moves |
| `[fit]` | iteration 600 | the iteration count the time budget allows |

### Configuration as submitted

105 000 iterations, batch 16, LR patch 64, AdamW (β = 0.9 / 0.99) at lr 3e-4
with 2000-step warmup and cosine decay to 1e-7, EMA 0.999, fp32,
`synth_ratio` 0.35. About 3.5 hours on one T4 at 8.2 it/s, peak memory 1.85 GB.

> **On mixed precision:** fp16 produces non-finite gradients on this model —
> 120/120 steps skipped, weights never move — and offers no speedup here
> (9.12 vs 9.04 it/s), so the submitted model is trained in fp32. bf16 runs on a
> T4 but without tensor-core acceleration, since Turing (SM 7.5) has no bf16
> units. `check_amp.py` measures all three and reports which to use.

### Checkpoint size

A training checkpoint carries raw weights, EMA weights, and AdamW optimizer
state — about 4× the model, or 291 MB. `strip_checkpoint.py` keeps only the EMA
weights:

```bash
python strip_checkpoint.py runs/run4/best.pth models/best.pth
```

291 MB → 72.7 MB, which fits under GitHub's 100 MB per-file limit and is what
allows the weights to ship inside the repository rather than behind a download.

---

## Requirements

**Python 3.10–3.13**, PyTorch 2.x, NumPy. A GPU is optional — `run.py` falls
back to CPU automatically.

```bash
pip install -r requirements.txt
```

On Linux the default PyPI torch wheel is the CUDA build, so this installs a
GPU-capable PyTorch with no extra index or flag.

> PyTorch does not yet publish wheels for Python 3.14. On 3.14, pip installs a
> partial build that fails with `No module named 'torch._strobelight'`.

For CPU-only environments, a much smaller download:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install numpy
```

---

## References

- L. Chen, X. Chu, X. Zhang, J. Sun. *Simple Baselines for Image Restoration.* ECCV 2022. — NAFNet backbone
- W. Shi et al. *Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel CNN.* CVPR 2016. — PixelShuffle
- Z. Wang et al. *Image Quality Assessment: From Error Visibility to Structural Similarity.* IEEE TIP 2004. — SSIM
- R. Zhang et al. *The Unreasonable Effectiveness of Deep Features as a Perceptual Metric.* CVPR 2018. — LPIPS
