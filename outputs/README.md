# Restored test outputs

The model's restorations of the 400 released test images
(`Test_NoisyLR/NoisyLR`), produced with:

```bash
python restore.py --input_dir <test>/NoisyLR --output_dir outputs/
```

| | |
|---|---|
| files | 400 |
| input | 128×128, float32 `.npy`, values reaching 1.54 (unclipped speckle) |
| output | 256×256, float32 `.npy`, clipped to [0, 1] on write |
| naming | identical to the input filename |
| checkpoint | `weights/best.pth`, EMA weights, iteration 105 000 |
| runtime | 29.5 ms per image on a single T4, fp32, no self-ensemble |

No ground truth was released for these images, so they are not scored here. The
quantitative numbers in the top-level README come from held-out group `c04`, where
ground truth exists.

## Verifying an output

```python
import numpy as np, glob
f = sorted(glob.glob("outputs/*.npy"))
a = np.load(f[0])
print(len(f), a.shape, a.dtype, a.min(), a.max(), len(np.unique(a)))
# 400 (256, 256) float32 0.0... 0.9... ~60000
```

A unique-value count in the tens of thousands confirms the float data survived.
If it reads 2, the output was written through an 8-bit path — `restore.py` refuses
to do that, but a manual re-export could.

Predictions are inside predictions.zip. 
