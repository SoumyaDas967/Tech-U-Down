# Final technical check

| Requirement | Status |
|---|---|
| `team_name/run.py` | present |
| `team_name/requirements.txt` | present, versions pinned |
| `team_name/README.md` | present |
| `team_name/models/` | present, contains architecture + `best.pth` |
| runs as `python run.py <input-dir> <output-dir>` | positional args |
| reads all `.npy` from the input directory | recursive, skips macOS junk |
| creates the output directory if absent | `mkdir(parents=True, exist_ok=True)` |
| one restored `.npy` per input | verified at the end of every run |
| output filename matches input | verified |
| grayscale `(H, W)` or `(H, W, 1)` | writes `(2H, 2W)` float32 |
| values within `[0, 1]`, no NaN/Inf | `nan_to_num` then `clip`, then verified |
| correct target resolution | exactly 2×, asserted per file |
| weights and supporting files included | `models/best.pth`, 76 MB, in-repo |
| dependencies with versions | `torch==2.5.1`, `numpy==2.1.3` |
| runs on NVIDIA GPU | CUDA auto-detected, `cudnn.benchmark` on |
| no internet, API keys, downloads, interaction, manual config | none of these are used |

`run.py` prints a PASS/FAIL block after every run and exits non-zero if any
output constraint is violated.

## Verify before submitting

```bash
python run.py <a folder of test .npy> /tmp/check
```

Must end with `All checks passed.`

Then confirm the outputs independently:

```python
import numpy as np, glob, os
ins  = sorted(glob.glob("<input-dir>/*.npy"))
outs = sorted(glob.glob("/tmp/check/*.npy"))
assert len(ins) == len(outs), "count mismatch"
assert {os.path.basename(p) for p in ins} == {os.path.basename(p) for p in outs}
for p in outs:
    a = np.load(p)
    assert a.ndim in (2, 3) and a.dtype == np.float32
    assert np.isfinite(a).all() and a.min() >= 0.0 and a.max() <= 1.0
i, o = np.load(ins[0]), np.load(outs[0])
assert o.shape[:2] == (i.shape[0] * 2, i.shape[1] * 2)
print("independent verification passed")
```
