# Submission checklist

| # | Requirement | Where it is | Status |
|---|---|---|---|
| 1 | README with complete setup instructions | `README.md` | done |
| 2 | Standalone `.py` evaluation script, input dir + output dir, no manual edits | `restore.py` | done |
| 3 | Training script reproducing training from scratch | `train.py`, plus `notebooks/quickstart.ipynb` | done |
| 4 | Trained model weights, downloadable | `weights/` — **add `best.pth` and paste the download link** | **action needed** |
| 5 | Restored test outputs | `outputs/` — **add the 400 `.npy` predictions** | **action needed** |
| 6 | `requirements.txt` from a pip freeze | `requirements.txt` + **`requirements-frozen.txt`** | **action needed** |

## The three remaining actions

**4 — weights.** Copy `best.pth` into `weights/`, then edit `weights/README.md` and
replace the placeholder with your real Google Drive or Hugging Face link. The file is
too large for plain git; either use Git LFS (`.gitattributes` is already configured)
or host it externally. An external link is safer — GitHub's free LFS quota is 1 GB of
storage and 1 GB/month of bandwidth.

**5 — outputs.** Copy the 400 restored `.npy` files into `outputs/`. At ~256 KB each
that is roughly 100 MB, which is over what plain git handles comfortably. Either
enable LFS for `*.npy` (already in `.gitattributes`) or commit a single
`outputs/predictions.zip` and note that in `outputs/README.md`.

**6 — frozen requirements.** Run this in the environment that trained the model:

```bash
pip freeze > requirements-frozen.txt
```

On Kaggle, in a notebook cell:

```python
!pip freeze > /kaggle/working/requirements-frozen.txt
```

Then download it and commit it alongside `requirements.txt`. Keep both: the frozen
file documents exactly what was used, and `requirements.txt` is the short list a
reviewer can actually install without version conflicts on a different machine.

## Verify before you push

```bash
python smoke_test.py                    # end-to-end pipeline, ~2 min
python restore.py --input_dir <a folder of test images> --output_dir /tmp/check
```

The second command is what a reviewer will run. Test it from a **fresh clone** in a
**different directory**, with only `requirements.txt` installed, to confirm nothing
depends on your local setup.
