# Trained model weights

`restore.py` expects the checkpoint at **`weights/best.pth`**.

## Download

| File | Size | Link |
|---|---|---|
| `best.pth` | ~291 MB | https://drive.google.com/file/d/1BWimh8iaAsISkys6lAy2XELVjWVHVPXg/view?usp=sharing |

```bash
# Google Drive
pip install gdown
gdown 1BWimh8iaAsISkys6lAy2XELVjWVHVPXg -O weights/best.pth

# or Hugging Face
huggingface-cli download <user>/<repo> best.pth --local-dir weights/
```

Or download it in a browser and drop it in this folder.

## What is inside

A dict with `model` (raw weights), `ema` (exponential moving average — this is what
`restore.py` loads by default and what the reported scores were measured with), `opt`,
`iter`, `args`, and `val`.

`restore.py` reads the architecture preset from `args`, so nothing needs configuring.
A bare `state_dict` also loads, in which case the `medium` preset is assumed.

| | |
|---|---|
| architecture | NAFNet-SR, `medium` preset |
| parameters | 19.02 M (~73 MB of fp32 weights) |
| iteration | 105 000 |
| validation | PSNR 23.671 · SSIM 0.6097 · LPIPS 0.3412 on held-out group c04 |

## If the file is tracked with Git LFS instead

```bash
git lfs install
git lfs pull
```

The repository's `.gitattributes` already routes `*.pth` through LFS. Note that
GitHub's free LFS quota is 1 GB of storage and 1 GB/month of bandwidth, so a
Drive or Hugging Face link is usually the safer choice for a file this size.
