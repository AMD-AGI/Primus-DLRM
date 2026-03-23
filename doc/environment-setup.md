# Environment Setup

## Verified Stack

| Component     | Version                        |
|---------------|--------------------------------|
| Python        | 3.12.3                         |
| PyTorch       | 2.10.0+rocm7.1                 |
| TorchRec      | 1.4.0                          |
| FBGEMM_GPU    | 1.5.0+rocm7.1.25424            |
| ROCm (system) | 7.1.1                          |
| GPU           | 8x AMD Instinct MI355X (gfx950)|

## Quick Start

```bash
cd /mnt/vast/huzhao/projects/dlrm

# 1. Create venv
python3.12 -m venv .venv
source .venv/bin/activate

# 2. Install PyTorch + ROCm (must come from ROCm index)
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm7.1

# 3. Install FBGEMM_GPU (must match PyTorch ROCm version)
pip install fbgemm-gpu --index-url https://download.pytorch.org/whl/rocm7.1

# 4. Install TorchRec
pip install torchrec

# 5. Install remaining dependencies
pip install -r requirements.txt

# 6. Install primus-dlrm in editable mode
pip install -e ".[dev]"
```

## Verify Installation

```bash
python -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'ROCm: {torch.version.hip}')
print(f'GPUs: {torch.cuda.device_count()}x {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')
import torchrec; print(f'TorchRec {torchrec.__version__}')
import fbgemm_gpu; print('FBGEMM_GPU OK')
"
```

Expected output:

```
PyTorch 2.10.0+rocm7.1
ROCm: 7.1.25424
GPUs: 8x AMD Instinct MI355X
TorchRec 1.4.0
FBGEMM_GPU OK
```

## Run Tests

```bash
python -m pytest tests/ -v
```

## MI355X (gfx950) Notes

- MI355X uses the gfx950 GPU architecture.
- PyTorch 2.10.0+rocm7.1 supports gfx950 natively. Earlier PyTorch ROCm builds (rocm6.3 and below) do not include gfx950 kernels and will fail at runtime with `hipErrorNoBinaryForGpu`.
- If you must use an older PyTorch build, set `HSA_OVERRIDE_GFX_VERSION=9.4.2` to map gfx950 to gfx942. This lets PyTorch detect the GPU but kernel execution may fail because gfx950 is not binary compatible with gfx942.
- Always prefer the ROCm 7.1 wheels for MI355X.

## fsspec Compatibility

There is a known version conflict: PyTorch bundles `fsspec>=2025.12.0` but HuggingFace `datasets` requires `fsspec<=2025.10.0`. After installing PyTorch, downgrade fsspec:

```bash
pip install "fsspec<=2025.10.0"
```

This does not affect functionality.

## Data Pipeline

After environment setup, download and preprocess the dataset:

```bash
# Download Yambda-50M from HuggingFace (~2 min)
python scripts/download_data.py --data-dir data/raw --size 50m

# Preprocess: temporal split, session segmentation (~30s)
python scripts/preprocess.py --raw-dir data/raw --out-dir data/processed --size 50m
```

## Training

```bash
# Quick validation: 5K steps + eval (~8 min)
python -u scripts/quick_validate.py --config configs/dlrm_quick.yaml \
    --processed-dir data/processed --device cuda:0 --max-steps 5000

# Full training: 1 epoch (~50 min on single MI355X)
python -u scripts/train.py --config configs/dlrm_quick.yaml \
    --processed-dir data/processed --device cuda:0
```
