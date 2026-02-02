#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python3.10}"

# PyTorch / CUDA selection
# - Set CUDA_TAG to match the PyTorch wheel index, e.g. cu118, cu121, cu124 (or cpu)
# - Keep torch/torchvision/torchaudio versions aligned
CUDA_TAG="${CUDA_TAG:-cu121}"
TORCH_VERSION="${TORCH_VERSION:-2.1.0}"
TORCHVISION_VERSION="${TORCHVISION_VERSION:-0.16.0}"
TORCHAUDIO_VERSION="${TORCHAUDIO_VERSION:-2.1.0}"
TORCH_SCATTER_VERSION="${TORCH_SCATTER_VERSION:-2.1.2}"

case "$CUDA_TAG" in
  cu118|cu121|cu124|cpu) ;;
  *)
    echo "ERROR: Unsupported CUDA_TAG='$CUDA_TAG'. Use one of: cu118, cu121, cu124, cpu" >&2
    exit 2
    ;;
esac

TORCH_INDEX_URL="https://download.pytorch.org/whl/${CUDA_TAG}"
TORCH_SCATTER_FIND_LINK="https://data.pyg.org/whl/torch-${TORCH_VERSION}+${CUDA_TAG}.html"

echo "=== bootstrap_uv ==="
echo "python: $PYTHON_BIN"
echo "CUDA_TAG: $CUDA_TAG"
echo "torch: $TORCH_VERSION  torchvision: $TORCHVISION_VERSION  torchaudio: $TORCHAUDIO_VERSION"
echo "torch-scatter: $TORCH_SCATTER_VERSION"
if command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi:"; nvidia-smi || true
else
  echo "nvidia-smi: not found (GPU may be unavailable)"
fi

# Clean env
rm -rf .venv
uv venv -p "$PYTHON_BIN"
source .venv/bin/activate

# Ensure pip exists in venv (sometimes needed)
python -m ensurepip --upgrade || true

# Upgrade pip tooling
uv pip install -U pip setuptools wheel

# -------------------------
# 1) Install PyTorch (CUDA 12.1)
# -------------------------
uv pip install \
  "torch==${TORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}" \
  "torchaudio==${TORCHAUDIO_VERSION}" \
  --index-url "$TORCH_INDEX_URL"

# -------------------------
# 2) Install torch-scatter matched to torch 2.1.0 + cu121
# -------------------------
uv pip install \
  "torch-scatter==${TORCH_SCATTER_VERSION}" \
  -f "$TORCH_SCATTER_FIND_LINK"

# -------------------------
# 3) Install the rest (pinned)
# -------------------------
if [[ ! -f requirements.txt ]]; then
  if [[ -f uv.lock ]]; then
    uv export --format requirements.txt --output-file requirements.txt --frozen --no-hashes --no-emit-project
  else
    echo "ERROR: requirements.txt not found and uv.lock missing; can't install remaining deps." >&2
    exit 2
  fi
fi
uv pip install -r requirements.txt

# -------------------------
# 4) Verify
# -------------------------
python - <<'PY'
import numpy as np
import torch
import torch_scatter

x = torch.randn(3, device="cuda")
print("✅ numpy:", np.__version__)
print("✅ torch:", torch.__version__, "cuda:", torch.version.cuda, "cuda_available:", torch.cuda.is_available())
print("✅ to numpy:", x.cpu().numpy().shape)
print("✅ torch_scatter:", torch_scatter.__version__)
PY
