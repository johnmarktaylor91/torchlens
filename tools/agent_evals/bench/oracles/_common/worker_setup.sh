#!/usr/bin/env bash
# Create the CPU test environment for the oracles. Run from anywhere; it works
# in the oracles directory. Plain PyTorch, transformers and numerical libraries only.
set -euo pipefail
cd "$(dirname "$0")/.."
if [ ! -x .venv/bin/python ]; then
  if command -v uv >/dev/null 2>&1; then
    uv venv --quiet --python 3.12 .venv
  else
    "${PYTHON:-python3}" -m venv .venv
  fi
fi
. .venv/bin/activate
PKGS=(torch torchvision "transformers>=4.40" numpy scipy safetensors)
if command -v uv >/dev/null 2>&1; then
  uv pip install --quiet --python .venv/bin/python --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match "${PKGS[@]}"
else
  python -m pip install --quiet --upgrade pip
  python -m pip install --quiet --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple "${PKGS[@]}"
fi
python - <<'PY'
import torch, transformers, numpy, scipy, torchvision
print("torch", torch.__version__, "transformers", transformers.__version__,
      "numpy", numpy.__version__, "torchvision", torchvision.__version__)
PY
