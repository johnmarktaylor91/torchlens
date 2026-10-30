#!/usr/bin/env bash
# Create the CPU test environment for the oracles. Run from the oracles directory.
# Plain PyTorch, transformers and numerical libraries only.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
if [ ! -x .venv/bin/python ]; then
  "$PY" -m venv .venv
fi
. .venv/bin/activate
python -m pip install --quiet --upgrade pip
python -m pip install --quiet --index-url https://download.pytorch.org/whl/cpu torch torchvision
python -m pip install --quiet "transformers>=4.40" numpy scipy safetensors
python - <<'PY'
import torch, transformers, numpy, scipy, torchvision
print("torch", torch.__version__, "transformers", transformers.__version__,
      "numpy", numpy.__version__, "torchvision", torchvision.__version__)
PY
