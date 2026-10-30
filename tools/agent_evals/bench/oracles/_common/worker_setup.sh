#!/usr/bin/env bash
# Create the benchmark's CPU environment for the oracles, pinned to the same
# versions the subject agents run in. Works in the oracles directory.
set -euo pipefail
cd "$(dirname "$0")/.."
PY_VERSION=${ORACLE_PYTHON:-3.10}
PKGS=(torch==2.13.0 torchvision==0.28.0 transformers==5.18.0 numpy scipy safetensors)
if [ ! -x .venv/bin/python ]; then
  if command -v uv >/dev/null 2>&1; then
    uv venv --quiet --python "$PY_VERSION" .venv
  else
    "python$PY_VERSION" -m venv .venv
  fi
fi
. .venv/bin/activate
if command -v uv >/dev/null 2>&1; then
  uv pip install --quiet --python .venv/bin/python --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match "${PKGS[@]}"
else
  python -m pip install --quiet --upgrade pip
  python -m pip install --quiet --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple "${PKGS[@]}"
fi
python - <<'PY'
import sys, torch, transformers, numpy, scipy, torchvision
print("python", sys.version.split()[0], "torch", torch.__version__, "transformers", transformers.__version__,
      "numpy", numpy.__version__, "torchvision", torchvision.__version__)
PY
