#!/usr/bin/env bash
# Build a scratch venv with TransformerLens and nnsight next to torchlens's own deps.
set -euo pipefail
V="$HOME/scratch/torchlens-eff-seat2/peers"
mkdir -p "$(dirname "$V")" out
uv venv --python 3.10 "$V" >out/setup.log 2>&1
export VIRTUAL_ENV="$V"
uv pip install --python "$V/bin/python" --index-url https://download.pytorch.org/whl/cpu "torch==2.13.0" >>out/setup.log 2>&1
echo "torch==2.13.0" > out/constraints.txt
uv pip install --python "$V/bin/python" -c out/constraints.txt transformer_lens nnsight py-spy >>out/setup.log 2>&1 || echo "PEERS_INSTALL_FAILED" >>out/setup.log
uv pip install --python "$V/bin/python" -c out/constraints.txt . >>out/setup.log 2>&1 || echo "TL_INSTALL_FAILED" >>out/setup.log
"$V/bin/python" - >>out/setup.log 2>&1 <<'PY'
import importlib
for m in ("torch", "transformers", "transformer_lens", "nnsight", "torchlens"):
    try:
        mod = importlib.import_module(m)
        print("VERSION", m, getattr(mod, "__version__", "?"))
    except Exception as e:
        print("IMPORTFAIL", m, type(e).__name__, str(e)[:300])
PY
tail -20 out/setup.log
