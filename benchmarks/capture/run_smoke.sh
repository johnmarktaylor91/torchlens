#!/usr/bin/env bash
# CPU smoke of the capture benchmarks: build the pinned environment, add the peer
# packages under a constraint that keeps its torch, run the harness tests, then the
# smoke matrix. Results go to outputs/ beside the checkout.
set -u
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p outputs
python3 -c 'import sys; sys.path.insert(0, "scripts"); import smoke_ci_parity as s; s.ensure_venv(s.DEFAULT_VENV, False)' || exit 1
PY=.venv-ci-smoke/bin/python
"$PY" -c 'import torch, torchvision, transformers; print(f"torch=={torch.__version__}\ntorchvision=={torchvision.__version__}\ntransformers=={transformers.__version__}")' > outputs/constraints.txt
# thingsvision pins numpy 1.22, pandas 1.4, matplotlib 3.5, scipy 1.8 and torchtyping 0.1.4
# exactly; those builds cannot load next to numpy 2 and torch 2.13. The environment's own
# numpy, pandas and matplotlib override them, plus the exact versions in peer-overrides.txt.
{ uv pip freeze --python "$PY" | grep -i -E '^(numpy|pandas|matplotlib)=='
  grep -v '^#' benchmarks/capture/peer-overrides.txt | grep .; } > outputs/overrides.txt
cat outputs/constraints.txt outputs/overrides.txt
"$PY" -m pytest tests/test_capture_benchmark_harness.py -q -p no:cacheprovider 2>&1 | tail -15
grep -v '^#' benchmarks/capture/peers.txt | grep . | while read -r peer; do
  if ! uv pip install --python "$PY" -q -c outputs/constraints.txt --overrides outputs/overrides.txt "$peer" > "outputs/install-${peer%%=*}.log" 2>&1; then
    echo "peer install failed: $peer"; tail -5 "outputs/install-${peer%%=*}.log"
  fi
done
for mod in captum.attr thingsvision nnsight transformer_lens; do
  "$PY" -c "import $mod" > "outputs/import-$mod.log" 2>&1 && echo "import ok: $mod" || { echo "import failed: $mod"; tail -3 "outputs/import-$mod.log"; }
done | tee outputs/peer-imports.txt
uv pip freeze --python "$PY" | grep -i -E '^(torch|torchvision|transformers|nnsight|captum|thingsvision|transformer.lens|psutil)==' | tee outputs/peer-freeze.txt
"$PY" benchmarks/capture/driver.py --smoke --processes "${SMOKE_PROCESSES:-1}" --warmups "${SMOKE_WARMUPS:-2}" \
  --samples "${SMOKE_SAMPLES:-10}" --budget-s 20 ${SMOKE_EXTRA:-} --out outputs/capture-smoke.json 2>&1
