#!/usr/bin/env bash
# Diagnostic-only, not part of the repo history: reproduce the Weekly slow-tier
# environment (torch 2.7.1+cpu / torchvision 0.22.1+cpu, dev+test+tabular
# extras) and run one shard of tests/test_real_world_models.py's slow tier,
# one test per process, to get the exact current failing node ids for the
# known-failures ledger (FJ-weekly-green). Deleted before this branch is
# pushed. Needs SHARD_INDEX and NUM_SHARDS in the environment.
set -ex

if ! command -v dot >/dev/null 2>&1; then
  echo "graphviz 'dot' not found; continuing without it (only affects one render-path test, not validate_forward_pass failures)"
fi

# This worker's ambient pip config injects an unreachable extra index
# (pypi.ngc.nvidia.com) that causes intermittent DNS failures; pin to only
# the two indices this install actually needs.
export PIP_CONFIG_FILE=/dev/null
unset PIP_EXTRA_INDEX_URL PIP_INDEX_URL PIP_INDEX

cat > /tmp/torch-2.7-constraints.txt <<'EOF'
torch==2.7.1+cpu
torchvision==0.22.1+cpu
EOF

# The system python3 on some workers is 3.8 (pyproject requires >=3.10 and
# modern setuptools refuses to build on it); use uv for an isolated 3.11 venv
# instead of relying on whatever interpreters happen to be on PATH.
uv venv --python 3.11 "/tmp/tl-weekly-venv-shard${SHARD_INDEX}"
export VIRTUAL_ENV="/tmp/tl-weekly-venv-shard${SHARD_INDEX}"
export PATH="/tmp/tl-weekly-venv-shard${SHARD_INDEX}/bin:$PATH"

uv pip install -c /tmp/torch-2.7-constraints.txt \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple \
  --index-strategy unsafe-best-match \
  -e ".[dev,test,tabular]" \
  "torch==2.7.1+cpu" \
  "torchvision==0.22.1+cpu"

python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__)"

python _diag_collect_real_model_failures.py \
  --repo "$(pwd)" \
  --target tests/test_real_world_models.py \
  --marker "slow and not rare" \
  --shard-index "${SHARD_INDEX}" \
  --num-shards "${NUM_SHARDS}" \
  --out "shard_results_${SHARD_INDEX}.json"
