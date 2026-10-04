#!/usr/bin/env bash
# Diagnostic-only, not part of the repo history: reproduce the Weekly slow-tier
# environment (torch 2.7.1+cpu / torchvision 0.22.1+cpu, graphviz, dev+test+tabular
# extras) and run tests/test_real_world_models.py's slow tier, one test per
# process, to get the exact current failing node ids for the known-failures
# ledger (FJ-weekly-green). Deleted before this branch is pushed.
set -ex

if ! command -v dot >/dev/null 2>&1; then
  echo "graphviz 'dot' not found and no passwordless sudo on this worker; continuing without it (only affects render-path tests, not validate_forward_pass failures)"
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

python3 -m pip install --upgrade pip
python3 -m pip install -c /tmp/torch-2.7-constraints.txt \
  --index-url https://download.pytorch.org/whl/cpu \
  --extra-index-url https://pypi.org/simple \
  -e ".[dev,test,tabular]" \
  "torch==2.7.1+cpu" \
  "torchvision==0.22.1+cpu"

python3 -c "import torch, torchvision; print(torch.__version__, torchvision.__version__)"

python3 _diag_collect_real_model_failures.py \
  --repo "$(pwd)" \
  --target tests/test_real_world_models.py \
  --marker "slow and not rare" \
  --workers 3 \
  --out realmodel_results.json
