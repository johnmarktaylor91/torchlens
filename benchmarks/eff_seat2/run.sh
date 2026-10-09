#!/usr/bin/env bash
# Run a list of benchmark commands (one per line in $1) with the checkout on PYTHONPATH.
# Usage: run.sh CMDFILE [PYTHON]
set -uo pipefail
PY="${2:-$HOME/scratch/torchlens-r236/smoke/bin/python}"
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export TL_COMMIT="$(git rev-parse --short HEAD 2>/dev/null || echo "${RUN_ON_COMMIT:-?}")"
mkdir -p out
echo "host=$(hostname) cpu=$(lscpu | sed -n 's/^Model name: *//p') cores=$(nproc) load=$(cut -d' ' -f1-3 /proc/loadavg) commit=$TL_COMMIT" | tee -a out/env.txt
while IFS= read -r line; do
  [ -z "$line" ] && continue
  case "$line" in \#*) continue;; esac
  echo ">>> $line" >>out/log.txt
  eval "\"$PY\" $line" >>out/log.txt 2>&1 || echo "FAILED rc=$? : $line" | tee -a out/log.txt
done <"$1"
tail -c 3000 out/log.txt
