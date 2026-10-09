#!/usr/bin/env bash
# Fingerprint the base commit and this checkout with the same scripts, then compare.
# Usage: exact_pair.sh BASE_SHA PYTHON [CONFIGS]
set -uo pipefail
BASE="$1"; PY="$2"; ONLY="${3:-}"
mkdir -p out/basesrc
git archive "$BASE" torchlens | tar -x -C out/basesrc
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
for m in gpt2 q3s; do
  PYTHONPATH="$PWD/out/basesrc" "$PY" benchmarks/eff_seat2/exact.py --model $m --out out/exact_base.jsonl ${ONLY:+--only $ONLY} >>out/log.txt 2>&1
  PYTHONPATH="$PWD" "$PY" benchmarks/eff_seat2/exact.py --model $m --out out/exact_head.jsonl ${ONLY:+--only $ONLY} >>out/log.txt 2>&1
done
"$PY" benchmarks/eff_seat2/exact.py --compare out/exact_base.jsonl out/exact_head.jsonl | tee out/compare.txt
