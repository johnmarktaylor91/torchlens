#!/usr/bin/env bash
# Usage: _common/run_all.sh t01-jacobian-lens t07-rsa ...   (from the oracles directory)
set -uo pipefail
cd "$(dirname "$0")/.."
. .venv/bin/activate
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-8}
rc=0
for t in "$@"; do
  python _common/test_task.py "$t" || rc=1
done
echo "ALL_TASKS_RC=$rc"
exit $rc
