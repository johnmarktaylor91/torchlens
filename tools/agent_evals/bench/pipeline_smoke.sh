#!/usr/bin/env bash
# Model-free smoke of the benchmark pipeline: build arm environments, audit them, check that
# every guard fires, check the stub task's oracle, then drive one run with the scripted fake
# agent through the real runner, grader and record. Needs uv and network for the environment
# builds only. Usage (from the repository root): tools/agent_evals/bench/pipeline_smoke.sh OUT [ARMS]
set -uo pipefail

OUT=$(mkdir -p "${1:-outputs/bench-smoke}" && cd "${1:-outputs/bench-smoke}" && pwd)
ARMS=${2:-A B C D}
REPO=$(pwd)
ENVS="$OUT/envs"
cd tools || exit 2
status=0

for arm in $ARMS; do
  echo "== build-env $arm"
  if ! python3 -m agent_evals.bench build-env --arm "$arm" --dest "$ENVS/$arm" --repo "$REPO" \
      >"$OUT/build-$arm.log" 2>&1; then
    echo "build-env $arm FAILED"; tail -n 30 "$OUT/build-$arm.log"; status=1
  else
    tail -n 1 "$OUT/build-$arm.log"
  fi
  [ -f "$ENVS/$arm/env.json" ] && cp "$ENVS/$arm/env.json" "$OUT/env-$arm.json"
done

PY="$ENVS/A/bin/python"
[ -x "$PY" ] || { echo "arm A environment missing; stopping"; exit 1; }

echo "== check-task T0"
"$PY" -m agent_evals.bench check-task --oracles agent_evals/bench/stub --task T0 --python "$PY" \
  --out "$OUT/check-T0" || status=1

cat >"$OUT/fake-tiers.json" <<EOF
{"tiers": {"small": {"agent_cmd": ["$PY", "$REPO/tools/agent_evals/bench/fake_agent.py",
  "--mode", "solve", "--solution", "$REPO/tools/agent_evals/bench/stub/T0/reference.py"],
  "caps": {"wall_s": 300, "checkpoints_s": [60, 120, 300]}}}}
EOF
echo "== fake-agent run T0 arm A small"
"$PY" -m agent_evals.bench run --oracles agent_evals/bench/stub --task T0 --arm A --tier small \
  --config "$OUT/fake-tiers.json" --envs "$ENVS" --out "$OUT/runs/T0-A-small-r0" || status=1
"$PY" -m agent_evals.bench summarize --results "$OUT/runs"
exit $status
