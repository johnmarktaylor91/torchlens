#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.." && exec _common/run_all.sh t03-causal-tracing
