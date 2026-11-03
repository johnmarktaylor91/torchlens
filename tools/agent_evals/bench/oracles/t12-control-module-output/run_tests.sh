#!/usr/bin/env bash
# Runs on the worker from the oracles directory after _common/worker_setup.sh.
cd "$(dirname "$0")/.." && exec _common/run_all.sh t12-control-module-output
