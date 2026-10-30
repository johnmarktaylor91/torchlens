"""Deterministic grader for the harness stub: exact equality with a direct recomputation."""

import argparse
import importlib.util
import json
import os

import numpy as np
import torch


def oracle(inputs: str) -> np.ndarray:
    spec = importlib.util.spec_from_file_location("model", os.path.join(inputs, "model.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    seed = json.load(open(os.path.join(inputs, "config.json")))["seed"]
    model = module.build_model()
    with torch.no_grad():
        return model.fc2(model.act(model.fc1(module.make_inputs(seed)))).numpy()


def bar(name: str, outputs: str | None, inputs: str) -> dict:
    path = os.path.join(outputs or "", "outputs.npz")
    if not outputs or not os.path.isfile(path):
        return {"name": name, "passed": False, "value": "missing", "threshold": "exact"}
    with np.load(path) as data:
        got = data["out"] if "out" in data else None
    want = oracle(inputs)
    if got is None or got.shape != want.shape:
        return {"name": name, "passed": False, "value": "wrong shape", "threshold": "exact"}
    return {
        "name": name,
        "passed": bool(np.array_equal(got, want)),
        "value": float(abs(got - want).max()),
        "threshold": "exact",
    }


parser = argparse.ArgumentParser()
parser.add_argument("--task-dir", required=True)
parser.add_argument("--outputs", required=True)
parser.add_argument("--heldout-outputs")
parser.add_argument("--cache")
args = parser.parse_args()
bars = [
    bar("visible exact", args.outputs, os.path.join(args.task_dir, "inputs")),
    bar("held-out exact", args.heldout_outputs, os.path.join(args.task_dir, "heldout")),
]
print(json.dumps({"task": "T0", "passed": all(b["passed"] for b in bars), "bars": bars}))
