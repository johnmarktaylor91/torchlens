"""T12 grader: exact comparison with an independent computation of the module output."""
import importlib.util
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_common"))
from grade_utils import Report, grader_main, load_npz, max_abs  # noqa: E402


def oracle(inputs_dir):
    cfg = json.load(open(os.path.join(inputs_dir, "config.json")))
    spec = importlib.util.spec_from_file_location("task_model_oracle", os.path.join(inputs_dir, "model.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    model = m.build_model(cfg["weight_seed"])
    x = torch.from_numpy(m.make_stimuli(cfg["stimulus_seed"], cfg["n_inputs"], cfg["input_shape"]))
    # Independent route: run the submodules explicitly up to the named module,
    # one input at a time (batch-size independence is part of the check).
    outs = []
    with torch.no_grad():
        for i in range(len(x)):
            h = model.stem(x[i:i + 1])
            h = model.block1(h)
            h = model.block2.conv(h)
            h = model.block2.bn(h)
            outs.append(h)
    return torch.cat(outs).numpy(), cfg


def grade_one(rep, label, inputs_dir, outputs_dir):
    ref, cfg = oracle(inputs_dir)
    try:
        acts = load_npz(os.path.join(outputs_dir, "activations.npz"))["activations"]
    except Exception as e:  # missing or malformed
        rep.fail_all([f"{label}:shape", f"{label}:exact"], f"missing output: {e}")
        return
    rep.add(f"{label}:shape", tuple(acts.shape) == ref.shape, note=f"shape={acts.shape}")
    err = max_abs(acts, ref)
    rep.add(f"{label}:exact", err <= 1e-4, err, 1e-4)


def grade(task_dir, outputs, heldout_outputs, cache):
    rep = Report("t12-control-module-output")
    grade_one(rep, "visible", os.path.join(task_dir, "inputs"), outputs)
    if heldout_outputs:
        grade_one(rep, "heldout", os.path.join(task_dir, "heldout"), heldout_outputs)
    return rep


if __name__ == "__main__":
    sys.exit(grader_main("t12", grade))
