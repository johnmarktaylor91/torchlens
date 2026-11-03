import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

def load(inputs):
    inputs = Path(inputs)
    cfg = json.loads((inputs / "config.json").read_text())
    spec = importlib.util.spec_from_file_location("task_model", inputs / "model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return cfg, module

def run(inputs, outputs, variant=None):
    cfg, module = load(inputs)
    model = module.build_model(cfg["weight_seed"])
    x = torch.from_numpy(module.make_inputs(cfg["stimulus_seed"]))
    logits, acts = model(x, return_intermediates=True)
    cls = 0 if variant == "class_zero" else cfg["objective_class"]
    grads = torch.autograd.grad(logits[:, cls].sum(), tuple(acts.values()))
    result = {}
    for (key, activation), gradient in zip(acts.items(), grads):
        if variant == "modules_only" and key not in ("stem", "conv2"):
            continue
        value = gradient if variant == "gradient_only" else activation * gradient
        if variant == "activation_only":
            value = activation
        arr = value.detach().numpy()
        if variant == "transpose" and arr.ndim == 4:
            arr = arr.swapaxes(-1, -2)
        if variant == "stimulus_order":
            arr = arr[::-1].copy()
        result[key] = arr
    outputs = Path(outputs)
    outputs.mkdir(parents=True, exist_ok=True)
    np.savez(outputs / "grad_act.npz", **result)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", required=True)
    p.add_argument("--outputs", required=True)
    a = p.parse_args()
    run(a.inputs, a.outputs)
