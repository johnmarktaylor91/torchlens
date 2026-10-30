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

@torch.no_grad()
def trajectory(model, tokens):
    state = None
    states, logits = [], []
    for time in range(tokens.shape[1]):
        out, state = model(tokens[:, time:time + 1], state)
        logits.append(out[:, 0])
        states.append(state.clone())
    return torch.stack(logits, 1), states


@torch.no_grad()
def run(inputs, outputs, variant=None):
    cfg, module = load(inputs)
    model = module.build_model(cfg["weight_seed"])
    clean, corrupt = [torch.from_numpy(x) for x in module.make_pairs(
        cfg["stimulus_seed"], cfg["n_pairs"], cfg["length"])]
    cl, cs = trajectory(model, clean)
    co, os = trajectory(model, corrupt)
    base, donor = (clean, os) if variant == "reversed_roles" else (corrupt, cs)
    patched = torch.full((len(clean), len(cfg["steps"]), cfg["length"], 64), float("nan"))
    for j, k in enumerate(cfg["steps"]):
        state = None
        for time in range(cfg["length"]):
            out, state = model(base[:, time:time + 1], state)
            when = k + 1 if variant == "off_by_one" else k
            if time == when or (variant == "reinject" and time > k):
                if variant == "top_only":
                    state = torch.stack((state[0], donor[time][1]))
                else:
                    state = donor[time].clone()
            if time > k:
                patched[:, j, time] = out[:, 0]
    arrays = dict(logits=patched.numpy(), clean_logits=cl.numpy(), corrupt_logits=co.numpy())
    if variant == "transpose":
        arrays["logits"] = arrays["logits"].transpose(1, 0, 2, 3)
    if variant == "stimulus_order":
        arrays = {key: value[::-1].copy() for key, value in arrays.items()}
    outputs = Path(outputs)
    outputs.mkdir(parents=True, exist_ok=True)
    np.savez(outputs / "patched.npz", **arrays)

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", required=True)
    p.add_argument("--outputs", required=True)
    a = p.parse_args()
    run(a.inputs, a.outputs)
