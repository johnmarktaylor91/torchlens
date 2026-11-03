"""Reference: forward hook on the named module, batched, eval mode."""
import argparse
import importlib.util
import json
import os

import numpy as np
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", required=True)
    p.add_argument("--outputs", required=True)
    a = p.parse_args()
    cfg = json.load(open(os.path.join(a.inputs, "config.json")))
    spec = importlib.util.spec_from_file_location("task_model", os.path.join(a.inputs, "model.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    model = m.build_model(cfg["weight_seed"])
    x = torch.from_numpy(m.make_stimuli(cfg["stimulus_seed"], cfg["n_inputs"], cfg["input_shape"]))
    module = dict(model.named_modules())[cfg["module"]]
    chunks = []
    h = module.register_forward_hook(lambda mod, inp, out: chunks.append(out.detach().clone()))
    with torch.no_grad():
        for i in range(0, len(x), 16):
            model(x[i:i + 16])
    h.remove()
    acts = torch.cat(chunks).numpy().astype(np.float32)
    os.makedirs(a.outputs, exist_ok=True)
    np.savez(os.path.join(a.outputs, "activations.npz"), activations=acts)


if __name__ == "__main__":
    main()
