"""Seeded wrong solution: wrong_module."""
import argparse, importlib.util, json, os
import numpy as np
import torch

VARIANT = "wrong_module"


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
    name = cfg["module"]
    if VARIANT == "wrong_module":
        name = "block2.conv"
    if VARIANT == "stimulus_order":
        x = torch.flip(x, dims=[0])
    if VARIANT == "train_mode_bn":
        model.train()
    module = dict(model.named_modules())[name]
    chunks = []
    h = module.register_forward_hook(lambda mod, inp, out: chunks.append(out.detach().clone()))
    with torch.no_grad():
        for i in range(0, len(x), 16):
            model(x[i:i + 16])
    h.remove()
    acts = torch.cat(chunks).numpy().astype(np.float32)
    if VARIANT == "transpose":
        acts = np.ascontiguousarray(acts.transpose(0, 1, 3, 2))
    os.makedirs(a.outputs, exist_ok=True)
    np.savez(os.path.join(a.outputs, "activations.npz"), activations=acts)


if __name__ == "__main__":
    main()
