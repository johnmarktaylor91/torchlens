import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_common"))
from grade_utils import Report, grader_main, load_npz, rel_fro

def load(inputs):
    inputs = Path(inputs)
    cfg = json.loads((inputs / "config.json").read_text())
    spec = importlib.util.spec_from_file_location("task_model", inputs / "model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return cfg, module

def oracle(inputs, cache_file):
    if cache_file.exists():
        return load_npz(cache_file)
    from torch.nn import functional as F
    cfg, module = load(inputs)
    m = module.build_model(cfg["weight_seed"])
    x = torch.from_numpy(module.make_inputs(cfg["stimulus_seed"]))
    # Independently reconstruct the forward, retaining each nonleaf gradient.
    stem = F.conv2d(x, m.stem.weight, m.stem.bias)
    conv1 = F.conv2d(stem, m.conv_weight, m.conv_bias, padding=1)
    relu1 = torch.relu(conv1)
    pool1 = F.max_pool2d(relu1, kernel_size=2, stride=2)
    conv2 = F.conv2d(pool1, m.conv2.weight, m.conv2.bias, padding=1)
    relu2 = torch.relu(conv2)
    add1 = pool1 + relu2
    ln1 = F.layer_norm(add1, add1.shape[1:], m.norm_weight, m.norm_bias, 1e-5)
    flat = ln1.reshape(len(x), -1)
    fc = flat @ m.fc_weight.T + m.fc_bias
    acts = dict(stem=stem, conv1=conv1, relu1=relu1, pool1=pool1, conv2=conv2,
                relu2=relu2, add1=add1, ln1=ln1, flat=flat, fc=fc)
    for value in acts.values():
        value.retain_grad()
    fc[:, cfg["objective_class"]].sum().backward()
    result = {key: (value * value.grad).detach().numpy() for key, value in acts.items()}
    np.savez(cache_file, **result)
    return result


def grade_one(rep, label, inputs, outputs, cache):
    names = [f"{label}:ids_and_shapes", f"{label}:max_relative_frobenius"]
    ref = oracle(inputs, Path(cache) / f"t10_{label}.npz")
    try:
        got = load_npz(Path(outputs) / "grad_act.npz")
        missing = [key for key in ref if key not in got or got[key].shape != ref[key].shape
                   or got[key].dtype != np.float32 or not np.isfinite(got[key]).all()]
        errors = {key: rel_fro(got[key], value) for key, value in ref.items() if key not in missing}
        worst = max(errors.values()) if errors and not missing else float("inf")
    except (OSError, ValueError, TypeError, KeyError) as exc:
        rep.fail_all(names, str(exc))
        return
    rep.add(names[0], not missing, len(ref) - len(missing), len(ref), note=f"missing={missing}")
    rep.add(names[1], worst < 1e-4, worst, 1e-4,
            note=" ".join(f"{k}={v:.6g}" for k, v in errors.items()))


def grade(task_dir, outputs, heldout_outputs, cache):
    rep = Report("t10-grad-times-activation")
    grade_one(rep, "visible", Path(task_dir) / "inputs", outputs, cache)
    if heldout_outputs:
        grade_one(rep, "heldout", Path(task_dir) / "heldout", heldout_outputs, cache)
    return rep


if __name__ == "__main__":
    sys.exit(grader_main("t10", grade))
