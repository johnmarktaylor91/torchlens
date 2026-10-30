import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_common"))
from grade_utils import Report, grader_main, load_npz, iou

def load(inputs):
    inputs = Path(inputs)
    cfg = json.loads((inputs / "config.json").read_text())
    spec = importlib.util.spec_from_file_location("task_model", inputs / "model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return cfg, module

# Each tuple is (input size, kernel, stride, padding, dilation).
ARITHMETIC = [(64, 5, 1, 2, 1), (64, 2, 2, 0, 1),
              (32, 3, 2, 1, 1), (16, 3, 1, 2, 2)]
DEPTH = {"relu1": 1, "pool1": 2, "relu2": 3, "relu3": 4}


def analytic_axis(index, depth):
    indices = {index}
    for size, kernel, stride, pad, dilation in reversed(ARITHMETIC[:depth]):
        indices = {out * stride - pad + dilation * offset
                   for out in indices for offset in range(kernel)
                   if 0 <= out * stride - pad + dilation * offset < size}
    return sorted(indices)


def oracle(inputs, cache_file):
    if cache_file.exists():
        return load_npz(cache_file)
    cfg, module = load(inputs)
    m = module.build_model(cfg["weight_seed"])
    images = module.make_inputs(cfg["stimulus_seed"])
    fields, gradients, effects = [], [], []
    for unit in cfg["units"]:
        rows = analytic_axis(unit["row"], DEPTH[unit["layer"]])
        columns = analytic_axis(unit["column"], DEPTH[unit["layer"]])
        mask = np.zeros((64, 64), dtype=bool)
        mask[np.ix_(rows, columns)] = True
        total = np.zeros((64, 64), dtype=np.float64)
        changes = []
        for image in images:
            x = torch.tensor(image[None], requires_grad=True)
            # Capture the actual activation, independently of the reference route.
            recorded = []
            handle = getattr(m, unit["layer"]).register_forward_hook(
                lambda mod, args, value: recorded.append(value))
            try:
                native = m(x)
            finally:
                handle.remove()
            scalar = recorded[0][0, unit["channel"], unit["row"], unit["column"]]
            derivative = torch.autograd.grad(scalar, x)[0][0, 0]
            total += derivative.abs().numpy()
            ablated = x.detach().clone()
            ablated[0, 0, torch.from_numpy(mask)] = 0
            with torch.no_grad():
                changes.append(float(m(ablated)[0, 0] - native[0, 0]))
        fields.append(mask)
        gradients.append(total / len(images) > cfg["gradient_threshold"])
        effects.append(np.mean(changes))
    result = dict(geometric=np.stack(fields), gradient=np.stack(gradients),
                  effects=np.array(effects, dtype=np.float32))
    np.savez(cache_file, **result)
    return result


def grade_one(rep, label, inputs, outputs, cache):
    names = [f"{label}:shapes", f"{label}:geometric_exact", f"{label}:gradient_iou",
             f"{label}:ablation_scaled_error"]
    ref = oracle(inputs, Path(cache) / f"t08_{label}.npz")
    try:
        geo = load_npz(Path(outputs) / "geometric.npz")["masks"]
        grad = load_npz(Path(outputs) / "gradient.npz")["masks"]
        effect = load_npz(Path(outputs) / "ablation.npz")["effects"]
        valid = (geo.shape == ref["geometric"].shape and grad.shape == ref["gradient"].shape
                 and effect.shape == ref["effects"].shape and geo.dtype == bool
                 and grad.dtype == bool and effect.dtype == np.float32 and np.isfinite(effect).all())
        overlap = min(iou(a, b) for a, b in zip(grad, ref["gradient"])) if valid else 0.0
        allowance = np.where(np.abs(ref["effects"]) < 1e-2, 1e-4, .01 * np.abs(ref["effects"]))
        error = float(np.max(np.abs(effect - ref["effects"]) / allowance)) if valid else float("inf")
    except (OSError, ValueError, TypeError, KeyError) as exc:
        rep.fail_all(names, str(exc))
        return
    rep.add(names[0], valid, int(valid), 1)
    rep.add(names[1], valid and np.array_equal(geo, ref["geometric"]),
            int(np.count_nonzero(geo != ref["geometric"])) if valid else None, 0)
    rep.add(names[2], valid and overlap >= .95, overlap, .95)
    rep.add(names[3], valid and error <= 1, error, 1,
            note=f"oracle_effects={ref['effects'].tolist()}")


def grade(task_dir, outputs, heldout_outputs, cache):
    rep = Report("t08-receptive-field-ablation")
    grade_one(rep, "visible", Path(task_dir) / "inputs", outputs, cache)
    if heldout_outputs:
        grade_one(rep, "heldout", Path(task_dir) / "heldout", heldout_outputs, cache)
    return rep


if __name__ == "__main__":
    sys.exit(grader_main("t08", grade))
