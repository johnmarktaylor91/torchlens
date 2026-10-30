import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_common"))
from grade_utils import Report, grader_main, load_npz, max_abs

def load(inputs):
    inputs = Path(inputs)
    cfg = json.loads((inputs / "config.json").read_text())
    spec = importlib.util.spec_from_file_location("task_model", inputs / "model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return cfg, module

def cell(model, tokens, state):
    x = model.embedding(tokens)
    result = []
    for layer in range(2):
        wi = getattr(model.gru, f"weight_ih_l{layer}")
        wh = getattr(model.gru, f"weight_hh_l{layer}")
        bi = getattr(model.gru, f"bias_ih_l{layer}")
        bh = getattr(model.gru, f"bias_hh_l{layer}")
        ir, iz, inn = (x @ wi.T + bi).chunk(3, -1)
        hr, hz, hn = (state[layer] @ wh.T + bh).chunk(3, -1)
        r, z = torch.sigmoid(ir + hr), torch.sigmoid(iz + hz)
        candidate = torch.tanh(inn + r * hn)
        x = (1 - z) * candidate + z * state[layer]
        result.append(x)
    return torch.stack(result)


@torch.no_grad()
def oracle(inputs, cache_file):
    if cache_file.exists():
        return load_npz(cache_file)
    cfg, module = load(inputs)
    model = module.build_model(cfg["weight_seed"])
    clean, corrupt = [torch.from_numpy(x) for x in module.make_pairs(
        cfg["stimulus_seed"], cfg["n_pairs"], cfg["length"])]
    states, native = [], []
    for sequence in (clean, corrupt):
        state = torch.zeros(2, len(clean), 48)
        history, logits = [], []
        for time in range(cfg["length"]):
            state = cell(model, sequence[:, time], state)
            history.append(state)
            logits.append(model.readout(state[-1]))
        states.append(history)
        native.append(torch.stack(logits, 1).numpy())
    patched = np.full((len(clean), len(cfg["steps"]), cfg["length"], 64), np.nan, np.float32)
    for j, k in enumerate(cfg["steps"]):
        state = states[0][k].clone()
        for time in range(k + 1, cfg["length"]):
            state = cell(model, corrupt[:, time], state)
            patched[:, j, time] = model.readout(state[-1]).numpy()
    result = dict(logits=patched, clean_logits=native[0], corrupt_logits=native[1])
    np.savez(cache_file, **result)
    return result


def grade_one(rep, label, inputs, outputs, cache):
    names = [f"{label}:shape_and_mask", f"{label}:max_abs"]
    ref = oracle(inputs, Path(cache) / f"t09_{label}.npz")
    try:
        got = load_npz(Path(outputs) / "patched.npz")
        valid = all(k in got and got[k].shape == v.shape and got[k].dtype == np.float32
                    and np.array_equal(np.isnan(got[k]), np.isnan(v))
                    and np.isfinite(got[k][np.isfinite(v)]).all() for k, v in ref.items())
        error = max(max_abs(got[k][np.isfinite(v)], v[np.isfinite(v)])
                    for k, v in ref.items()) if valid else float("inf")
    except Exception as exc:  # Malformed solution files fail at this boundary.
        rep.fail_all(names, str(exc))
        return
    rep.add(names[0], valid, int(valid), 1)
    rep.add(names[1], error < 1e-5, error, 1e-5)


def grade(task_dir, outputs, heldout_outputs, cache):
    rep = Report("t09-recurrent-state-patch")
    grade_one(rep, "visible", Path(task_dir) / "inputs", outputs, cache)
    if heldout_outputs:
        grade_one(rep, "heldout", Path(task_dir) / "heldout", heldout_outputs, cache)
    return rep


if __name__ == "__main__":
    sys.exit(grader_main("t09", grade))
