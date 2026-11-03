import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_common"))
from grade_utils import Report, grader_main, load_npz, load_json, max_abs

from transformers import AutoModelForCausalLM, AutoTokenizer


@torch.no_grad()
def oracle(inputs, cache_file):
    torch.set_num_threads(1)
    if cache_file.exists():
        return load_npz(cache_file)
    cfg = load_json(Path(inputs) / "config.json")
    prompts = load_json(Path(inputs) / "prompts.json")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"])
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], torch_dtype=torch.float32,
                                                attn_implementation=cfg["attention_implementation"]).eval()
    block = model.gpt_neox.layers[cfg["layer"]]
    vectors = []
    recorded = []
    def collect(mod, args, output):
        h = output[0] if isinstance(output, (tuple, list)) else output
        recorded.append(h[0, -1].clone())
    handle = block.register_forward_hook(collect)
    try:
        for group in ("positive", "negative"):
            rows = []
            for text in prompts[group]:
                ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids
                model(ids, use_cache=False)
                rows.append(recorded.pop())
            vectors.append(torch.stack(rows).mean(dim=0))
    finally:
        handle.remove()
    direction = (vectors[0] - vectors[1]).float()
    unit = direction / torch.linalg.vector_norm(direction)
    tokens, native, projections, first = [], [], [], []
    for text in prompts["generation"]:
        start = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids
        rows = []
        for steering in (False, True):
            observed = []
            def intervene(mod, args, output):
                h = output[0] if isinstance(output, (tuple, list)) else output
                observed.append((h[0, -1] * unit).sum().item())
                changed = h + cfg["coefficient"] * unit.view(1, 1, -1)
                if isinstance(output, tuple):
                    return (changed,) + output[1:]
                if isinstance(output, list):
                    return [changed] + output[1:]
                return changed
            handle = block.register_forward_hook(intervene) if steering else None
            ids = start.clone()
            row = []
            try:
                for step in range(cfg["new_tokens"]):
                    scores = model(input_ids=ids, use_cache=False).logits[0, -1]
                    if steering and step == 0:
                        first.append(scores.numpy())
                    next_id = int(torch.argmax(scores))
                    row.append(next_id)
                    ids = torch.cat([ids, torch.tensor([[next_id]])], 1)
            finally:
                if handle is not None:
                    handle.remove()
            rows.append(row)
            if steering:
                projections.append(observed)
        native.append(rows[0])
        tokens.append(rows[1])
    result = dict(direction=direction.numpy(), logits=np.stack(first), tokens=np.array(tokens),
                  native=np.array(native), projections=np.array(projections, dtype=np.float32))
    np.savez(cache_file, **result)
    return result


def grade_one(rep, label, inputs, outputs, cache):
    names = [f"{label}:{name}" for name in ("shapes", "direction_cosine", "first_logits_relative_max",
             "steered_token_agreement", "native_token_agreement", "projection_max_abs", "time_ratio")]
    ref = oracle(inputs, Path(cache) / f"t05_{label}.npz")
    try:
        direction = load_npz(Path(outputs) / "direction.npz")["direction"]
        logits = load_npz(Path(outputs) / "steered_first_logits.npz")["logits"]
        items = load_json(Path(outputs) / "generation.json")["items"]
        timing = load_json(Path(outputs) / "timing.json")
        tokens = np.asarray([row["steered_token_ids"] for row in items])
        native = np.asarray([row["native_token_ids"] for row in items])
        projections = np.asarray([row["projections"] for row in items], dtype=np.float64)
        valid = (direction.shape == ref["direction"].shape and logits.shape == ref["logits"].shape
                 and tokens.shape == ref["tokens"].shape and native.shape == ref["native"].shape
                 and projections.shape == ref["projections"].shape and direction.dtype == np.float32
                 and logits.dtype == np.float32 and np.issubdtype(tokens.dtype, np.integer)
                 and np.issubdtype(native.dtype, np.integer)
                 and all(np.isfinite(x).all() for x in (direction, logits, projections)))
        cosine = float(direction.astype(float) @ ref["direction"].astype(float) /
                       (np.linalg.norm(direction.astype(float)) * np.linalg.norm(ref["direction"].astype(float)))) if valid and np.linalg.norm(direction) > 0 else -1.0
        error = max_abs(logits, ref["logits"]) / float(np.max(np.abs(ref["logits"]))) if valid else float("inf")
        agreement = float(np.mean(tokens == ref["tokens"])) if valid else 0.0
        native_agreement = float(np.mean(native == ref["native"])) if valid else 0.0
        projection_error = max_abs(projections, ref["projections"]) if valid else float("inf")
        nt, st = float(timing["native_seconds"]), float(timing["steered_seconds"])
        ratio = st / nt if np.isfinite(nt) and np.isfinite(st) and nt > 0 and st > 0 else float("inf")
    except Exception as exc:  # Malformed solution files fail at this boundary.
        rep.fail_all(names, str(exc))
        return
    rep.add(names[0], valid, int(valid), 1)
    rep.add(names[1], cosine >= .999, cosine, .999)
    rep.add(names[2], error <= .01, error, .01)
    rep.add(names[3], agreement >= .9, agreement, .9)
    rep.add(names[4], native_agreement == 1, native_agreement, 1)
    # Float32 mean reduction and scalar projection can accumulate sub-milliscale noise.
    rep.add(names[5], projection_error < 1e-3, projection_error, 1e-3)
    rep.add(names[6], ratio <= 1.5, ratio, 1.5, note="same-process measurement; advisory across machines")


def grade(task_dir, outputs, heldout_outputs, cache):
    rep = Report("t05-diff-of-means-steering")
    grade_one(rep, "visible", Path(task_dir) / "inputs", outputs, cache)
    if heldout_outputs:
        grade_one(rep, "heldout", Path(task_dir) / "heldout", heldout_outputs, cache)
    return rep


if __name__ == "__main__":
    sys.exit(grader_main("t05", grade))
