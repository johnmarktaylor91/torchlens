import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "_common"))
from grade_utils import Report, grader_main, load_npz, load_json, max_abs

from transformers import AutoModelForCausalLM, AutoTokenizer


def aligned(tokenizer, text, span):
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    # Independent alignment: exact positive overlap of token and character spans.
    indices = [i for i, (start, end) in enumerate(encoded["offset_mapping"])
               if end > start and end > span[0] and start < span[1]]
    if not indices or indices != list(range(indices[0], indices[-1] + 1)):
        raise ValueError("Expected a nonempty contiguous token span")
    return torch.tensor([encoded["input_ids"]]), [indices[0], indices[-1] + 1]


def hidden_argument(args, kwargs):
    return args[0] if args else kwargs["hidden_states"]


@torch.no_grad()
def oracle(inputs, cache_file):
    if cache_file.exists():
        return load_npz(cache_file)
    cfg, items = load_json(Path(inputs) / "config.json"), load_json(Path(inputs) / "items.json")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"], use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], torch_dtype=torch.float32,
                                                attn_implementation=cfg["attention_implementation"]).eval()
    # The input to block 7 is exactly the residual after block 6. This avoids
    # the reference's hidden-state output extraction and block-forward replacement.
    next_block = model.transformer.h[cfg["layer"] + 1]
    base, patched, alignments = [], [], []
    for item in items:
        source, sr = aligned(tokenizer, item["source"], item["source_span"])
        target, tr = aligned(tokenizer, item["target"], item["target_span"])
        saved = []
        def capture(mod, args, kwargs):
            saved.append(hidden_argument(args, kwargs)[:, sr[0]:sr[1]].clone())
        handle = next_block.register_forward_pre_hook(capture, with_kwargs=True)
        try:
            model(input_ids=source, use_cache=False)
        finally:
            handle.remove()
        representation = saved[0]
        source_length, target_length = sr[1] - sr[0], tr[1] - tr[0]
        indices = torch.arange(target_length).clamp(max=source_length - 1)
        donor = representation.index_select(1, indices)
        mode = "none" if source_length == target_length else (
            "truncate" if source_length > target_length else "repeat_last")
        base.append(model(input_ids=target, use_cache=False).logits[0, -1].numpy())
        def intervene(mod, args, kwargs):
            h = hidden_argument(args, kwargs).clone()
            h[:, tr[0]:tr[1]] = donor
            if args:
                return (h, *args[1:]), kwargs
            return args, {**kwargs, "hidden_states": h}
        handle = next_block.register_forward_pre_hook(intervene, with_kwargs=True)
        try:
            patched.append(model(input_ids=target, use_cache=False).logits[0, -1].numpy())
        finally:
            handle.remove()
        alignments.append(dict(source_range=sr, target_range=tr, adaptation=mode))
    result = dict(patched_logits=np.stack(patched), baseline_logits=np.stack(base),
                  alignment=np.array(json.dumps(alignments)))
    np.savez(cache_file, **result)
    return result


def grade_one(rep, label, inputs, outputs, cache):
    names = [f"{label}:{key}" for key in ("shapes", "alignment_exact", "patched_max_abs", "baseline_max_abs")]
    ref = oracle(inputs, Path(cache) / f"t11_{label}.npz")
    try:
        got = load_npz(Path(outputs) / "composition.npz")
        alignment = load_json(Path(outputs) / "alignment.json")
        expected_alignment = json.loads(str(ref["alignment"]))
        valid = all(key in got and got[key].shape == ref[key].shape and got[key].dtype == np.float32
                    and np.isfinite(got[key]).all() for key in ("patched_logits", "baseline_logits"))
        patched_error = max_abs(got["patched_logits"], ref["patched_logits"]) if valid else float("inf")
        baseline_error = max_abs(got["baseline_logits"], ref["baseline_logits"]) if valid else float("inf")
        align_ok = alignment == expected_alignment
    except Exception as exc:  # Malformed solution files fail at this boundary.
        rep.fail_all(names, str(exc))
        return
    rep.add(names[0], valid, int(valid), 1)
    rep.add(names[1], align_ok, int(align_ok), 1)
    rep.add(names[2], patched_error < 1e-4, patched_error, 1e-4)
    rep.add(names[3], baseline_error < 1e-4, baseline_error, 1e-4)


def grade(task_dir, outputs, heldout_outputs, cache):
    rep = Report("t11-two-stage-composition")
    grade_one(rep, "visible", Path(task_dir) / "inputs", outputs, cache)
    if heldout_outputs:
        grade_one(rep, "heldout", Path(task_dir) / "heldout", heldout_outputs, cache)
    return rep


if __name__ == "__main__":
    sys.exit(grader_main("t11", grade))
