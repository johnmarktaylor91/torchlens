"""Oracle: full position-by-position Jacobians from torch.func.jacrev."""
import hashlib
import importlib.util
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_common"))
from grade_utils import Report, grader_main, load_json, load_npz, rel_fro, topk_overlap


def load_model(inputs):
    cfg = load_json(os.path.join(inputs, "config.json"))
    module_spec = importlib.util.spec_from_file_location("oracle_decoder", os.path.join(inputs, "model.py"))
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return cfg, module.build_model(cfg["weight_seed"]).requires_grad_(False)


def read_texts(path):
    return [json.loads(line)["text"] for line in open(path) if line.strip()]


def encode(text, length=None):
    return torch.from_numpy(np.frombuffer(text.encode("utf-8"), dtype=np.uint8).copy().astype(np.int64))[:length]


def exact_fit(model, corpus, cfg):
    matrices = {l: torch.zeros(64, 64, dtype=torch.float64) for l in cfg["source_layers"]}
    for text in corpus:
        ids = encode(text, cfg["max_seq_len"])
        valid = torch.arange(cfg["skip_first_positions"], len(ids) - 1)
        if not len(valid):
            raise ValueError("no valid positions")
        with torch.no_grad():
            sources = model.states(ids)
        for layer in cfg["source_layers"]:
            def remaining(residual):
                h = residual
                for block in model.blocks[layer + 1:cfg["target_layer"] + 1]:
                    h = block(h)
                return h
            # Axes: target position, output dimension, source position, input
            # dimension. No cotangent sum or estimator from the solution is used.
            jacobian = torch.func.jacrev(remaining, chunk_size=64)(sources[layer])
            selected = jacobian.index_select(0, valid).index_select(2, valid)
            matrices[layer] += selected.double().sum(dim=(0, 2)) / len(valid)
    return {l: (value / len(corpus)).float() for l, value in matrices.items()}


@torch.no_grad()
def oracle_readout(model, prompts, matrices, cfg):
    result, own = [], []
    for prompt in prompts:
        ids = encode(prompt)
        residual = model.embed(ids)
        row = []
        for layer, block in enumerate(model.blocks):
            residual = block(residual)
            if layer in cfg["source_layers"]:
                transported = residual[-1] @ matrices[layer].T
                logits = model.unembed(model.final_norm(transported))
                row.append(torch.argsort(logits, descending=True)[:10].tolist())
        result.append(row)
        logits = model.unembed(model.final_norm(residual[-1]))
        own.append(torch.argsort(logits, descending=True)[:10].tolist())
    return np.array(result), np.array(own)


def oracle(inputs, cache):
    fingerprint = hashlib.sha256()
    for name in ("config.json", "model.py", "corpus.jsonl"):
        with open(os.path.join(inputs, name), "rb") as f:
            fingerprint.update(f.read())
    path = os.path.join(cache, "t02-v1-" + fingerprint.hexdigest()[:16] + ".npz")
    cfg, model = load_model(inputs)
    if os.path.isfile(path):
        z = load_npz(path)
        matrices = {l: torch.from_numpy(z[f"J_{l}"]) for l in cfg["source_layers"]}
    else:
        matrices = exact_fit(model, read_texts(os.path.join(inputs, "corpus.jsonl")), cfg)
        np.savez(path, **{f"J_{l}": value.numpy() for l, value in matrices.items()})
    top, own = oracle_readout(model, read_texts(os.path.join(inputs, "readout_prompts.jsonl")), matrices, cfg)
    return cfg, {l: v.numpy() for l, v in matrices.items()}, top, own


def grade_one(rep, label, inputs, outputs, cache):
    names = [f"{label}:{n}" for n in ("shapes", "jacobian", "readout")]
    try:
        z = load_npz(os.path.join(outputs, "lens.npz"))
        rd = load_json(os.path.join(outputs, "readout.json"))
        cfg, ref, expected, own = oracle(inputs, cache)
        top = np.asarray(rd["top10"])
        model_top = np.asarray(rd["model_top10"])
        shapes = list(rd["layers"]) == cfg["source_layers"] and top.shape == expected.shape and model_top.shape == own.shape
        shapes = shapes and all(f"J_{l}" in z and z[f"J_{l}"].shape == (64, 64)
                                and np.isfinite(z[f"J_{l}"]).all() for l in ref)
        shapes = shapes and np.issubdtype(top.dtype, np.integer) and np.all((top >= 0) & (top < 256))
        rep.add(names[0], shapes)
        errors = {l: rel_fro(z.get(f"J_{l}", np.array([])), value) for l, value in ref.items()}
        worst = max(errors.values())
        rep.add(names[1], worst < 1e-3, worst, 1e-3, " ".join(f"L{l}={v:.5g}" for l, v in errors.items()))
        overlap = topk_overlap(top, expected, 10) if top.shape == expected.shape else 0.0
        own_overlap = topk_overlap(model_top, own, 10) if model_top.shape == own.shape else 0.0
        rep.add(names[2], overlap >= 0.9 and own_overlap >= 0.9, overlap, 0.9,
                f"model_top10_overlap={own_overlap:.6g}")
    except (OSError, KeyError, ValueError, TypeError) as exc:
        rep.fail_all(names, str(exc))


def grade(task_dir, outputs, heldout_outputs, cache):
    rep = Report("t02-jacobian-lens-new-architecture")
    grade_one(rep, "visible", os.path.join(task_dir, "inputs"), outputs, cache)
    if heldout_outputs:
        grade_one(rep, "heldout", os.path.join(task_dir, "heldout"), heldout_outputs, cache)
    else:
        rep.fail_all(["heldout:required"], "held-out output directory required")
    return rep


if __name__ == "__main__":
    sys.exit(grader_main("t02", grade))
