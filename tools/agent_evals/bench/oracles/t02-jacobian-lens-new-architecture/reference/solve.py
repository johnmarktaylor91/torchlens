"""Exact estimator using batched cotangents summed over target positions."""
import argparse
import importlib.util
import json
import os

import numpy as np
import torch


def load(inputs):
    cfg = json.load(open(os.path.join(inputs, "config.json")))
    path = os.path.join(inputs, "model.py")
    spec = importlib.util.spec_from_file_location("byte_decoder", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return cfg, module.build_model(cfg["weight_seed"]).requires_grad_(False)


def texts(inputs, filename):
    return [json.loads(line)["text"] for line in open(os.path.join(inputs, filename)) if line.strip()]


def encode(text, limit=None):
    return torch.tensor(list(text.encode("utf-8")[:limit]), dtype=torch.long)


def fit(model, corpus, cfg, variant):
    sources = cfg["source_layers"]
    total = {layer: torch.zeros(64, 64, dtype=torch.float64) for layer in sources}
    for text in corpus:
        ids = encode(text, cfg["max_seq_len"])
        valid = torch.arange(0 if variant == "no_position_mask" else cfg["skip_first_positions"],
                             len(ids) if variant == "no_position_mask" else len(ids) - 1)
        if not len(valid):
            raise ValueError("corpus text has no valid positions")
        state = model.embed(ids).detach().requires_grad_(True)
        states = []
        for block in model.blocks:
            state = block(state)
            states.append(state)
        target_positions = valid[-1:] if variant == "last_target_only" else valid
        targets = states[cfg["target_layer"]][target_positions].sum(dim=0)
        derivatives = torch.autograd.grad(targets, [states[l] for l in sources],
                                          grad_outputs=torch.eye(64), is_grads_batched=True)
        for layer, derivative in zip(sources, derivatives):
            total[layer] += derivative[:, valid].double().sum(dim=1) / len(valid)
    return {layer: (value / len(corpus)).float() for layer, value in total.items()}


@torch.no_grad()
def readout(model, prompts, lenses, cfg, variant):
    if variant == "shuffled_prompts":
        prompts = prompts[1:] + prompts[:1]
    top, model_top = [], []
    for text in prompts:
        states = model.states(encode(text))
        row = []
        for layer in cfg["source_layers"]:
            matrix = lenses[layer].T if variant == "transpose" else lenses[layer]
            transported = matrix @ states[layer][-1]
            if variant != "no_final_norm":
                transported = model.final_norm(transported)
            row.append(model.unembed(transported).topk(cfg["readout_top_k"]).indices.tolist())
        top.append(row)
        model_top.append(model.unembed(model.final_norm(states[-1][-1])).topk(10).indices.tolist())
    return dict(layers=cfg["source_layers"], top10=top, model_top10=model_top)


def run(inputs, outputs, variant=None):
    cfg, model = load(inputs)
    matrices = fit(model, texts(inputs, "corpus.jsonl"), cfg, variant)
    results = readout(model, texts(inputs, "readout_prompts.jsonl"), matrices, cfg, variant)
    os.makedirs(outputs, exist_ok=True)
    np.savez(os.path.join(outputs, "lens.npz"), **{
        f"J_{layer}": (matrix.T if variant == "transpose" else matrix).numpy()
        for layer, matrix in matrices.items()})
    with open(os.path.join(outputs, "readout.json"), "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--outputs", required=True)
    args = parser.parse_args()
    run(args.inputs, args.outputs)
