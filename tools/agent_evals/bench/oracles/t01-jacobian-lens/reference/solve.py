"""Reference solution. Hidden states come from the model's own
``output_hidden_states`` return (block outputs 0..4) plus one forward hook for
the final block's pre-norm output; batched cotangents give 32 rows of each
Jacobian per backward pass. ``run(inputs, outputs, variant)`` is shared with
the seeded wrong solutions.
"""
import argparse
import json
import math
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DIM_BATCH = 32


def read_jsonl(path):
    return [json.loads(l)["text"] for l in open(path) if l.strip()]


def final_block_output(model):
    store = {}

    def hook(mod, inp, out):
        store["h"] = out[0] if isinstance(out, (tuple, list)) else out
    handle = model.gpt_neox.layers[-1].register_forward_hook(hook)
    return store, handle


def fit(model, tok, texts, cfg, variant=None):
    src = cfg["source_layers"]
    d = model.config.hidden_size
    skip = 0 if variant == "no_position_mask" else cfg["skip_first_positions"]
    acc = {l: torch.zeros(d, d, dtype=torch.float64) for l in src}
    n = 0
    store, handle = final_block_output(model)
    for text in texts:
        ids = tok(text, return_tensors="pt", add_special_tokens=False,
                  truncation=True, max_length=cfg["max_seq_len"]).input_ids
        T = ids.shape[1]
        hi = T if variant == "no_position_mask" else T - 1
        valid = torch.arange(skip, hi)
        if len(valid) == 0:
            continue
        ids_b = ids.expand(DIM_BATCH, -1)
        out = model(ids_b, output_hidden_states=True)
        hs = out.hidden_states  # hs[l + 1] is the output of block l
        h_final = store["h"]
        sources = [hs[l + 1] for l in src]
        per_text = {l: torch.zeros(d, d, dtype=torch.float64) for l in src}
        n_pass = math.ceil(d / DIM_BATCH)
        for k in range(n_pass):
            d0 = k * DIM_BATCH
            nd = min(DIM_BATCH, d - d0)
            cot = torch.zeros_like(h_final)
            b = torch.arange(nd)
            targets = valid[-1:] if variant == "last_target_only" else valid
            cot[b[:, None], targets[None, :], (d0 + b)[:, None]] = 1.0
            grads = torch.autograd.grad(h_final, sources, grad_outputs=cot,
                                        retain_graph=k < n_pass - 1)
            for l, g in zip(src, grads):
                rows = g[:nd][:, valid, :].double().mean(dim=1)  # [nd, d]
                per_text[l][d0:d0 + nd] = rows
        for l in src:
            acc[l] += per_text[l]
        n += 1
    handle.remove()
    return {l: (acc[l] / n).float() for l in src}


@torch.no_grad()
def readout(model, tok, prompts, lens, cfg, variant=None):
    src = cfg["source_layers"]
    k = cfg["readout_top_k"]
    store, handle = final_block_output(model)
    top, model_top = [], []
    for text in prompts:
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids
        out = model(ids, output_hidden_states=True)
        h_final = store["h"][0, -1]
        row = []
        for l in src:
            h = out.hidden_states[l + 1][0, -1]
            J = lens[l]
            if variant == "transpose":
                J = J.T
            t = J @ h
            if variant != "no_final_norm":
                t = model.gpt_neox.final_layer_norm(t)
            logits = model.get_output_embeddings()(t)
            row.append(torch.topk(logits, k).indices.tolist())
        top.append(row)
        model_top.append(torch.topk(model.get_output_embeddings()(model.gpt_neox.final_layer_norm(h_final)), k).indices.tolist())
    handle.remove()
    return {"layers": src, "top10": top, "model_top10": model_top}


def run(inputs, outputs, variant=None):
    cfg = json.load(open(os.path.join(inputs, "config.json")))
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], torch_dtype=torch.float32).eval()
    texts = read_jsonl(os.path.join(inputs, "corpus.jsonl"))
    prompts = read_jsonl(os.path.join(inputs, "readout_prompts.jsonl"))
    if variant == "shuffled_prompts":
        prompts = prompts[1:] + prompts[:1]
    lens = fit(model, tok, texts, cfg, variant=variant)
    saved = {f"J_{l}": (J.T if variant == "transpose" else J).numpy().astype(np.float32)
             for l, J in lens.items()}
    os.makedirs(outputs, exist_ok=True)
    np.savez(os.path.join(outputs, "lens.npz"), **saved)
    rd = readout(model, tok, prompts, lens, cfg, variant=variant)
    json.dump(rd, open(os.path.join(outputs, "readout.json"), "w"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", required=True)
    p.add_argument("--outputs", required=True)
    a = p.parse_args()
    run(a.inputs, a.outputs)
