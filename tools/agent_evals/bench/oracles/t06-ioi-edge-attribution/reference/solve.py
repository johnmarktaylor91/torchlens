"""Outgoing residual-edge attribution via module recordings and local VJPs."""

import argparse
import importlib.util
import json
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Prompts per pass; the score is a per-prompt mean, so this only bounds memory.
CHUNK = 32


def examples(inputs, cfg, tokenizer):
    spec = importlib.util.spec_from_file_location("ioi_prompts", os.path.join(inputs, "prompts.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = module.make_prompts(cfg["prompt_seed"], cfg["n_prompts"], cfg["names"])
    tokenized = [
        [tokenizer.encode(row[key], add_special_tokens=False) for row in rows]
        for key in ("clean", "corrupt")
    ]
    lengths = {len(ids) for batch in tokenized for ids in batch}
    if len(lengths) != 1:
        raise ValueError("all prompt pairs must have equal token lengths")
    targets = [
        [tokenizer.encode(row[key], add_special_tokens=False) for row in rows]
        for key in ("positive_token", "negative_token")
    ]
    if any(len(ids) != 1 for names in targets for ids in names):
        raise ValueError("metric names must be one token")
    return (
        [torch.tensor(batch) for batch in tokenized],
        [torch.tensor([ids[0] for ids in names]) for names in targets],
    )


def recorded_forward(model, ids, positive, negative, gradients):
    joined, residuals, handles, final = {}, {}, [], {}

    def pre_projection(layer):
        def save(module, args):
            joined[layer] = args[0]

        return save

    def pre_norm(layer):
        def save(module, args):
            residuals[layer] = args[0].detach()

        return save

    def pre_final(module, args):
        final["state"] = args[0]

    try:
        for layer, block in enumerate(model.transformer.h):
            handles.append(block.attn.c_proj.register_forward_pre_hook(pre_projection(layer)))
            handles.append(block.ln_1.register_forward_pre_hook(pre_norm(layer)))
        handles.append(model.transformer.ln_f.register_forward_pre_hook(pre_final))
        embeddings = model.transformer.wte(ids).detach().requires_grad_(gradients)
        state = model.transformer(inputs_embeds=embeddings, use_cache=False).last_hidden_state
        logits = model.lm_head(state[:, -1])
        batch = torch.arange(len(ids))
        metric = logits[batch, positive] - logits[batch, negative]
    finally:
        for handle in handles:
            handle.remove()
    return metric, [joined[layer] for layer in range(12)], residuals, final["state"]


def receiver_gradients(block, residual, global_gradient):
    raw = residual.detach().requires_grad_(True)
    saved = {}

    def save(module, args):
        saved["joined"] = args[0]

    handle = block.attn.c_proj.register_forward_pre_hook(save)
    length = raw.shape[1]
    causal = torch.zeros(length, length).masked_fill(
        torch.ones(length, length, dtype=torch.bool).triu(1), torch.finfo(raw.dtype).min
    )
    try:
        block.attn(block.ln_1(raw), attention_mask=causal[None, None])
    finally:
        handle.remove()
    # One cotangent per receiver head; batch samples remain independent.
    heads = global_gradient.reshape(len(raw), length, 12, 64)
    cotangents = torch.eye(12)[:, None, None, :, None] * heads[None]
    derivatives = torch.autograd.grad(
        saved["joined"],
        raw,
        grad_outputs=cotangents.reshape(12, *global_gradient.shape),
        is_grads_batched=True,
    )[0]
    return derivatives.detach()


def score_edges(model, good, bad, residuals, derivatives, signed=False):
    scores = torch.zeros(12, 12)
    receivers = [
        receiver_gradients(block, residuals[layer], derivatives[layer])
        for layer, block in enumerate(model.transformer.h)
    ]
    final = derivatives[-1]
    for layer, block in enumerate(model.transformer.h):
        delta = (bad[layer] - good[layer]).detach().reshape(*good[layer].shape[:2], 12, 64)
        projection = block.attn.c_proj.weight.detach().reshape(12, 64, 768)
        difference = torch.einsum("bthd,hdw->bthw", delta, projection)
        direct = torch.einsum("bthw,btw->bh", difference, final)
        contribution = direct if signed else direct.abs()
        for receiver in range(layer + 1, 12):
            edge = torch.einsum("bthw,kbtw->bhk", difference, receivers[receiver])
            contribution = contribution + (edge if signed else edge.abs()).sum(-1)
        scores[layer] = contribution.mean(0)
    return scores.numpy()


def chunk_scores(model, clean, corrupt, positive, negative, signed=False):
    with torch.no_grad():
        _, bad, _, _ = recorded_forward(model, corrupt, positive, negative, False)
    metric, good, residuals, final = recorded_forward(model, clean, positive, negative, True)
    derivatives = torch.autograd.grad(metric.sum(), [*good, final])
    return score_edges(model, good, bad, residuals, derivatives, signed=signed)


def run(inputs, outputs, variant=None):
    cfg = json.load(open(os.path.join(inputs, "config.json")))
    os.makedirs(outputs, exist_ok=True)
    if variant == "random_scores":
        scores = np.random.default_rng(cfg["random_score_seed"]).random((12, 12)).astype(np.float32)
    else:
        tokenizer = AutoTokenizer.from_pretrained(cfg["model"])
        model = AutoModelForCausalLM.from_pretrained(
            cfg["model"], torch_dtype=torch.float32, attn_implementation="eager"
        ).eval()
        model.requires_grad_(False)
        (clean, corrupt), (positive, negative) = examples(inputs, cfg, tokenizer)
        if variant == "single_prompt":
            clean, corrupt, positive, negative = [
                x[:1] for x in (clean, corrupt, positive, negative)
            ]
        if variant == "reversed_roles":
            clean, corrupt = corrupt, clean
        scores = np.zeros((12, 12))
        for start in range(0, len(clean), CHUNK):
            part = slice(start, start + CHUNK)
            mean = chunk_scores(
                model,
                clean[part],
                corrupt[part],
                positive[part],
                negative[part],
                signed=variant == "reversed_roles",
            )
            scores += mean * len(clean[part]) / len(clean)
        if variant == "transpose":
            scores = scores.T
    np.savez(os.path.join(outputs, "head_scores.npz"), scores=scores.astype(np.float32))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--outputs", required=True)
    args = parser.parse_args()
    run(args.inputs, args.outputs)
