"""Exact and first-order head interventions using projection-input recordings."""
import argparse
import json
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def metric(model, embeddings, positive, negative):
    state = model.transformer(inputs_embeds=embeddings, use_cache=False).last_hidden_state[:, -1]
    logits = model.lm_head(state)
    return logits[:, positive] - logits[:, negative]


def record(model, embeddings, positive, negative):
    values, handles = {}, []
    def save(layer):
        def pre(module, args):
            values[layer] = args[0]
        return pre
    try:
        for layer, block in enumerate(model.transformer.h):
            handles.append(block.attn.c_proj.register_forward_pre_hook(save(layer)))
        score = metric(model, embeddings, positive, negative)
    finally:
        for handle in handles:
            handle.remove()
    return score, [values[l] for l in range(len(model.transformer.h))]


def exact_changes(model, clean, donor, positive, negative, baseline, variant):
    changes = np.empty((12, 12), np.float32)
    with torch.no_grad():
        for layer, block in enumerate(model.transformer.h):
            def patch(module, args):
                value = args[0].reshape(12, clean.shape[1], 12, 64).clone()
                source = donor[layer].reshape(1, clean.shape[1], 12, 64)
                if variant == "all_heads":
                    value[:] = source
                else:
                    for head in range(12):
                        value[head, :, head] = source[0, :, head]
                return (value.reshape(12, clean.shape[1], 768),) + args[1:]
            handle = block.attn.c_proj.register_forward_pre_hook(patch)
            try:
                scores = metric(model, clean.expand(12, -1, -1), positive, negative)
                changes[layer] = (scores - baseline).numpy()
            finally:
                handle.remove()
    return changes


def run(inputs, outputs, variant=None):
    cfg = json.load(open(os.path.join(inputs, "config.json")))
    prompts = json.load(open(os.path.join(inputs, "prompts.json")))
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"], torch_dtype=torch.float32, attn_implementation="eager").eval()
    model.requires_grad_(False)
    ids = [tokenizer(p, return_tensors="pt", add_special_tokens=False).input_ids
           for p in (prompts["clean"], prompts["corrupt"])]
    if ids[0].shape != ids[1].shape:
        raise ValueError("paired prompts must have equal token length")
    names = [tokenizer.encode(prompts[k], add_special_tokens=False)
             for k in ("positive_token", "negative_token")]
    if any(len(n) != 1 for n in names):
        raise ValueError("each metric token must encode to one token")
    positive, negative = names[0][0], names[1][0]
    if variant == "reversed_roles":
        ids.reverse()
    clean = model.transformer.wte(ids[0]).detach().requires_grad_(True)
    corrupt = model.transformer.wte(ids[1]).detach()
    with torch.no_grad():
        bad_metric, bad = record(model, corrupt, positive, negative)
    clean_metric, good = record(model, clean, positive, negative)
    gradients = torch.autograd.grad(clean_metric.sum(), good)
    attribution = []
    for grad, donor, original in zip(gradients, bad, good):
        product = (grad * (donor - original)).reshape(1, clean.shape[1], 12, 64)
        if variant == "last_position_only":
            product = product[:, -1:]
        attribution.append(product.detach().sum(dim=(0, 1, 3)))
    attribution = torch.stack(attribution).numpy()
    exact = exact_changes(model, clean.detach(), bad, positive, negative,
                          clean_metric.detach(), variant)
    if variant == "transpose":
        exact, attribution = exact.T, attribution.T
    os.makedirs(outputs, exist_ok=True)
    np.savez(os.path.join(outputs, "patching.npz"), exact=exact, attribution=attribution,
             clean_metric=clean_metric.detach().numpy().reshape(()),
             corrupt_metric=bad_metric.numpy().reshape(()))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--outputs", required=True)
    args = parser.parse_args()
    run(args.inputs, args.outputs)
