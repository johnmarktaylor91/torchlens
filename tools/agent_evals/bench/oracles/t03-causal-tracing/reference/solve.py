"""Causal restoration using cloned module outputs and batched interventions."""
import argparse
import json
import os

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def tensor(out):
    return out[0] if isinstance(out, (tuple, list)) else out


def replace(out, value):
    if isinstance(out, tuple):
        return (value,) + out[1:]
    if isinstance(out, list):
        return [value] + out[1:]
    return value


def modules(model):
    return {(site, l): (block if site == "resid" else getattr(block, site))
            for l, block in enumerate(model.transformer.h)
            for site in ("attn", "mlp", "resid")}


def logits(model, embeddings):
    h = model.transformer(inputs_embeds=embeddings, use_cache=False).last_hidden_state
    return model.lm_head(h[:, -1])


def capture(model, embeddings, sites):
    saved, handles = {}, []
    def record(key):
        def hook(mod, args, out):
            saved[key] = tensor(out).detach().clone()
        return hook
    try:
        for key, mod in sites.items():
            handles.append(mod.register_forward_hook(record(key)))
        result = logits(model, embeddings)
    finally:
        for handle in handles:
            handle.remove()
    return result, saved


def restored(model, base, module, donor, batch_size):
    values = []
    for start in range(0, base.shape[1], batch_size):
        positions = torch.arange(start, min(start + batch_size, base.shape[1]))
        def patch(mod, args, out):
            value = tensor(out).clone()
            value[torch.arange(len(positions)), positions] = donor[0, positions]
            return replace(out, value)
        handle = module.register_forward_hook(patch)
        try:
            values.append(logits(model, base.expand(len(positions), -1, -1)))
        finally:
            handle.remove()
    return torch.cat(values)


@torch.no_grad()
def run(inputs, outputs, variant=None):
    cfg = json.load(open(os.path.join(inputs, "config.json")))
    prompts = json.load(open(os.path.join(inputs, "prompts.json")))
    tok = AutoTokenizer.from_pretrained(cfg["model"], use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"], torch_dtype=torch.float32, attn_implementation="eager").eval()
    model.requires_grad_(False)
    rng = np.random.default_rng(cfg["noise_seed"] + (1 if variant == "noise_seed" else 0))
    std = float(model.transformer.wte.weight.std(unbiased=False)) * cfg["noise_std_multiplier"]
    sites = modules(model)
    encoded = [tok(p["text"], add_special_tokens=False, return_offsets_mapping=True) for p in prompts]
    width = max(len(e["input_ids"]) for e in encoded)
    result = {s: np.full((len(prompts), cfg["n_layers"], width), np.nan, np.float32)
              for s in ("attn", "mlp", "resid")}
    clean_prob, corrupt_prob = [], []
    for i, (p, enc) in enumerate(zip(prompts, encoded)):
        ids = torch.tensor([enc["input_ids"]])
        clean = model.transformer.wte(ids)
        corrupt = clean.clone()
        subject = [j for j, (a, b) in enumerate(enc["offset_mapping"])
                   if b > p["subject_start"] and a < p["subject_end"]]
        noise = rng.standard_normal((len(subject), clean.shape[-1])).astype(np.float32)
        corrupt[0, subject] += torch.from_numpy(noise * np.float32(std))
        target = tok.encode(p["correct_token"], add_special_tokens=False)
        if len(target) != 1 or not subject:
            raise ValueError("subject must overlap tokens and answer must be one token")
        c_logits, saved = capture(model, clean, sites)
        x_logits, bad_saved = capture(model, corrupt, sites)
        clean_prob.append(c_logits.softmax(-1)[0, target[0]].item())
        corrupt_prob.append(x_logits.softmax(-1)[0, target[0]].item())
        if variant == "reversed_roles":
            corrupt, saved = clean, bad_saved
        for (site, layer), mod in sites.items():
            values = restored(model, corrupt, mod, saved[site, layer], cfg["batch_size"])
            result[site][i, layer, :ids.shape[1]] = values.softmax(-1)[:, target[0]].numpy()
        if variant == "position_off_by_one":
            for site in result:
                result[site][i, :, :ids.shape[1]] = np.roll(
                    result[site][i, :, :ids.shape[1]], -1, axis=-1)
    if variant == "site_mixup":
        result["attn"], result["mlp"] = result["mlp"], result["attn"]
    if variant == "reversed_roles":
        clean_prob, corrupt_prob = corrupt_prob, clean_prob
    os.makedirs(outputs, exist_ok=True)
    np.savez(os.path.join(outputs, "heatmap.npz"), **result,
             clean_prob=np.array(clean_prob, np.float32), corrupt_prob=np.array(corrupt_prob, np.float32))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--outputs", required=True)
    args = parser.parse_args()
    run(args.inputs, args.outputs)
