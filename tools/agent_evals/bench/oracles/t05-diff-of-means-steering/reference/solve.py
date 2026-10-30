import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

import time
from transformers import AutoModelForCausalLM, AutoTokenizer


def unpack(output):
    return output[0] if isinstance(output, (tuple, list)) else output


def replace(output, value):
    if isinstance(output, tuple):
        return (value, *output[1:])
    if isinstance(output, list):
        return [value, *output[1:]]
    return value


@torch.no_grad()
def fit(model, tokenizer, prompts, layer):
    means = []
    for group in ("positive", "negative"):
        rows = []
        for text in prompts[group]:
            ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids
            result = model(ids, output_hidden_states=True, use_cache=False)
            rows.append(result.hidden_states[layer + 1][0, -1])
        means.append(torch.stack(rows).mean(0))
    return means[0] - means[1]


@torch.no_grad()
def generate(model, tokenizer, texts, cfg, direction=None, variant=None):
    items, first_logits = [], []
    block = model.gpt_neox.layers[cfg["layer"]]
    original = block.forward
    for text in texts:
        ids = tokenizer(text, add_special_tokens=False, return_tensors="pt").input_ids
        tokens, projections = [], []
        if direction is not None:
            def modified(*args, **kwargs):
                result = original(*args, **kwargs)
                hidden = unpack(result)
                changed = hidden.clone()
                if variant == "last_only":
                    changed[:, -1] += cfg["coefficient"] * direction
                else:
                    changed += cfg["coefficient"] * direction
                measured = changed if variant == "projection_after" else hidden
                projections.append(float(measured[0, -1] @ direction))
                return replace(result, changed)
            block.forward = modified
        try:
            for step in range(cfg["new_tokens"]):
                logits = model(ids, use_cache=False).logits[:, -1]
                if step == 0:
                    first_logits.append(logits[0].clone())
                token = logits.argmax(-1)
                tokens.append(int(token[0]))
                ids = torch.cat((ids, token[:, None]), dim=1)
        finally:
            block.forward = original
        items.append(dict(token_ids=tokens, projections=projections))
    return items, torch.stack(first_logits).numpy()


def run(inputs, outputs, variant=None):
    torch.set_num_threads(1)
    inputs, outputs = Path(inputs), Path(outputs)
    cfg = json.loads((inputs / "config.json").read_text())
    prompts = json.loads((inputs / "prompts.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"])
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], torch_dtype=torch.float32,
                                                attn_implementation=cfg["attention_implementation"]).eval()
    direction = fit(model, tokenizer, prompts, cfg["layer"])
    if variant == "reversed_roles":
        direction = -direction
    unit = direction if variant == "no_normalisation" else direction / direction.norm()
    started = time.perf_counter()
    native, _ = generate(model, tokenizer, prompts["generation"], cfg)
    native_seconds = time.perf_counter() - started
    started = time.perf_counter()
    steered, logits = generate(model, tokenizer, prompts["generation"], cfg, unit, variant)
    steered_seconds = time.perf_counter() - started
    items = [dict(native_token_ids=a["token_ids"], steered_token_ids=b["token_ids"],
                  projections=b["projections"]) for a, b in zip(native, steered)]
    if variant == "stimulus_order":
        items, logits = items[::-1], logits[::-1]
    if variant == "transpose":
        logits = logits.T
    outputs.mkdir(parents=True, exist_ok=True)
    np.savez(outputs / "direction.npz", direction=direction.numpy())
    np.savez(outputs / "steered_first_logits.npz", logits=logits)
    (outputs / "generation.json").write_text(json.dumps(dict(items=items)))
    (outputs / "timing.json").write_text(json.dumps(dict(native_seconds=native_seconds,
                                                       steered_seconds=steered_seconds)))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", required=True)
    p.add_argument("--outputs", required=True)
    a = p.parse_args()
    run(a.inputs, a.outputs)
