import argparse
import importlib.util
import json
from pathlib import Path

import numpy as np
import torch

from transformers import AutoModelForCausalLM, AutoTokenizer


def encode(tokenizer, text, span):
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    start, last = encoded.char_to_token(span[0]), encoded.char_to_token(span[1] - 1)
    if start is None or last is None:
        raise ValueError("Span must contain tokenized characters")
    return torch.tensor([encoded["input_ids"]]), [start, last + 1]


def adapt(representation, length):
    source_length = representation.shape[1]
    if source_length > length:
        return representation[:, :length], "truncate"
    if source_length < length:
        extra = representation[:, -1:].expand(-1, length - source_length, -1)
        return torch.cat((representation, extra), 1), "repeat_last"
    return representation, "none"


def patch_output(output, replacement, span, first_only=False):
    hidden = output[0] if isinstance(output, (tuple, list)) else output
    result = hidden.clone()
    start, end = span
    count = 1 if first_only else end - start
    result[:, start:start + count] = replacement[:, :count]
    if isinstance(output, tuple):
        return (result, *output[1:])
    if isinstance(output, list):
        return [result, *output[1:]]
    return result


@torch.no_grad()
def run(inputs, outputs, variant=None):
    inputs, outputs = Path(inputs), Path(outputs)
    cfg = json.loads((inputs / "config.json").read_text())
    items = json.loads((inputs / "items.json").read_text())
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"], use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], torch_dtype=torch.float32,
                                                attn_implementation=cfg["attention_implementation"]).eval()
    layer = cfg["layer"] - (1 if variant == "block_five" else 0)
    block = model.transformer.h[layer]
    baseline, patched, alignment = [], [], []
    for item in items:
        source, source_chars = item["source"], item["source_span"]
        target, target_chars = item["target"], item["target_span"]
        if variant == "reversed_roles":
            source, target, source_chars, target_chars = target, source, target_chars, source_chars
        source_ids, sr = encode(tokenizer, source, source_chars)
        target_ids, tr = encode(tokenizer, target, target_chars)
        if variant == "off_by_one":
            sr, tr = [v + 1 for v in sr], [v + 1 for v in tr]
        hidden = model(source_ids, output_hidden_states=True, use_cache=False).hidden_states[layer + 1]
        donor, adaptation = adapt(hidden[:, sr[0]:sr[1]].clone(), tr[1] - tr[0])
        baseline.append(model(target_ids, use_cache=False).logits[0, -1].numpy())
        original = block.forward
        def modified(*args, **kwargs):
            return patch_output(original(*args, **kwargs), donor, tr, variant == "first_only")
        block.forward = modified
        try:
            patched.append(model(target_ids, use_cache=False).logits[0, -1].numpy())
        finally:
            block.forward = original
        alignment.append(dict(source_range=sr, target_range=tr, adaptation=adaptation))
    baseline, patched = np.stack(baseline), np.stack(patched)
    if variant == "transpose":
        patched = patched.T
    if variant == "stimulus_order":
        baseline, patched, alignment = baseline[::-1], patched[::-1], alignment[::-1]
    outputs.mkdir(parents=True, exist_ok=True)
    np.savez(outputs / "composition.npz", patched_logits=patched, baseline_logits=baseline)
    (outputs / "alignment.json").write_text(json.dumps(alignment))

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", required=True)
    p.add_argument("--outputs", required=True)
    a = p.parse_args()
    run(a.inputs, a.outputs)
