"""Independent oracle: explicit attention and one replacement per forward."""
import hashlib
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_common"))
from grade_utils import Report, grader_main, load_npz, max_abs, spearman


def head_results(block, state):
    q, k, v = block.attn.c_attn(block.ln_1(state)).chunk(3, -1)
    batch, length, _ = q.shape
    q, k, v = [x.reshape(batch, length, 12, 64).transpose(1, 2) for x in (q, k, v)]
    scores = (q @ k.transpose(-1, -2)) / 8.0
    future = torch.ones(length, length, dtype=torch.bool).triu(1)
    probs = scores.masked_fill(future, torch.finfo(scores.dtype).min).softmax(-1)
    return (probs @ v).transpose(1, 2).contiguous()


def evaluate(model, ids, positive, negative, replacement=None):
    length = ids.shape[1]
    state = model.transformer.wte(ids) + model.transformer.wpe(torch.arange(length))[None]
    if torch.is_grad_enabled():
        state.requires_grad_(True)
    recorded = []
    for layer, block in enumerate(model.transformer.h):
        heads = head_results(block, state)
        if replacement is not None and layer == replacement[0]:
            head, donor = replacement[1:]
            heads = torch.cat((heads[:, :, :head], donor[:, :, head:head + 1],
                               heads[:, :, head + 1:]), dim=2)
        recorded.append(heads)
        state = state + block.attn.c_proj(heads.reshape(ids.shape[0], length, 768))
        normalized = block.ln_2(state)
        state = state + block.mlp.c_proj(block.mlp.act(block.mlp.c_fc(normalized)))
    logits = model.lm_head(model.transformer.ln_f(state[:, -1]))
    return logits[:, positive] - logits[:, negative], recorded


def oracle(inputs, cache):
    digest = hashlib.sha256()
    for filename in ("config.json", "prompts.json"):
        with open(os.path.join(inputs, filename), "rb") as f:
            digest.update(f.read())
    path = os.path.join(cache, "t04-v1-" + digest.hexdigest()[:16] + ".npz")
    if os.path.isfile(path):
        return load_npz(path)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = json.load(open(os.path.join(inputs, "config.json")))
    prompts = json.load(open(os.path.join(inputs, "prompts.json")))
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"], torch_dtype=torch.float32, attn_implementation="eager").eval()
    model.requires_grad_(False)
    clean, corrupt = [tokenizer(x, return_tensors="pt", add_special_tokens=False).input_ids
                      for x in (prompts["clean"], prompts["corrupt"])]
    positive, negative = [tokenizer.encode(prompts[k], add_special_tokens=False)[0]
                          for k in ("positive_token", "negative_token")]
    if clean.shape != corrupt.shape:
        raise ValueError("unequal pair lengths")
    with torch.no_grad():
        corrupt_metric, donors = evaluate(model, corrupt, positive, negative)
        actual = model.lm_head(model.transformer(corrupt, use_cache=False).last_hidden_state[:, -1])
        if not torch.allclose(corrupt_metric, actual[:, positive] - actual[:, negative], atol=1e-4):
            raise AssertionError("explicit attention disagrees with model baseline")
    clean_metric, heads = evaluate(model, clean, positive, negative)
    derivatives = torch.autograd.grad(clean_metric.sum(), heads)
    attribution = torch.stack([
        torch.einsum("bthd,bthd->h", derivative, donor - original).detach()
        for derivative, donor, original in zip(derivatives, donors, heads)]).numpy()
    exact = np.empty((12, 12), np.float32)
    with torch.no_grad():
        for layer in range(12):
            for head in range(12):
                restored, _ = evaluate(model, clean, positive, negative, (layer, head, donors[layer]))
                exact[layer, head] = (restored - clean_metric).item()
    answer = dict(exact=exact, attribution=attribution,
                  clean_metric=clean_metric.detach().numpy().reshape(()),
                  corrupt_metric=corrupt_metric.numpy().reshape(()))
    np.savez(path, **answer)
    return answer


def grade_one(rep, label, inputs, outputs, cache):
    names = [f"{label}:{key}" for key in ("shape", "exact", "attribution", "clean_metric", "corrupt_metric")]
    try:
        agent = load_npz(os.path.join(outputs, "patching.npz"))
        ref = oracle(inputs, cache)
        valid = all(key in agent and agent[key].shape == value.shape
                    and np.isfinite(agent[key]).all() for key, value in ref.items())
        rep.add(names[0], valid)
        for key in ("exact", "attribution"):
            arr = agent.get(key, np.array([]))
            score = spearman(arr, ref[key]) if arr.shape == ref[key].shape and np.isfinite(arr).all() else float("nan")
            rep.add(f"{label}:{key}", score >= 0.98, score, 0.98)
        for key in ("clean_metric", "corrupt_metric"):
            err = max_abs(agent.get(key, np.array([])), ref[key])
            rep.add(f"{label}:{key}", err <= 1e-4, err, 1e-4)
    except (OSError, KeyError, ValueError, TypeError) as exc:
        rep.fail_all(names, str(exc))


def grade(task_dir, outputs, heldout_outputs, cache):
    rep = Report("t04-attribution-vs-activation-patching")
    grade_one(rep, "visible", os.path.join(task_dir, "inputs"), outputs, cache)
    if heldout_outputs:
        grade_one(rep, "heldout", os.path.join(task_dir, "heldout"), heldout_outputs, cache)
    else:
        rep.fail_all(["heldout:required"], "held-out output directory required")
    return rep


if __name__ == "__main__":
    sys.exit(grader_main("t04", grade))
