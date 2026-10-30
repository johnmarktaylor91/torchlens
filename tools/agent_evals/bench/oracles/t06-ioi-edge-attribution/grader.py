"""Independent oracle with separate residual inputs for every receiving head."""
import hashlib
import importlib.util
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_common"))
from grade_utils import Report, grader_main, load_npz, roc_auc, spearman

IOI_HEADS = [(9, 9), (9, 6), (10, 0), (10, 7), (11, 10),
             (7, 3), (7, 9), (8, 6), (8, 10), (5, 5), (5, 8), (5, 9), (6, 9),
             (0, 1), (0, 10), (3, 0), (2, 2), (4, 11),
             (9, 0), (9, 7), (10, 1), (10, 2), (10, 6), (10, 10), (11, 2), (11, 9)]


def head_attention(block, separate):
    normalized = block.ln_1(separate)
    weights = block.attn.c_attn.weight.reshape(768, 3, 12, 64)
    biases = block.attn.c_attn.bias.reshape(3, 12, 64)
    qkv = torch.einsum("hbtw,wkhd->khbtd", normalized, weights)
    qkv = qkv + biases[:, :, None, None, :]
    q, k, v = qkv.unbind(0)
    scores = q @ k.transpose(-1, -2) / 8.0
    length = q.shape[-2]
    future = torch.ones(length, length, dtype=torch.bool).triu(1)
    probability = scores.masked_fill(future, torch.finfo(scores.dtype).min).softmax(-1)
    return probability @ v


def manual_forward(model, ids, positive, negative):
    state = model.transformer.wte(ids) + model.transformer.wpe(torch.arange(ids.shape[1]))[None]
    if torch.is_grad_enabled():
        state.requires_grad_(True)
    receivers, results = [], []
    for block in model.transformer.h:
        # Each receiving head gets an independent copy of the raw residual.
        # Its derivative excludes the other heads and the block's MLP branch.
        separate = state.unsqueeze(0).expand(12, -1, -1, -1).clone()
        z = head_attention(block, separate)
        receivers.append(separate)
        results.append(z.permute(1, 2, 0, 3))
        joined = results[-1].reshape(*ids.shape, 768)
        state = state + block.attn.c_proj(joined)
        state = state + block.mlp.c_proj(block.mlp.act(block.mlp.c_fc(block.ln_2(state))))
    logits = model.lm_head(model.transformer.ln_f(state[:, -1]))
    batch = torch.arange(len(ids))
    return logits[batch, positive] - logits[batch, negative], results, receivers, state


def oracle(inputs, cache):
    digest = hashlib.sha256()
    for name in ("config.json", "prompts.py"):
        with open(os.path.join(inputs, name), "rb") as f:
            digest.update(f.read())
    path = os.path.join(cache, "t06-v1-" + digest.hexdigest()[:16] + ".npz")
    if os.path.isfile(path):
        return load_npz(path)["scores"]
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = json.load(open(os.path.join(inputs, "config.json")))
    spec = importlib.util.spec_from_file_location("oracle_prompts", os.path.join(inputs, "prompts.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = module.make_prompts(cfg["prompt_seed"], cfg["n_prompts"], cfg["names"])
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"], torch_dtype=torch.float32, attn_implementation="eager").eval().requires_grad_(False)
    clean, corrupt = [torch.tensor([tokenizer.encode(row[key], add_special_tokens=False) for row in rows])
                      for key in ("clean", "corrupt")]
    targets = [[tokenizer.encode(row[key], add_special_tokens=False) for row in rows]
               for key in ("positive_token", "negative_token")]
    if any(len(ids) != 1 for batch in targets for ids in batch):
        raise ValueError("names must be one token")
    positive, negative = [torch.tensor([ids[0] for ids in batch]) for batch in targets]
    with torch.no_grad():
        _, bad, _, _ = manual_forward(model, corrupt, positive, negative)
        actual = model.lm_head(model.transformer(clean, use_cache=False).last_hidden_state[:, -1])
        actual_metric = actual[torch.arange(len(clean)), positive] - actual[torch.arange(len(clean)), negative]
    metric, good, receiver_inputs, final = manual_forward(model, clean, positive, negative)
    if not torch.allclose(metric, actual_metric, atol=1e-4, rtol=1e-4):
        raise AssertionError("independent attention disagrees with model metric")
    grads = torch.autograd.grad(metric.sum(), [*receiver_inputs, final])
    final_grad = grads[-1].reshape(len(clean), -1)
    scores = []
    with torch.no_grad():
        for layer, block in enumerate(model.transformer.h):
            delta = bad[layer] - good[layer]
            weight = block.attn.c_proj.weight.reshape(12, 64, 768)
            projected = torch.stack([delta[:, :, h] @ weight[h] for h in range(12)], dim=1)
            sender = projected.reshape(len(clean), 12, -1)
            value = (sender * final_grad[:, None]).sum(-1).abs()
            for receiver in range(layer + 1, 12):
                incoming = grads[receiver].permute(1, 0, 2, 3).reshape(len(clean), 12, -1)
                edge = sender @ incoming.transpose(-1, -2)
                value += edge.abs().sum(-1)
            scores.append(value.mean(0))
    result = torch.stack(scores).numpy()
    np.savez(path, scores=result)
    return result


def labels():
    result = np.zeros((12, 12), dtype=bool)
    for layer, head in IOI_HEADS:
        result[layer, head] = True
    return result


def grade(task_dir, outputs, heldout_outputs, cache):
    rep = Report("t06-ioi-edge-attribution")
    ref = oracle(os.path.join(task_dir, "inputs"), cache)
    own_auc = roc_auc(ref, labels())
    try:
        assert own_auc >= 0.85, "oracle scores fail independent circuit-membership calibration"
    except AssertionError as exc:
        rep.add("oracle_auc", False, own_auc, 0.85, str(exc))
        return rep
    rep.add("oracle_auc", True, own_auc, 0.85)
    try:
        agent = load_npz(os.path.join(outputs, "head_scores.npz"))["scores"]
        shape = agent.shape == (12, 12) and np.isfinite(agent).all() and (agent >= 0).all()
        rep.add("shape", shape)
        auc = roc_auc(agent, labels()) if shape else float("nan")
        correlation = spearman(agent, ref) if shape else float("nan")
        rep.add("membership_auc", shape and auc >= 0.85, auc, 0.85)
        rep.add("oracle_rank", shape and correlation >= 0.8, correlation, 0.8)
    except (OSError, KeyError, ValueError, TypeError) as exc:
        rep.fail_all(["shape", "membership_auc", "oracle_rank"], str(exc))
    return rep


if __name__ == "__main__":
    sys.exit(grader_main("t06", grade))
