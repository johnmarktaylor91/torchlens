"""Independent oracle: explicit attention equations and residual additions."""
import hashlib
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_common"))
from grade_utils import Report, grader_main, load_npz, max_abs, pearson


def attention(block, h):
    x = block.ln_1(h)
    q, k, v = block.attn.c_attn(x).chunk(3, dim=-1)
    batch, length, width = q.shape
    heads = block.attn.num_heads
    def split(t):
        return t.reshape(batch, length, heads, width // heads).transpose(1, 2)
    q, k, v = map(split, (q, k, v))
    weights = q @ k.transpose(-1, -2) / (width // heads) ** 0.5
    mask = torch.ones(length, length, dtype=torch.bool).triu(1)
    weights = weights.masked_fill(mask, torch.finfo(weights.dtype).min).softmax(-1)
    joined = (weights @ v).transpose(1, 2).reshape(batch, length, width)
    return block.attn.c_proj(joined)


def forward(model, initial, patch=None):
    h = initial
    records = {}
    for layer, block in enumerate(model.transformer.h):
        a = attention(block, h)
        a = intervene(a, patch, "attn", layer)
        h = h + a
        m = block.mlp.c_proj(block.mlp.act(block.mlp.c_fc(block.ln_2(h))))
        m = intervene(m, patch, "mlp", layer)
        h = intervene(h + m, patch, "resid", layer)
        records["attn", layer], records["mlp", layer], records["resid", layer] = a, m, h
    logits = model.lm_head(model.transformer.ln_f(h[:, -1]))
    return logits, records


def intervene(value, patch, site, layer):
    if patch is None or (site, layer) != patch[:2]:
        return value
    donor, positions = patch[2:]
    result = value.clone()
    result[torch.arange(len(positions)), positions] = donor[0, positions]
    return result


@torch.no_grad()
def oracle(inputs, cache):
    fingerprint = hashlib.sha256()
    for name in ("config.json", "prompts.json"):
        with open(os.path.join(inputs, name), "rb") as f:
            fingerprint.update(f.read())
    path = os.path.join(cache, "t03-v1-" + fingerprint.hexdigest()[:16] + ".npz")
    if os.path.isfile(path):
        return load_npz(path)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = json.load(open(os.path.join(inputs, "config.json")))
    prompts = json.load(open(os.path.join(inputs, "prompts.json")))
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"], use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"], torch_dtype=torch.float32, attn_implementation="eager").eval()
    model.requires_grad_(False)
    rng = np.random.default_rng(cfg["noise_seed"])
    noise_scale = np.float32(cfg["noise_std_multiplier"] * float(model.transformer.wte.weight.std(unbiased=False)))
    encodings = [tokenizer(p["text"], add_special_tokens=False, return_offsets_mapping=True) for p in prompts]
    max_length = max(len(e["input_ids"]) for e in encodings)
    answer = {s: np.full((len(prompts), 12, max_length), np.nan, np.float32)
              for s in ("attn", "mlp", "resid")}
    answer.update(clean_prob=np.empty(len(prompts), np.float32), corrupt_prob=np.empty(len(prompts), np.float32))
    for row, (p, enc) in enumerate(zip(prompts, encodings)):
        ids = torch.tensor([enc["input_ids"]])
        length = ids.shape[1]
        embeddings = model.transformer.wte(ids)
        noisy = embeddings.clone()
        indices = [i for i, (start, end) in enumerate(enc["offset_mapping"])
                   if start < p["subject_end"] and end > p["subject_start"]]
        noise = rng.standard_normal((len(indices), embeddings.shape[-1])).astype(np.float32) * noise_scale
        noisy[:, indices] += torch.from_numpy(noise)
        positions = model.transformer.wpe(torch.arange(length))[None]
        clean_logits, clean = forward(model, embeddings + positions)
        corrupt_logits, _ = forward(model, noisy + positions)
        actual = model.transformer(inputs_embeds=noisy, use_cache=False).last_hidden_state
        actual_logits = model.lm_head(actual[:, -1])
        if not torch.allclose(actual_logits, corrupt_logits, atol=1e-4, rtol=1e-5):
            raise AssertionError("explicit oracle disagrees with model logits")
        target_ids = tokenizer.encode(p["correct_token"], add_special_tokens=False)
        if len(target_ids) != 1 or not indices:
            raise ValueError("invalid subject or target")
        token = target_ids[0]
        answer["clean_prob"][row] = clean_logits.softmax(-1)[0, token]
        answer["corrupt_prob"][row] = corrupt_logits.softmax(-1)[0, token]
        for site in ("attn", "mlp", "resid"):
            for layer in range(12):
                for start in range(0, length, cfg["batch_size"]):
                    pos = torch.arange(start, min(start + cfg["batch_size"], length))
                    out, _ = forward(model, (noisy + positions).expand(len(pos), -1, -1),
                                     (site, layer, clean[site, layer], pos))
                    answer[site][row, layer, start:start + len(pos)] = out.softmax(-1)[:, token].numpy()
    np.savez(path, **answer)
    return answer


def grade_one(rep, label, inputs, outputs, cache):
    names = [f"{label}:{s}" for s in ("shape", "attn", "mlp", "resid", "clean_prob", "corrupt_prob")]
    try:
        agent = load_npz(os.path.join(outputs, "heatmap.npz"))
        ref = oracle(inputs, cache)
        shape_ok = all(k in agent and agent[k].shape == v.shape for k, v in ref.items())
        rep.add(names[0], shape_ok)
        for site in ("attn", "mlp", "resid"):
            expected = ref[site]
            value = agent.get(site)
            valid = np.isfinite(expected)
            ok = value is not None and value.shape == expected.shape
            ok = ok and np.array_equal(np.isnan(value), np.isnan(expected)) and np.isfinite(value[valid]).all()
            corr = pearson(value[valid], expected[valid]) if ok else float("nan")
            rep.add(f"{label}:{site}", ok and corr >= 0.99, corr, 0.99)
        for key in ("clean_prob", "corrupt_prob"):
            err = max_abs(agent.get(key, np.array([])), ref[key])
            rep.add(f"{label}:{key}", err <= 1e-4, err, 1e-4)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        rep.fail_all(names, str(exc))


def grade(task_dir, outputs, heldout_outputs, cache):
    rep = Report("t03-causal-tracing")
    grade_one(rep, "visible", os.path.join(task_dir, "inputs"), outputs, cache)
    if heldout_outputs:
        grade_one(rep, "heldout", os.path.join(task_dir, "heldout"), heldout_outputs, cache)
    else:
        rep.fail_all(["heldout:required"], "held-out output directory required")
    return rep


if __name__ == "__main__":
    sys.exit(grader_main("t03", grade))
