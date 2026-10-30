"""T1 grader. The oracle is a port of the published estimator (companion code
of the July 2026 Jacobian lens paper, fitting module at commit 581d398):
forward hooks on the blocks, parameters frozen, the earliest source residual
marked as the graph root, one-hot cotangents at every valid target position
for 16 output dimensions per backward pass, mean over valid source positions,
running mean over texts. Readout: unembed(final_norm(J_l h)) at the last position.
"""
import json
import math
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_common"))
from grade_utils import Report, grader_main, inputs_hash, load_json, load_npz, rel_fro, topk_overlap  # noqa: E402

DIM_BATCH = 16


def read_jsonl(path):
    return [json.loads(l)["text"] for l in open(path) if l.strip()]


class Recorder:
    def __init__(self, blocks, at, root=None):
        self.blocks, self.at, self.root, self.acts, self.handles = blocks, at, root, {}, []

    def _hook(self, i):
        def hook(mod, inp, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            if i == self.root:
                t.requires_grad_(True)
            self.acts[i] = t
        return hook

    def __enter__(self):
        for i in self.at:
            self.handles.append(self.blocks[i].register_forward_hook(self._hook(i)))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()


def oracle_fit(model, tok, texts, cfg):
    src = list(cfg["source_layers"])
    tgt = cfg["target_layer"]
    skip = cfg["skip_first_positions"]
    d = model.config.hidden_size
    model.requires_grad_(False)
    blocks = model.gpt_neox.layers
    total = {l: torch.zeros(d, d) for l in src}
    n = 0
    for text in texts:
        ids = tok(text, return_tensors="pt", add_special_tokens=False,
                  truncation=True, max_length=cfg["max_seq_len"]).input_ids
        T = ids.shape[1]
        mask = torch.zeros(T, dtype=torch.bool)
        mask[skip:T - 1] = True
        if mask.sum() == 0:
            continue
        valid = mask.nonzero(as_tuple=True)[0]
        J = {l: torch.zeros(d, d) for l in src}
        with Recorder(blocks, [*src, tgt], root=min(src)) as rec, torch.enable_grad():
            model(ids.expand(DIM_BATCH, -1))
            h_t = rec.acts[tgt]
            h_s = [rec.acts[l] for l in src]
            bidx = torch.arange(DIM_BATCH)
            cot = torch.zeros_like(h_t)
            n_pass = math.ceil(d / DIM_BATCH)
            for k, d0 in enumerate(range(0, d, DIM_BATCH)):
                nd = min(DIM_BATCH, d - d0)
                cot.zero_()
                cot[bidx[:nd, None], valid[None, :], d0 + bidx[:nd, None]] = 1.0
                grads = torch.autograd.grad(outputs=h_t, inputs=h_s, grad_outputs=cot,
                                            retain_graph=k < n_pass - 1)
                for l, g in zip(src, grads):
                    J[l][d0:d0 + nd, :] = g[:nd, valid, :].float().mean(dim=1)
        for l in src:
            total[l] += J[l]
        n += 1
    return {l: total[l] / n for l in src}


def self_check(model, tok, text, cfg, lens_rows, n_dims=3):
    """Consistency check of the batched estimator against per-target VJPs on
    one short text: for a few output dimensions, the row obtained with the
    cotangent at every valid target at once must equal the sum of the rows
    obtained one target position at a time."""
    src = list(cfg["source_layers"])
    tgt = cfg["target_layer"]
    skip = cfg["skip_first_positions"]
    blocks = model.gpt_neox.layers
    ids = tok(text, return_tensors="pt", add_special_tokens=False,
              truncation=True, max_length=skip + 8).input_ids
    T = ids.shape[1]
    valid = torch.arange(skip, T - 1)
    dims = torch.randperm(model.config.hidden_size, generator=torch.Generator().manual_seed(0))[:n_dims]
    with Recorder(blocks, [*src, tgt], root=min(src)) as rec, torch.enable_grad():
        model(ids)
        h_t, h_s = rec.acts[tgt], [rec.acts[l] for l in src]
        for d in dims.tolist():
            cot = torch.zeros_like(h_t)
            cot[0, valid, d] = 1.0
            batched = torch.autograd.grad(h_t, h_s, grad_outputs=cot, retain_graph=True)
            summed = [torch.zeros_like(g) for g in batched]
            for p in valid.tolist():
                cot = torch.zeros_like(h_t)
                cot[0, p, d] = 1.0
                for acc, g in zip(summed, torch.autograd.grad(h_t, h_s, grad_outputs=cot, retain_graph=True)):
                    acc += g
            for b, sm in zip(batched, summed):
                assert torch.allclose(b[0, valid], sm[0, valid], atol=1e-5, rtol=1e-4), \
                    "oracle self-check failed: batched cotangent differs from per-target sum"


@torch.no_grad()
def oracle_readout(model, tok, prompts, lens, cfg):
    src = list(cfg["source_layers"])
    tgt = cfg["target_layer"]
    k = cfg["readout_top_k"]
    blocks = model.gpt_neox.layers
    top, model_top = [], []
    for text in prompts:
        ids = tok(text, return_tensors="pt", add_special_tokens=False).input_ids
        with Recorder(blocks, [*src, tgt]) as rec:
            model(ids)
        row = []
        for l in src:
            h = rec.acts[l][0, -1].float()
            logits = model.get_output_embeddings()(model.gpt_neox.final_layer_norm(h @ lens[l].T))
            row.append(torch.topk(logits, k).indices.tolist())
        top.append(row)
        hf = rec.acts[tgt][0, -1].float()
        model_top.append(torch.topk(model.get_output_embeddings()(model.gpt_neox.final_layer_norm(hf)), k).indices.tolist())
    return top, model_top


def oracle(task_dir, cache):
    inputs = os.path.join(task_dir, "inputs")
    heldout = os.path.join(task_dir, "heldout")
    key = inputs_hash(inputs, heldout, os.path.abspath(__file__))
    f = os.path.join(cache, f"t01_oracle_{key}.npz")
    if os.path.isfile(f):
        with np.load(f, allow_pickle=True) as z:
            return ({int(k[2:]): z[k] for k in z.files if k.startswith("J_")},
                    z["top_visible"], z["top_heldout"], z["model_visible"], z["model_heldout"])
    from transformers import AutoModelForCausalLM, AutoTokenizer
    cfg = json.load(open(os.path.join(inputs, "config.json")))
    assert open(os.path.join(inputs, "corpus.jsonl")).read() == open(os.path.join(heldout, "corpus.jsonl")).read(), \
        "held-out corpus must equal the visible corpus (only the readout prompts differ)"
    tok = AutoTokenizer.from_pretrained(cfg["model"])
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], torch_dtype=torch.float32).eval()
    texts = read_jsonl(os.path.join(inputs, "corpus.jsonl"))
    lens = oracle_fit(model, tok, texts, cfg)
    self_check(model, tok, texts[0], cfg, lens)
    tv, mv = oracle_readout(model, tok, read_jsonl(os.path.join(inputs, "readout_prompts.jsonl")), lens, cfg)
    th, mh = oracle_readout(model, tok, read_jsonl(os.path.join(heldout, "readout_prompts.jsonl")), lens, cfg)
    out = {f"J_{l}": J.numpy() for l, J in lens.items()}
    np.savez(f, top_visible=np.array(tv), top_heldout=np.array(th),
             model_visible=np.array(mv), model_heldout=np.array(mh), **out)
    return {l: J.numpy() for l, J in lens.items()}, np.array(tv), np.array(th), np.array(mv), np.array(mh)


def grade(task_dir, outputs, heldout_outputs, cache):
    rep = Report("t01-jacobian-lens")
    cfg = json.load(open(os.path.join(task_dir, "inputs", "config.json")))
    src = list(cfg["source_layers"])
    J_ref, top_v, top_h, model_v, model_h = oracle(task_dir, cache)
    names = ["shapes", "jacobian", "readout_visible", "model_top10_visible", "readout_heldout"]
    try:
        z = load_npz(os.path.join(outputs, "lens.npz"))
        rd = load_json(os.path.join(outputs, "readout.json"))
        J = {l: z[f"J_{l}"] for l in src}
        top = np.array(rd["top10"])
        mtop = np.array(rd["model_top10"])
    except Exception as e:
        rep.fail_all(names, f"missing output: {e}")
        return rep
    d = J_ref[src[0]].shape[0]
    shapes_ok = all(J[l].shape == (d, d) for l in src) and top.shape == top_v.shape \
        and mtop.shape == model_v.shape and list(rd.get("layers", [])) == src
    rep.add("shapes", shapes_ok, note=f"top10 shape {top.shape}, expected {top_v.shape}")
    errs = {l: rel_fro(J[l], J_ref[l]) for l in src}
    worst = max(errs.values())
    rep.add("jacobian", worst < 1e-3, worst, 1e-3, note=" ".join(f"L{l}={e:.2e}" for l, e in errs.items()))
    ov = topk_overlap(top, top_v, 10) if top.shape == top_v.shape else 0.0
    rep.add("readout_visible", ov >= 0.9, ov, 0.9)
    mo = topk_overlap(mtop, model_v, 10) if mtop.shape == model_v.shape else 0.0
    rep.add("model_top10_visible", mo >= 1.0, mo, 1.0)
    if heldout_outputs:
        try:
            th = np.array(load_json(os.path.join(heldout_outputs, "readout.json"))["top10"])
            ovh = topk_overlap(th, top_h, 10) if th.shape == top_h.shape else 0.0
            rep.add("readout_heldout", ovh >= 0.9, ovh, 0.9)
        except Exception as e:
            rep.fail_all(["readout_heldout"], f"missing output: {e}")
    return rep


if __name__ == "__main__":
    sys.exit(grader_main("t01", grade))
