"""Reference: forward hooks on submodules, with the residual additions
recovered from the hooked tensors (cloned before any in-place update).

``run(inputs, outputs, variant=None)`` is reused by the seeded wrong solutions.
"""
import argparse
import importlib.util
import json
import os

import numpy as np
import torch
import torchvision
from transformers import AutoModelForImageClassification


def load_inputs(inputs):
    cfg = json.load(open(os.path.join(inputs, "config.json")))
    layers = json.load(open(os.path.join(inputs, "layers.json")))
    spec = importlib.util.spec_from_file_location("stimuli", os.path.join(inputs, "stimuli.py"))
    st = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(st)
    images, labels = st.make_stimuli(cfg["stimulus_seed"], cfg["n_categories"],
                                     cfg["per_category"], cfg["image_size"])
    mean = torch.tensor(cfg["normalize_mean"]).view(1, 3, 1, 1)
    std = torch.tensor(cfg["normalize_std"]).view(1, 3, 1, 1)
    x = (torch.from_numpy(images) - mean) / std
    return cfg, layers, x, labels


def cnn_features(x):
    model = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1).eval()
    rec = {}
    handles = []

    def save(name):
        def hook(mod, inp, out):
            rec[name] = out.detach().clone()
        return hook

    def save_relu(prefix):
        # torchvision's block ReLU is shared by both activations and runs in
        # place, so its hook input is already overwritten when the hook fires:
        # the residual sum is rebuilt from the cloned bn2 output and the shortcut.
        count = {"n": 0}

        def hook(mod, inp, out):
            count["n"] += 1
            if count["n"] == 1:
                rec[f"{prefix}.relu1"] = out.detach().clone()
            else:
                rec[f"{prefix}.out"] = out.detach().clone()
        return hook

    def save_block_input(prefix):
        def pre(mod, args):
            rec[f"{prefix}._in"] = args[0].detach().clone()
        return pre

    handles.append(model.relu.register_forward_hook(save("stem.relu")))
    handles.append(model.maxpool.register_forward_hook(save("stem.maxpool")))
    for stage in (1, 2, 3, 4):
        for b in (0, 1):
            blk = getattr(model, f"layer{stage}")[b]
            p = f"layer{stage}.{b}"
            for sub in ("conv1", "bn1", "conv2", "bn2"):
                handles.append(getattr(blk, sub).register_forward_hook(save(f"{p}.{sub}")))
            if blk.downsample is not None:
                handles.append(blk.downsample.register_forward_hook(save(f"{p}.downsample")))
            handles.append(blk.relu.register_forward_hook(save_relu(p)))
            handles.append(blk.register_forward_pre_hook(save_block_input(p)))
    handles.append(model.avgpool.register_forward_hook(save("avgpool")))
    handles.append(model.fc.register_forward_hook(save("fc")))
    with torch.no_grad():
        model(x)
    for h in handles:
        h.remove()
    for stage in (1, 2, 3, 4):
        for b in (0, 1):
            p = f"layer{stage}.{b}"
            shortcut = rec[f"{p}.downsample"] if f"{p}.downsample" in rec else rec[f"{p}._in"]
            rec[f"{p}.add"] = rec[f"{p}.bn2"] + shortcut
            rec.pop(f"{p}._in")
    return rec


def vit_blocks(vit):
    return vit.layers if hasattr(vit, "layers") else vit.encoder.layer


def vit_features(x):
    model = AutoModelForImageClassification.from_pretrained("facebook/deit-tiny-patch16-224").eval()
    vit = model.vit
    rec = {}
    handles = []

    def save(name, index=None):
        def hook(mod, inp, out):
            t = out if index is None else out[index]
            rec[name] = t.detach().clone()
        return hook

    def save_input(name):
        def pre(mod, args):
            rec[name] = args[0].detach().clone()
        return pre

    handles.append(vit.embeddings.register_forward_hook(save("embeddings")))
    blocks = vit_blocks(vit)
    for i, layer in enumerate(blocks):
        p = f"block{i}"
        handles.append(layer.register_forward_pre_hook(save_input(f"{p}._in")))
        handles.append(layer.layernorm_before.register_forward_hook(save(f"{p}.ln1")))
        handles.append(layer.attention.register_forward_hook(save(f"{p}.attn", 0)))
        handles.append(layer.layernorm_after.register_forward_hook(save(f"{p}.ln2")))
        if hasattr(layer, "mlp"):  # fc1 -> activation (a plain function) -> fc2
            handles.append(layer.mlp.fc1.register_forward_hook(save(f"{p}._fc1")))
            handles.append(layer.mlp.fc2.register_forward_hook(save(f"{p}.mlp_out")))
        else:  # older layout: intermediate (dense + act) and output (dense + residual)
            handles.append(layer.intermediate.register_forward_hook(save(f"{p}.mlp_hidden")))
            handles.append(layer.output.dense.register_forward_hook(save(f"{p}.mlp_out")))
        handles.append(layer.register_forward_hook(save(f"{p}.add2")))
    handles.append(vit.layernorm.register_forward_hook(save("final_ln")))
    handles.append(model.classifier.register_forward_hook(save("classifier")))
    with torch.no_grad():
        model(pixel_values=x)
    for h in handles:
        h.remove()
    for i, layer in enumerate(blocks):
        p = f"block{i}"
        a2 = rec[f"{p}.add2"]
        a2 = a2[0] if isinstance(a2, (tuple, list)) else a2
        rec[f"{p}.add2"] = a2
        rec[f"{p}.add1"] = rec[f"{p}.attn"] + rec.pop(f"{p}._in")
        if f"{p}._fc1" in rec:
            with torch.no_grad():
                rec[f"{p}.mlp_hidden"] = layer.mlp.activation_fn(rec.pop(f"{p}._fc1"))
    return rec


def rdm(feats):
    f = feats.reshape(feats.shape[0], -1).astype(np.float64)
    c = np.corrcoef(f)
    d = 1.0 - c
    np.fill_diagonal(d, 0.0)
    return d


def category_rdm(labels):
    return (labels[:, None] != labels[None, :]).astype(np.float64)


def match(a, b):
    iu = np.triu_indices(a.shape[0], k=1)
    return float(np.corrcoef(a[iu], b[iu])[0, 1])


def run(inputs, outputs, variant=None):
    cfg, layers, x, labels = load_inputs(inputs)
    if variant == "stimulus_order":
        perm = np.random.default_rng(0).permutation(len(x))
        x = x[perm]  # labels left as they were: a stimulus-order error
    os.makedirs(outputs, exist_ok=True)
    best = {}
    for net, feats in (("cnn", cnn_features(x)), ("vit", vit_features(x))):
        cat = category_rdm(labels)
        rdms = {}
        scores = {}
        for item in layers[net]:
            lid = item["id"]
            if variant == "module_outputs_only" and (lid.endswith(".add") or lid.endswith(".add1")):
                continue
            t = feats[lid].numpy()
            if variant == "inplace_alias" and lid.endswith(".bn2"):
                t = feats[lid.replace(".bn2", ".add")].numpy()  # what an un-cloned hook would hold
            if variant == "bn1_alias" and lid.endswith(".bn1"):
                t = feats[lid.replace(".bn1", ".relu1")].numpy()  # bn1 overwritten by the in-place ReLU
            if variant == "drop_cls" and net == "vit" and t.ndim == 3 and t.shape[1] == 197:
                t = t[:, 1:, :]  # patch tokens only
            if variant == "transpose":
                t = np.ascontiguousarray(t.reshape(t.shape[0], -1).T).reshape(t.shape[0], -1)
            rdms[lid] = rdm(t)
            scores[lid] = match(rdms[lid], cat)
        np.savez(os.path.join(outputs, f"rdms_{net}.npz"), **rdms)
        best[net] = max(scores, key=scores.get)
    json.dump(best, open(os.path.join(outputs, "best_layer.json"), "w"), indent=1)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", required=True)
    p.add_argument("--outputs", required=True)
    a = p.parse_args()
    run(a.inputs, a.outputs)
