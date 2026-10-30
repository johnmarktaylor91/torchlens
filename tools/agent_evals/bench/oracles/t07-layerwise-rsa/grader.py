"""T7 grader. The oracle re-computes every listed layer explicitly from the
submodules (no hooks), checks its own result against the networks' logits,
and compares RDMs and best layers with the agent's outputs."""
import importlib.util
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "_common"))
from grade_utils import Report, grader_main, load_json, load_npz, rdm_correlation  # noqa: E402


def load_inputs(inputs):
    cfg = json.load(open(os.path.join(inputs, "config.json")))
    layers = json.load(open(os.path.join(inputs, "layers.json")))
    spec = importlib.util.spec_from_file_location("stimuli_oracle", os.path.join(inputs, "stimuli.py"))
    st = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(st)
    images, labels = st.make_stimuli(cfg["stimulus_seed"], cfg["n_categories"],
                                     cfg["per_category"], cfg["image_size"])
    mean = torch.tensor(cfg["normalize_mean"]).view(1, 3, 1, 1)
    std = torch.tensor(cfg["normalize_std"]).view(1, 3, 1, 1)
    return cfg, layers, (torch.from_numpy(images) - mean) / std, labels


def _rdm(t):
    f = t.reshape(t.shape[0], -1).to(torch.float64).numpy()
    d = 1.0 - np.corrcoef(f)
    np.fill_diagonal(d, 0.0)
    return d


@torch.no_grad()
def cnn_oracle(x):
    import torchvision
    m = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1).eval()
    out = {}
    h = m.relu(m.bn1(m.conv1(x)))
    out["stem.relu"] = _rdm(h)
    h = m.maxpool(h)
    out["stem.maxpool"] = _rdm(h)
    for stage in (1, 2, 3, 4):
        for b in (0, 1):
            blk = getattr(m, f"layer{stage}")[b]
            p = f"layer{stage}.{b}"
            c1 = blk.conv1(h)
            out[f"{p}.conv1"] = _rdm(c1)
            n1 = blk.bn1(c1)
            out[f"{p}.bn1"] = _rdm(n1)
            r1 = torch.relu(n1)
            out[f"{p}.relu1"] = _rdm(r1)
            c2 = blk.conv2(r1)
            out[f"{p}.conv2"] = _rdm(c2)
            n2 = blk.bn2(c2)
            out[f"{p}.bn2"] = _rdm(n2)
            if blk.downsample is not None:
                idn = blk.downsample(h)
                out[f"{p}.downsample"] = _rdm(idn)
            else:
                idn = h
            s = n2 + idn
            out[f"{p}.add"] = _rdm(s)
            h = torch.relu(s)
            out[f"{p}.out"] = _rdm(h)
    h = m.avgpool(h)
    out["avgpool"] = _rdm(h)
    logits = m.fc(torch.flatten(h, 1))
    out["fc"] = _rdm(logits)
    assert torch.allclose(logits, m(x), atol=1e-4), "oracle CNN recomputation disagrees with the model"
    return out


@torch.no_grad()
def vit_oracle(x):
    from transformers import AutoModelForImageClassification
    model = AutoModelForImageClassification.from_pretrained("facebook/deit-tiny-patch16-224").eval()
    vit = model.vit
    out = {}
    h = vit.embeddings(x)
    out["embeddings"] = _rdm(h)
    for i, layer in enumerate(vit.encoder.layer):
        p = f"block{i}"
        ln1 = layer.layernorm_before(h)
        out[f"{p}.ln1"] = _rdm(ln1)
        attn = layer.attention(ln1)
        attn = attn[0] if isinstance(attn, (tuple, list)) else attn
        out[f"{p}.attn"] = _rdm(attn)
        h = attn + h
        out[f"{p}.add1"] = _rdm(h)
        ln2 = layer.layernorm_after(h)
        out[f"{p}.ln2"] = _rdm(ln2)
        mh = layer.intermediate(ln2)
        out[f"{p}.mlp_hidden"] = _rdm(mh)
        mo = layer.output.dense(mh)
        out[f"{p}.mlp_out"] = _rdm(mo)
        h = mo + h
        out[f"{p}.add2"] = _rdm(h)
    h = vit.layernorm(h)
    out["final_ln"] = _rdm(h)
    logits = model.classifier(h[:, 0])
    out["classifier"] = _rdm(logits)
    assert torch.allclose(logits, model(pixel_values=x).logits, atol=1e-4), \
        "oracle ViT recomputation disagrees with the model"
    return out


def oracle(inputs, cache_file):
    if os.path.isfile(cache_file):
        with np.load(cache_file, allow_pickle=True) as z:
            return {k: z[k] for k in z.files}, json.loads(str(z["_best"]))
    cfg, layers, x, labels = load_inputs(inputs)
    cat = (labels[:, None] != labels[None, :]).astype(np.float64)
    rd = {}
    best = {}
    for net, o in (("cnn", cnn_oracle(x)), ("vit", vit_oracle(x))):
        scores = {}
        for item in layers[net]:
            lid = item["id"]
            rd[f"{net}/{lid}"] = o[lid]
            scores[lid] = rdm_correlation(o[lid], cat)
        best[net] = max(scores, key=scores.get)
    np.savez(cache_file, _best=np.array(json.dumps(best)), **rd)
    return rd, best


def grade_one(rep, label, task_dir, inputs, outputs, cache):
    tag = os.path.basename(inputs)
    ref, ref_best = oracle(inputs, os.path.join(cache, f"t07_oracle_{tag}.npz"))
    layers = json.load(open(os.path.join(inputs, "layers.json")))
    names = [f"{label}:layers_present", f"{label}:rdm_agreement", f"{label}:best_layer"]
    try:
        agent = {}
        for net in ("cnn", "vit"):
            z = load_npz(os.path.join(outputs, f"rdms_{net}.npz"))
            agent.update({f"{net}/{k}": v for k, v in z.items()})
        best = load_json(os.path.join(outputs, "best_layer.json"))
    except Exception as e:
        rep.fail_all(names, f"missing output: {e}")
        return
    n = next(iter(ref.values())).shape[0]
    missing = [k for k in ref if k not in agent or agent[k].shape != (n, n)]
    rep.add(names[0], not missing, len(ref) - len(missing), len(ref),
            note=f"missing={missing[:5]}" if missing else "")
    corrs = {k: rdm_correlation(agent[k], ref[k]) for k in ref if k not in missing}
    worst = min(corrs.values()) if corrs else float("nan")
    worst_k = min(corrs, key=corrs.get) if corrs else None
    rep.add(names[1], (not missing) and worst >= 0.999, worst, 0.999, note=f"worst layer {worst_k}")
    rep.add(names[2], best == ref_best, note=f"agent={best} oracle={ref_best}")


def grade(task_dir, outputs, heldout_outputs, cache):
    rep = Report("t07-layerwise-rsa")
    grade_one(rep, "visible", task_dir, os.path.join(task_dir, "inputs"), outputs, cache)
    if heldout_outputs:
        grade_one(rep, "heldout", task_dir, os.path.join(task_dir, "heldout"), heldout_outputs, cache)
    return rep


if __name__ == "__main__":
    sys.exit(grader_main("t07", grade))
