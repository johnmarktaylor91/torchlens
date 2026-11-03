"""Extraction correctness with an explicit stimulus-order error control.

Run ``python examples/gallery/neuro_rdms.py --out DIR``. Uses rsatoolbox 0.3.2
as the independent correlation-distance implementation, not a neural benchmark.
Fashion-MNIST test images (Zalando, MIT license) have a pinned compressed SHA256.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import struct
import time
import traceback
import urllib.request
from pathlib import Path
from typing import Any

import _gallery as gallery
import numpy as np
import rsatoolbox
import torch
from torch import nn
from torchvision import models
from torchvision.transforms import functional as TF

import torchlens as tl

CLAIM = "Per-layer extraction correctness and correlation RDMs against rsatoolbox 0.3.2"
SEED = 17
SHUFFLE_SEED = 123
CORRELATION_MIN = 0.999
RDM_ATOL = 1e-6
SHUFFLED_MAX = 0.5
DATA_URL = (
    "https://raw.githubusercontent.com/zalandoresearch/fashion-mnist/master/"
    "data/fashion/t10k-images-idx3-ubyte.gz"
)
DATA_SHA256 = "346e55b948d973a97e58d2351dde16a484bd415d4595297633bb08f03db6a073"
NORMALIZE_MEAN = (0.485, 0.456, 0.406)
NORMALIZE_STD = (0.229, 0.224, 0.225)
ADD_NAMES = {"add", "__add__", "__iadd__"}


def stimuli(count: int) -> tuple[torch.Tensor, list[str]]:
    """Fetch and verify the pinned images, preserving test-split order.

    Parameters
    ----------
    count:
        Number of consecutive test images, starting at index zero.

    Returns
    -------
    tuple
        Normalized RGB batch and stable stimulus identifiers.
    """
    if not 4 <= count <= 92:
        raise ValueError("count must be between 4 and 92")
    with urllib.request.urlopen(DATA_URL, timeout=60) as response:
        compressed = response.read(8_000_001)
    if hashlib.sha256(compressed).hexdigest() != DATA_SHA256:
        raise ValueError("Fashion-MNIST compressed input hash mismatch")
    raw = gzip.decompress(compressed)
    if struct.unpack(">IIII", raw[:16]) != (2051, 10000, 28, 28):
        raise ValueError("unexpected Fashion-MNIST IDX header")
    images = np.frombuffer(raw, dtype=np.uint8, offset=16).reshape(10000, 28, 28)
    x = torch.from_numpy(images[:count].copy()).unsqueeze(1).float().div(255)
    x = TF.resize(x.repeat(1, 3, 1, 1), [224, 224], antialias=True)
    x = TF.normalize(x, NORMALIZE_MEAN, NORMALIZE_STD)
    return x, [f"fashion-mnist:test:{i:05d}" for i in range(count)]


# region: plain
def plain_extract(model: nn.Module, x: torch.Tensor, stages: list[str]) -> dict[str, torch.Tensor]:
    """Extract stages and recompute every ResNet residual sum independently.

    Parameters
    ----------
    model, x, stages:
        Eval model, pinned stimulus batch and module outputs to retain.

    Returns
    -------
    dict
        Module outputs and pre-ReLU residual sums in execution order.
    """
    outputs: dict[str, torch.Tensor] = {}
    hooks: list[Any] = []

    def capture(name: str) -> Any:
        """Build a module hook that snapshots values before in-place edits."""

        def collect(_module: nn.Module, _args: Any, output: torch.Tensor) -> None:
            """Retain one independent module output."""
            outputs[name] = output.detach().clone()

        return collect

    for stage in stages:
        hooks.append(model.get_submodule(stage).register_forward_hook(capture(stage)))
    blocks = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, models.resnet.BasicBlock)
    ]
    for index, (_name, block) in enumerate(blocks):
        hooks.append(block.bn2.register_forward_hook(capture(f"main_{index}")))
        if block.downsample is not None:
            hooks.append(block.downsample.register_forward_hook(capture(f"skip_{index}")))
        else:

            def remember(_module: nn.Module, args: Any, i: int = index) -> None:
                """Retain the identity-branch input."""
                outputs[f"skip_{i}"] = args[0].detach().clone()

            hooks.append(block.register_forward_pre_hook(remember))
    vit_blocks = list(model.encoder.layers.children()) if not blocks else []
    if vit_blocks:

        def position(_module: nn.Module, args: Any) -> None:
            """Recompute the positional embedding addition."""
            outputs["position"] = args[0].detach().clone() + model.encoder.pos_embedding.detach()

        hooks.append(model.encoder.register_forward_pre_hook(position))
        for index, block in enumerate(vit_blocks):

            def remember_vit(_module: nn.Module, args: Any, i: int = index) -> None:
                """Retain the first residual input."""
                outputs[f"vit_skip_{i}"] = args[0].detach().clone()

            def remember_second(_module: nn.Module, args: Any, i: int = index) -> None:
                """Retain the second residual input."""
                outputs[f"vit_second_skip_{i}"] = args[0].detach().clone()

            hooks.append(block.register_forward_pre_hook(remember_vit))
            hooks.append(block.dropout.register_forward_hook(capture(f"attention_{index}")))
            hooks.append(block.ln_2.register_forward_pre_hook(remember_second))
            hooks.append(block.mlp.register_forward_hook(capture(f"mlp_{index}")))
    try:
        with torch.no_grad():
            model(x)
    finally:
        for hook in hooks:
            hook.remove()
    result = {stage: outputs[stage] for stage in stages}
    result.update(
        {f"residual_{i}": outputs[f"main_{i}"] + outputs[f"skip_{i}"] for i in range(len(blocks))}
    )
    if vit_blocks:
        result["residual_0"] = outputs["position"]
        for i in range(len(vit_blocks)):
            result[f"residual_{2 * i + 1}"] = outputs[f"vit_skip_{i}"] + outputs[f"attention_{i}"]
            result[f"residual_{2 * i + 2}"] = outputs[f"vit_second_skip_{i}"] + outputs[f"mlp_{i}"]
    return result


def reference_rdm(features: torch.Tensor) -> np.ndarray:
    """Compute correlation distances with rsatoolbox's own Dataset pipeline.

    Parameters
    ----------
    features:
        Independent activations with stimuli on the first axis.

    Returns
    -------
    numpy.ndarray
        Condensed upper-triangle dissimilarities in presentation order.
    """
    matrix = features.flatten(1).double().numpy()
    dataset = rsatoolbox.data.Dataset(measurements=matrix)
    return rsatoolbox.rdm.calc_rdm(dataset, method="correlation").dissimilarities[0]


# endregion


def pearson(left: np.ndarray, right: np.ndarray) -> float:
    """Compare condensed RDMs with Pearson correlation.

    Parameters
    ----------
    left, right:
        Condensed dissimilarities.

    Returns
    -------
    float
        Pearson coefficient (NaN fails all the fixed checks).
    """
    return float(np.corrcoef(left, right)[0, 1])


# region: torchlens
def measure(
    model: nn.Module, x: torch.Tensor, ids: list[str], arch: str, out: Path
) -> gallery.Entry:
    """Compare all selected sites and every residual sum, then replay an RDM.

    Parameters
    ----------
    model, x, ids, arch, out:
        Model, images, presentation order, architecture and output directory.

    Returns
    -------
    gallery.Entry
        Oracle, order-error control and artifact evidence.
    """
    stages = (
        ["layer1", "layer2", "layer3", "layer4", "avgpool"]
        if arch == "resnet18"
        else ["conv_proj", "encoder.layers.encoder_layer_5", "encoder.ln"]
    )
    selector = tl.func("add") | tl.func("__add__") | tl.func("__iadd__")
    for stage in stages:
        selector = selector | tl.module(stage)
    with torch.no_grad():
        trace = tl.trace(model, x, save=selector)
    adds = [op for op in trace.ops if op.func_name in ADD_NAMES]
    sites = stages + [op.label for op in adds]
    plain = plain_extract(model, x, stages)
    if arch == "resnet18" and len(adds) != 8:
        raise ValueError(f"expected all eight ResNet residual adds, found {len(adds)}")
    if arch == "vit_b_16" and len(adds) != 25:
        raise ValueError(f"expected 24 ViT residual adds and the positional add, found {len(adds)}")
    labels = [trace[site].label for site in sites]
    keys = stages + [f"residual_{i}" for i in range(len(adds))]
    rdm = tl.neuro.rdms(
        trace, sites=sites, metric="correlation", pattern_descriptors={"stimulus_id": ids}
    )
    rdm_labels = list(rdm.rdm_descriptors["site"])
    measured = {
        key: rdm.dissimilarities[rdm_labels.index(label)]
        for key, label in zip(keys, labels, strict=True)
    }
    reference = {key: reference_rdm(value) for key, value in plain.items()}
    correlations = {key: pearson(measured[key], reference[key]) for key in keys}
    errors = {key: float(np.max(np.abs(measured[key] - reference[key]))) for key in keys}
    permutation = np.random.default_rng(SHUFFLE_SEED).permutation(len(ids))
    shuffled_ids = [ids[i] for i in permutation]
    shuffled = {key: pearson(measured[key], reference_rdm(plain[key][permutation])) for key in keys}
    # A synthetic extraction target, deliberately not a neural recording or a brain score.
    target = reference[stages[-1]]
    best_plain = max(keys, key=lambda key: pearson(reference[key], target))
    best_tl = max(keys, key=lambda key: pearson(measured[key], target))
    entry = gallery.Entry(
        "neuro_rdms",
        CLAIM,
        {
            "seed": SEED,
            "shuffle_seed": SHUFFLE_SEED,
            "arch": arch,
            "weights": "IMAGENET1K_V1",
            "data_sha256": DATA_SHA256,
            "stimulus_ids": ids,
            "preprocessing": "28x28 grayscale to RGB; antialiased bilinear 224x224; ImageNet mean/std",
            "dtype": "float32",
            "rsatoolbox": "0.3.2",
        },
    )
    entry.oracles = {
        "per_layer_rdm": gallery.Check(
            f"each Pearson >= {CORRELATION_MIN}",
            correlations,
            all(value >= CORRELATION_MIN for value in correlations.values()),
        ),
        "rsatoolbox_pipeline": gallery.Check(
            f"each max absolute error <= {RDM_ATOL}",
            errors,
            all(value <= RDM_ATOL for value in errors.values()),
        ),
        "best_layer": gallery.Check(
            "same best layer against the declared synthetic target",
            {"plain": best_plain, "torchlens": best_tl},
            best_plain == best_tl,
        ),
    }
    entry.controls = {
        "shuffled_order": gallery.Check(
            f"every shuffled Pearson < {SHUFFLED_MAX}",
            shuffled,
            all(value < SHUFFLED_MAX for value in shuffled.values()),
        ),
        "order_mismatch_flag": gallery.Check(
            "identity comparison flags the one-sided permutation",
            {"stimulus_order_mismatch": ids != shuffled_ids, "shuffled_ids": shuffled_ids},
            ids != shuffled_ids,
        ),
    }
    trace.annotations["gallery"] = {
        "stimulus_ids": ids,
        "sites": sites,
        "keys": keys,
        "preprocessing": entry.pinned["preprocessing"],
        "rdms": {k: v.tolist() for k, v in measured.items()},
        "shuffled_ids": shuffled_ids,
        "stimulus_order_mismatch": ids != shuffled_ids,
    }
    artifact = out / "neuro_rdms.tlspec"
    trace.save(artifact, level="portable", overwrite=True)

    def replay(loaded: Any) -> dict[str, Any]:
        """Recompute one RDM from the reloaded portable activations."""
        meta = loaded.annotations["gallery"]
        reloaded = tl.neuro.rdms(
            loaded,
            sites=[meta["sites"][0]],
            metric="correlation",
            pattern_descriptors={"stimulus_id": meta["stimulus_ids"]},
        )
        error = float(np.max(np.abs(reloaded.dissimilarities[0] - measured[keys[0]])))
        return {
            "matched": error <= RDM_ATOL,
            "max_abs_error": error,
            "stimulus_order_mismatch": meta["stimulus_ids"] != meta["shuffled_ids"],
        }

    entry.artifact = gallery.artifact_record(artifact, replay)
    entry.measured = {
        "synthetic_target": stages[-1],
        "residual_adds": len(adds),
        "stimulus_order_mismatch": ids != shuffled_ids,
    }
    entry.verbs = ("capture", "select", "read", "compare", "save")
    entry.architectures = ("ResNet-18" if arch == "resnet18" else "ViT-B/16",)
    return entry


# endregion


def main() -> None:
    """Run extraction and the fixed order-error control."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--quick", action="store_true", help="eight stimuli instead of 24")
    parser.add_argument("--arch", choices=("resnet18", "vit_b_16"), default="resnet18")
    parser.add_argument("--stimuli", type=int, default=None)
    args = parser.parse_args()
    started = time.time()
    torch.set_num_threads(2)
    torch.manual_seed(SEED)
    args.out.mkdir(parents=True, exist_ok=True)
    try:
        x, ids = stimuli(args.stimuli or (8 if args.quick else 24))
        weights = (
            models.ResNet18_Weights.IMAGENET1K_V1
            if args.arch == "resnet18"
            else models.ViT_B_16_Weights.IMAGENET1K_V1
        )
        model = getattr(models, args.arch)(weights=weights).eval()
        entry = measure(model, x, ids, args.arch, args.out)
    except Exception as exc:
        traceback.print_exc()
        entry = gallery.Entry(
            "neuro_rdms", CLAIM, {"seed": SEED, "arch": args.arch, "data_sha256": DATA_SHA256}
        )
        entry.oracles["execution"] = gallery.Check(
            "measurement completes without a capture gap",
            {"type": type(exc).__name__, "message": str(exc)},
            False,
        )
    entry.lines = {
        name: gallery.count_region_lines(Path(__file__), name) for name in ("torchlens", "plain")
    }
    entry.lines["reference"] = "not measured"
    summary = gallery.report(entry, args.out, "cpu", started)
    summary["stamp"]["stimulus_order_mismatch"] = entry.measured.get("stimulus_order_mismatch")
    (args.out / "neuro_rdms.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    print("STAMP: " + json.dumps(summary["stamp"], sort_keys=True))


if __name__ == "__main__":
    main()
