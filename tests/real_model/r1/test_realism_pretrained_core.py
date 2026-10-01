"""R1 core plumbing tests (memo A1/A2/A6): the three-checkpoint core, offline.

Minimal real-checkpoint rows proving the registry -> preflight -> offline
fixture pipeline end to end; the RG gallery (B1) builds on these fixtures.
Every assertion is payload identity against an independent same-process
rerun -- never the machinery's own report.
"""

from __future__ import annotations

import json

import pytest
import torch

from tests.real_model.registry import NATURAL_INPUTS_DIR

pytestmark = [pytest.mark.heavy, pytest.mark.real_model, pytest.mark.real_checkpoint]


def _prompt(prompt_id: str) -> dict:
    for line in (NATURAL_INPUTS_DIR / "prompts.jsonl").read_text().splitlines():
        if line.strip() and json.loads(line)["id"] == prompt_id:
            return json.loads(line)
    raise AssertionError(f"prompt {prompt_id!r} not in the committed corpus")


def test_distilgpt2_real_tokenizer_trace_matches_direct(r1_loader):
    import torchlens as tl

    loaded = r1_loader("r1-distilgpt2")
    model, tokenizer = loaded["model"], loaded["tokenizer"]
    encoded = tokenizer(_prompt("clean-factual-1")["clean"], return_tensors="pt")
    with torch.no_grad():
        direct = model(**encoded).logits
    trace = tl.trace(model, (), dict(encoded))
    assert trace.outcome.status.name == "COMPLETE"
    matches = [
        op
        for op in trace.output_ops
        if op.out.shape == direct.shape and torch.equal(op.out, direct)
    ]
    assert matches, "traced logits != direct forward on the real checkpoint"


def test_albert_real_cross_layer_sharing_is_multi_pass(r1_loader):
    import torchlens as tl

    loaded = r1_loader("r1-albert-base-v2")
    model, tokenizer = loaded["model"], loaded["tokenizer"]
    encoded = tokenizer(_prompt("encoder-sentence-1")["text"], return_tensors="pt")
    trace = tl.trace(model, (), dict(encoded))
    assert trace.outcome.status.name == "COMPLETE"
    max_passes = 0
    for layer in trace.layers:
        ops = getattr(layer, "ops", None)
        if ops is not None:
            max_passes = max(max_passes, len(ops))
    assert max_passes >= 12, (
        "albert-base-v2 shares ONE encoder layer across 12 positions; the"
        f" capture saw at most {max_passes} passes -- real cross-layer weight"
        " sharing (the pass-blind-donor substrate) is not being grouped"
    )


def test_resnet18_real_weights_natural_image_matches_direct(r1_loader):
    from torchvision.io import read_image
    from torchvision.transforms.functional import center_crop, resize

    import torchlens as tl

    loaded = r1_loader("r1-resnet18-imagenet1k-v1")
    model = loaded["model"]
    image = read_image(str(NATURAL_INPUTS_DIR / "pd_astronaut_256.jpg"))
    batch = center_crop(resize(image, 256, antialias=True), 224).unsqueeze(0).float() / 255.0
    with torch.no_grad():
        direct = model(batch)
    trace = tl.trace(model, batch)
    assert trace.outcome.status.name == "COMPLETE"
    assert torch.equal(trace.output_ops[0].out, direct), (
        "traced resnet18 output != direct forward on real weights"
    )
    # First real vision weights in the suite: the BN buffers are the point.
    assert model.bn1.running_mean.abs().sum() > 0, "pretrained BN stats missing?"
