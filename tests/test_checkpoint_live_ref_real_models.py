"""Checkpoint live-ref guard, real-model legs (A-CKPT; foldB MEMO s4.2 tests 1-2).

Leg 2 (zero-network): torchvision resnet18 + the real IMAGENET1K_V1 checkpoint
plus one deterministic SGD step between two captures -- the exact measured
repro of the defect (weight_norm_diff false zero, diff_pair "identical",
aggregate('std') all zeros, out/grad serving today's bytes). Runs offline
today against the cached checkpoint; proves the class is not
transformer-specific.

Leg 1 (transformer): EleutherAI/pythia-70m at published training revisions
step0 and step64 -- genuinely different immutable published bytes, loaded into
ONE model object and traced twice with fixed retained token ids. The fetch is
the named P04 prerequisite; when the revision pair is not in the local HF
cache this leg reports itself BLOCKED-ON-P04 loudly instead of pretending to
cover the row (the skip does NOT close the row; see the A-CKPT lane report).
"""

from __future__ import annotations

import hashlib
from typing import Any

import pytest
import torch

import torchlens as tl
from torchlens.errors import CheckpointSeriesLiveParamsError

_CODE = "checkpoint_series_live_params"


@pytest.mark.heavy
def test_resnet18_real_checkpoint_series_refuses_typed() -> None:
    """resnet18 + real ImageNet weights + one SGD step: refusal, not false zeros."""

    torchvision = pytest.importorskip("torchvision")
    from torchvision.models import ResNet18_Weights, resnet18

    del torchvision
    torch.manual_seed(0)
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.eval()
    x = torch.randn(2, 3, 64, 64)

    fc_before = model.fc.weight.detach().clone()
    trace_ck0 = tl.trace(model, x)
    optimizer = torch.optim.SGD(model.parameters(), lr=1.0)
    model(x).sum().backward()
    optimizer.step()
    trace_ck64 = tl.trace(model, x)
    fc_after = model.fc.weight.detach().clone()

    # Independent checkpoint truth: the step moved the weight.
    moved = torch.linalg.vector_norm(fc_after - fc_before).item()
    assert moved > 0.0

    # The defect mechanism: both members resolve ONE live parameter object,
    # so any per-member "capture value" claim would be today's bytes twice.
    assert trace_ck0.params["fc.weight"].value is trace_ck64.params["fc.weight"].value
    assert trace_ck0.params["fc.weight"].value_basis.basis == "live_ref"

    bundle = tl.bundle({"ck0": trace_ck0, "ck64": trace_ck64})
    view = bundle.params["fc.weight"]
    for read in (
        lambda: view.weight_norm_diff,
        lambda: view.diff_pair(),
        lambda: view.aggregate("std"),
        lambda: view.out,
        lambda: view.grad,
    ):
        with pytest.raises(CheckpointSeriesLiveParamsError) as excinfo:
            read()
        exc = excinfo.value
        assert exc.fields["code"] == _CODE
        assert set(exc.fields["members"]) == {"ck0", "ck64"}
        assert exc.fields["param_address"] == "fc.weight"
        assert "'ck0'" in str(exc) and "'ck64'" in str(exc)


def _pythia_revision_pair() -> tuple[Any, Any, Any]:
    """Return (config_loader, both revision state dicts) from the LOCAL cache only."""

    transformers = pytest.importorskip("transformers")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        model = AutoModelForCausalLM.from_pretrained(
            "EleutherAI/pythia-70m", revision="step0", local_files_only=True
        )
        state_step64 = AutoModelForCausalLM.from_pretrained(
            "EleutherAI/pythia-70m", revision="step64", local_files_only=True
        ).state_dict()
        tokenizer = AutoTokenizer.from_pretrained(
            "EleutherAI/pythia-70m", revision="step0", local_files_only=True
        )
    except Exception:  # noqa: BLE001 - any cache miss is the named P04 gap
        pytest.skip(
            "BLOCKED-ON-P04: EleutherAI/pythia-70m revisions step0/step64 are not "
            "in the local HF cache (named P04 fetch prerequisite, foldB MEMO s5 "
            "row 3). This skip does NOT close the real-model row; the A-CKPT lane "
            "report carries it as the named blocked row."
        )
    del transformers
    return model, state_step64, tokenizer


@pytest.mark.slow
def test_pythia_70m_published_revision_pair_refuses_typed() -> None:
    """pythia-70m step0/step64 in ONE model object: the RLHF-forensics shape."""

    model, state_step64, tokenizer = _pythia_revision_pair()
    token_ids = tokenizer("The checkpoint series", return_tensors="pt").input_ids

    probe_address = "gpt_neox.layers.0.attention.dense.weight"
    digest_step0 = hashlib.sha256(model.state_dict()[probe_address].numpy().tobytes()).hexdigest()
    digest_step64 = hashlib.sha256(state_step64[probe_address].numpy().tobytes()).hexdigest()
    # Independent checkpoint truth: the published revisions genuinely differ.
    assert digest_step0 != digest_step64

    trace_step0 = tl.trace(model, token_ids)
    model.load_state_dict(state_step64)
    trace_step64 = tl.trace(model, token_ids)

    bundle = tl.bundle({"step0": trace_step0, "step64": trace_step64})
    view = bundle.params[probe_address]
    with pytest.raises(CheckpointSeriesLiveParamsError) as excinfo:
        _ = view.weight_norm_diff
    exc = excinfo.value
    assert exc.fields["code"] == _CODE
    assert set(exc.fields["members"]) == {"step0", "step64"}
    assert exc.fields["param_address"] == probe_address
    assert "'step0'" in str(exc) and "'step64'" in str(exc)
