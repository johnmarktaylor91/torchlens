"""A02 pretrained real-GPT-2 row: residual + final-norm anchors on real weights.

Venue-gated exactly like ``tests/real_model/r1`` (GATE-ID R1_OFFLINE_VENUE):
runs only under the offline preflighted cache env, loads THROUGH the artifact
registry's checkpoint-evidence hook (pinned revision, never a bare model id),
and skips with the gate id everywhere else. The R0 rows in
``test_sem_resid_a02.py`` carry the structural gate; this row proves the same
anchors on the actual 12-layer, 768-dim checkpoint.
"""

from __future__ import annotations

import os
import warnings

import pytest
import torch

import torchlens as tl
from tests.real_model.registry import load_registry

pytestmark = [pytest.mark.heavy, pytest.mark.real_model]

_IN_VENUE = (
    os.environ.get("HF_HUB_OFFLINE") == "1" and os.environ.get("TRANSFORMERS_OFFLINE") == "1"
)


@pytest.mark.skipif(
    not _IN_VENUE,
    reason=(
        "GATE R1_OFFLINE_VENUE: not in the offline preflighted venue; run "
        "scripts/preflight_fetch_artifacts.py fetch, export its print-env, then rerun."
    ),
)
def test_pretrained_gpt2_residual_and_final_norm_anchors() -> None:
    from transformers import AutoModelForCausalLM

    row = load_registry().checkpoint_evidence("r1-gpt2")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = AutoModelForCausalLM.from_pretrained(row.model_id, revision=row.revision).eval()
    ids = torch.randint(
        0, model.config.vocab_size, (1, 8), generator=torch.Generator().manual_seed(20260826)
    )
    trace = tl.trace(model, ids)
    assert trace.outcome.status.name == "COMPLETE"

    block_5 = trace.modules["transformer.h.5"]
    pre = block_5.facets.resid_pre.value
    assert pre.shape == (1, 8, model.config.n_embd)
    assert pre.dtype.is_floating_point
    post_4 = trace.modules["transformer.h.4"].facets.resid_post.value
    assert torch.equal(pre, post_4)

    facets = trace.modules["self"].facets
    assert facets.final_norm_kind == "layer_norm"
    assert torch.equal(facets.final_norm_gamma.value, model.transformer.ln_f.weight)
    assert torch.equal(facets.final_norm_beta.value, model.transformer.ln_f.bias)
    assert torch.equal(facets.unembed_weight.value, model.lm_head.weight)
    assert model.lm_head.weight is model.transformer.wte.weight
    last = model.config.num_hidden_layers - 1
    post_last = trace.modules[f"transformer.h.{last}"].facets.resid_post.value
    assert torch.equal(facets.final_norm_input.value, post_last)
