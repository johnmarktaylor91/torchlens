"""F03 item 7b: the head-ablation sugar + THE SAME-FILE ORACLE (heavy tier).

The sugar's v-facet lowering is checked candidate-by-candidate against a
hand-written torch forward_pre_hook zeroing head i's slice of the c_proj
input (no TorchLens in the oracle path) on a config-built GPT-2 running
through the REAL HF ``from_dict`` parsing path — agreement to tolerance and
identical ranking. The oracle is the LAW for architecture-encapsulating
sugar because it is what caught the panel's own flagship mis-spelling (the
unscoped 36-site broadcast). Heavy tier: each row runs a config GPT-2
capture per head plus the manual-hook oracle loop.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.errors.episode import BundleExperimentError
from torchlens.experiment import head_ablation_candidates, site_sweep

pytestmark = pytest.mark.heavy


def _gpt2() -> nn.Module:
    pytest.importorskip("transformers")
    import sys

    sys.path.insert(0, ".")
    from tests.real_model.r0.families import build_gpt2

    return build_gpt2("eager").eval()


def test_head_ablation_sugar_matches_hand_written_hook_oracle() -> None:
    """12/12-style agreement row: sugar v-facet zero == manual z-slice zero.

    The oracle is a plain torch forward_pre_hook on the block's c_proj
    zeroing head i's input slice — NO TorchLens in the oracle path. For
    standard MHA attn_out is linear in V_i, so the two spellings must agree
    per candidate with identical ranking.
    """

    model = _gpt2()
    torch.manual_seed(0)
    ids = torch.randint(0, 100, (1, 8))
    config = model.config
    n_heads, d_head = config.n_head, config.n_embd // config.n_head
    module = "transformer.h.1.attn"
    target_token = 17

    def metric(member: tl.Trace) -> float:
        label = member.output_layers[0]
        return float(member[label].out[0, -1, target_token].item())

    baseline = tl.trace(model, ids, capture=tl.options.CaptureOptions(intervention_ready=True))
    candidates = head_ablation_candidates(baseline, module, heads=n_heads)
    assert list(candidates) == [f"h{i}" for i in range(n_heads)]
    bundle = site_sweep(
        baseline,
        candidates=candidates,
        edit=tl.zero_ablate(),
        metric=metric,
        retain="none",
        model=model,
        x=ids,
    )
    sweep_values = {
        row.candidate_id: row.value
        for row in bundle.effects().rows
        if row.candidate_id != "__baseline__" and row.status in ("completed", "released")
    }
    assert set(sweep_values) == set(candidates), bundle.effects().rows

    # ---- the hand-written oracle (no TorchLens) ---------------------------
    c_proj = model.get_submodule(f"{module}.c_proj")
    oracle_values: dict[str, float] = {}
    for i in range(n_heads):

        def zero_head_slice(mod: nn.Module, args: tuple, *, head: int = i) -> tuple:
            (hidden,) = args
            hidden = hidden.clone()
            hidden[..., head * d_head : (head + 1) * d_head] = 0.0
            return (hidden,)

        handle = c_proj.register_forward_pre_hook(zero_head_slice)
        try:
            with torch.no_grad():
                logits = model(ids).logits
        finally:
            handle.remove()
        oracle_values[f"h{i}"] = float(logits[0, -1, target_token].item())

    for candidate_id, oracle_value in oracle_values.items():
        assert sweep_values[candidate_id] == pytest.approx(oracle_value, abs=1e-4), (
            candidate_id,
            sweep_values[candidate_id],
            oracle_value,
        )
    sweep_rank = sorted(sweep_values, key=lambda c: sweep_values[c])
    oracle_rank = sorted(oracle_values, key=lambda c: oracle_values[c])
    assert sweep_rank == oracle_rank


def test_head_sugar_refuses_gqa_and_unresolved_module() -> None:
    model = _gpt2()
    torch.manual_seed(0)
    ids = torch.randint(0, 100, (1, 8))
    baseline = tl.trace(model, ids, capture=tl.options.CaptureOptions(intervention_ready=True))
    with pytest.raises(BundleExperimentError) as excinfo:
        head_ablation_candidates(baseline, "transformer.h.9.attn", heads=2)
    assert excinfo.value.fields["code"] == "head_ablation_module_unresolved"
    with pytest.raises(BundleExperimentError) as excinfo:
        head_ablation_candidates(baseline, "transformer.h.1.attn", heads=99)
    assert excinfo.value.fields["code"] == "head_ablation_heads_invalid"


def test_unscoped_head_selector_is_refused_as_sweep_candidate() -> None:
    """The negative row: the measured 36-site broadcast cannot enter a sweep."""

    model = _gpt2()
    torch.manual_seed(0)
    ids = torch.randint(0, 100, (1, 8))
    baseline = tl.trace(model, ids, capture=tl.options.CaptureOptions(intervention_ready=True))

    def metric(member: tl.Trace) -> float:
        label = member.output_layers[0]
        return float(member[label].out[0, -1, 17].item())

    bundle = site_sweep(
        baseline,
        candidates={"unscoped": tl.head(0)},
        edit=tl.zero_ablate(),
        metric=metric,
        retain="none",
        model=model,
        x=ids,
    )
    (row,) = [r for r in bundle.effects().rows if r.candidate_id == "unscoped"]
    assert row.status == "refused"
    assert row.resolved_site_count is None
