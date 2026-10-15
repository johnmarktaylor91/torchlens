"""Flagship head-ablation experiment on pretrained GPT-2 small (R1 venue).

The workflow gallery's head-ablation row on the pinned
``openai-community/gpt2`` checkpoint: arm a file-backed ledger with the
hypothesis and metric declared before the first material step, capture an
intervention-ready baseline on a real prompt, sweep the 12 heads of
``transformer.h.5.attn`` (scoped v-facet candidates, one site each) on the
live-hook engine with every member retained, read the effect table, explain
the top member, record an observation and a verdict with its basis, save and
reload. An independent oracle in this file zeroes each head's slice of the
block's ``c_proj`` input with a plain torch forward pre-hook (no TorchLens in
its path); every candidate must agree with it and rank identically. Runs only
in the preflighted offline venue (GATE-ID R1_OFFLINE_VENUE, ``conftest.py``
here), where it can never skip.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

pytestmark = [pytest.mark.real_model, pytest.mark.real_checkpoint, pytest.mark.slow]

_PROMPT = "The Eiffel Tower is located in the city of"
_MODULE = "transformer.h.5.attn"


def test_head_ablation_flagship_on_pretrained_gpt2(r1_loader, tmp_path: Path) -> None:
    import torchlens as tl
    from torchlens.experiment import head_ablation_candidates, ledger, site_sweep

    loaded = r1_loader("r1-gpt2")
    model, tokenizer = loaded["model"], loaded["tokenizer"]
    model.eval()
    config = model.config
    assert (config.n_layer, config.n_head, config.n_embd) == (12, 12, 768)
    n_heads, d_head = config.n_head, config.n_embd // config.n_head
    ids = tokenizer(_PROMPT, return_tensors="pt")["input_ids"]
    (target,) = tokenizer(" Paris")["input_ids"]

    def metric(member: tl.Trace) -> float:
        return float(member[member.output_layers[0]].out[0, -1, target].item())

    led = ledger(
        tmp_path / "flagship.tlledger",
        question=f"which head of {_MODULE} carries ' Paris'?",
        hypothesis="one head dominates the ' Paris' logit",
        metric="logit[' Paris'] at the last position",
    )
    try:
        baseline = tl.trace(model, ids, capture=tl.options.CaptureOptions(intervention_ready=True))
        candidates = head_ablation_candidates(baseline, _MODULE, heads=n_heads)
        bundle = site_sweep(
            baseline,
            candidates=candidates,
            edit=tl.zero_ablate(),
            metric=metric,
            retain="all",
            model=model,
            x=ids,
        )
        assert bundle.names[0] == "baseline" and len(bundle.names) == n_heads + 1
        view = bundle.effects()
        rows = {row.candidate_id: row for row in view.rows if row.candidate_id != "__baseline__"}
        assert set(rows) == set(candidates)
        assert all(row.resolved_site_count == 1 for row in rows.values())
        assert all(row.status == "completed" for row in rows.values())
        top_candidate, top_delta = view.most_changed(top_n=1)[0]
        report = bundle.why(top_candidate)
        assert report.lineage_status == "exact"
        assert report.member_suffix and report.member_suffix[0].fire_count >= 1

        step_ids = list(led.entries()["e1"].steps)
        assert len(step_ids) == 1, "a sweep is one material step, never one per candidate"
        led.observe(f"ablating {top_candidate} moved the logit by {top_delta:.4f}", value=top_delta)
        led.verdict("supports", text=f"{top_candidate} dominates", basis=[step_ids[0]])

        bundle_path = tmp_path / "flagship_bundle"
        bundle.save(str(bundle_path))
        reloaded = tl.load(str(bundle_path))
        assert [row.to_payload() for row in reloaded.effects().rows] == [
            row.to_payload() for row in view.rows
        ]
        assert reloaded.why(top_candidate).member_suffix == report.member_suffix
    finally:
        led.close()

    # ---- the independent oracle (no TorchLens) -------------------------------
    c_proj = model.get_submodule(f"{_MODULE}.c_proj")
    oracle: dict[str, float] = {}
    for head in range(n_heads):

        def zero_slice(_module: torch.nn.Module, args: tuple, *, index: int = head) -> tuple:
            (hidden,) = args
            hidden = hidden.clone()
            hidden[..., index * d_head : (index + 1) * d_head] = 0.0
            return (hidden,)

        handle = c_proj.register_forward_pre_hook(zero_slice)
        try:
            with torch.no_grad():
                oracle[f"h{head}"] = float(model(ids).logits[0, -1, target].item())
        finally:
            handle.remove()
    swept = {name: row.value for name, row in rows.items()}
    for name, value in oracle.items():
        assert swept[name] == pytest.approx(value, abs=1e-3), (name, swept[name], value)
    assert sorted(swept, key=swept.__getitem__) == sorted(oracle, key=oracle.__getitem__)
