"""F03 item 12: the commissioned workflow gallery, end to end (heavy tier).

The flagship head-ablation experiment on a config-built GPT-2 (the real HF
``from_dict`` parsing path, zero network): arm a file-backed ledger with
hypothesis and metric declared BEFORE the first material step -> baseline
capture (intervention_ready) -> site_sweep over one named block's heads
(candidates SCOPED, resolved_site_count == 1 asserted per candidate) ->
effects table -> why(top member) -> explicit observation -> verdict with
basis -> save -> reload -> serve over the read-only tools. The pretrained
openai-community/gpt2 12-head row is the R1/network-tier variant of this
same scenario (see the lane report); the mechanism assertions are
weight-independent.

Also carries the funded emission-overhead measurement: armed vs unarmed
sweep, asserted bounded (one step per operation = two fsyncs).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.experiment import (
    head_ablation_candidates,
    ledger,
    ledger_entry,
    ledger_evidence,
    ledger_overview,
    site_sweep,
)
from torchlens.experiment._ledger import EvidenceRef, _artifact_digest

pytestmark = pytest.mark.heavy


def _gpt2() -> nn.Module:
    pytest.importorskip("transformers")
    sys.path.insert(0, ".")
    from tests.real_model.r0.families import build_gpt2

    return build_gpt2("eager").eval()


def test_flagship_gallery_end_to_end(tmp_path: Path) -> None:
    model = _gpt2()
    torch.manual_seed(0)
    ids = torch.randint(0, 100, (1, 8))
    n_heads = model.config.n_head
    module = "transformer.h.1.attn"
    target_token = 17

    def metric(member: tl.Trace) -> float:
        return float(member[member.output_layers[0]].out[0, -1, target_token].item())

    ledger_path = tmp_path / "experiment.tlledger"
    led = ledger(
        ledger_path,
        question=f"which head of {module} carries token {target_token}?",
        hypothesis="one head dominates the target logit",
        metric=f"logit[{target_token}] at the last position",
    )
    try:
        # ---- material phase -------------------------------------------------
        baseline = tl.trace(model, ids, capture=tl.options.CaptureOptions(intervention_ready=True))
        candidates = head_ablation_candidates(baseline, module, heads=n_heads)
        bundle = site_sweep(
            baseline,
            candidates=candidates,
            edit=tl.zero_ablate(),
            metric=metric,
            retain="all",
            model=model,
            x=ids,
        )
        # Members baseline-first; one exact construction answer per variant.
        assert bundle.names[0] == "baseline"
        view = bundle.effects()
        per_candidate = {
            row.candidate_id: row for row in view.rows if row.candidate_id != "__baseline__"
        }
        assert set(per_candidate) == set(candidates)
        assert all(row.resolved_site_count == 1 for row in per_candidate.values())
        assert all(row.status == "completed" for row in per_candidate.values())
        top_candidate, top_delta = view.most_changed(top_n=1)[0]
        report = bundle.why(top_candidate)
        assert report.lineage_status == "exact"
        assert report.member_suffix, "the variant has no construction suffix"
        assert report.member_suffix[0].fire_count >= 1, "zero-fire is a failure"

        # ---- semantic phase --------------------------------------------------
        step_ids = list(led.entries()["e1"].steps)
        assert len(step_ids) == 1, "a sweep is ONE material step, never per-candidate"
        led.observe(
            f"ablating {top_candidate} moved the target logit by {top_delta:.4f}",
            value=top_delta,
        )
        led.verdict("supports", text=f"{top_candidate} dominates", basis=[step_ids[0]])

        # ---- persistence + re-serving ---------------------------------------
        bundle_path = tmp_path / "sweep_bundle"
        bundle.save(str(bundle_path))
        led.relink(
            EvidenceRef(
                kind="bundle",
                uri="sweep_bundle",
                object_id=bundle.bundle_id,
                digest=_artifact_digest(bundle_path),
            ),
            reason="bundle persisted after the sweep step",
        )
        loaded = tl.load(str(bundle_path))
        loaded_view = loaded.effects()
        assert [row.to_payload() for row in loaded_view.rows] == [
            row.to_payload() for row in view.rows
        ]
        loaded_report = loaded.why(top_candidate)
        assert loaded_report.lineage_status == report.lineage_status
        assert loaded_report.member_suffix == report.member_suffix

        # Ledger artifact is reference-only: no tensor payloads, no audit copies.
        text = ledger_path.read_text(encoding="utf-8")
        assert "tensor(" not in text
        assert "intervention_audit" not in text

        # ---- the agent-recovery reads (bounded) ------------------------------
        overview = ledger_overview(str(ledger_path))
        (line,) = overview["entries"]
        assert "verdict=supports (basis=1)" in line
        assert len(line) < 400, "the overview line must stay tens-of-tokens bounded"
        entry = ledger_entry(str(ledger_path), "e1")
        assert entry["metric_declared_at_seq"] < min(
            event["seq"] for event in entry["events"] if event["kind"] == "material_step_started"
        ), "declaration order is a checkable fact (never post-hoc pre-registration)"
        evidence = ledger_evidence(str(ledger_path), "e1", ref_index=1)
        assert evidence["ref"]["availability"] == "persisted"
        served_tables = evidence["served"]["effect_tables"]
        assert list(served_tables.values())[0]["rows"], "served effect table is empty"
    finally:
        led.close()


def test_emission_overhead_is_bounded(tmp_path: Path) -> None:
    """Armed vs unarmed sweep: one step per operation is near-free."""

    class _Tiny(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(3, 3)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.relu(self.linear(x))

    torch.manual_seed(0)
    model = _Tiny().eval()
    x = torch.randn(2, 3)

    def metric(member: tl.Trace) -> float:
        return float(member[member.output_layers[0]].out.sum().item())

    def one_sweep() -> None:
        baseline = tl.trace(model, x, capture=tl.options.CaptureOptions(intervention_ready=True))
        site_sweep(
            baseline,
            candidates=["relu_1_2"],
            edit=tl.zero_ablate(),
            metric=metric,
            retain="none",
            model=model,
            x=x,
        )

    one_sweep()  # warm the wrappers
    t0 = time.perf_counter()
    one_sweep()
    unarmed = time.perf_counter() - t0

    led = ledger(tmp_path / "bench.tlledger", hypothesis="h", metric="m")
    try:
        t0 = time.perf_counter()
        one_sweep()
        armed = time.perf_counter() - t0
    finally:
        led.close()
    # One started + one finalized event (two fsyncs) per operation: bounded
    # well under 2x + a generous absolute grace for fs jitter.
    assert armed <= unarmed * 2.0 + 0.5, (armed, unarmed)
