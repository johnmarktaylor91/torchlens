"""F03 item 12: the commissioned workflow gallery, end to end (heavy tier).

The flagship head-ablation experiment on a config-built GPT-2 (the real HF
``from_dict`` parsing path, zero network): baseline capture (replayable) ->
site_sweep over one named block's heads (candidates SCOPED,
resolved_site_count == 1 asserted per candidate) -> effects table -> why(top
member) -> save -> reload -> the effect table and the operation chronology
read back from the artifact. The pretrained openai-community/gpt2 12-head
row is the R1/network-tier variant of this same scenario (see the lane
report); the mechanism assertions are weight-independent.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.experiment import head_ablation_candidates, sweep_sites

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

    # ---- material phase -----------------------------------------------------
    baseline = tl.trace(model, ids, capture=tl.options.CaptureOptions(replayable=True))
    candidates = head_ablation_candidates(baseline, module, heads=n_heads)
    bundle = sweep_sites(
        baseline,
        candidates=candidates,
        edit=tl.zero_ablate(),
        metric=metric,
        retain="all",
        model=model,
        inputs=ids,
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
    top_candidate, top_delta = view.rank_changes(k=1)[0]
    assert isinstance(top_delta, float)
    report = bundle.why(top_candidate)
    assert report.lineage_status == "exact"
    assert report.member_suffix, "the variant has no construction suffix"
    assert report.member_suffix[0].fire_count >= 1, "zero-fire is a failure"

    # ---- the record is the Bundle: one sweep operation, the table beside it --
    sweeps = [op for op in bundle.history if op.kind == "site_sweep"]
    assert len(sweeps) == 1, "a sweep is ONE operation, never one per candidate"
    assert sweeps[0].params["attempted"] == n_heads

    # ---- persistence ---------------------------------------------------------
    bundle_path = tmp_path / "sweep_bundle"
    bundle.save(str(bundle_path))
    loaded = tl.load(str(bundle_path))
    loaded_view = loaded.effects()
    assert [row.to_payload() for row in loaded_view.rows] == [row.to_payload() for row in view.rows]
    loaded_report = loaded.why(top_candidate)
    assert loaded_report.lineage_status == report.lineage_status
    assert loaded_report.member_suffix == report.member_suffix
    assert [op.kind for op in loaded.history] == [op.kind for op in bundle.history]
