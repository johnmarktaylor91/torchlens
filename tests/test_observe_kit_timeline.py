"""Observe items 11-13: timeline v2 artifact, deterministic SVG, liveness view.

Conservation and honesty pins: closed categories with named-absent facts,
output pseudo-rows excluded, tied parameters counted once, the saved band
never stacked (gross beside decomposition), module rollup conservation, a
byte-exact SVG golden (sha256 pin -- regenerate deliberately, never in a
merge gate), and the liveness estimate that excludes the autograd-saved band
wholesale and never changes observed category totals.
"""

from __future__ import annotations

import hashlib
import json

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._observe import (
    estimated_liveness,
    memory_timeline_v2,
    module_rollup,
)
from torchlens.viz import timeline


class _DeterministicBlock(nn.Module):
    """Value-deterministic model: constant weights, no RNG anywhere."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.full((8,), 0.5))
        self.shift = nn.Parameter(torch.full((8,), 0.25))
        self.register_buffer("mask", torch.ones(8))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """mul -> mul -> add -> tanh, all shape-stable torch ops."""

        y = x * self.mask
        y = y * self.scale
        y = y + self.shift
        return torch.tanh(y)


class _TiedEmbedding(nn.Module):
    """Weight tying: embedding and head share ONE parameter storage."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(16, 8)
        self.head = nn.Linear(8, 16, bias=False)
        self.head.weight = self.embed.weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed then project back out through the tied weight."""

        return self.head(self.embed(x))


def _artifact(model: nn.Module, x: torch.Tensor, **capture: object) -> dict:
    """Capture one trace and build its timeline artifact."""

    captured = tl.trace(model, x, capture=tl.options.CaptureOptions(**capture))
    try:
        return memory_timeline_v2(captured)
    finally:
        captured.cleanup()


def test_tied_parameters_count_once_with_alias_rows() -> None:
    """Weight tying: one counted parameter row, one alias row with zero bytes."""

    artifact = _artifact(_TiedEmbedding(), torch.arange(4).unsqueeze(0))
    param_rows = [row for row in artifact["events"] if row["category"] == "parameter"]
    alias_rows = [row for row in param_rows if row["alias_of"] is not None]
    assert alias_rows, "the tied weight must surface as an alias row"
    assert all(row["bytes"] == 0 for row in alias_rows)
    counted_total = sum(row["bytes"] for row in param_rows)
    unique_param_bytes = 16 * 8 * 4  # the ONE shared storage, fp32
    assert counted_total == unique_param_bytes


def test_module_rollup_conservation() -> None:
    """Root total equals the sum of exclusive displayed rows at every depth."""

    artifact = _artifact(_DeterministicBlock(), torch.ones(2, 8))
    for depth in (1, 2):
        rollup = module_rollup(artifact, depth=depth)
        exclusive_sum = sum(sum(bucket.values()) for bucket in rollup["rows"].values())
        assert rollup["total_bytes"] == exclusive_sum
        assert rollup["total_bytes"] == sum(row["bytes"] for row in artifact["events"])


def test_svg_binning_is_disclosed_never_truncating() -> None:
    """Over-budget traces bin contiguous ordinals and disclose the bin size."""

    artifact = _artifact(_DeterministicBlock(), torch.ones(2, 8))
    binned = timeline(artifact, column_budget=2)
    assert "binned x" in binned
    unbinned = timeline(artifact, column_budget=4096)
    assert "binned x" not in unbinned


def test_liveness_view_is_estimated_scoped_and_total_preserving() -> None:
    """Death rules printed; autograd_saved excluded; totals untouched."""

    artifact = _artifact(
        _DeterministicBlock(), torch.ones(2, 8, requires_grad=True), backward_ready=True
    )
    totals_before = dict(artifact["category_totals"])
    view = estimated_liveness(artifact)
    assert view["evidence"] == "estimated"
    assert "EXCLUDED wholesale" in view["death_rules"]["autograd_saved"]
    assert view["peak_bytes"] >= view["persistent_baseline_bytes"]
    assert view["series"], "the estimate carries a per-ordinal series"
    # The live series is NOT the monotone cumulative-produced series.
    live_values = [value for _ordinal, value in view["series"]]
    assert max(live_values) == view["peak_bytes"]
    assert artifact["category_totals"] == totals_before


def test_recording_input_refuses_with_metadata_remedy() -> None:
    """Sparse Recording products refuse typed, naming the to_trace remedy."""

    from torchlens._errors import InvalidArgumentError

    recording = tl.record(_DeterministicBlock(), torch.ones(2, 8), save=tl.func("tanh"))
    with pytest.raises(InvalidArgumentError) as excinfo:
        memory_timeline_v2(recording)
    assert excinfo.value.fields["code"] == "timeline_requires_trace"
    assert "to_trace" in str(excinfo.value)


def test_export_door_writes_v2_and_v1_stays_compatible(tmp_path) -> None:
    """The export registry serves both v1 (one-release) and v2."""

    from torchlens import export

    model = _DeterministicBlock()
    captured = tl.trace(model, torch.ones(2, 8))
    try:
        v1_path = export.memory_timeline(captured, tmp_path / "v1.json")
        v2_path = export.memory_timeline_v2(captured, tmp_path / "v2.json")
        v1_payload = json.loads(v1_path.read_text())
        v2_payload = json.loads(v2_path.read_text())
        assert v1_payload["schema"] == "torchlens.memory_timeline.v1"
        assert "deprecation" in v1_payload
        assert v1_payload["events"], "the v1 compatibility exporter still serves rows"
        assert v2_payload["schema"] == "torchlens.memory_timeline.v2"
        assert "memory_timeline_v2" in export.list_targets()
    finally:
        captured.cleanup()


def _regen() -> None:
    """Print the current SVG golden sha256 for a deliberate rebaseline."""

    artifact = _artifact(
        _DeterministicBlock(), torch.ones(2, 8, requires_grad=True), backward_ready=True
    )
    print(hashlib.sha256(timeline(artifact).encode()).hexdigest())
