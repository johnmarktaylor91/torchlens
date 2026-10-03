"""Grouped backward attribution floor pins (L9 memo 1.1).

Read-only per-site rollups over BACKWARD records keyed on the merged L1
site_key surface: per-site GradFn/GradFnCall aggregation (fire counts,
per-fire timing once memo 1.3 evidence exists, pass coverage). Accessor-level
only -- no persisted fields, no writes to L1 or L6 files. Spelling
DOCUMENTED-UNSTABLE pending the naming session.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens._errors import InvalidArgumentError
from torchlens.quantities import Duration


class _ReusedModule(nn.Module):
    """One Linear called twice: both calls share a site key."""

    def __init__(self) -> None:
        super().__init__()
        self.entry = nn.Linear(4, 8)
        self.shared = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.relu(self.entry(x))
        h = torch.relu(self.shared(h))
        return self.shared(h)


def _backward_trace(model: nn.Module | None = None) -> tl.Trace:
    torch.manual_seed(0)
    trace = tl.trace(
        model if model is not None else _ReusedModule(),
        torch.randn(3, 4),
        save_mode="reference",
        capture=tl.options.CaptureOptions(backward_ready=True),
    )
    trace.log_backward(trace.output_ops[0].out.sum())
    return trace


def test_rollup_groups_reused_module_calls_under_one_site() -> None:
    trace = _backward_trace()
    summary = trace.grad_fn_site_summary
    keyed = {key: entry for key, entry in summary.items() if key is not None}
    assert keyed, "no keyed sites in the rollup"
    shared_entries = [
        entry for key, entry in keyed.items() if "|shared|" in key or key.startswith("s1|shared")
    ]
    assert shared_entries, f"no shared-module site among {sorted(keyed)}"
    shared_fires = sum(entry["fire_count"] for entry in shared_entries)
    # The reused module fired backward once per call instance.
    assert shared_fires >= 2
    for entry in summary.values():
        assert entry["fire_count"] >= len(entry["pass_coverage"])
        assert entry["pass_coverage"] == tuple(sorted(entry["pass_coverage"]))


@pytest.mark.smoke
def test_rollup_carries_live_timing_evidence() -> None:
    trace = _backward_trace()
    summary = trace.grad_fn_site_summary
    total_fires = sum(entry["fire_count"] for entry in summary.values())
    total_timed = sum(entry["timed_fire_count"] for entry in summary.values())
    assert total_fires == total_timed > 0
    for entry in summary.values():
        if entry["timed_fire_count"]:
            assert isinstance(entry["total_fire_duration"], Duration)
            assert float(entry["total_fire_duration"]) > 0
        else:
            assert entry["total_fire_duration"] is None


def test_unattributed_grad_fns_group_under_none_bucket() -> None:
    trace = _backward_trace()
    summary = trace.grad_fn_site_summary
    assert None in summary, "AccumulateGrad nodes should land in the None bucket"
    unattributed = summary[None]
    assert unattributed["fire_count"] > 0
    assert any(label.startswith("accumulategrad") for label in unattributed["grad_fn_labels"])


def test_rollup_counts_survive_load_without_timing(tmp_path) -> None:
    trace = _backward_trace()
    live_summary = trace.grad_fn_site_summary
    path = tmp_path / "site_summary.tlspec"
    with tl._io.prerelease.activate_prerelease_fields():
        tl.save(trace, str(path))
        loaded = tl.load(str(path))
    loaded_summary = loaded.grad_fn_site_summary
    assert set(loaded_summary) == set(live_summary)
    for key, live_entry in live_summary.items():
        loaded_entry = loaded_summary[key]
        assert loaded_entry["grad_fn_labels"] == live_entry["grad_fn_labels"]
        assert loaded_entry["fire_count"] == live_entry["fire_count"]
        assert loaded_entry["pass_coverage"] == live_entry["pass_coverage"]
        # Loaded traces carry no runtime timing evidence: honest zeros/None,
        # never a false duration.
        assert loaded_entry["timed_fire_count"] == 0
        assert loaded_entry["total_fire_duration"] is None


def test_keyless_op_backed_rollup_refuses_site_key_unavailable(monkeypatch) -> None:
    trace = _backward_trace()
    for op in trace.layer_list:
        monkeypatch.setattr(type(op), "site_key", None, raising=False)
    with pytest.raises(InvalidArgumentError) as excinfo:
        _ = trace.grad_fn_site_summary
    assert excinfo.value.fields["code"] == "site_key_unavailable"
