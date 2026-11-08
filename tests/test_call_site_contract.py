"""The call-site grouping contract (``callsite_v2``) is a live tripwire.

Grouping rule section 3 keeps the call-site key internal, so ``equivalence_class``
keeps its argument/parameter fingerprint and step 7 groups by a raw-graph
workspace side table. Two locked loop-detection invariants that encoded the old
grouper's rule were replaced for call-site grouped traces:

- old: layers sharing (func_name, param barcodes, equivalence_class) are one layer;
  new: two different layers never share a call-site key (contract (ii));
- old: every pass of a layer has one equivalence_class; new: the same except for
  the components the rule says are not evidence (argument/shape hash rows 49 and
  50, parameter identity row 51, module path rows 30 and 31).

These tests prove the replacement still fires: planted bugs that merge two call
sites, split one, or leave a recurrent group spanning two must fail, step 7 at
capture and the metadata invariants afterwards; a loaded trace (no side table)
reports the contract as not checked rather than passing it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._postprocess import _call_site_contract as contract, loop_detection
from torchlens.errors import MetadataInvariantError
from torchlens.validation import check_metadata_invariants
from torchlens.validation._diagnostics import get_validation_diagnostics


class _TwoReluLines(nn.Module):
    """Row 14: the same functional op on two lines at root is two layers."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(x - 0.1)
        x = torch.relu(x - 0.2)
        return x


class _TanhLoop(nn.Module):
    """Row 3: a one-op loop at root is one layer with 3 passes."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(3):
            x = torch.tanh(x)
        return x


def _x() -> torch.Tensor:
    return torch.randn(2, 4)


def _invariant_error_in_chain(exc: BaseException) -> MetadataInvariantError | None:
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        if isinstance(current, MetadataInvariantError):
            return current
        seen.add(id(current))
        current = current.__cause__ or current.__context__
    return None


def _capture_must_breach(model: nn.Module, clause: str) -> None:
    with pytest.raises(Exception) as excinfo:  # noqa: PT011 (the chain is inspected below)
        tl.trace(model, _x())
    error = _invariant_error_in_chain(excinfo.value)
    assert error is not None, f"expected a call-site contract breach, got {excinfo.value!r}"
    assert f"({clause})" in str(error), str(error)


# Kept by shrink-r4 (per-id subset rule): sole coverage of _postprocess/_call_site_contract.py.
def test_planted_merge_of_two_call_sites_fails_step7(monkeypatch: pytest.MonkeyPatch) -> None:
    """A grouping key that fuses the two relu lines into one layer breaches (i)."""

    original = loop_detection._call_site_group_key

    def _merging_key(node, classes):
        site = classes.get(node._label_raw, "")
        if site.startswith("relu"):
            return ("call_site", "planted-merge")
        return original(node, classes)

    monkeypatch.setattr(loop_detection, "_call_site_group_key", _merging_key)
    _capture_must_breach(_TwoReluLines(), "i")


# Kept by shrink-r4 (per-id subset rule): sole coverage of _postprocess/_call_site_contract.py.
def test_planted_recurrent_group_spanning_two_sites_fails_step7(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pass rebuild that joins the two relu layers' groups breaches (iii)."""

    original = loop_detection._rebuild_pass_assignments

    def _joining_rebuild(trace):
        original(trace)
        relus = [entry for entry in trace if entry.func_name == "relu"]
        joined = sorted(entry._label_raw for entry in relus)
        for entry in relus:
            entry._graph_recurrent_ops = joined

    monkeypatch.setattr(loop_detection, "_rebuild_pass_assignments", _joining_rebuild)
    _capture_must_breach(_TwoReluLines(), "iii")


# Kept by shrink-r4 (per-id subset rule): sole coverage of validation/_invariants_equivalence.py, campaign mutant kills.
def test_equivalence_clause_still_fires_on_a_layer_type_mismatch() -> None:
    """The narrowed equivalence clause still refuses a pass of a different layer type."""

    torch.manual_seed(0)
    trace = tl.trace(_TanhLoop(), _x())
    passes = [op for op in trace._layer_list if op.func_name == "tanh"]
    assert len(passes) == 3
    passes[1]._internal_set("equivalence_class", "sigmoid_no_args")
    with pytest.raises(MetadataInvariantError, match="type mismatch"):
        check_metadata_invariants(trace)


# Kept by shrink-r4 (per-id subset rule): sole coverage of campaign mutant kills.
def test_loaded_trace_reports_the_contract_unchecked(tmp_path: Path) -> None:
    """A loaded call-site grouped trace has no side table: re-validation says so, never passes it."""

    torch.manual_seed(0)
    trace = tl.trace(_TanhLoop(), _x())
    spec = tmp_path / "loop.tlspec"
    trace.save(str(spec))
    loaded = tl.load(str(spec))
    assert loaded.options.grouping_policy["detector"] == "callsite_v2"
    assert contract.call_site_classes_for(loaded) is None
    check_metadata_invariants(loaded)
    unchecked = [
        d
        for d in get_validation_diagnostics(loaded)
        if d.check_name == "call_site_contract_unchecked"
    ]
    assert len(unchecked) == 1
    assert "not checked" in unchecked[0].message
