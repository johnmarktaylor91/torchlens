"""The call-site grouping contract (``callsite_v1``) is a live tripwire.

Grouping rule section 3 keeps the call-site key internal, so ``equivalence_class``
keeps its argument/parameter fingerprint and step 7 groups by a raw-graph
workspace side table. Two locked loop-detection invariants that encoded the old
grouper's rule were replaced for ``callsite_v1`` traces:

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
from torchlens.validation import MetadataInvariantError, check_metadata_invariants
from torchlens.validation.diagnostics import get_validation_diagnostics


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


class _ScalarLoop(nn.Module):
    """Row 49: a varying scalar in a loop is one layer; its argument hash varies."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(1, 4):
            x = torch.tanh(x * float(i))
        return x


class _SharedWeightTwoLines(nn.Module):
    """Row 25: one raw parameter on two lines is two single-pass matmul layers."""

    def __init__(self) -> None:
        super().__init__()
        self.w = nn.Parameter(torch.randn(4, 4) * 0.1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x @ self.w
        x = torch.tanh(x)
        x = x @ self.w
        return x


class _ParameterListLoop(nn.Module):
    """Row 51: a loop over a ParameterList is one matmul layer, one param per pass."""

    def __init__(self) -> None:
        super().__init__()
        self.ws = nn.ParameterList([nn.Parameter(torch.randn(4, 4) * 0.1) for _ in range(3)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for w in self.ws:
            x = torch.tanh(x @ w)
        return x


class _Inner(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(x) * 2.0


class _Parent(nn.Module):
    def __init__(self, inner: nn.Module) -> None:
        super().__init__()
        self.inner = inner

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.inner(x) + 1.0


class _SharedInstance(nn.Module):
    """Row 30: one module instance under two parents is one layer, 2 passes."""

    def __init__(self) -> None:
        super().__init__()
        inner = _Inner()
        self.a = _Parent(inner)
        self.b = _Parent(inner)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.b(self.a(x))


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


def test_planted_split_of_one_call_site_fails_step7(monkeypatch: pytest.MonkeyPatch) -> None:
    """A grouping key that splits the tanh loop line into three layers breaches (ii)."""

    original = loop_detection._call_site_group_key

    def _splitting_key(node, classes):
        if classes.get(node._label_raw, "").startswith("tanh"):
            return ("planted-split", node._label_raw)
        return original(node, classes)

    monkeypatch.setattr(loop_detection, "_call_site_group_key", _splitting_key)
    _capture_must_breach(_TanhLoop(), "ii")


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
            entry.recurrent_ops = joined

    monkeypatch.setattr(loop_detection, "_rebuild_pass_assignments", _joining_rebuild)
    _capture_must_breach(_TwoReluLines(), "iii")


@pytest.mark.parametrize(
    "model_cls",
    [
        _TwoReluLines,
        _TanhLoop,
        _ScalarLoop,
        _SharedWeightTwoLines,
        _ParameterListLoop,
        _SharedInstance,
    ],
)
def test_contract_and_narrowed_invariants_hold_on_rule_rows(model_cls: type[nn.Module]) -> None:
    """Rows 3, 14, 25, 30, 49 and 51 pass every metadata invariant, re-checked live."""

    torch.manual_seed(0)
    trace = tl.trace(model_cls(), _x())
    assert trace.grouping_policy["detector"] == "callsite_v1"
    assert contract.call_site_classes_for(trace) is not None
    assert check_metadata_invariants(trace) is True
    assert not [
        d for d in get_validation_diagnostics(trace) if d.check == "call_site_contract_unchecked"
    ]


def test_row_shapes_are_what_the_replaced_invariants_refused() -> None:
    """The narrowed components really differ on these rows (the replacement is load-bearing)."""

    torch.manual_seed(0)
    scalar = tl.trace(_ScalarLoop(), _x())
    muls = [op for op in scalar.layer_list if op.layer_type == "mul"]
    assert len({op.layer_label for op in muls}) == 1
    assert len({op.equivalence_class for op in muls}) == 3  # row 49: argument hash
    shared = tl.trace(_SharedWeightTwoLines(), _x())
    matmuls = [op for op in shared.layer_list if op.layer_type == "matmul"]
    assert len({op.layer_label for op in matmuls}) == 2  # row 25: shared W, two layers
    assert len({op.equivalence_class for op in matmuls}) == 1


def test_post_hoc_recheck_fires_on_a_corrupted_side_table() -> None:
    """Metadata invariants re-check (ii) from the live side table."""

    torch.manual_seed(0)
    trace = tl.trace(_TwoReluLines(), _x())
    classes = dict(contract.call_site_classes_for(trace) or {})
    relu_raw = [op._label_raw for op in trace.layer_list if op.func_name == "relu"]
    assert len(relu_raw) == 2
    for raw in relu_raw:
        classes[raw] = "relu@cs-planted"
    contract.remember_call_site_classes(trace, classes)
    with pytest.raises(MetadataInvariantError, match=r"\(ii\)"):
        check_metadata_invariants(trace)


def test_equivalence_clause_still_fires_on_a_layer_type_mismatch() -> None:
    """The narrowed equivalence clause still refuses a pass of a different layer type."""

    torch.manual_seed(0)
    trace = tl.trace(_TanhLoop(), _x())
    passes = [op for op in trace.layer_list if op.func_name == "tanh"]
    assert len(passes) == 3
    passes[1]._internal_set("equivalence_class", "sigmoid_no_args")
    with pytest.raises(MetadataInvariantError, match="type mismatch"):
        check_metadata_invariants(trace)


def test_loaded_trace_reports_the_contract_unchecked(tmp_path: Path) -> None:
    """A loaded callsite_v1 trace has no side table: re-validation says so, never passes it."""

    torch.manual_seed(0)
    trace = tl.trace(_TanhLoop(), _x())
    spec = tmp_path / "loop.tlspec"
    tl.save(trace, str(spec))
    loaded = tl.load(str(spec))
    assert loaded.grouping_policy["detector"] == "callsite_v1"
    assert contract.call_site_classes_for(loaded) is None
    check_metadata_invariants(loaded)
    unchecked = [
        d for d in get_validation_diagnostics(loaded) if d.check == "call_site_contract_unchecked"
    ]
    assert len(unchecked) == 1
    assert "not checked" in unchecked[0].message
