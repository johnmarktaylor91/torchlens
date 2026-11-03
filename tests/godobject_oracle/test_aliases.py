"""aliases-v1: the identity and mutation-effect matrix for public records.

Each test is one row of the converged-design alias matrix
(docs/internal/reference/trace_core_design.md section 6). These are direct behavioral
assertions rather than goldens because identity relations (``is``, weakref
liveness, mutation visibility) cannot be value-snapshotted.
"""

from __future__ import annotations

import gc
import warnings
import weakref

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._errors import RecordBindingError
from torchlens._trace_state import TraceState
from torchlens.intervention.errors import DirectActivationWriteWarning

_SEED = 20260812


class _AliasCNN(nn.Module):
    """Small deterministic conv model for alias assertions."""

    def __init__(self) -> None:
        """Initialize conv and head layers."""

        super().__init__()
        self.conv = nn.Conv2d(1, 2, kernel_size=3, padding=1)
        self.head = nn.Linear(2, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run conv-relu-pool-linear."""

        y = torch.relu(self.conv(x))
        return self.head(y.mean(dim=(2, 3)))


class _RecurrentCell(nn.Module):
    """Three equivalent tanh(linear) steps producing a recurrence group."""

    def __init__(self) -> None:
        """Initialize the shared cell."""

        super().__init__()
        self.cell = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the shared cell three times."""

        state = x
        for _ in range(3):
            state = torch.tanh(self.cell(state))
        return state


class _MultiOutput(nn.Module):
    """Model whose central op returns multiple sibling outputs."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Split into two chunks and recombine."""

        first, second = torch.split(x, 2, dim=1)
        return torch.cat([torch.relu(first), torch.sigmoid(second)], dim=1)


def _capture_cnn() -> tl.Trace:
    """Capture the deterministic CNN trace."""

    torch.manual_seed(_SEED)
    return tl.trace(_AliasCNN(), torch.linspace(-1.0, 1.0, 16).reshape(1, 1, 4, 4))


def _capture_recurrent() -> tl.Trace:
    """Capture the recurrence-group trace."""

    torch.manual_seed(_SEED)
    return tl.trace(_RecurrentCell(), torch.linspace(-1.0, 1.0, 4).reshape(1, 4))


def _capture_multi_output() -> tl.Trace:
    """Capture the multi-output-op trace."""

    torch.manual_seed(_SEED)
    return tl.trace(_MultiOutput(), torch.linspace(-1.0, 1.0, 4).reshape(1, 4))


def test_multi_output_siblings_share_call_facts_not_identity() -> None:
    """Row 4: multi-output siblings agree on call facts, stay distinct rows."""

    trace = _capture_multi_output()
    split_ops = [op for op in trace.ops.values() if op.func_name == "split"]
    assert len(split_ops) >= 2, "split did not produce sibling output ops"
    first, second = split_ops[0], split_ops[1]
    assert first is not second
    assert first.call.call_id == second.call.call_id
    assert first.call.args_template == second.call.args_template
    assert first.multi_output_index != second.multi_output_index
    assert first.layer_label != second.layer_label

    # Relation views are immutable (FORK-1), so sibling cross-talk through
    # in-place mutation is impossible by construction. Assignment stays
    # per-record: rebinding one sibling's cell never reaches the other.
    original_first, original_second = first.children, second.children
    first.children = tuple(original_first) + ("__aliases_v1_sibling__",)
    try:
        assert "__aliases_v1_sibling__" not in second.children
        assert second.children == original_second
    finally:
        first.children = original_first


def test_fork_isolation() -> None:
    """Row 7: forked traces are fully isolated record graphs."""

    trace = _capture_cnn()
    fork = trace.fork()
    assert fork is not trace
    for label, op in trace.ops.items():
        assert fork.ops[label] is not op

    label = trace.ops.keys()[0]
    original = trace.ops[label].children
    fork.ops[label].children = tuple(original) + ("__aliases_v1_fork__",)
    assert "__aliases_v1_fork__" not in trace.ops[label].children
    trace.ops[label].children = tuple(original) + ("__aliases_v1_parent__",)
    assert "__aliases_v1_parent__" not in fork.ops[label].children


def test_direct_write_warning_and_dirty_transition() -> None:
    """Row 8: guarded direct writes warn ONCE and mark the trace dirty.

    This exact contract (warn, never raise; DIRECT_WRITE_DIRTY state; one
    warning per trace) must be preserved byte-identically through the overlay
    path. Hard-freezing finished records would convert this warning into an
    exception and MUST fail here.
    """

    trace = _capture_cnn()
    op = next(op for op in trace.ops.values() if isinstance(op.out, torch.Tensor))
    assert trace.state is not TraceState.DIRECT_WRITE_DIRTY

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        op.out = torch.zeros_like(op.out)
    direct = [w for w in caught if issubclass(w.category, DirectActivationWriteWarning)]
    assert len(direct) == 1
    assert trace.state is TraceState.DIRECT_WRITE_DIRTY

    other = next(o for o in trace.ops.values() if o is not op and isinstance(o.out, torch.Tensor))
    with warnings.catch_warnings(record=True) as caught_again:
        warnings.simplefilter("always")
        other.out = torch.zeros_like(other.out)
    repeat = [w for w in caught_again if issubclass(w.category, DirectActivationWriteWarning)]
    assert repeat == []


def test_source_trace_weakref_lifetime() -> None:
    """Row 9: op->trace is weak; a collected Trace is not resurrected.

    The read after collection refuses typed (r5 b7-opus R52-B: never a bare
    ``None`` behind the ``-> Trace`` signature) rather than returning ``None``.
    """

    trace = _capture_cnn()
    op = next(iter(trace.ops.values()))
    assert op.source_trace is trace

    trace_ref = weakref.ref(trace)
    del trace
    gc.collect()
    assert trace_ref() is None
    with pytest.raises(RecordBindingError) as exc_info:
        _ = op.source_trace
    assert exc_info.value.fields["code"] == "trace_reference_collected"


def test_equivalence_group_shared_live_view() -> None:
    """Row 10 (FORK-1): one shared LIVE view per group, mutation impossible.

    Every member of an equivalence group reads THE same immutable view
    object (O(1), no per-read copies), and the one sanctioned group
    mutation — removal scrub — rebinds the group row once so every member
    reflects it. Caller mutation is impossible by construction.
    """

    trace = _capture_recurrent()
    grouped = [op for op in trace.ops.values() if op.graph.equivalent_ops]
    by_group: dict[frozenset, list] = {}
    for op in grouped:
        by_group.setdefault(frozenset(op.graph.equivalent_ops), []).append(op)
    multi = [members for members in by_group.values() if len(members) >= 2]
    assert multi, "no multi-member equivalence group captured"
    first, second = multi[0][0], multi[0][1]
    assert first.graph.equivalent_ops == second.graph.equivalent_ops
    assert first.graph.equivalent_ops is second.graph.equivalent_ops, (
        "group members must share ONE cached view"
    )
    with pytest.raises(AttributeError):
        first.graph.equivalent_ops.add("__aliases_v1_group__")

    # LIVE through the sanctioned scrub path: filtering the group once is
    # observed by every member on its next read, sharing intact.
    from torchlens._data_classes.cleanup import _scrub_per_op_equivalence_lists

    group_labels = sorted(first.graph.equivalent_ops)
    removed = group_labels[-1]
    _scrub_per_op_equivalence_lists(multi[0], {removed})
    assert removed not in first.graph.equivalent_ops
    assert removed not in second.graph.equivalent_ops
    assert first.graph.equivalent_ops is second.graph.equivalent_ops
