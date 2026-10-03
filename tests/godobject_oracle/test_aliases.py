"""aliases-v1: the identity and mutation-effect matrix for public records.

Each test is one row of the converged-design alias matrix
(docs/reference/trace_core_design.md section 6). These are direct behavioral
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


@pytest.mark.smoke
def test_repeated_lookup_identity() -> None:
    """Row 1: repeated lookups return the SAME object, across every accessor."""

    trace = _capture_cnn()
    for label, op in trace.ops.items():
        assert trace.ops[label] is op
        assert trace[label] is op
    for label in trace.layers.keys():
        assert trace.layers[label] is trace.layers[label]
    for label in trace.modules.keys():
        assert trace.modules[label] is trace.modules[label]


@pytest.mark.heavy
def test_record_lifetime_pinned_by_trace() -> None:
    """Row 1b: records stay alive while the Trace lives, without user refs."""

    trace = _capture_cnn()
    label = trace.ops.keys()[0]

    # Frozen storage contract: Op is slots-only with no __weakref__ today, so
    # user code cannot weak-reference it. A facade Op must keep this refusal
    # (or its relaxation is a declared public change, not an accident).
    with pytest.raises(TypeError):
        weakref.ref(trace.ops[label])

    layer_label = trace.layers.keys()[0]
    layer_ref = weakref.ref(trace.layers[layer_label])
    gc.collect()
    assert layer_ref() is not None, "record died while its Trace is alive"
    assert trace.layers[layer_label] is layer_ref()

    del trace
    gc.collect()
    assert layer_ref() is None, "record outlived its collected Trace (leak)"


@pytest.mark.smoke
def test_group_fields_return_live_immutable_views() -> None:
    """Row 2 (FORK-1, decided 2026-08-12): live immutable group views.

    ``equivalent_ops``/``recurrent_ops`` reads resolve to their group's ONE
    cached immutable view — identity-stable, O(1), and alias-safe because
    the view cannot be mutated. This natively replaces the historical
    fresh-mutable-copy-per-read barrier.
    """

    trace = _capture_recurrent()
    grouped = [op for op in trace.ops.values() if op.equivalent_ops]
    assert grouped, "recurrent capture produced no equivalence groups"
    op = grouped[0]

    first_read = op.equivalent_ops
    second_read = op.equivalent_ops
    assert first_read is second_read, "group view reads must be identity-stable"
    assert isinstance(first_read, frozenset)
    with pytest.raises(AttributeError):
        first_read.add("__aliases_v1_sentinel__")

    recurrent = [op for op in trace.ops.values() if op.recurrent_ops]
    assert recurrent, "recurrent capture produced no recurrence groups"
    rec_op = recurrent[0]
    rec_first = rec_op.recurrent_ops
    assert rec_first is rec_op.recurrent_ops
    assert isinstance(rec_first, tuple)
    with pytest.raises(AttributeError):
        rec_first.append("__aliases_v1_sentinel__")


@pytest.mark.smoke
def test_relation_reads_are_immutable_views() -> None:
    """Row 3 (FORK-1, decided 2026-08-12): relation reads are views.

    Finished-trace relation accessors return IMMUTABLE views — ``tuple`` for
    label sequences, ``frozenset`` for label sets — with identity-stable
    repeated reads. In-place mutation raises instead of sticking. This
    supersedes the historical hand-back-the-stored-list contract; the break
    is authorized and documented in trace_core_design.md section 3.2.
    """

    trace = _capture_cnn()
    op = next(op for op in trace.ops.values() if op.parents and op.children)

    children_read = op.children
    assert isinstance(children_read, tuple)
    assert children_read is op.children, "relation view reads must be identity-stable"
    with pytest.raises(AttributeError):
        children_read.append("__aliases_v1_child__")
    assert isinstance(op.parents, tuple)
    assert isinstance(op.input_ancestors, frozenset)
    assert isinstance(op.output_descendants, frozenset)
    assert isinstance(op.modules, tuple)
    with pytest.raises(AttributeError):
        op.input_ancestors.add("__aliases_v1_member__")

    # Direct ASSIGNMENT still works (the storage write path is unchanged) and
    # normalizes a raw builtin to the declared view type on a finished trace.
    original = op.children
    op.children = list(original) + ["__aliases_v1_assigned__"]
    assert isinstance(op.children, tuple)
    assert "__aliases_v1_assigned__" in op.children
    op.children = original
    assert op.children == original


@pytest.mark.smoke
def test_plain_mutable_containers_are_observably_mutable() -> None:
    """Row 3b: NON-relation container fields keep the stored-container contract.

    Dict-shaped metadata (``parent_arg_positions`` and friends) was excluded
    from the immutable-view decision: reads hand back the stored dict and
    mutation through a read IS visible on the next read.
    """

    trace = _capture_cnn()
    op = next(op for op in trace.ops.values() if op.parent_arg_positions)
    read = op.parent_arg_positions
    assert read is op.parent_arg_positions
    read["__aliases_v1_domain__"] = {}
    assert "__aliases_v1_domain__" in op.parent_arg_positions
    del op.parent_arg_positions["__aliases_v1_domain__"]
    assert "__aliases_v1_domain__" not in op.parent_arg_positions


@pytest.mark.smoke
def test_multi_output_siblings_share_call_facts_not_identity() -> None:
    """Row 4: multi-output siblings agree on call facts, stay distinct rows."""

    trace = _capture_multi_output()
    split_ops = [op for op in trace.ops.values() if op.func_name == "split"]
    assert len(split_ops) >= 2, "split did not produce sibling output ops"
    first, second = split_ops[0], split_ops[1]
    assert first is not second
    assert first.func_call_id == second.func_call_id
    assert first.args_template == second.args_template
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


@pytest.mark.smoke
def test_payload_identity_and_distinctness() -> None:
    """Row 5: payload reads are identity-stable; equal values stay distinct."""

    trace = _capture_cnn()
    ops = list(trace.ops.values())
    saved = [op for op in ops if isinstance(op.out, torch.Tensor)]
    assert saved, "default capture saved no activations"
    op = saved[0]
    assert op.out is op.out

    torch.manual_seed(_SEED)
    other_trace = _capture_cnn()
    other_op = other_trace.ops[op.layer_label]
    assert isinstance(other_op.out, torch.Tensor)
    assert torch.equal(op.out, other_op.out)
    assert op.out is not other_op.out


@pytest.mark.smoke
def test_op_copy_selective_depth() -> None:
    """Row 6: Op.copy() honors the documented share/deep split, field by field."""

    trace = _capture_cnn()
    op = next(op for op in trace.ops.values() if isinstance(op.out, torch.Tensor))
    clone = op.copy()
    assert clone is not op
    assert type(clone) is type(op)

    # Documented shared-by-reference fields.
    assert clone.func is op.func
    assert clone.out is op.out
    if op.saved_args is not None:
        assert clone.saved_args is op.saved_args
    if op.container_spec is not None:
        assert clone.container_spec is op.container_spec
    if op.parent_params:
        assert clone.parent_params is op.parent_params

    # Relation metadata: equal content; immutable views MAY be shared across
    # records (FORK-1), so no distinct-identity requirement anymore.
    assert clone.children == op.children
    assert isinstance(clone.children, tuple)
    # Rebinding the clone's cell never reaches the source op.
    clone.children = tuple(op.children) + ("__aliases_v1_copy__",)
    assert "__aliases_v1_copy__" not in op.children


@pytest.mark.smoke
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


@pytest.mark.smoke
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


@pytest.mark.smoke
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


@pytest.mark.smoke
def test_equivalence_group_shared_live_view() -> None:
    """Row 10 (FORK-1): one shared LIVE view per group, mutation impossible.

    Every member of an equivalence group reads THE same immutable view
    object (O(1), no per-read copies), and the one sanctioned group
    mutation — removal scrub — rebinds the group row once so every member
    reflects it. Caller mutation is impossible by construction.
    """

    trace = _capture_recurrent()
    grouped = [op for op in trace.ops.values() if op.equivalent_ops]
    by_group: dict[frozenset, list] = {}
    for op in grouped:
        by_group.setdefault(frozenset(op.equivalent_ops), []).append(op)
    multi = [members for members in by_group.values() if len(members) >= 2]
    assert multi, "no multi-member equivalence group captured"
    first, second = multi[0][0], multi[0][1]
    assert first.equivalent_ops == second.equivalent_ops
    assert first.equivalent_ops is second.equivalent_ops, "group members must share ONE cached view"
    with pytest.raises(AttributeError):
        first.equivalent_ops.add("__aliases_v1_group__")

    # LIVE through the sanctioned scrub path: filtering the group once is
    # observed by every member on its next read, sharing intact.
    from torchlens.data_classes.cleanup import _scrub_per_op_equivalence_lists

    group_labels = sorted(first.equivalent_ops)
    removed = group_labels[-1]
    _scrub_per_op_equivalence_lists(multi[0], {removed})
    assert removed not in first.equivalent_ops
    assert removed not in second.equivalent_ops
    assert first.equivalent_ops is second.equivalent_ops
