"""W3 capture-kernel hardening gates (round 26).

Guards the silent capture-emit gap class the W3 audit found on shipped
v2.32.4 -- traces that were wrong or incomplete yet validated True:

- F1: an in-place op through a VIEW never linked the mutation to consumers of
  the BASE tensor (dead-end mutation node, stale parent on the consumer).
- F3: tuple ``out=`` destinations lost their producer parent edges and the
  pre-allocated ``torch.empty`` producer was orphan-pruned (executed op gone).
- F4: tuple pass-through outputs (``broadcast_tensors(x, x)``) minted phantom
  sibling ops, stole the live input's label, and injected a false dependency
  into downstream direct consumers of the input.
- F5: ops whose output is a ``torch.Tensor`` SUBCLASS were never logged
  (whole subclass regions missing).
- F6: the ``nn.Identity`` / pass-through module boundary op dangled;
  downstream consumers bypassed the module's recorded output op.
- F7: the in-place version baseline snapshot was never session-scoped, so a
  tensor surviving across captures kept a stale baseline.
- F9: ``arg_names`` collapsed ``add`` / ``add_`` / ``__add__`` onto one
  stripped key, recording the wrong signature for the in-place and dunder
  spellings.

Every test asserts the RECORDED TRACE (edges, presence, labels), not just
values, and -- where the finding was silent -- that validation passes
legitimately on the corrected capture.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens._user_funcs import validate_forward_pass

pytestmark = pytest.mark.heavy  # validation replays make these multi-second tests


def _ops_by_func(log, func_name):
    # ``op_labels`` yields pass-qualified raw labels (``add_2_3:1``) while
    # ``parents``/``children`` use the layer-label space (``add_2_3``);
    # normalize so edge assertions compare within one space.
    return [lab.split(":")[0] for lab in log.op_labels if log[lab].func_name == func_name]


def _single_op(log, func_name):
    labs = _ops_by_func(log, func_name)
    assert len(labs) == 1, f"expected exactly one {func_name!r} op, got {labs}"
    return labs[0]


def _has_path(log, source, target):
    """Return whether a directed parent->child path connects source to target."""
    frontier = [source]
    seen = set()
    while frontier:
        lab = frontier.pop()
        if lab == target:
            return True
        if lab in seen:
            continue
        seen.add(lab)
        frontier.extend(log[lab].children)
    return False


# ---------------------------------------------------------------------------
# F1: view-mediated in-place mutation must link to consumers of the base
# ---------------------------------------------------------------------------


class ChunkViewInplace(nn.Module):
    def forward(self, x):
        a, b = torch.chunk(x, 2, dim=0)
        a.mul_(2.0)
        return x.sum() + b.sum()


def test_view_inplace_base_consumer_edge_chunk_disjoint_sibling():
    log = tl.trace(ChunkViewInplace(), torch.ones(4, 2))
    mutation = _single_op(log, "mul_")
    sums = _ops_by_func(log, "sum")
    assert len(sums) == 2
    # x overlaps the mutated chunk -> x.sum() binds through the mutation op.
    x_sum_parents = {p for lab in sums for p in log[lab].parents}
    assert mutation in x_sum_parents
    # b is the DISJOINT sibling chunk: its consumer must NOT acquire a false
    # dependency on the mutation op.
    b_sum = [lab for lab in sums if mutation not in log[lab].parents]
    assert len(b_sum) == 1, "disjoint sibling chunk consumer gained a false mutation edge"
    assert any("chunk" in p for p in log[b_sum[0]].parents)
    assert validate_forward_pass(ChunkViewInplace(), torch.ones(4, 2))


def test_out_view_destination_links_base_consumer():
    """out= into a VIEW of a live tensor is the same mutation class as F1."""

    class OutView(nn.Module):
        def forward(self, x):
            y = x + 0.0
            torch.add(x[0], 50.0, out=y[0])
            return y * 1.0

    log = tl.trace(OutView(), torch.zeros(2, 3))
    out_op = _single_op(log, "add")
    consumer = _single_op(log, "__mul__")
    assert out_op in log[consumer].parents
    assert validate_forward_pass(OutView(), torch.zeros(2, 3))


# ---------------------------------------------------------------------------
# F3: tuple out= destinations keep their producer edges
# ---------------------------------------------------------------------------


class SortTupleOut(nn.Module):
    def forward(self, x):
        v = torch.empty_like(x)
        i = torch.empty(x.shape, dtype=torch.long)
        torch.sort(x, dim=0, out=(v, i))
        return v * 1.0


def test_out_tuple_destination_edges_sort():
    log = tl.trace(SortTupleOut(), torch.randn(4, 2))
    empty_like = _single_op(log, "empty_like")
    # The plain torch.empty producer is an EXECUTED op and must not vanish.
    empty = _single_op(log, "empty")
    sort_ops = _ops_by_func(log, "sort")
    assert len(sort_ops) == 2
    for sort_op in sort_ops:
        parents = log[sort_op].parents
        assert empty_like in parents, f"{sort_op} lost its values-destination producer edge"
        assert empty in parents, f"{sort_op} lost its indices-destination producer edge"
    # Downstream consumer binds to the sort values output.
    consumer = _single_op(log, "__mul__")
    assert any(p in sort_ops for p in log[consumer].parents)


def test_out_tuple_destination_sort_validates():
    """F3 completeness pin: the tuple ``out=`` capture validates end to end.

    The structural edge assertions above pin the recorded graph; this pins the
    full forward-replay verdict, so a regression that re-drops the tuple
    destination edges (or re-orphan-prunes the executed ``torch.empty``
    producer) fails the public validation contract, not just edge inspection.
    """

    assert tl.validate(SortTupleOut(), torch.randn(4, 2), scope="forward") is True


# ---------------------------------------------------------------------------
# F4: tuple pass-through outputs must not steal the input's label
# ---------------------------------------------------------------------------


class BroadcastUnused(nn.Module):
    def forward(self, x):
        torch.broadcast_tensors(x, x)  # pure pass-through, result unused
        return x * 1.0


def test_tuple_passthrough_no_label_steal_or_false_dep():
    log = tl.trace(BroadcastUnused(), torch.randn(2, 3))
    bt_ops = _ops_by_func(log, "broadcast_tensors")
    # Both executed pass-through entries are still recorded (honest accounting)...
    assert len(bt_ops) == 2
    for bt_op in bt_ops:
        assert log[bt_op].parents == ("input_1",)
    # ...but the downstream direct consumer of x keeps its true parent: no
    # false dependency through an op whose result the user never used.
    consumer = _single_op(log, "__mul__")
    assert log[consumer].parents == ("input_1",)
    assert validate_forward_pass(BroadcastUnused(), torch.randn(2, 3))


# ---------------------------------------------------------------------------
# F5: Tensor-subclass outputs are logged
# ---------------------------------------------------------------------------


class _TracedSubclass(torch.Tensor):
    pass


class SubclassRegion(nn.Module):
    def forward(self, x):
        y = x.as_subclass(_TracedSubclass)
        z = y + 1.0
        return (z * 2.0).as_subclass(torch.Tensor)


def test_subclass_output_capture():
    log = tl.trace(SubclassRegion(), torch.zeros(3))
    add_op = _single_op(log, "add")
    mul_op = _single_op(log, "mul")
    # The subclass region is fully present and fully connected (torch's
    # __torch_function__ machinery may interpose as_subclass re-wrap hops,
    # so assert path connectivity, not direct parenthood).
    first_cast = _ops_by_func(log, "as_subclass")[0]
    assert log[first_cast].parents == ("input_1",)
    assert _has_path(log, first_cast, add_op)
    assert _has_path(log, add_op, mul_op)
    # Nothing in the trace is provenance-broken.
    non_input_ops = [lab for lab in log.op_labels if log[lab].func_name != "none"]
    for lab in non_input_ops:
        assert log[lab].parents, f"{lab} lost input provenance"
    assert torch.equal(log[mul_op].out, torch.full((3,), 2.0))
    assert validate_forward_pass(SubclassRegion(), torch.zeros(3))


def test_parameter_outputs_stay_excluded():
    """The nn.Parameter carve-out of the emit gate must survive the isinstance fix."""

    from torchlens.backends.torch._ops_exhaustive import _output_should_be_logged

    assert not _output_should_be_logged(nn.Parameter(torch.zeros(2)), True)
    assert _output_should_be_logged(torch.zeros(2), True)
    assert _output_should_be_logged(torch.zeros(2).as_subclass(_TracedSubclass), True)


# ---------------------------------------------------------------------------
# F6: pass-through module boundary op links to downstream consumers
# ---------------------------------------------------------------------------


class IdentityModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.idn = nn.Identity()
        self.dp = nn.Dropout(0.5)

    def forward(self, x):
        y = self.dp(x)
        z = self.idn(y)
        return z * 1.0


# ---------------------------------------------------------------------------
# F7: in-place version baselines are session-scoped
# ---------------------------------------------------------------------------


def test_label_version_snapshot_session_scoped():
    from torchlens._utils._tensor_label import begin_label_session, end_label_session
    from torchlens.backends.torch._ops_capture_records import (
        _label_version_baseline,
        _record_label_version_snapshot,
    )

    t = torch.zeros(3)
    try:
        begin_label_session()
        _record_label_version_snapshot(t)
        assert _label_version_baseline(t) == t._version
        end_label_session()

        # Mutated BETWEEN captures: the stale entry must be inert next session,
        # or a non-mutating identity return would be misclassified as in-place.
        t.add_(1.0)
        begin_label_session()
        assert _label_version_baseline(t) is None
        # Re-recording under the new session serves the fresh baseline.
        _record_label_version_snapshot(t)
        assert _label_version_baseline(t) == t._version
    finally:
        end_label_session()


# ---------------------------------------------------------------------------
# F9: arg_names respect the exact registered spelling
# ---------------------------------------------------------------------------
