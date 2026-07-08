"""Regression tests for registered-buffer version capture."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.backends.torch import buffer_writes
from torchlens.data_classes.cleanup import _scrub_per_op_equivalence_lists


TensorFactory = Callable[[], torch.Tensor]


def test_removed_buffer_raw_label_is_scrubbed_from_per_op_graph_fields() -> None:
    """Deleted buffer nodes leave no stale raw labels for final label mapping."""

    op = SimpleNamespace(
        parents=["buffer_3_raw", "add_1_raw"],
        root_ancestors={"buffer_3_raw", "input_1_raw"},
        children=["mul_1_raw", "buffer_3_raw"],
        input_ancestors={"input_1_raw"},
        output_descendants={"buffer_3_raw", "output_1_raw"},
        internal_source_parents=["buffer_3_raw"],
        internal_source_ancestors={"buffer_3_raw"},
        conditional_entry_children=["buffer_3_raw"],
        conditional_then_children=["buffer_3_raw", "relu_1_raw"],
        conditional_else_children=[],
        equivalent_ops=["buffer_3_raw", "add_1_raw"],
        recurrent_ops=["buffer_3_raw"],
        parent_arg_positions={
            "args": {0: "buffer_3_raw", 1: "add_1_raw"},
            "kwargs": {"bias": "buffer_3_raw"},
        },
        out_versions_by_child={"buffer_3_raw": torch.ones(1), "add_1_raw": torch.zeros(1)},
        conditional_elif_children={0: ["buffer_3_raw", "add_1_raw"]},
        conditional_arm_children={1: {"then": ["buffer_3_raw"], "else": ["add_1_raw"]}},
    )

    _scrub_per_op_equivalence_lists([op], {"buffer_3_raw"})

    assert op.parents == ["add_1_raw"]
    assert op.root_ancestors == {"input_1_raw"}
    assert op.children == ["mul_1_raw"]
    assert op.output_descendants == {"output_1_raw"}
    assert op.internal_source_parents == []
    assert op.internal_source_ancestors == set()
    assert op.conditional_entry_children == []
    assert op.conditional_then_children == ["relu_1_raw"]
    # NOTE: op_equivalence_classes is a Trace-level dict, never a per-op field;
    # the dead per-op scrub for it was removed by the cert round-1 data-model fix.
    assert op.equivalent_ops == ["add_1_raw"]
    assert op.recurrent_ops == []
    assert op.parent_arg_positions == {"args": {1: "add_1_raw"}, "kwargs": {}}
    assert set(op.out_versions_by_child) == {"add_1_raw"}
    assert op.conditional_elif_children == {0: ["add_1_raw"]}
    assert op.conditional_arm_children == {1: {"then": [], "else": ["add_1_raw"]}}


class RecurrentReassign(nn.Module):
    """Top-level recurrent reassignment model."""

    def __init__(self, steps: int = 4) -> None:
        """Initialize the recurrent buffer."""

        super().__init__()
        self.steps = steps
        self.register_buffer("h", torch.zeros(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run repeated buffer reassignment."""

        for _ in range(self.steps):
            self.h = torch.tanh(self.h + x)
        return self.h + x


class InplaceOps(nn.Module):
    """Explicit in-place mutator model."""

    def __init__(self) -> None:
        """Initialize the buffer."""

        super().__init__()
        self.register_buffer("b", torch.ones(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run in-place ``mul_`` and ``add_`` writes."""

        self.b.mul_(2)
        self.b.add_(x)
        return self.b + x


class CopyWrite(nn.Module):
    """Explicit ``copy_`` write model."""

    def __init__(self) -> None:
        """Initialize the buffer."""

        super().__init__()
        self.register_buffer("b", torch.zeros(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Copy into the registered buffer."""

        self.b.copy_(x)
        return self.b + x


class DataCopyWrite(nn.Module):
    """Explicit ``.data.copy_`` write model."""

    def __init__(self) -> None:
        """Initialize the buffer."""

        super().__init__()
        self.register_buffer("b", torch.zeros(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Copy through the buffer's ``.data`` tensor."""

        self.b.data.copy_(x)
        return self.b + x


class SliceWrite(nn.Module):
    """View/slice write model."""

    def __init__(self) -> None:
        """Initialize the buffer."""

        super().__init__()
        self.register_buffer("b", torch.zeros(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Write into a slice of the buffer."""

        self.b[:2].copy_(x)
        return self.b.sum()


class SetItemWrite(nn.Module):
    """``__setitem__`` write model."""

    def __init__(self) -> None:
        """Initialize the buffer."""

        super().__init__()
        self.register_buffer("b", torch.zeros(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Write into the buffer with item assignment."""

        self.b[:2] = x
        return self.b.sum()


class DirectBuffersWrite(nn.Module):
    """Direct ``_buffers`` reassignment model."""

    def __init__(self) -> None:
        """Initialize the buffer."""

        super().__init__()
        self.register_buffer("b", torch.zeros(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Replace the buffer through ``_buffers``."""

        self._buffers["b"] = x + 1
        return self.b + x


class TwoLoops(nn.Module):
    """Same buffer written in two loops."""

    def __init__(self) -> None:
        """Initialize the buffer."""

        super().__init__()
        self.register_buffer("h", torch.zeros(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Write the same buffer in two separate loops."""

        for _ in range(2):
            self.h = self.h + x
        for _ in range(2):
            self.h = torch.tanh(self.h)
        return self.h + x


class DualRoleInplace(nn.Module):
    """In-place write whose return is also used directly."""

    def __init__(self) -> None:
        """Initialize the buffer."""

        super().__init__()
        self.register_buffer("b", torch.ones(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Use the in-place result and the mutated buffer."""

        y = self.b.add_(x)
        return y * self.b


class StaticReadOnly(nn.Module):
    """Static read-only buffer model."""

    def __init__(self) -> None:
        """Initialize the buffer."""

        super().__init__()
        self.register_buffer("b", torch.arange(2.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Read a static buffer."""

        return self.b + x


class DataSetter(nn.Module):
    """``.data = tensor`` buffer storage reassignment model."""

    def __init__(self) -> None:
        """Initialize the buffer."""

        super().__init__()
        self.register_buffer("b", torch.zeros(2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Replace the buffer storage through the data setter."""

        self.b.data = x + 1
        return self.b


class AliasWrite(nn.Module):
    """Shared/overlapping registered-buffer model."""

    def __init__(self) -> None:
        """Initialize aliased registered buffers."""

        super().__init__()
        base = torch.zeros(4)
        self.register_buffer("b", base)
        self.register_buffer("c", base[:2])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Write through the aliased buffer view."""

        self.c.add_(x)
        return self.b.sum()


class AliasReadThenWholeWrite(nn.Module):
    """Read a buffer view, mutate the whole buffer, then read the view again."""

    def __init__(self) -> None:
        """Initialize overlapping registered buffers."""

        super().__init__()
        base = torch.zeros(4)
        self.register_buffer("whole", base)
        self.register_buffer("halfview", base[:2])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Read the view before and after mutating the full buffer."""

        first = self.halfview.sum()
        self.whole.add_(x)
        second = self.halfview.sum()
        return first + second


def test_storage_aliased_buffer_first_read_keeps_pre_mutation_snapshot() -> None:
    """A buffer-view read before an overlapping write keeps its original payload."""

    trace = tl.trace(AliasReadThenWholeWrite(), torch.ones(4))

    assert torch.equal(trace["buffer_1"].out, torch.zeros(2))
    assert torch.equal(trace["sum_1_1"].out, torch.tensor(0.0))
    assert torch.equal(trace["buffer_4"].out, torch.ones(2))
    assert torch.equal(trace["sum_2_3"].out, torch.tensor(2.0))


class ManyBufferWrites(nn.Module):
    """Model with many registered buffers but writes to only one of them."""

    def __init__(self, num_extra_buffers: int = 48, steps: int = 8, extra_ops: int = 0) -> None:
        """Initialize a repeated-write model with many unrelated buffers.

        Parameters
        ----------
        num_extra_buffers:
            Number of read-free registered buffers used to stress lookup scale.
        steps:
            Number of in-place writes to the target buffer.
        extra_ops:
            Number of additional non-buffer-writing ops appended to the forward.
        """

        super().__init__()
        self.steps = steps
        self.extra_ops = extra_ops
        self.register_buffer("target", torch.zeros(2))
        for index in range(num_extra_buffers):
            self.register_buffer(f"unused_{index}", torch.full((2,), float(index)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Repeatedly mutate the target buffer while passing a non-buffer tensor arg."""

        for _ in range(self.steps):
            self.target.add_(x)
        out = self.target + x
        for _ in range(self.extra_ops):
            out = out + 1.0
        return out


@pytest.mark.parametrize(
    ("model_factory", "input_factory", "expected_overwrites"),
    [
        (lambda: RecurrentReassign(), lambda: torch.ones(2), {"h": 4}),
        (lambda: InplaceOps(), lambda: torch.ones(2), {"b": 2}),
        (lambda: CopyWrite(), lambda: torch.ones(2), {"b": 1}),
        (lambda: DataCopyWrite(), lambda: torch.ones(2), {"b": 1}),
        (lambda: SliceWrite(), lambda: torch.ones(2), {"b": 1}),
        (lambda: SetItemWrite(), lambda: torch.ones(2), {"b": 1}),
        (lambda: DirectBuffersWrite(), lambda: torch.ones(2), {"b": 1}),
        (lambda: TwoLoops(), lambda: torch.ones(2), {"h": 4}),
        (lambda: DualRoleInplace(), lambda: torch.ones(2), {"b": 1}),
        (lambda: StaticReadOnly(), lambda: torch.ones(2), {"b": 0}),
        (lambda: DataSetter(), lambda: torch.ones(2), {"b": 1}),
        (lambda: AliasWrite(), lambda: torch.ones(2), {"c": 1}),
        (
            lambda: nn.BatchNorm1d(3).train(),
            lambda: torch.randn(4, 3),
            {
                "num_batches_tracked": 1,
                "running_mean": 1,
                "running_var": 1,
            },
        ),
        (
            lambda: nn.BatchNorm2d(3).train(),
            lambda: torch.randn(2, 3, 4, 4),
            {
                "num_batches_tracked": 1,
                "running_mean": 1,
                "running_var": 1,
            },
        ),
        (
            lambda: nn.InstanceNorm1d(3, track_running_stats=True).train(),
            lambda: torch.randn(2, 3, 4),
            {"running_mean": 1, "running_var": 1},
        ),
    ],
)
def test_buffer_write_models_validate_and_expose_entities(
    model_factory: Callable[[], nn.Module],
    input_factory: TensorFactory,
    expected_overwrites: dict[str, int],
) -> None:
    """Validate stress models and assert buffer entity metadata."""

    model = model_factory()
    x = input_factory()
    assert tl.validation.validate_forward_pass(
        model_factory(), x.clone(), random_seed=123, validate_metadata=True
    )

    trace = tl.trace(model, x, save_arg_values=True)
    for address, overwrite_count in expected_overwrites.items():
        assert address in trace.buffers
        buffer = trace.buffers[address]
        assert buffer.versions
        assert buffer.final_value is not None
        assert buffer.num_overwrites == overwrite_count
    assert not any(op.is_buffer for op in trace.compute_ops)


def test_batchnorm_buffer_reads_materialize_in_raw_index_order() -> None:
    """Assert deferred BatchNorm buffer reads do not precede earlier raw ops."""

    model = nn.BatchNorm1d(3).train()
    trace = tl.trace(model, torch.randn(4, 3), save_arg_values=True)
    raw_indices = [op.raw_index for op in trace.layer_list]
    add_op = trace.layer_dict_all_keys["add_1_1"]
    buffer_2_op = trace.layer_dict_all_keys["buffer_2"]
    batchnorm_op = trace.layer_dict_all_keys["batchnorm_1_2"]
    buffer_5_op = trace.layer_dict_all_keys["buffer_5"]

    assert raw_indices == sorted(raw_indices)
    assert add_op.raw_index < buffer_2_op.raw_index
    assert trace.layer_list.index(add_op) < trace.layer_list.index(buffer_2_op)
    assert batchnorm_op.raw_index < buffer_5_op.raw_index
    assert trace.layer_list.index(batchnorm_op) < trace.layer_list.index(buffer_5_op)


def test_buffer_write_lookup_avoids_per_op_full_storage_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Storage-key calls scale with buffer WRITES, never with total op count.

    The overlapping-alias refresh deliberately re-derives every registered
    buffer's storage key on each journaled WRITE (a stale-index-safe full scan;
    cert round-1 backends fix), so the write path is allowed to cost
    O(writes x buffers). The regression this test guards against is the
    pre-index pathology where EVERY traced op paid a full storage scan
    (O(ops x buffers)): adding non-buffer-writing ops must cost only a few
    per-op resolution lookups, never a per-op sweep over all buffers.
    """

    call_count = 0
    original_storage_key = buffer_writes.storage_key

    def counted_storage_key(tensor: torch.Tensor) -> tuple[Any, ...] | None:
        """Count storage-key computations while preserving real behavior."""

        nonlocal call_count
        call_count += 1
        return original_storage_key(tensor)

    monkeypatch.setattr(buffer_writes, "storage_key", counted_storage_key)

    num_buffers = 48 + 1
    extra_ops = 16

    trace = tl.trace(ManyBufferWrites(num_extra_buffers=48, steps=8), torch.ones(2))
    base_count = call_count

    call_count = 0
    trace_extra = tl.trace(
        ManyBufferWrites(num_extra_buffers=48, steps=8, extra_ops=extra_ops), torch.ones(2)
    )
    extra_count = call_count

    assert trace.graph_shape_hash is not None
    assert trace_extra.graph_shape_hash is not None
    # A per-op full scan would add ~num_buffers calls for EACH extra op
    # (~16 * 49 = 784); genuine per-op resolution adds only a handful.
    assert extra_count - base_count < extra_ops * (num_buffers // 6)


def _assert_buffer_op_accessors_partition_buffer_ops(trace: tl.Trace) -> None:
    """Assert derived buffer op accessors partition the flat buffer op population."""

    expected_read_ops = [
        op.label for op in trace.layer_list if op.is_buffer and op.buffer_write_kind is None
    ]
    expected_write_ops = [
        op.label for op in trace.layer_list if op.is_buffer and op.buffer_write_kind is not None
    ]
    all_buffer_ops = [op.label for op in trace.layer_list if op.is_buffer]

    assert trace.buffer_read_ops == expected_read_ops
    assert trace.buffer_write_ops == expected_write_ops
    assert trace.num_buffer_read_ops == len(trace.buffer_read_ops)
    assert trace.num_buffer_write_ops == len(trace.buffer_write_ops)
    assert trace.num_buffer_source_ops == len(trace.buffer_read_ops)
    assert trace.num_buffer_sink_ops == len(trace.buffer_write_ops)
    assert set(trace.buffer_read_ops).isdisjoint(trace.buffer_write_ops)
    assert set(trace.buffer_read_ops) | set(trace.buffer_write_ops) == set(all_buffer_ops)
    assert {trace[op_label].layer_label for op_label in all_buffer_ops} == set(trace.buffer_layers)
    assert trace.num_buffer_layers == len(trace.buffer_layers) == len(all_buffer_ops)


def _assert_op_level_buffer_accessors(trace: tl.Trace) -> None:
    """Assert Op-side buffer_source_ops/buffer_sink_ops resolve (LIVE traces only).

    Op-level resolved accessors need a live source Trace, so this is skipped for
    traces loaded from disk (their Ops are intentionally detached).
    """

    for op in trace.compute_ops:
        expected_sources = [
            trace[parent_label].label
            for parent_label in op.parents
            if trace[parent_label].is_buffer and trace[parent_label].buffer_write_kind is None
        ]
        expected_sinks = [
            trace[child_label].label
            for child_label in op.children
            if trace[child_label].is_buffer and trace[child_label].buffer_write_kind is not None
        ]
        # OpAccessor iterates call-index keys (like op.input_ops); .get(key) resolves the Op
        # (int __getitem__ is 0-based list position, not the sparse parent-index key).
        source_ops = op.buffer_source_ops
        sink_ops = op.buffer_sink_ops
        assert [source_ops.get(k).label for k in source_ops] == expected_sources
        assert [sink_ops.get(k).label for k in sink_ops] == expected_sinks


def test_buffer_op_accessors_partition_read_and_write_versions() -> None:
    """Read/write op accessors partition buffer versions, including dual-role buffers."""

    trace = tl.trace(DualRoleInplace(), torch.ones(2), save_arg_values=True)

    _assert_buffer_op_accessors_partition_buffer_ops(trace)
    _assert_op_level_buffer_accessors(trace)
    assert trace.buffer_read_ops
    assert trace.buffer_write_ops

    b_versions = {version.label for version in trace.buffers["b"].versions}
    assert b_versions & set(trace.buffer_read_ops)
    assert b_versions & set(trace.buffer_write_ops)


def test_buffer_op_accessors_round_trip_through_tlspec(tmp_path: Path) -> None:
    """Derived buffer op accessors remain correct after portable ``.tlspec`` load."""

    pytest.importorskip("safetensors")
    trace = tl.trace(DualRoleInplace(), torch.ones(2), save_arg_values=True)
    path = tmp_path / "buffer_ops.tlspec"

    trace.save(path, level="portable")
    loaded = tl.load(path)

    _assert_buffer_op_accessors_partition_buffer_ops(loaded)
    assert loaded.buffer_read_ops == trace.buffer_read_ops
    assert loaded.buffer_write_ops == trace.buffer_write_ops
    assert loaded.num_buffer_read_ops == trace.num_buffer_read_ops
    assert loaded.num_buffer_write_ops == trace.num_buffer_write_ops
    assert loaded.num_buffer_source_ops == trace.num_buffer_source_ops
    assert loaded.num_buffer_sink_ops == trace.num_buffer_sink_ops


def test_reassignment_double_count_is_exact() -> None:
    """Assert N top-level reassignments produce exactly N write events."""

    trace = tl.trace(RecurrentReassign(steps=5), torch.ones(2), save_arg_values=True)
    events = [event for event in trace._buffer_write_events if event.address == "h"]
    assert len(events) == 5
    assert trace.buffers["h"].num_overwrites == 5


class RecurrentCell(nn.Module):
    """RNN-style cell: reassigns a state buffer in a loop around a submodule.

    The inner ``nn.Linear`` makes the loop body recurrent, so loop detection
    engages over the reassigned buffer's version nodes. Regression guard for a
    crash where merging the initial buffer node left a dangling label in the
    output node's ``equivalent_ops`` (``'buffer_1_raw' is not a known raw
    label``) that loop detection then dereferenced mid-pass.
    """

    def __init__(self, dim: int = 8, steps: int = 4) -> None:
        """Initialize the recurrent cell and its hidden-state buffer."""

        super().__init__()
        self.steps = steps
        self.cell = nn.Linear(dim, dim)
        self.register_buffer("h", torch.zeros(1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reset then recurrently reassign the hidden-state buffer."""

        self.h = torch.zeros_like(self.h)
        for _ in range(self.steps):
            self.h = torch.tanh(self.cell(x) + self.h)
        return self.h


def test_recurrent_cell_reassignment_does_not_break_loop_detection() -> None:
    """An RNN cell reassigning its state buffer must trace and validate."""

    model = RecurrentCell()
    x = torch.randn(1, 8)
    assert tl.validation.validate_forward_pass(
        RecurrentCell(), x.clone(), random_seed=7, validate_metadata=True
    )
    trace = tl.trace(model, x, save_arg_values=True)
    assert "h" in trace.buffers
    assert trace.buffers["h"].num_overwrites == 5  # one reset + four loop steps


class GradRecurrent(nn.Module):
    """Recurrent reassignment with learnable params (non-detached hidden state)."""

    def __init__(self, dim: int = 4, steps: int = 3) -> None:
        """Initialize the recurrent cell and its state buffer."""

        super().__init__()
        self.steps = steps
        self.lin = nn.Linear(dim, dim)
        self.register_buffer("h", torch.zeros(1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Reassign the state buffer through the autograd graph each step."""

        self.h = torch.zeros_like(self.h)
        for _ in range(self.steps):
            self.h = torch.tanh(self.lin(x) + self.h)
        return self.h.sum()


class GradBatchNorm(nn.Module):
    """Learnable model whose forward updates fused BatchNorm running stats."""

    def __init__(self, dim: int = 4) -> None:
        """Initialize the linear + BatchNorm stack."""

        super().__init__()
        self.lin = nn.Linear(dim, dim)
        self.bn = nn.BatchNorm1d(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the fused-buffer-writing forward."""

        return self.bn(self.lin(x)).sum()


def _param_grads(model: nn.Module, x: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return a fresh-backward gradient snapshot for every parameter."""

    model.zero_grad(set_to_none=True)
    model(x).backward()
    return {
        name: param.grad.detach().clone()
        for name, param in model.named_parameters()
        if param.grad is not None
    }


@pytest.mark.parametrize("model_factory", [GradRecurrent, GradBatchNorm])
def test_buffer_capture_preserves_gradient_flow(
    model_factory: Callable[[], nn.Module],
) -> None:
    """Capture hooks must be observational: tracing must not break autograd.

    A reassigned state buffer carries ``grad_fn`` exactly like a non-detached
    RNN hidden state; the fused-write snapshot reads (never replaces) the live
    buffer. So gradients through a traced model must match an untraced run.
    """

    import copy

    torch.manual_seed(0)
    reference = model_factory().train()
    traced_model = copy.deepcopy(reference).train()
    x = torch.randn(8, 4)

    expected = _param_grads(reference, x.clone())

    # Tracing performs the fused/reassignment writes; it must leave the live
    # autograd path untouched, so a subsequent backward still matches.
    tl.trace(traced_model, x.clone())
    actual = _param_grads(traced_model, x.clone())

    assert expected.keys() == actual.keys()
    for name in expected:
        assert torch.allclose(expected[name], actual[name], atol=1e-5), name


def test_data_setter_reconciliation_records_buffer_write() -> None:
    """Assert ``.data = tensor`` changes are recorded as buffer writes."""

    trace = tl.trace(DataSetter(), torch.ones(2), save_arg_values=True)
    writes = [trace[label] for label in trace.buffer_write_ops]

    assert len(writes) == 1
    assert writes[0].buffer_write_kind == "data_reassign"
    assert torch.equal(writes[0].out, torch.full((2,), 2.0))
