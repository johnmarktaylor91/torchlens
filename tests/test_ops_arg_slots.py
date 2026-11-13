"""The per-op argument slot snapshot gives the four live derivations their op-time values.

Parents, ``parent_arg_positions``, the unattributed / dropped-edge witness and the
argument templates are read from one slot snapshot per op instead of four walks over
the arguments. Each subject is traced twice: once with the snapshot disabled (every
derivation falls back to its per-argument live lookups, the former code path) and once
with it; the recorded values must be equal op for op. The subjects cover in-place
relabels, buffer and closure-constant source logging (labels assigned in the middle of
the shared builder), ``out=`` writes, ``.data`` rebinds, nested list arguments,
keyword tensors and mid-forward parameter mutation.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens.backends.torch import _ops_arg_slots, _ops_argument_fields, _ops_shared_fields
from torchlens.intervention._types import CapturedArgTemplate, LiteralTensor, ParentRef

_CLOSURE_CONSTANT = torch.linspace(-1.0, 1.0, 8)


class _InplaceMLP(nn.Module):
    """In-place relu, augmented add and a mul_ after use."""

    def __init__(self) -> None:
        """Build two linear layers."""
        super().__init__()
        self.a = nn.Linear(8, 8)
        self.b = nn.Linear(8, 8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mix in-place and out-of-place ops on shared tensors."""
        h = self.a(x)
        h.relu_()
        y = self.b(h)
        y += h
        h.mul_(2.0)
        return y + h


class _BufferModel(nn.Module):
    """BatchNorm running stats plus a registered buffer updated in place."""

    def __init__(self) -> None:
        """Build a conv, a batch norm and a counter buffer."""
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3)
        self.bn = nn.BatchNorm2d(4)
        self.register_buffer("scale", torch.full((4, 1, 1), 0.5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize, scale by the buffer, then bump the buffer."""
        h = self.bn(self.conv(x))
        out = h * self.scale
        self.scale.add_(1.0)
        return out.sum()


class _ClosureConstantModel(nn.Module):
    """Consumes a module-level constant tensor and an unregistered tensor attribute."""

    def __init__(self) -> None:
        """Hold a plain (unregistered) tensor attribute."""
        super().__init__()
        self.lin = nn.Linear(8, 8)
        self.offset = torch.arange(8.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add both constants around a linear layer."""
        h = self.lin(x + _CLOSURE_CONSTANT)
        return torch.cat([h, self.offset.expand_as(h)], dim=1).sum()


class _OutDataModel(nn.Module):
    """A loop with an ``out=`` write and a ``.data`` rebind."""

    def __init__(self) -> None:
        """Build one linear layer."""
        super().__init__()
        self.lin = nn.Linear(6, 6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Loop, write through out=, rebind .data."""
        h = x
        for _ in range(2):
            h = self.lin(h).tanh()
        dest = torch.empty_like(h)
        torch.add(h.detach(), x, out=dest)
        z = dest * 2
        z.data = h.detach() + 1
        return z + dest


class _NestedArgsModel(nn.Module):
    """Nested list arguments, keyword tensors and a shape list."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Stack, concatenate, reshape and use a keyword tensor."""
        a = x * 2
        b = x + 1
        stacked = torch.stack([a, b, a])
        joined = torch.cat((stacked[0], stacked[1]), dim=0)
        shape = [2, -1]
        flat = joined.reshape(shape)
        return torch.where(flat > 0, flat, other=torch.zeros_like(flat)).sum()


class _ParamMutModel(nn.Module):
    """A parameter multiplied in place mid-forward, then read again."""

    def __init__(self) -> None:
        """Build one linear layer."""
        super().__init__()
        self.lin = nn.Linear(5, 5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Use, mutate, reuse the weight."""
        h = self.lin(x)
        with torch.no_grad():
            self.lin.weight.mul_(0.5)
        return self.lin(h) + torch.nn.functional.linear(h, self.lin.weight)


_SUBJECTS: tuple[tuple[str, type[nn.Module], tuple[int, ...]], ...] = (
    ("inplace_mlp", _InplaceMLP, (2, 8)),
    ("buffer", _BufferModel, (2, 3, 6, 6)),
    ("closure_constant", _ClosureConstantModel, (2, 8)),
    ("loop_out_data", _OutDataModel, (2, 6)),
    ("nested_args", _NestedArgsModel, (3, 4)),
    ("param_mutation", _ParamMutModel, (2, 5)),
)


def _norm_component(component: Any) -> Any:
    """Return a comparable form of one template component (tensors by value)."""

    if isinstance(component, ParentRef):
        layout = component._capture_layout
        return ("parent", component.parent_label, None if layout is None else repr(layout))
    if isinstance(component, LiteralTensor):
        value = component.value
        return (
            "tensor",
            tuple(value.shape),
            str(value.dtype),
            value.detach().flatten().tolist(),
            component.param_barcode is None,
            component._capture_param_fingerprint is None,
        )
    if isinstance(component, tuple):
        return tuple(_norm_component(item) for item in component)
    return repr(component)


def _norm_template(template: Any) -> Any:
    """Return a comparable form of a captured argument template (or ``None``)."""

    if template is None:
        return None
    assert isinstance(template, CapturedArgTemplate)
    return (
        _norm_component(template.args),
        tuple((key, _norm_component(value)) for key, value in template.kwargs),
        template.notes,
        repr(template.func_id),
    )


def _derived_fields(model_cls: type[nn.Module], shape: tuple[int, ...]) -> list[tuple[Any, ...]]:
    """Trace ``model_cls`` (fixed seed) and return each op's four derived fields."""

    torch.manual_seed(0)
    model = model_cls()
    x = torch.randn(*shape)
    trace = tl.trace(model, x)
    return [
        (
            index,
            list(op.parents),
            {kind: dict(op.call.parent_arg_positions[kind]) for kind in ("args", "kwargs")},
            tuple(op.call.unattributed_tensor_args),
            tuple(op.call.dropped_edge_tensor_args),
            _norm_template(op.call.args_template),
            _norm_template(op.call.kwargs_template),
        )
        for index, op in enumerate(trace.ops)
    ]


@pytest.mark.parametrize(("name", "model_cls", "shape"), _SUBJECTS, ids=[s[0] for s in _SUBJECTS])
def test_slot_derivations_equal_per_argument_lookups(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    model_cls: type[nn.Module],
    shape: tuple[int, ...],
) -> None:
    """Every op's parents, positions, witness pair and templates match the lookup path."""

    with_slots = _derived_fields(model_cls, shape)
    with monkeypatch.context() as patch:
        patch.setattr(_ops_argument_fields, "snapshot_arg_slots", lambda *_args: ())
        without_slots = _derived_fields(model_cls, shape)
    assert len(with_slots) == len(without_slots), name
    diffs = [
        (left[0], index)
        for left, right in zip(without_slots, with_slots)
        for index in range(1, 7)
        if left[index] != right[index]
    ]
    assert diffs == [], f"{name}: slot derivations differ (op, field index): {diffs[:6]}"


@pytest.fixture
def label_reads_per_op(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[tuple[int, int]]]:
    """Count ``get_tensor_label`` calls from each op's snapshot to the end of its witness."""

    import sys

    from torchlens.backends.torch import _tl

    seen: list[tuple[int, int]] = []
    window: dict[str, Any] = {"open": False, "reads": 0, "slots": 0}
    real_label = _tl.get_tensor_label

    def counting_label(t: Any) -> Any:
        """Count one label read while a window is open."""
        if window["open"]:
            window["reads"] += 1
        return real_label(t)

    for module_name, module in list(sys.modules.items()):
        if module_name.startswith("torchlens") and module is not None:
            for attr, value in list(vars(module).items()):
                if value is real_label:
                    monkeypatch.setattr(module, attr, counting_label)
    real_snapshot = _ops_argument_fields.snapshot_arg_slots
    real_witness = _ops_shared_fields.witness_from_slots

    def snapshot(*args: Any) -> Any:
        """Open the window, then take the snapshot."""
        window.update(open=True, reads=0)
        slots = real_snapshot(*args)
        window["slots"] = len(slots)
        return slots

    def witness(*args: Any) -> Any:
        """Run the witness, then close the window."""
        try:
            return real_witness(*args)
        finally:
            seen.append((window["reads"], window["slots"]))
            window["open"] = False

    monkeypatch.setattr(_ops_argument_fields, "snapshot_arg_slots", snapshot)
    monkeypatch.setattr(_ops_shared_fields, "witness_from_slots", witness)
    yield seen


@pytest.mark.parametrize(
    ("model_cls", "shape"),
    [(_InplaceMLP, (2, 8)), (_BufferModel, (2, 3, 6, 6)), (_NestedArgsModel, (3, 4))],
)
def test_one_label_read_per_tensor_slot(
    label_reads_per_op: list[tuple[int, int]],
    model_cls: type[nn.Module],
    shape: tuple[int, ...],
) -> None:
    """Between the snapshot and the witness each op reads at most one label per tensor slot."""

    torch.manual_seed(0)
    tl.trace(model_cls(), torch.randn(*shape))
    assert label_reads_per_op, "no op was observed"
    over = [(reads, slots) for reads, slots in label_reads_per_op if reads > slots]
    assert over == [], f"ops reading a label more than once per slot: {over[:6]}"


#: Measured ``get_tensor_label`` calls per trace (fixed seed). Before the slot snapshot the
#: same traces read 91, 186 and 159 labels: every derivation read each tensor again.
#: The receiver-write fast lane and the registration ladder's labeled-argument skip took
#: ``inplace_mlp`` from 64 to 57 and ``buffer`` from 120 to 118; the merged held-record
#: lanes measure 50 and 110. The nested-call exit skips the outer ``F.batch_norm`` emission,
#: whose label reads take ``buffer`` from 110 to 103. A single-output call now reuses its
#: loggable-output verdict instead of reading the output's label a second time in the emit
#: loop, which takes the three traces to 44, 97 and 59. The thin emit logs a plain single
#: output without the emit loop's label reads, which takes them to 38, 91 and 50.
_PINNED_LABEL_READS = (
    ("inplace_mlp", _InplaceMLP, (2, 8), 38),
    ("buffer", _BufferModel, (2, 3, 6, 6), 91),
    ("nested_args", _NestedArgsModel, (3, 4), 50),
)


@pytest.mark.parametrize(
    ("name", "model_cls", "shape", "expected"),
    _PINNED_LABEL_READS,
    ids=[entry[0] for entry in _PINNED_LABEL_READS],
)
def test_label_reads_per_trace_are_pinned(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    model_cls: type[nn.Module],
    shape: tuple[int, ...],
    expected: int,
) -> None:
    """A whole trace makes exactly the measured number of gated label reads."""

    import sys

    from torchlens.backends.torch import _tl

    real_label = _tl.get_tensor_label
    reads = [0]

    def counting_label(t: Any) -> Any:
        """Count one gated label read."""
        reads[0] += 1
        return real_label(t)

    for module_name, module in list(sys.modules.items()):
        if module_name.startswith("torchlens") and module is not None:
            for attr, value in list(vars(module).items()):
                if value is real_label:
                    monkeypatch.setattr(module, attr, counting_label)
    torch.manual_seed(0)
    model, x = model_cls(), torch.randn(*shape)
    tl.trace(model, x)
    assert reads[0] == expected, f"{name}: {reads[0]} label reads per trace, pinned at {expected}"


class _OwnedTensorModel(nn.Module):
    """A lazy batch norm plus a plain tensor attribute that is not a registered buffer."""

    def __init__(self) -> None:
        """Build the lazy norm and the attribute."""
        super().__init__()
        self.norm = nn.LazyBatchNorm1d()
        self.offset = torch.full((4,), 0.25)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize, then shift by the attribute."""
        return self.norm(x) + self.offset


@pytest.mark.parametrize(
    ("model_cls", "shape", "expects_address"),
    [(_BufferModel, (2, 3, 6, 6), True), (_OwnedTensorModel, (3, 4), False)],
    ids=["buffer", "owned_tensor"],
)
def test_buffer_address_resolution_never_labels_a_tensor(
    monkeypatch: pytest.MonkeyPatch,
    model_cls: type[nn.Module],
    shape: tuple[int, ...],
    expects_address: bool,
) -> None:
    """The registration ladder reads a first-use tensor's label once: resolution never sets one."""

    from torchlens.backends.torch import _tl, wrappers

    real_resolve = wrappers.resolve_or_late_index_buffer_address
    seen: list[tuple[Any, Any, Any]] = []

    def recording(trace: Any, t: torch.Tensor) -> Any:
        """Record the tensor's label before and after one address resolution."""
        before = _tl.get_tensor_label(t)
        address = real_resolve(trace, t)
        seen.append((before, address, _tl.get_tensor_label(t)))
        return address

    monkeypatch.setattr(wrappers, "resolve_or_late_index_buffer_address", recording)
    torch.manual_seed(0)
    tl.trace(model_cls(), torch.randn(*shape))
    assert seen, "no first-use tensor reached address resolution"
    assert not expects_address or any(address is not None for _before, address, _after in seen)
    assert [entry for entry in seen if entry[0] is not None or entry[2] is not None] == []


def test_slot_record_is_frozen() -> None:
    """``_ArgSlot`` records are immutable once the snapshot is taken."""

    slot = _ops_arg_slots._ArgSlot(
        kind="args",
        key=0,
        value_id=0,
        label=None,
        known_provenance=False,
        session_parent_labels=(),
        data_alias=False,
        param_unmutated=False,
        param_validated=False,
    )
    with pytest.raises(AttributeError):
        slot.kind = "kwargs"  # type: ignore[misc]
