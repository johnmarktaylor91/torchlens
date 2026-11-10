"""Foreign data round trips in plain forward code (no ``autograd.Function``).

Data that leaves torch (``.numpy()``, ``__dlpack__`` / ``to_dlpack``, ``.tolist()``) and
comes back through a host constructor is one ``foreigndata`` op whose parent is the
departed tensor: a zero-copy or copy round trip replays exactly, a re-entry after host
computation TorchLens did not see is a typed validation failure (never a silent constant),
and a true constant in a capture with no departure stays the plain source op it is.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._record_type import record_type
from torchlens._utils._foreign_data_replay import foreigndata
from torchlens.backends.torch._foreign_data import (
    FOREIGN_DATA_EXPORT_WRAPPER_NAME,
    FOREIGN_DATA_FUNC_NAME,
    FOREIGN_DATA_REPLAY_REASON,
    FOREIGN_DATA_TRANSFORM_KIND,
    RESOLUTION_CONSTANT,
    RESOLUTION_COPY,
    RESOLUTION_UNRESOLVED,
    RESOLUTION_VIEW,
    foreign_data_resolution,
    is_foreign_data_recorder,
)
from torchlens.backends.torch.completeness_witness import (
    AUDITED_COMPLETENESS_BOUNDARIES,
)
from torchlens.backends.torch.wrappers import wrap_torch
from torchlens.errors import TorchLensWarning
from torchlens.validation import last_validation_failure, validate_forward_pass
from torchlens.validation._diagnostics import CHECK_REPLAY


class _RoundTripModel(nn.Module):
    """``fc -> detach -> <leave torch> -> <come back> -> relu`` with a pluggable round trip."""

    def __init__(self, round_trip) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)
        self.round_trip = round_trip

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.fc(x).detach()
        return torch.relu(self.round_trip(y))


def _numpy_round_trip(y: torch.Tensor) -> torch.Tensor:
    return torch.from_numpy(y.numpy())


def _cpu_numpy_round_trip(y: torch.Tensor) -> torch.Tensor:
    return torch.from_numpy(y.cpu().numpy())


def _numpy_transpose_round_trip(y: torch.Tensor) -> torch.Tensor:
    # A numpy view with different geometry shares the memory: still pure data movement.
    return torch.from_numpy(y.numpy().T)


def _as_tensor_round_trip(y: torch.Tensor) -> torch.Tensor:
    return torch.as_tensor(np.asarray(y))


def _dlpack_capsule_round_trip(y: torch.Tensor) -> torch.Tensor:
    return torch.from_dlpack(torch.utils.dlpack.to_dlpack(y))


def _dlpack_numpy_round_trip(y: torch.Tensor) -> torch.Tensor:
    return torch.from_dlpack(np.from_dlpack(y))


def _tolist_round_trip(y: torch.Tensor) -> torch.Tensor:
    return torch.tensor(y.tolist())


def _tensor_of_ndarray_round_trip(y: torch.Tensor) -> torch.Tensor:
    return torch.tensor(y.numpy())


def _numpy_host_compute_round_trip(y: torch.Tensor) -> torch.Tensor:
    # The old false pass: from_numpy replayed the ndarray as a constant.
    return torch.from_numpy(y.numpy() * 2)


def _tolist_host_compute_round_trip(y: torch.Tensor) -> torch.Tensor:
    return torch.tensor([[v * 2 for v in row] for row in y.tolist()])


def _foreign_ops(trace: tl.Trace) -> list:
    """Return the ``foreigndata`` ops of a trace."""

    return [op for op in trace._layer_list if op.func_name == FOREIGN_DATA_FUNC_NAME]


def _trace_unresolved(model: nn.Module, x: torch.Tensor) -> tl.Trace:
    """Trace a model with an unresolved round trip; assert the one capture-time warning."""

    with pytest.warns(TorchLensWarning, match="re-entered torch after host computation") as rec:
        trace = tl.trace(model, x)
    codes = [w.message.fields.get("code") for w in rec if isinstance(w.message, TorchLensWarning)]
    assert codes.count("foreign_data_roundtrip_unresolved") == 1
    return trace


def _trace_quiet(model: nn.Module, x: torch.Tensor) -> tl.Trace:
    with warnings.catch_warnings():
        warnings.simplefilter("error", TorchLensWarning)
        return tl.trace(model, x)


def _assert_one_round_trip(trace: tl.Trace, resolution: str, parent_type: str) -> None:
    """Assert exactly one ``foreigndata`` op with the departed tensor as its parent."""

    ops = _foreign_ops(trace)
    assert len(ops) == 1
    op = ops[0]
    assert op.call.is_func_transform is True
    assert op.call.func_transform_kind == FOREIGN_DATA_TRANSFORM_KIND
    assert op.call.func_transform_config["resolution"] == resolution
    assert is_foreign_data_recorder(op.func)
    assert foreign_data_resolution(op.func) == resolution
    assert [record_type(trace[label]) for label in op.parents] == [parent_type]
    assert [record_type(trace[label]) for label in op.children] == ["relu"]
    # The graph hole is closed: the output descends from the model input.
    assert trace.output_ops[0].input_ancestors


def _validate(model: nn.Module, x: torch.Tensor) -> bool:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return validate_forward_pass(model, [x], input_kwargs={})


@pytest.mark.parametrize(
    ("round_trip", "resolution", "parent_type"),
    [
        pytest.param(_numpy_round_trip, RESOLUTION_VIEW, "detach", marks=pytest.mark.smoke),
        (_numpy_transpose_round_trip, RESOLUTION_VIEW, "detach"),
        pytest.param(
            _dlpack_capsule_round_trip, RESOLUTION_VIEW, "detach", marks=pytest.mark.smoke
        ),
        pytest.param(_tolist_round_trip, RESOLUTION_COPY, "detach", marks=pytest.mark.smoke),
    ],
)
def test_pure_round_trip_is_one_op_with_the_departed_parent_and_validates(
    round_trip, resolution, parent_type
) -> None:
    """numpy, dlpack and tolist round trips outside any Function replay exactly."""

    model = _RoundTripModel(round_trip)
    x = torch.randn(3, 4)
    trace = _trace_quiet(model, x)
    _assert_one_round_trip(trace, resolution, parent_type)
    op = _foreign_ops(trace)[0]
    # The replay callable rebuilds the re-entered value from the departed tensor and the
    # recorded literals alone, and its key names the portable replay function.
    departed = trace[op.parents[0]].out
    expected = model.round_trip(departed.clone())
    assert torch.equal(op.func(departed, *op.call.non_tensor_pos_args), expected)
    assert (op.func.__module__, op.func.__qualname__) == (
        foreigndata.__module__,
        foreigndata.__qualname__,
    )
    assert _validate(model, x) is True
    assert last_validation_failure() is None


@pytest.mark.smoke
@pytest.mark.parametrize(
    "round_trip", [_numpy_host_compute_round_trip, _tolist_host_compute_round_trip]
)
def test_host_computation_between_export_and_reentry_is_a_typed_failure(round_trip) -> None:
    """The former false pass: host arithmetic in the gap is refused, never a constant."""

    model = _RoundTripModel(round_trip)
    x = torch.randn(3, 4)
    trace = _trace_unresolved(model, x)
    _assert_one_round_trip(trace, RESOLUTION_UNRESOLVED, "detach")
    assert _validate(model, x) is False
    failure = last_validation_failure()
    assert failure is not None
    # The capture-time warning opens with validation's own sentence for this op.
    with pytest.warns(TorchLensWarning) as rec:
        tl.trace(model, x)
    sentence = failure.message.split("; the recorded value")[0]
    assert str(rec[0].message).startswith(sentence)
    assert failure.check_name == CHECK_REPLAY
    assert failure.func_name == FOREIGN_DATA_FUNC_NAME
    assert failure.extra["reason"] == FOREIGN_DATA_REPLAY_REASON
    assert failure.op_label == _foreign_ops(trace)[0].label
    assert failure.extra["parents"] == list(_foreign_ops(trace)[0].parents)


class _ConstantAfterExportModel(nn.Module):
    """A host constant built after a plain export: linked and refused by design."""

    def __init__(self, export_first: bool) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)
        self.export_first = export_first
        self.table = np.ones((3, 4), dtype=np.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.fc(x)
        if self.export_first:
            y.detach().numpy()
        return y + torch.from_numpy(self.table)


class _ScalarAfterTolistModel(nn.Module):
    """A Python-scalar constant after a ``tolist``: list and scalar arguments link ``tolist`` only."""

    def __init__(self, export: str) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)
        self.export = export

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.fc(x)
        if self.export == "numpy":
            y.detach().numpy()
        else:
            y.detach().tolist()
        return y * torch.tensor(0.5)


def test_scalar_constructor_links_only_tolist_departures() -> None:
    """``torch.tensor(0.5)`` after a numpy export stays a source; after a tolist it is linked."""

    numpy_model = _ScalarAfterTolistModel("numpy")
    trace = _trace_quiet(numpy_model, torch.randn(3, 4))
    assert _foreign_ops(trace) == []
    tolist_model = _ScalarAfterTolistModel("tolist")
    trace = _trace_unresolved(tolist_model, torch.randn(3, 4))
    ops = _foreign_ops(trace)
    assert (
        len(ops) == 1 and ops[0].call.func_transform_config["resolution"] == RESOLUTION_UNRESOLVED
    )


class _ConsumedCapsuleModel(nn.Module):
    """``from_dlpack`` of a capsule made before the forward: not torch data of this capture."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)
        self.host = np.full((3, 4), 2.0, dtype=np.float32)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.from_dlpack(self.host.__dlpack__())
        return self.fc(x) * scale


def test_consumed_capsule_without_departure_replays_its_recorded_value() -> None:
    """Finding 2: a parentless ``from_dlpack`` no longer fails replay on its dead capsule."""

    model = _ConsumedCapsuleModel()
    x = torch.randn(3, 4)
    trace = _trace_quiet(model, x)
    ops = _foreign_ops(trace)
    assert len(ops) == 1
    assert ops[0].call.func_transform_config["resolution"] == RESOLUTION_CONSTANT
    assert list(ops[0].parents) == []
    assert torch.equal(ops[0].func(), torch.full((3, 4), 2.0))
    assert _validate(model, x) is True


class _ExportOnlyModel(nn.Module):
    """A DGL-shaped export in plain code (``to_dlpack`` of a size-1 dim) with no re-entry."""

    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.fc(x)
        # torch.s toDLPack normalises the stride of a size-1 dim through as_strided.
        torch.utils.dlpack.to_dlpack(y.detach().unsqueeze(0))
        return torch.relu(y)


@pytest.mark.smoke
def test_plain_export_dispatches_are_owned_by_the_export_token() -> None:
    """With the census armed, an export's own dispatches count once, owned by the export."""

    wrap_torch(completeness_witness=True)
    model = _ExportOnlyModel()
    trace = _trace_quiet(model, torch.randn(3, 4))
    assert trace.provenance.verified is True
    assert trace.completeness_diagnostics == []
    rows = [
        entry
        for entry in trace.health.completeness_decompositions
        if entry["owner_wrapper"] == FOREIGN_DATA_EXPORT_WRAPPER_NAME
    ]
    for row in rows:
        assert row["capture_accounted"] is True


@pytest.mark.smoke
def test_selector_and_audited_row_are_wired() -> None:
    """The op is selectable by its transform kind and the export token is an audited row."""

    model = _RoundTripModel(_numpy_round_trip)
    trace = tl.trace(model, torch.randn(2, 4), save=tl.func_transform(FOREIGN_DATA_TRANSFORM_KIND))
    saved = [op for op in trace._layer_list if op.has_saved_activation]
    assert [op.func_name for op in saved] == [FOREIGN_DATA_FUNC_NAME]
    rows = [
        row
        for row in AUDITED_COMPLETENESS_BOUNDARIES
        if row.wrapper_name == FOREIGN_DATA_EXPORT_WRAPPER_NAME
    ]
    assert len(rows) == 1 and rows[0].operator is None


def test_replay_function_checks_its_untrusted_literals() -> None:
    """The portable replay runs each mode and refuses literals that read past the source."""

    base = torch.arange(20.0)
    source = base[4:16].reshape(3, 4)  # storage larger than the departed tensor
    assert torch.equal(foreigndata(source, "strided", (4, 3), (1, 4), 0), source.t())
    assert torch.equal(foreigndata(source, "gather", (2,), (11, 0)), torch.tensor([15.0, 4.0]))
    copied = foreigndata(source, "copy", torch.float64, torch.device("cpu"))
    assert copied.dtype == torch.float64 and torch.equal(copied, source.double())
    with pytest.raises(RuntimeError):
        foreigndata(source, "strided", (3, 4), (4, 1), 1)  # one past the departed tensor
    with pytest.raises(RuntimeError):
        foreigndata(source, "strided", (3, 4), (-4, 1), 0)
    with pytest.raises(IndexError):
        foreigndata(source, "gather", (1,), (12,))
    with pytest.raises(KeyError):
        foreigndata(source, "eval", "print(1)")
    with pytest.raises(TypeError):
        foreigndata(source, "copy", "float64", "cpu")
