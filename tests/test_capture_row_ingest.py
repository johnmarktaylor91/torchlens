"""Default capture ingests its capture rows (capture C2c).

The exhaustive commit writes each op's cells into a capture row (C2b). Ingest
adopts that row into the trace's op store and writes only the join cells, the
neutral seeds and ``Op.__init__``'s defaults and conversions into it, so a
default capture builds no ``OpRecord`` from a draft, scatters no record into
cells, constructs no ``Op`` from a fields dict and clones no sealed journal
entry. These tests pin those counts, that each capture row becomes exactly one
``Op`` over the same row list, that every adopted op's cells equal the cells
the record path builds for the same entry, and that a handle whose capture
rows were handed off refuses to build.
"""

from __future__ import annotations

import weakref
from collections import Counter
from collections.abc import Callable
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
import torchlens._postprocess as postprocess_module
from torchlens._constants import LAYER_PASS_LOG_FIELD_ORDER
from torchlens._ir import _row_journal, op_record_scatter
from torchlens._postprocess import _materialize
from torchlens._trace_core.op_store import _MISSING, OpRowStore
from torchlens.backends.torch import _ops_container_base


class _ResidualMlp(nn.Module):
    """Two linear layers with a residual add and a nonlinearity."""

    def __init__(self) -> None:
        """Build the two layers."""

        super().__init__()
        self.fc1 = nn.Linear(6, 6)
        self.fc2 = nn.Linear(6, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the residual block and the head."""

        return self.fc2(torch.relu(self.fc1(x)) + x)


class _ConvNormModel(nn.Module):
    """Convolution, batch norm buffers, an in-place op and a split."""

    def __init__(self) -> None:
        """Build the conv, norm and linear layers."""

        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3)
        self.norm = nn.BatchNorm2d(4)
        self.fc = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convolve, normalize, rectify in place and project half the features."""

        y = self.norm(self.conv(x))
        y.relu_()
        left, _right = y.flatten(1)[:, :16].split(8, dim=1)
        return self.fc(left)


class _TinyEncoder(nn.Module):
    """A one-layer transformer encoder with a projection head."""

    def __init__(self) -> None:
        """Build the encoder layer and head."""

        super().__init__()
        self.encoder = nn.TransformerEncoderLayer(
            d_model=8, nhead=2, dim_feedforward=16, dropout=0.0, batch_first=True
        )
        self.head = nn.Linear(8, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Encode, then project the mean token."""

        return self.head(self.encoder(x).mean(dim=1))


def _subjects() -> list[tuple[str, nn.Module, torch.Tensor]]:
    """Return (name, model, input) probe subjects."""

    torch.manual_seed(0)
    return [
        ("residual_mlp", _ResidualMlp().eval(), torch.randn(2, 6)),
        ("conv_norm", _ConvNormModel().train(), torch.randn(2, 3, 6, 6)),
        ("tiny_encoder", _TinyEncoder().eval(), torch.randn(2, 5, 8)),
    ]


_SUBJECT_NAMES = [name for name, _model, _x in _subjects()]


def _subject(name: str) -> tuple[nn.Module, torch.Tensor]:
    """Return a fresh model and input for subject ``name``."""

    for subject_name, model, x in _subjects():
        if subject_name == name:
            return model, x
    raise KeyError(name)


def _counting(counts: Counter[str], key: str, function: Callable[..., Any]) -> Callable[..., Any]:
    """Return ``function`` wrapped to count its calls under ``key``."""

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        """Count one call and delegate."""

        counts[key] += 1
        return function(*args, **kwargs)

    return wrapper


@pytest.mark.parametrize("name", _SUBJECT_NAMES)
def test_default_capture_ingests_capture_rows_without_record_rebuilds(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    """No draft freeze, scatter, fields-dict Op or clone; one Op shell per capture row."""

    counts: Counter[str] = Counter()
    for module, attribute in (
        (_ops_container_base, "_op_record_from_log"),
        (op_record_scatter, "scatter_record_to_cells"),
        (_materialize, "materialize_log_from_fields"),
        (postprocess_module, "_clone_op_event_for_replay"),
        (_row_journal, "_record_from_row"),
    ):
        monkeypatch.setattr(
            module, attribute, _counting(counts, attribute, getattr(module, attribute))
        )
    handle_rows: list[list[Any]] = []
    original_handle = _row_journal._RowJournal.handle

    def recording_handle(self: Any, row_index: int, rest: Any) -> Any:
        """Record each committed capture row list, then build the handle."""

        handle_rows.append(self.store._rows[row_index])
        return original_handle(self, row_index, rest)

    monkeypatch.setattr(_row_journal._RowJournal, "handle", recording_handle)
    adopted: list[tuple[OpRowStore, list[Any]]] = []
    original_adopt = OpRowStore.adopt_row

    def recording_adopt(self: OpRowStore, cells: list[Any]) -> int:
        """Record each adopted row list with its store, then adopt it."""

        adopted.append((self, cells))
        return original_adopt(self, cells)

    monkeypatch.setattr(OpRowStore, "adopt_row", recording_adopt)
    model, x = _subject(name)

    trace = tl.trace(model, x)

    assert handle_rows, "the exhaustive capture committed no rows"
    assert counts["_op_record_from_log"] == 0
    assert counts["scatter_record_to_cells"] == 0
    assert counts["materialize_log_from_fields"] == 0
    assert counts["_clone_op_event_for_replay"] == 0
    assert counts["_record_from_row"] == len(handle_rows)
    trace_store = trace._trace_core.ops
    adopted_by_trace = [cells for store, cells in adopted if store is trace_store]
    capture_ids = {id(row) for row in handle_rows}
    assert sum(id(cells) in capture_ids for cells in adopted_by_trace) == len(handle_rows)
    assert len({id(cells) for cells in adopted_by_trace} & capture_ids) == len(handle_rows)


def _same(left: Any, right: Any) -> bool:
    """Return whether two cell values are equal in type and value (tensors by content)."""

    if left is right:
        return True
    if type(left) is not type(right):
        return False
    if isinstance(left, torch.Tensor):
        return left.dtype == right.dtype and left.shape == right.shape and torch.equal(left, right)
    if isinstance(left, weakref.ref):
        return left() is right()
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(_same(left[k], right[k]) for k in left)
    if isinstance(left, list | tuple):
        return len(left) == len(right) and all(map(_same, left, right))
    return bool(left == right)


@pytest.mark.parametrize("name", _SUBJECT_NAMES)
def test_adopted_capture_row_equals_the_record_path_cells(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    """Every adopted row holds exactly the cells the scatter-and-construct path builds."""

    from torchlens._data_classes.op import _OP_STORE_LAYOUT

    expected: dict[str, Any] = {}
    owning: list[Any] = []
    original_build = _materialize.build_ingest_inputs
    original_join = _materialize._apply_join_cells
    original_scatter = op_record_scatter.scatter_record_to_cells
    original_construct = _materialize.materialize_log_from_fields

    def recording_build(*args: Any, **kwargs: Any) -> Any:
        """Record the ingest inputs' owning trace."""

        inputs = original_build(*args, **kwargs)
        owning.append(inputs.owning_trace)
        return inputs

    def shadow_join(fields_dict: dict[str, object], record: Any, *rest: Any) -> None:
        """Join as ingest does, then build the record path's op for the same entry."""

        original_join(fields_dict, record, *rest)
        old: dict[str, object] = dict.fromkeys(LAYER_PASS_LOG_FIELD_ORDER)
        old.update(original_scatter(record, rest[0], owning[-1]))
        original_join(old, record, *rest)
        expected[record.label_raw] = original_construct(old, None)

    monkeypatch.setattr(_materialize, "build_ingest_inputs", recording_build)
    monkeypatch.setattr(_materialize, "_apply_join_cells", shadow_join)
    checked: list[str] = []
    original_apply = _materialize.apply_step0_result

    def comparing_apply(trace: Any, result: Any) -> None:
        """Compare each registered op's cells with the record path's op, then apply."""

        for label_raw, op_log in result.raw_log_registrations:
            old = expected[label_raw]
            for field_name, fid in _OP_STORE_LAYOUT.fid_by_name.items():
                got = op_log._core.cell_get(op_log._row, fid)
                want = old._core.cell_get(old._row, fid)
                assert (got is _MISSING) == (want is _MISSING), (label_raw, field_name)
                assert _same(got, want), (label_raw, field_name, got, want)
            checked.append(label_raw)
        original_apply(trace, result)

    monkeypatch.setattr(_materialize, "apply_step0_result", comparing_apply)
    model, x = _subject(name)

    tl.trace(model, x)

    assert checked and len(checked) == len(expected)


def test_handle_refuses_to_build_after_its_rows_are_handed_off() -> None:
    """A handle left unbuilt when ingest takes the capture rows raises instead of misreading."""

    from torchlens._data_classes.op import _OP_STORE_LAYOUT

    store = OpRowStore(_OP_STORE_LAYOUT)
    journal = _row_journal._RowJournal(store)
    row_index = store.adopt_row([_MISSING] * _OP_STORE_LAYOUT.n_fields)
    rest = _row_journal._RowResidual(**dict.fromkeys(_row_journal._RowResidual.__slots__[:-1]))
    handle = journal.handle(row_index, rest)

    journal.release_rows()

    with pytest.raises(RuntimeError, match="handed off"):
        handle.materialize()
