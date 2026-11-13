"""Row-backed exhaustive capture journal (capture C2b).

An exhaustive op commits as a row handle; its full ``OpRecord`` is built lazily,
once. These tests pin the build count (none through capture, seal and the raw
shape hash; each record once per trace), that the lazy record equals the eager
freeze field for field (raw and amended) as does every value a handle answers
from its row, that values read at commit are
snapshotted (a later ``requires_grad`` flip cannot leak in), and that the
commit-time residual is released once the record is built.
"""

from __future__ import annotations

import dataclasses
import gc
import warnings
import weakref
from collections.abc import Iterator
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
import torchlens._postprocess as postprocess_module
from torchlens._ir import _row_journal
from torchlens._ir.op_record import OpRecord, OpRecordAttributeError, record_with_path_updates
from torchlens._ir.refs import journal_record
from torchlens.backends.torch import _ops_row_journal, _thin_emit
from torchlens.backends.torch._ops_container_base import _op_record_from_log
from torchlens.errors import TraceCaptureError


@pytest.fixture(autouse=True)
def _thin_emit_off() -> Iterator[None]:
    """Pin today's emit: these tests spy on its commit (``commit_row_record``).

    The thin emit appends its rows without that commit; its rows are checked
    against today's in ``tests/test_thin_emit_soundness.py``.
    """

    previous = _thin_emit.set_thin_emit_enabled(False)
    yield
    _thin_emit.set_thin_emit_enabled(previous)


class _TinyTransformer(nn.Module):
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


class _InplaceBufferModel(nn.Module):
    """In-place ops, a batch-norm buffer and a multi-output split."""

    def __init__(self) -> None:
        """Build the conv, norm and linear layers."""

        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3)
        self.norm = nn.BatchNorm2d(4)
        self.fc = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Convolve, normalize in place-heavy fashion and split the result."""

        y = self.norm(self.conv(x))
        y.relu_()
        flat = y.flatten(1)[:, :16]
        flat[:, :8].add_(1.0)
        left, right = flat.split(8, dim=1)
        return self.fc(left), right * 2


def _subjects() -> list[tuple[str, nn.Module, torch.Tensor, dict[str, Any]]]:
    """Return (name, model, input, trace kwargs) probe subjects."""

    torch.manual_seed(0)
    return [
        ("transformer", _TinyTransformer().eval(), torch.randn(2, 5, 8), {}),
        ("inplace_buffer", _InplaceBufferModel().train(), torch.randn(2, 3, 6, 6), {}),
        (
            "module_save",
            nn.Sequential(nn.Linear(4, 4), nn.Tanh(), nn.Linear(4, 2)),
            torch.randn(3, 4),
            {"save": tl.module("0")},
        ),
    ]


def _same(left: Any, right: Any, path: str, diffs: list[str]) -> None:
    """Append every leaf where ``left`` and ``right`` differ (tensors by identity)."""

    if type(left) is not type(right):
        diffs.append(f"{path}: type {type(left).__name__} != {type(right).__name__}")
    elif isinstance(left, torch.Tensor):
        if left is not right:
            diffs.append(f"{path}: distinct tensors")
    elif dataclasses.is_dataclass(left) and not isinstance(left, type):
        for field in dataclasses.fields(left):
            _same(
                getattr(left, field.name), getattr(right, field.name), f"{path}.{field.name}", diffs
            )
    elif isinstance(left, dict):
        if list(left) != list(right):
            diffs.append(f"{path}: keys {list(left)!r} != {list(right)!r}")
        else:
            for key in left:
                _same(left[key], right[key], f"{path}[{key!r}]", diffs)
    elif isinstance(left, (list, tuple)):
        if len(left) != len(right):
            diffs.append(f"{path}: length {len(left)} != {len(right)}")
        else:
            for index, (a, b) in enumerate(zip(left, right)):
                _same(a, b, f"{path}[{index}]", diffs)
    elif left is not right and left != right:
        diffs.append(f"{path}: {left!r} != {right!r}")


def record_diffs(left: Any, right: Any) -> list[str]:
    """Return the field paths where two op records differ."""

    diffs: list[str] = []
    _same(left, right, "record", diffs)
    return diffs


def fast_read_diffs(handle: Any, record: OpRecord) -> list[str]:
    """Return the names a handle answers from its row differently from ``record``."""

    diffs: list[str] = []
    for name in _row_journal._FAST_READS:
        _same(getattr(handle, name), getattr(record, name), f"{handle.label_raw}.{name}", diffs)
    return diffs


class _CommitSpy:
    """Records, per committed handle, the eager record the freeze builds at commit."""

    def __init__(self) -> None:
        """Start with no commits and no ingest snapshot."""

        self.eager: list[tuple[Any, OpRecord]] = []
        self.residuals: list[weakref.ReferenceType[Any]] = []
        self.journal: Any = None
        self.on_ingest: Any = None
        self.ingest_seen = False
        self.build_eager = True
        self.fast_diffs: list[str] = []

    def commit(self, real: Any, draft: Any, layout: Any, edge_uses: Any) -> Any:
        """Build the eager record from ``draft``, then commit the row as production does."""

        self.journal = draft.trace.capture_events
        if not self.build_eager:
            return real(draft, layout, edge_uses)
        if draft.call_ref_box is None:
            # The commit resolves a lone output through a fresh box; share one so
            # the eager freeze and the row commit see the same function ref.
            draft.call_ref_box = []
        eager = _op_record_from_log(
            draft.trace,
            draft.fields_dict,
            draft.tensor,
            draft.fire_results,
            module_stack=draft.module_stack,
            call_ref_box=draft.call_ref_box,
        )
        handle = real(draft, layout, edge_uses)
        self.fast_diffs.extend(f"commit {d}" for d in fast_read_diffs(handle, eager))
        self.eager.append((handle, eager))
        self.residuals.append(weakref.ref(handle._rest))
        return handle


@pytest.fixture
def commit_spy(monkeypatch: pytest.MonkeyPatch) -> Iterator[_CommitSpy]:
    """Wrap the exhaustive row commit and the hand-off to ingest with a spy."""

    spy = _CommitSpy()
    real_commit = _ops_row_journal.commit_row_record
    monkeypatch.setattr(
        _ops_row_journal,
        "commit_row_record",
        lambda draft, layout, edge_uses: spy.commit(real_commit, draft, layout, edge_uses),
    )
    real_hand_off = postprocess_module.hand_off_capture_rows

    def hand_off(*args: Any) -> Any:
        """Run the ingest hook once, before the hand-off to ingest builds anything."""

        if not spy.ingest_seen:
            spy.ingest_seen = True
            if spy.on_ingest is not None:
                spy.on_ingest(spy)
        return real_hand_off(*args)

    monkeypatch.setattr(postprocess_module, "hand_off_capture_rows", hand_off)
    yield spy
    spy.eager.clear()


@pytest.fixture
def build_counter(monkeypatch: pytest.MonkeyPatch) -> Iterator[dict[str, int]]:
    """Count lazy record builds and every ``OpRecord`` construction."""

    counts = {"builds": 0, "records": 0}
    real_build = _row_journal._record_from_row
    real_init = OpRecord.__init__

    def counting_build(handle: Any) -> OpRecord:
        """Count one lazy build."""

        counts["builds"] += 1
        return real_build(handle)

    def counting_init(self: OpRecord, *args: Any, **kwargs: Any) -> None:
        """Count one record construction."""

        counts["records"] += 1
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(_row_journal, "_record_from_row", counting_build)
    monkeypatch.setattr(OpRecord, "__init__", counting_init)
    yield counts


def _eager_folded(spy: _CommitSpy) -> dict[int, OpRecord]:
    """Return the eager records with the journal's amendments folded, keyed by seq."""

    by_seq: dict[int, OpRecord] = {}
    for handle, eager in spy.eager:
        by_seq[handle.seq] = dataclasses.replace(
            eager, core=dataclasses.replace(eager.core, seq=handle.seq)
        )
    for amendment in spy.journal.op_amendments:
        by_seq[amendment.target_seq] = record_with_path_updates(
            by_seq[amendment.target_seq], amendment.patch
        )
    return by_seq


@pytest.mark.parametrize("subject", range(3), ids=["transformer", "inplace_buffer", "module_save"])
def test_capture_builds_no_record_before_ingest_and_each_once(
    subject: int, commit_spy: _CommitSpy, build_counter: dict[str, int]
) -> None:
    """Capture, seal and the raw hash build nothing; a trace builds each record once."""

    name, model, x, kwargs = _subjects()[subject]
    seen: dict[str, int] = {}

    def at_ingest(spy: _CommitSpy) -> None:
        """Refuse an unknown name on a handle, then snapshot the build counters."""

        with pytest.raises(OpRecordAttributeError):
            _ = spy.journal.op_events[0].not_a_record_name  # refused without a build
        seen.update(build_counter)

    commit_spy.build_eager = False
    commit_spy.on_ingest = at_ingest
    trace = tl.trace(model, x, **kwargs)
    try:
        assert commit_spy.journal is not None, f"{name}: no exhaustive commits"
        assert seen == {"builds": 0, "records": 0}, f"{name}: built before ingest {seen}"
        built = build_counter["builds"]
        events = trace.op_events
        assert all(type(event) is OpRecord for event in events)
        assert build_counter["builds"] == built, f"{name}: op_events access built records"
        assert built == len(events), f"{name}: {built} builds for {len(events)} records"
    finally:
        trace.cleanup()


@pytest.mark.parametrize("subject", range(3), ids=["transformer", "inplace_buffer", "module_save"])
def test_lazy_record_equals_eager_raw_and_amended(subject: int, commit_spy: _CommitSpy) -> None:
    """Every handle's lazy record equals the eager freeze, before and after amendments."""

    name, model, x, kwargs = _subjects()[subject]
    diffs: list[str] = []

    def at_ingest(spy: _CommitSpy) -> None:
        """Compare raw handles, then the amended fold, before ingest builds anything."""

        folded_expected = _eager_folded(spy)
        folded = spy.journal.amended_op_records()
        assert len(folded) == len(folded_expected)
        for event in folded:
            diffs.extend(f"fast {d}" for d in fast_read_diffs(event, folded_expected[event.seq]))
            built = journal_record(event)
            diffs.extend(f"folded {d}" for d in record_diffs(built, folded_expected[event.seq]))
        for handle, eager in spy.eager:
            raw = _row_journal._record_from_row(handle) if handle._row is not None else None
            if raw is not None:
                expected = dataclasses.replace(
                    eager, core=dataclasses.replace(eager.core, seq=handle.seq)
                )
                diffs.extend(f"raw {d}" for d in record_diffs(raw, expected))

    commit_spy.on_ingest = at_ingest
    trace = tl.trace(model, x, **kwargs)
    try:
        assert commit_spy.ingest_seen, f"{name}: ingest hook never ran"
        assert commit_spy.fast_diffs == [], f"{name}: {commit_spy.fast_diffs[:10]}"
        assert diffs == [], f"{name}: {diffs[:10]}"
    finally:
        trace.cleanup()


def test_requires_grad_flip_after_capture_cannot_leak(commit_spy: _CommitSpy) -> None:
    """Flipping ``requires_grad`` on params and outputs before the lazy build changes nothing."""

    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
    diffs: list[str] = []

    def at_ingest(spy: _CommitSpy) -> None:
        """Flip every param and output tensor, then build and compare each record."""

        for param in model.parameters():
            param.requires_grad_(False)
        for handle, _eager in spy.eager:
            tensor = handle.output.tensor.payload
            if isinstance(tensor, torch.Tensor):
                tensor.detach_()
        folded_expected = _eager_folded(spy)
        for event in spy.journal.amended_op_records():
            built = journal_record(event)
            diffs.extend(record_diffs(built, folded_expected[event.seq]))

    commit_spy.on_ingest = at_ingest
    trace = tl.trace(model, torch.randn(3, 4))
    try:
        assert any(handle.params for handle, _ in commit_spy.eager)
        assert diffs == []
    finally:
        trace.cleanup()


def test_residual_released_after_materialization(commit_spy: _CommitSpy) -> None:
    """No commit-time residual outlives the trace's record builds."""

    name, model, x, kwargs = _subjects()[0]
    trace = tl.trace(model, x, **kwargs)
    try:
        events = trace.op_events
        assert events, name
        commit_spy.eager.clear()
        gc.collect()
        alive = [ref for ref in commit_spy.residuals if ref() is not None]
        assert alive == [], f"{len(alive)} of {len(commit_spy.residuals)} residuals alive"
    finally:
        trace.cleanup()


_KEPT_LEAVES: list[torch.Tensor] = []
"""The leaf ``_InPlaceLeafModel`` creates, kept where the transform can reach it."""


class _InPlaceLeafModel(nn.Module):
    """A factory leaf, scaled in place (the in-place op's output is the leaf), then added."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Create a leaf with ``torch.ones``, keep it, scale it in place and add it."""

        leaf = torch.ones(3)
        _KEPT_LEAVES.append(leaf)
        leaf.mul_(2.0)
        return x + leaf


def test_activation_transform_flip_keeps_op_output_requires_grad() -> None:
    """A transform flipping an op output's live ``requires_grad`` leaves the op's record unchanged.

    The record's output ``requires_grad`` is the op output's own flag, read on the
    output when the op returns. The transform receives a saved copy, so it reaches
    the live output through the kept leaf: the in-place ``mul_`` returns the leaf
    itself, and its transform (the first one after the leaf is kept) flips the
    leaf's flag after that read and before the commit. The ``mul_`` record keeps
    the op-output value.
    """

    model = _InPlaceLeafModel().eval()
    flips: list[bool] = []

    def flip_leaf(tensor: torch.Tensor) -> torch.Tensor:
        """Flip the kept leaf's ``requires_grad`` once, at the first save after it is kept."""

        if _KEPT_LEAVES and not flips and _KEPT_LEAVES[0].is_leaf:
            leaf = _KEPT_LEAVES[0]
            flips.append(leaf.requires_grad)
            leaf.requires_grad_(not leaf.requires_grad)
        return tensor

    _KEPT_LEAVES.clear()
    trace = tl.trace(
        model,
        torch.randn(2, 3),
        capture=tl.options.CaptureOptions(activation_transform=flip_leaf),
    )
    try:
        assert flips == [False], f"transform flipped {flips}"
        assert _KEPT_LEAVES[0].requires_grad is True
        records = [journal_record(event) for event in trace.op_events]
        scaled = [record for record in records if "mul" in record.layer_type]
        assert len(scaled) == 1, [record.layer_type for record in records]
        assert scaled[0].output.tensor.requires_grad is False
        ones = [record for record in records if record.layer_type == "ones"]
        assert len(ones) == 1, [record.layer_type for record in records]
        assert ones[0].output.tensor.requires_grad is False
    finally:
        trace.cleanup()
        _KEPT_LEAVES.clear()


class _ScrubInterrupt(BaseException):
    """BaseException standing in for KeyboardInterrupt."""


class _RaiseAfterOps(nn.Module):
    """Model that commits two ops, then raises the parametrized error."""

    def __init__(self, error: BaseException) -> None:
        super().__init__()
        self.fc = nn.Linear(3, 4)
        self.error = error

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        torch.relu(self.fc(x))
        raise self.error


@pytest.mark.parametrize(
    ("error", "raised"),
    [(_ScrubInterrupt("interrupted"), _ScrubInterrupt), (RuntimeError("boom"), TraceCaptureError)],
    ids=["interrupt", "error"],
)
def test_failed_capture_scrub_settles_pending_rows(
    error: BaseException, raised: type[BaseException]
) -> None:
    """The failed-capture scrub drops ``capture_events`` first; pending rows still settle."""

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with pytest.raises(raised):
            tl.trace(_RaiseAfterOps(error).eval(), torch.ones(1, 3))
    failed = [str(w.message) for w in caught if "also failed while handling" in str(w.message)]
    assert failed == []
