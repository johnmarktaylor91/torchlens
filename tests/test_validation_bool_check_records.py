"""One capture record per conditional test of a reused bool tensor.

``Op.terminal_conditional_id`` names one conditional, so a bool tensor tested by
two conditional tests used to fail metadata invariant 8 (``if flag: ... if flag:``
and a ternary followed by an ``if`` on one predicate). Capture now logs a ``clone``
of the bool at every later conditional test, and that clone is the test's terminal
bool. Models that test each bool once record exactly what they did before.

Models live at module level so the conditional classifier can read their source.
"""

from __future__ import annotations

import warnings
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch
import torch.nn as nn

import torchlens as tl
from torchlens import trace as trace_fn
from torchlens._data_classes.op import Op
from torchlens._data_classes.trace import Trace
from torchlens._runnable import PathFaithfulness
from torchlens.errors import MetadataInvariantError
from torchlens.options import CaptureOptions
from torchlens.validation import check_metadata_invariants

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")


class TwoIfsOneBool(nn.Module):
    """One bool tensor tests two ``if`` statements."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run both gated branches."""
        flag = (x > 0).any()
        if flag:
            x = x + 1
        if flag:
            x = x * 2
        return x


class TernaryThenIf(nn.Module):
    """One bool tensor tests a conditional expression and a later ``if``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the ternary and the gated branch."""
        c = x.sum() > 0
        y = x + 1 if c else x - 1
        if c:
            y = y * 3
        return y


class ThreeIfsOneBoolElse(nn.Module):
    """One bool tensor tests three ``if`` statements, the last with an ``else``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run three gated branches."""
        ok = (x > 0).all()
        if ok:
            x = torch.relu(x)
        x = x * 2
        if ok:
            x = torch.sigmoid(x)
        if ok:
            x = x - 1
        else:
            x = torch.tanh(x)
            x = x + 1
        return x


class LoopRetestsOneSite(nn.Module):
    """One bool tensor tested by the same ``if`` on every loop iteration."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the gated loop body three times."""
        flag = (x > 0).any()
        for _ in range(3):
            if flag:
                x = x - 0.5
        return x


class AssertThenIf(nn.Module):
    """One bool tensor tested by an ``assert`` and then one ``if``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Assert, then run the gated branch."""
        m = torch.isnan(x).any()
        assert not m
        if not m:
            x = x + 1
        return x


class OneIfOneBool(nn.Module):
    """One bool tensor tested once."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the gated branch."""
        flag = (x > 0).any()
        if flag:
            x = x + 1
        return x


def _trace(model: nn.Module, x: torch.Tensor) -> Trace:
    """Trace one forward pass with warnings silenced."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return trace_fn(model, x)


def _ops_named(trace: Trace, func_name: str) -> list[Op]:
    """Return the non-output ops whose captured function name matches."""

    return [op for op in trace._layer_list if op.func_name == func_name and not op.is_output]


@pytest.fixture
def positive_input() -> Iterator[torch.Tensor]:
    """Yield a deterministic input that takes every ``then`` arm."""

    yield torch.ones(2, 3)


@pytest.mark.parametrize(
    ("model_cls", "n_tests"),
    [(TwoIfsOneBool, 2), (TernaryThenIf, 2), (ThreeIfsOneBoolElse, 3)],
)
def test_reused_bool_gets_one_terminal_record_per_conditional(
    model_cls: type[nn.Module], n_tests: int, positive_input: torch.Tensor
) -> None:
    """Every conditional test owns exactly one terminal bool that points back to it."""

    trace = _trace(model_cls(), positive_input)
    try:
        events = sorted(trace.graph.conditional_records, key=lambda event: event.id)
        assert len(events) == n_tests
        owners = [event.bool_layers for event in events]
        assert all(len(labels) == 1 for labels in owners), owners
        assert len({labels[0] for labels in owners}) == n_tests
        for event in events:
            assert trace[event.bool_layers[0]].terminal_conditional_id == event.id
        clones = _ops_named(trace, "clone")
        assert len(clones) == n_tests - 1
        first_bool = trace[events[0].bool_layers[0]]
        for clone in clones:
            assert list(clone.parents) == [first_bool.layer_label]
            assert clone.control_flow.is_scalar_bool
            assert clone.control_flow.is_terminal_conditional_bool
        assert check_metadata_invariants(trace)
    finally:
        trace.cleanup()


@pytest.mark.parametrize("model_cls", [TwoIfsOneBool, TernaryThenIf, ThreeIfsOneBoolElse])
def test_reused_bool_model_validates(model_cls: type[nn.Module]) -> None:
    """A model that tests one bool in several conditionals validates end to end."""

    for x in (torch.ones(2, 3), -torch.ones(2, 3)):
        assert tl.validate(model_cls(), x, scope="forward")


@pytest.mark.parametrize("model_cls", [LoopRetestsOneSite, AssertThenIf, OneIfOneBool])
def test_single_conditional_bool_adds_no_record(model_cls: type[nn.Module]) -> None:
    """A bool that decides one conditional test (however often it runs) adds no op."""

    trace = _trace(model_cls(), torch.ones(2, 3))
    try:
        assert _ops_named(trace, "clone") == []
        assert check_metadata_invariants(trace)
    finally:
        trace.cleanup()
    assert tl.validate(model_cls(), torch.ones(2, 3), scope="forward")


def test_clone_linked_to_the_first_conditional_still_fails_invariant_8(
    positive_input: torch.Tensor,
) -> None:
    """Planted bug: one record listed by two conditionals (the old shared shape) still fails."""

    trace = _trace(TwoIfsOneBool(), positive_input)
    try:
        events = sorted(trace.graph.conditional_records, key=lambda event: event.id)
        clone_label = events[1].bool_layers[0]
        events[0].bool_layers.append(clone_label)
        with pytest.raises(MetadataInvariantError, match="Invariant 8"):
            check_metadata_invariants(trace)
    finally:
        trace.cleanup()


def test_unrecorded_clone_falls_back_and_still_fails(
    monkeypatch: pytest.MonkeyPatch, positive_input: torch.Tensor
) -> None:
    """Planted bug: without the per-test record the shared bool still fails its invariants.

    The second conditional's arm then runs before any recorded test of that conditional
    (invariant 6), or its bool points at the first conditional (invariant 8).
    """

    from torchlens.backends.torch import _bool_check_records

    monkeypatch.setattr(_bool_check_records, "_log_clone_label", lambda trace, source: None)
    trace = _trace(TwoIfsOneBool(), positive_input)
    try:
        assert _ops_named(trace, "clone") == []
        with pytest.raises(MetadataInvariantError, match=r"Invariant [68]:"):
            check_metadata_invariants(trace)
    finally:
        trace.cleanup()
    ternary = _trace(TernaryThenIf(), positive_input)
    try:
        with pytest.raises(MetadataInvariantError, match=r"Invariant [68]:"):
            check_metadata_invariants(ternary)
    finally:
        ternary.cleanup()


def test_reused_bool_model_saves_runnable_and_verifies(tmp_path: Path) -> None:
    """Under the replayable witness belt each later test logs one record and replays VERIFIED."""

    model = ThreeIfsOneBoolElse().eval()
    value = torch.ones(2, 3)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        trace = tl.trace(
            model,
            value,
            capture=CaptureOptions(replayable=True, capture_container_structure=True, cache=False),
            save="all",
        )
    assert len(_ops_named(trace, "clone")) == 2
    path = tmp_path / "reused-bool.tlspec"
    trace.save(path, level="runnable", include_weights=True)
    result = tl.load(path).run_transaction(inputs=value, seed=0)
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert torch.equal(result.output, model(value))
