"""Module records are built on first lookup, and every door sees the same records.

Step 16 seals the module facts (multi-output role names, the root call's
argument summaries, the uncalled module census, gradient-record owner naming)
and leaves the ``Module`` and ``ModuleCall`` records, the buffer accessor and
the flat module-call accessor to be built the first time a door reads them.
These tests hold the deferred build to the eager one: whichever door reads
first, the records come out the same; a fork, ``repr`` and ``to_pandas`` do
not build; save and pickle build first and keep their format.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._data_classes.module import Module, ModuleCall

_LAZY = ("_module_logs", "_buffer_accessor", "_param_logs_by_module")


class _Net(nn.Module):
    """A batch norm (buffers), an LSTM (named outputs), a reused block and an unused layer."""

    def __init__(self) -> None:
        """Build the layers."""

        super().__init__()
        self.norm = nn.BatchNorm1d(5)
        self.lstm = nn.LSTM(5, 3)
        self.block = nn.Linear(3, 3)
        self.unused = nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalise, run the LSTM, then apply the block twice.

        Parameters
        ----------
        x:
            Input of shape ``(seq, batch, 5)``.

        Returns
        -------
        torch.Tensor
            Output of shape ``(seq, batch, 3)``.
        """

        seq, batch, width = x.shape
        x = self.norm(x.reshape(seq * batch, width)).reshape(seq, batch, width)
        out, (h_n, c_n) = self.lstm(x)
        return self.block(self.block(out)) + h_n + c_n


def _trace() -> Any:
    """Capture a fresh trace of ``_Net``."""

    torch.manual_seed(0)
    model = _Net().train()
    return tl.trace(model, torch.randn(4, 2, 5))


def _pending(trace: Any) -> bool:
    """Whether the trace's module records are still waiting for a lookup."""

    state = trace.__dict__
    return "_module_pending" in state and not any(name in state for name in _LAZY)


def _facts(trace: Any) -> dict[str, Any]:
    """The module, call and buffer facts a user reads, in a comparable form."""

    return {
        "modules": [
            (m.address, m.num_calls, list(m.call_labels), list(m.layer_labels))
            for m in trace.modules
        ],
        "calls": [(c.label, c.forward_args_summary) for c in trace.module_calls],
        "buffers": [(b.address, [v.label for v in b.versions]) for b in trace.buffers],
        "root": trace.root_module.address,
        "uncalled": list(trace.uncalled_modules),
    }


def _held_records(value: Any) -> list[Any]:
    """Module and ModuleCall objects held directly by a value or its slots and items."""

    found = []
    stack = [value]
    seen: set[int] = set()
    while stack:
        item = stack.pop()
        if id(item) in seen:
            continue
        seen.add(id(item))
        if isinstance(item, (Module, ModuleCall)):
            found.append(item)
        elif isinstance(item, dict):
            stack.extend(item.values())
        elif isinstance(item, (list, tuple, set)):
            stack.extend(item)
        elif hasattr(item, "__slots__") and item.__class__.__name__ == "PendingModules":
            stack.extend(getattr(item, name) for name in item.__slots__ if name != "build")
    return found


def test_capture_leaves_records_pending_with_no_module_objects() -> None:
    """After capture the trace holds plain facts and no Module or ModuleCall object."""

    trace = _trace()
    assert _pending(trace)
    assert _held_records(trace.__dict__["_module_pending"]) == []


@pytest.mark.parametrize(
    "door",
    [
        lambda t: t.modules,
        lambda t: t.module_calls,
        lambda t: t.buffers,
        lambda t: t.root_module,
        lambda t: len(t.modules),
        lambda t: t.modules["lstm"],
    ],
    ids=["modules", "module_calls", "buffers", "root_module", "len", "getitem"],
)
def test_every_first_door_builds_the_same_records(door: Any) -> None:
    """Whichever door reads first, the module, call and buffer facts match."""

    expected = _facts(_trace())
    trace = _trace()
    door(trace)
    assert not _pending(trace)
    assert _facts(trace) == expected


def test_seal_time_facts_exist_before_any_lookup() -> None:
    """Output role names and the uncalled census are written at the seal."""

    trace = _trace()
    names = sorted(op.multi_output_name for op in trace.ops if op.multi_output_name is not None)
    uncalled = list(trace.uncalled_modules)
    assert _pending(trace)
    assert names == ["c_n", "h_n", "output"]
    assert uncalled == ["unused"]


def test_root_live_arguments_are_released_at_the_seal() -> None:
    """The pending facts keep argument shapes, never the user's live tensors."""

    trace = _trace()
    args, _kwargs = trace.__dict__["_module_pending"].root_forward_args
    assert all(not isinstance(arg, torch.Tensor) for arg in args)
    assert trace.root_module.forward_args_summary != ""


def test_repr_and_to_pandas_do_not_build() -> None:
    """Display paths count modules from the pending facts."""

    trace = _trace()
    text = repr(trace)
    trace.to_pandas()
    assert _pending(trace)
    trace.modules
    assert repr(trace) == text


def test_fork_of_pending_trace_does_not_build() -> None:
    """A fork shares the pending facts and builds its own records, once, on lookup."""

    parent = _trace()
    expected = _facts(_trace())
    fork = parent.fork()
    assert _pending(parent)
    assert _pending(fork)
    assert _facts(fork) == expected
    assert _pending(parent)
    assert all(m._source_trace is fork for m in fork.modules)
    assert _facts(parent) == expected
    assert fork.modules["lstm"] is not parent.modules["lstm"]


def test_fork_of_built_trace_matches_fork_of_pending_trace() -> None:
    """Forking before or after the build gives the same records."""

    built = _trace()
    built.modules
    assert _facts(built.fork()) == _facts(_trace().fork())


def test_pickle_and_save_build_first(tmp_path: Path) -> None:
    """Pickle and save read a pending trace's records as if they were built eagerly."""

    expected = _facts(_trace())
    restored = pickle.loads(pickle.dumps(_trace()))
    assert _facts(restored) == expected
    path = tmp_path / "pending.tlspec"
    _trace().save(path)
    loaded = tl.load(path)
    reference = tmp_path / "built.tlspec"
    built = _trace()
    built.modules
    built.save(reference)
    assert _facts(loaded) == _facts(tl.load(reference))


def test_cleanup_of_pending_trace_does_not_build() -> None:
    """Cleanup drops the pending facts instead of building records to delete them."""

    trace = _trace()
    trace.cleanup()
    assert "_module_pending" not in trace.__dict__
    assert "_module_logs" not in trace.__dict__
