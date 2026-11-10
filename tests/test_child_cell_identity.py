"""Key-form child cells keep read identity: an in-place write to the container persists.

``Module.ops``, ``Module.params`` and ``Buffer.versions`` store their children
as row keys (``torchlens/_trace_core/record_refs.py``). The first read in a
trace builds the container and keeps it in the trace's side table, so later
reads return the same object and a write into it is seen by the next read, as
when the cell held the container itself.
"""

from __future__ import annotations

import gc
from typing import Any

import torch
from torch import nn

import torchlens as tl
from torchlens._trace_core.record_refs import CHILD_CELLS, RowKeys
from torchlens._trace_core.record_rows import RecordCellField


class _Counted(nn.Module):
    """Two linear layers and a buffer updated in place on every forward."""

    def __init__(self) -> None:
        """Build the layers and the buffer."""

        super().__init__()
        self.first = nn.Linear(4, 4)
        self.second = nn.Linear(4, 4)
        self.register_buffer("seen", torch.zeros(4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Count the input into the buffer and run both layers."""

        self.seen.add_(x.sum(0))
        return self.second(torch.relu(self.first(x))) + self.seen


def _raw(record: Any, name: str) -> Any:
    """Return the stored cell of ``record.name`` without the child-cell descriptor."""

    return RecordCellField.__get__(type(record).__dict__[name], record)


def _trace() -> Any:
    """Capture the model once."""

    torch.manual_seed(0)
    return tl.trace(_Counted(), torch.randn(3, 4))


def _keyed_module(trace: Any, name: str) -> Any:
    """Return a module whose ``name`` cell is in key form."""

    for module in trace.modules.values():
        if _raw(module, name).__class__ is RowKeys:
            return module
    raise AssertionError(f"no module holds {name} as row keys")


def test_module_ops_write_persists() -> None:
    """``module.ops`` is the same accessor on every read; a call set into it stays."""

    trace = _trace()
    module = _keyed_module(trace, "ops")
    calls = module.ops
    assert module.ops is calls
    first = calls[0]
    calls[99] = first
    gc.collect()
    assert module.ops is calls
    assert module.ops._dict[99] is first  # noqa: SLF001
    assert _raw(module, "ops").__class__ is RowKeys


def test_module_params_read_identity() -> None:
    """``module.params`` is the same accessor on every read."""

    trace = _trace()
    module = _keyed_module(trace, "params")
    params = module.params
    gc.collect()
    assert module.params is params


def test_buffer_versions_write_persists() -> None:
    """``buffer.versions`` is the same list on every read; an appended op stays."""

    trace = _trace()
    buffers = [b for b in trace.buffers.values() if _raw(b, "versions")]
    keyed = [b for b in buffers if _raw(b, "versions")[0].__class__ is int]
    assert keyed, "no buffer holds its versions as rows"
    buffer = keyed[0]
    versions = buffer.versions
    assert buffer.versions is versions
    count = len(versions)
    versions.append(versions[0])
    gc.collect()
    assert buffer.versions is versions
    assert len(buffer.versions) == count + 1


def test_containers_live_in_the_trace_side_table() -> None:
    """The kept containers sit in the reading trace's side table, not on the record."""

    trace = _trace()
    module = _keyed_module(trace, "ops")
    calls = module.ops
    table = trace._trace_core._strong_facades[CHILD_CELLS]  # noqa: SLF001
    assert any(entry[2] is calls for entry in table.values())
