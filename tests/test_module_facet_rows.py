"""Module and module-call facet views name their records by row, live and after a reload.

A facet view cached on a ``Module`` names the Module; one cached on a
``ModuleCall`` wraps a ``CallBoundModule`` that names the call and its Module.
Each name is kept as a row key (``torchlens/_trace_core/record_refs.py``), so
the trace's storage holds no record through the view, and each read still
answers with the record the trace hands out.
"""

from __future__ import annotations

import gc
import pickle
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._trace_core.record_refs import RowKey


class _TwoLayers(nn.Module):
    """Two linear layers under one parent."""

    def __init__(self) -> None:
        """Build the layers."""

        super().__init__()
        self.first = nn.Linear(3, 3)
        self.second = nn.Linear(3, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run both layers."""

        return self.second(torch.relu(self.first(x)))


def _trace() -> Any:
    """Return a trace of the two-layer model."""

    torch.manual_seed(0)
    return tl.trace(_TwoLayers(), torch.randn(2, 3), save="all")


def _assert_views_by_row(trace: Any) -> tuple[list[Any], list[Any]]:
    """Check every module and module-call view of ``trace``; return their keys."""

    module_keys = []
    for module in trace.modules:
        view = module.facets
        assert view.__dict__["_record"].__class__ is RowKey
        assert view._record is module
        module_keys.append(sorted(map(str, view.keys())))
    call_keys = []
    for call in trace.module_calls:
        subject = call.facets._record
        assert subject.__dict__["_call"].__class__ is RowKey
        assert subject.__dict__["_module"].__class__ is RowKey
        assert subject._call is call
        assert subject._module is call.module
        call_keys.append(sorted(map(str, call.facets.keys())))
    return module_keys, call_keys


def test_live_views_name_records_by_row() -> None:
    """A live trace's module and module-call views store row keys and read the records."""

    _assert_views_by_row(_trace())


def test_loaded_views_match_live(tmp_path: Path) -> None:
    """A saved and loaded trace builds the same views, again by row."""

    live = _trace()
    expected = _assert_views_by_row(live)
    path = tmp_path / "module_facets.tlspec"
    live.save(path)
    loaded = tl.load(path)
    assert _assert_views_by_row(loaded) == expected


def test_pickled_views_match_live() -> None:
    """A pickle round trip drops the cached views and rebuilds them by row."""

    live = _trace()
    expected = _assert_views_by_row(live)
    restored = pickle.loads(pickle.dumps(live))
    assert _assert_views_by_row(restored) == expected


def test_view_refuses_after_its_trace_is_collected() -> None:
    """A view kept past its trace refuses with the collected-trace code."""

    trace = _trace()
    view = trace.modules["first"].facets
    del trace
    gc.collect()
    with pytest.raises(tl.errors.RecordBindingError) as raised:
        _ = view._record
    assert raised.value.fields["code"] == "trace_reference_collected"
