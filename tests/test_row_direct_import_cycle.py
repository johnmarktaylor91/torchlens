"""The row-direct capture modules stay outside the package's largest import cycle.

The exhaustive commit writes rows through ``backends/torch/_ops_row_journal.py``
and ``backends/torch/_ops_row_cells.py``. ``ops.py`` (inside the cycle) calls
them, so any import from either module back into a cycle member pulls both into
the cycle and raises the layering ratchet. The helpers they need live in modules
with no import into the cycle: the record-ref builders in
``backends/torch/_ops_freeze_refs.py`` and the function-ref cells in
``_ir/op_record_scatter.py``.
"""

from __future__ import annotations

import importlib.util
import inspect
import sys
from collections.abc import Iterator
from pathlib import Path
from types import ModuleType

import pytest

REPO = Path(__file__).resolve().parent.parent
GRAPH_SOURCE = REPO / "tools" / "layering" / "graph.py"

#: Modules the row-direct commit imports at runtime; none may join the cycle.
ROW_DIRECT_MODULES = (
    "torchlens.backends.torch._ops_row_journal",
    "torchlens.backends.torch._ops_row_cells",
    "torchlens.backends.torch._ops_freeze_refs",
    "torchlens._ir.op_record_scatter",
    "torchlens._ir._row_journal",
)


def _load_graph_module() -> ModuleType:
    """Load ``tools/layering/graph.py`` by path (it is not an installed package)."""

    spec = importlib.util.spec_from_file_location("_tl_row_direct_graph", GRAPH_SOURCE)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {GRAPH_SOURCE}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)
    return module


@pytest.fixture(scope="module")
def cycle_and_edges() -> Iterator[tuple[frozenset[str], dict[str, set[str]]]]:
    """Yield the largest runtime import cycle's members and the runtime edges."""

    graph_module = _load_graph_module()
    graph = graph_module.build_graph(REPO / "torchlens")
    summary = graph_module.summarize(graph, members=True)
    edges = graph.edges()
    yield frozenset(summary["largest_scc_members"]), edges
    edges.clear()


@pytest.mark.parametrize("module", ROW_DIRECT_MODULES)
def test_row_direct_module_is_outside_the_largest_cycle(
    cycle_and_edges: tuple[frozenset[str], dict[str, set[str]]], module: str
) -> None:
    """Each row-direct module is a graph node and not a member of the largest cycle."""

    members, edges = cycle_and_edges
    assert module in edges, f"{module} is not a package module"
    assert module not in members


@pytest.mark.parametrize("module", ROW_DIRECT_MODULES)
def test_row_direct_module_imports_no_cycle_member(
    cycle_and_edges: tuple[frozenset[str], dict[str, set[str]]], module: str
) -> None:
    """No runtime import (module top or deferred) of a row-direct module enters the cycle."""

    members, edges = cycle_and_edges
    assert sorted(edges[module] & members) == []


def test_commit_row_record_requires_layout_and_edge_uses() -> None:
    """The commit takes the row layout and the edge uses as required arguments.

    ``ops.py`` supplies both (they come from modules inside the cycle); a
    default would let a caller skip the edge uses and commit a residual
    without them.
    """

    from torchlens.backends.torch import _ops_row_journal

    params = inspect.signature(_ops_row_journal.commit_row_record).parameters
    assert list(params) == ["draft", "layout", "edge_uses"]
    for name in ("layout", "edge_uses"):
        assert params[name].default is inspect.Parameter.empty, name
    with pytest.raises(TypeError):
        _ops_row_journal.commit_row_record(object())  # type: ignore[call-arg]
