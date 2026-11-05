"""Branches in source that only ``linecache`` holds are recorded.

IPython and Jupyter register each cell's text in ``linecache`` under a
pseudo-filename and compile the cell under that name. Conditional tracking
reads that text when no file exists, so a notebook ``if`` on a tensor records
its ``Conditional``, validates, and raises no scalar-escape warning. Source
with no readable text (``exec`` with no ``linecache`` entry) keeps warning, a
file on disk is indexed as before, and one pseudo-filename seen with two
different texts fails closed: it warns that the source changed under the same
name and records nothing more for that name.
"""

from __future__ import annotations

import linecache
import uuid
import warnings
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
import torch

import torchlens as tl
from torchlens._postprocess import ast_branches
from torchlens._postprocess._branch_source import read_interactive_source
from torchlens.errors import ScalarEscapeWarning

# No smoke mark: the smoke tier sits at its size ceiling (tests/conftest.py).

_CELL_SOURCE = """
import torch
from torch import nn


class Loop(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.fc2 = nn.Linear(4, 4)

    def forward(self, x):
        if x.sum() < 0:     # branch
            x = -x
        for _ in range(2):  # 2 passes
            x = self.fc2(torch.relu(self.fc1(x)))
        return x
"""

#: Same class name and line layout, different text: the test reads ``> 0``.
_OTHER_CELL_SOURCE = _CELL_SOURCE.replace("x.sum() < 0", "x.sum() > 0")


@pytest.fixture
def cell_names() -> Iterator[list[str]]:
    """Yield a list for pseudo-filenames a test registers; drop them afterwards."""

    names: list[str] = []
    yield names
    for name in names:
        linecache.cache.pop(name, None)
        ast_branches.invalidate_cache(name)


def _new_cell_name(names: list[str]) -> str:
    """Return a fresh IPython-style pseudo-filename and remember it for cleanup."""

    name = f"<ipython-input-1-{uuid.uuid4().hex[:12]}>"
    names.append(name)
    return name


def _register(name: str, source: str) -> None:
    """Register ``source`` in ``linecache`` the way IPython registers a cell."""

    linecache.cache[name] = (len(source), None, source.splitlines(True), name)


def _run_cell(name: str, source: str, *, register: bool = True) -> Any:
    """Compile ``source`` under ``name`` (optionally registered) and return its ``Loop``."""

    if register:
        _register(name, source)
    namespace: dict[str, Any] = {}
    exec(compile(source, name, "exec"), namespace)
    return namespace["Loop"]


def _trace(model_cls: Any) -> tuple[Any, Any, torch.Tensor, list[str], list[str]]:
    """Trace a fresh model on a negative input; return trace, model, input and warnings."""

    torch.manual_seed(0)
    model = model_cls()
    x = -torch.ones(1, 4)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        trace = tl.trace(model, x)
    escapes = [str(w.message) for w in caught if issubclass(w.category, ScalarEscapeWarning)]
    others = [str(w.message) for w in caught if not issubclass(w.category, ScalarEscapeWarning)]
    return trace, model, x, escapes, others


def test_linecache_cell_records_the_if_and_does_not_warn(cell_names: list[str]) -> None:
    """A registered cell records its ``if``, validates and raises no scalar escape."""

    name = _new_cell_name(cell_names)
    trace, model, x, escapes, _ = _trace(_run_cell(name, _CELL_SOURCE))

    assert escapes == []
    records = list(trace.graph.conditional_records)
    assert len(records) == 1
    branch_line = _CELL_SOURCE.splitlines().index("        if x.sum() < 0:     # branch") + 1
    assert records[0].source_file == name
    assert records[0].if_stmt_span[0] == branch_line
    assert tl.validation.validate_forward_pass(model, x) is True


def test_exec_without_linecache_entry_still_warns(cell_names: list[str]) -> None:
    """Source with no readable text records no branch and keeps the true warning."""

    name = _new_cell_name(cell_names)
    trace, _, _, escapes, _ = _trace(_run_cell(name, _CELL_SOURCE, register=False))

    assert len(escapes) == 1
    assert list(trace.graph.conditional_records) == []
    assert ast_branches.get_file_index(name) is None


def test_stale_linecache_entry_for_a_missing_file_is_not_read(tmp_path: Path) -> None:
    """An entry with a real mtime belongs to a disk file; once it is gone it is not read."""

    missing = str(tmp_path / "deleted_module.py")
    linecache.cache[missing] = (len(_CELL_SOURCE), 1.0, _CELL_SOURCE.splitlines(True), missing)
    try:
        assert read_interactive_source(missing) is None
        assert ast_branches.get_file_index(missing) is None
    finally:
        linecache.cache.pop(missing, None)
        ast_branches.invalidate_cache(missing)


def test_file_on_disk_is_indexed_by_filename(tmp_path: Path) -> None:
    """A real file keeps the disk path: indexed from disk, cached under its filename."""

    path = tmp_path / "loop_on_disk.py"
    path.write_text(_CELL_SOURCE, encoding="utf-8")
    filename = str(path)
    try:
        file_index = ast_branches.get_file_index(filename)
        assert file_index is not None
        assert file_index.mtime_ns == path.stat().st_mtime_ns
        assert len(file_index.conditionals) == 1
        assert ast_branches._file_cache[filename] is file_index
    finally:
        ast_branches.invalidate_cache(filename)


def test_distinct_cells_with_different_text_each_record(cell_names: list[str]) -> None:
    """Two cells under their own names index their own text."""

    first = _new_cell_name(cell_names)
    second = _new_cell_name(cell_names)
    first_trace, _, _, first_escapes, _ = _trace(_run_cell(first, _CELL_SOURCE))
    second_trace, _, _, second_escapes, _ = _trace(_run_cell(second, _OTHER_CELL_SOURCE))

    assert first_escapes == [] and second_escapes == []
    assert [record.source_file for record in first_trace.graph.conditional_records] == [first]
    assert [record.source_file for record in second_trace.graph.conditional_records] == [second]
    first_index = ast_branches.get_file_index(first)
    second_index = ast_branches.get_file_index(second)
    assert first_index is not None and "x.sum() < 0" in first_index.source
    assert second_index is not None and "x.sum() > 0" in second_index.source


def test_reregistered_name_with_new_text_fails_closed(cell_names: list[str]) -> None:
    """One name with two texts warns once that the source changed and records nothing."""

    name = _new_cell_name(cell_names)
    first_trace, _, _, first_escapes, _ = _trace(_run_cell(name, _CELL_SOURCE))
    assert first_escapes == []
    assert len(list(first_trace.graph.conditional_records)) == 1

    second_trace, _, _, escapes, others = _trace(_run_cell(name, _OTHER_CELL_SOURCE))

    assert list(second_trace.graph.conditional_records) == []
    assert len(escapes) == 1
    changed = [text for text in others if "changed: two different texts" in text]
    assert len(changed) == 1
    assert name in changed[0]
    # From then on the name is unreadable, even when the first text comes back.
    _register(name, _CELL_SOURCE)
    assert ast_branches.get_file_index(name) is None


def test_invalidate_cache_drops_a_cell_index_and_its_pin(cell_names: list[str]) -> None:
    """``invalidate_cache(name)`` clears a cell's digest-keyed index and its pin."""

    name = _new_cell_name(cell_names)
    _register(name, _CELL_SOURCE)
    assert ast_branches.get_file_index(name) is not None
    assert any(isinstance(key, tuple) and key[0] == name for key in ast_branches._file_cache)

    ast_branches.invalidate_cache(name)

    assert not any(isinstance(key, tuple) and key[0] == name for key in ast_branches._file_cache)
    _register(name, _OTHER_CELL_SOURCE)
    file_index = ast_branches.get_file_index(name)
    assert file_index is not None and "x.sum() > 0" in file_index.source
