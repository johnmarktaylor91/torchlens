"""Unit tests for ``torchlens._postprocess.ast_branches``."""

from __future__ import annotations

import ast
import os
import warnings
from collections.abc import Iterator
from pathlib import Path
from textwrap import dedent

import pytest

import torchlens._postprocess.ast_branches as ast_branches
from torchlens._data_classes.func_call_location import FuncCallLocation
from torchlens._postprocess.ast_branches import (
    BoolClassification,
    attribute_op,
    classify_bool,
    get_file_index,
    invalidate_cache,
)


@pytest.fixture(autouse=True)
def clear_ast_branch_cache() -> Iterator[None]:
    """Keep the AST index cache isolated across tests."""

    invalidate_cache()
    yield
    invalidate_cache()


def _write_source(tmp_path: Path, filename: str, source: str) -> Path:
    """Write synthetic source text to a temporary Python file.

    Parameters
    ----------
    tmp_path:
        Pytest temporary directory.
    filename:
        Output filename to create.
    source:
        Python source text.

    Returns
    -------
    Path
        Written file path.
    """

    path = tmp_path / filename
    path.write_text(dedent(source).lstrip("\n"), encoding="utf-8")
    return path


def _find_token(source: str, token: str, occurrence: int = 1) -> tuple[int, int]:
    """Return the line and column for a token occurrence.

    Parameters
    ----------
    source:
        Full source text to search.
    token:
        Token to locate.
    occurrence:
        One-based occurrence count.

    Returns
    -------
    Tuple[int, int]
        One-based line number and zero-based column offset.
    """

    seen = 0
    for line_number, line_text in enumerate(source.splitlines(), start=1):
        start = 0
        while True:
            column = line_text.find(token, start)
            if column < 0:
                break
            seen += 1
            if seen == occurrence:
                return line_number, column
            start = column + len(token)
    raise AssertionError(f"Token {token!r} occurrence {occurrence} not found.")


def _load_source(path: Path) -> str:
    """Read a temporary source file.

    Parameters
    ----------
    path:
        Source file to read.

    Returns
    -------
    str
        File contents.
    """

    return path.read_text(encoding="utf-8")


# Kept by shrink-r4 (per-id subset rule): sole coverage of _postprocess/ast_branches.py.
def test_line_only_assert_wrapping_ternary_fails_closed(tmp_path: Path) -> None:
    """Refuse line-only classification when an ``assert`` wraps a ternary.

    In ``assert left if cond else right`` the assert operand and the ternary
    test are distinct same-line consumption sites; classifying line-only
    evidence as the ternary TEST would wire the asserted VALUE into the
    ternary's conditional record (deep-hunt C1).
    """

    path = _write_source(
        tmp_path,
        "assert_ternary.py",
        """
        def forward():
            assert left_flag if cond_pick else right_flag
            return 1
        """,
    )
    source = _load_source(path)
    line, _col = _find_token(source, "cond_pick")

    classification = classify_bool(str(path), line, None)

    assert classification == BoolClassification("unknown", None, None, None)


# Kept by shrink-r4 (per-id subset rule): sole coverage of _postprocess/ast_branches.py.
def test_evicted_index_rebuild_fails_closed_on_source_drift(tmp_path: Path) -> None:
    """Refuse to rebuild from drifted disk content after LRU eviction.

    Deep-hunt C4, second half: the retained-source guarantee is per
    ``FileIndex``, so LRU eviction used to discard the retained source and
    the next query silently rebuilt from the (possibly edited) file on disk.
    The first-read digest is pinned outside the LRU: a rebuild from matching
    content succeeds, a rebuild from drifted content warns once and fails
    closed to ``None``.
    """

    path = _write_source(
        tmp_path,
        "evicted_case.py",
        """
        def forward():
            if cond_pinned:
                return 1
            return 0
        """,
    )

    first_index = get_file_index(str(path))
    assert first_index is not None

    # Same-content rebuild after eviction succeeds (mtime-only touch).
    ast_branches._file_cache.pop(str(path))
    os.utime(path, ns=(first_index.mtime_ns + 1_000_000,) * 2)
    rebuilt = get_file_index(str(path))
    assert rebuilt is not None
    assert rebuilt is not first_index

    # Drifted-content rebuild after eviction fails closed with one warning.
    ast_branches._file_cache.pop(str(path))
    path.write_text(
        dedent(
            """
            def forward():
                if cond_drifted:
                    return 2
                return 0
            """
        ).lstrip("\n"),
        encoding="utf-8",
    )
    with pytest.warns(UserWarning, match="changed on disk"):
        assert get_file_index(str(path)) is None
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert get_file_index(str(path)) is None

    # Explicit invalidation clears the pin and re-indexes deliberately.
    invalidate_cache(str(path))
    reindexed = get_file_index(str(path))
    assert reindexed is not None
    assert any("cond_drifted" in line for line in reindexed.source.splitlines())


_CALL_INDEX_SOURCE = """
def forward(self, x):
    hidden = torch.relu(self.fc1(x))
    gate = torch.sigmoid(hidden).sum()
    if gate > 0:
        scaled = torch.mul(hidden, torch.tensor(2.0))
    else:
        scaled = torch.sub(hidden, 1.0)
    same, span = torch.add(scaled, 1), torch.add(scaled, 1)
    return torch.cat([scaled, same, span], dim=0)


def helper(y):
    doubled = torch.mul(y, 2)
    return torch.relu(doubled)
"""


def _reference_scope_node(path: Path, qualname: str) -> ast.AST:
    """Re-parse a fixture file and return the named scope's function node.

    Parameters
    ----------
    path:
        Fixture source file.
    qualname:
        Scope qualname to look up.

    Returns
    -------
    ast.AST
        The scope's function node from an independent reference parse.
    """

    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    scopes, scope_nodes = ast_branches._collect_scopes(module)
    for scope, node in zip(scopes, scope_nodes, strict=True):
        if scope.qualname == qualname:
            return node
    raise AssertionError(f"scope {qualname!r} not found in {path}")


# Kept by shrink-r4 (per-id subset rule): sole coverage of _postprocess/ast_branches.py.
def test_scope_call_index_is_reused_across_queries(tmp_path: Path) -> None:
    """Walk each queried scope once, not once per candidate-call query."""

    path = _write_source(tmp_path, "call_index_reuse.py", _CALL_INDEX_SOURCE)
    index = get_file_index(str(path))
    assert index is not None
    forward_scope = next(scope for scope in index.scopes if scope.qualname == "forward")
    helper_scope = next(scope for scope in index.scopes if scope.qualname == "helper")

    reference_forward = _reference_scope_node(path, "forward")
    forward_calls = [node for node in ast.walk(reference_forward) if isinstance(node, ast.Call)]
    assert len(forward_calls) > 3, "Need several call sites to prove reuse."

    walks: list[ast.AST] = []
    real_walk = ast.walk

    def _counting_walk(node: ast.AST) -> Iterator[ast.AST]:
        walks.append(node)
        return iter(list(real_walk(node)))

    first_entries = index.scope_calls(forward_scope)
    try:
        ast.walk = _counting_walk  # type: ignore[assignment]
        for call_node in forward_calls:
            ast_branches._find_candidate_calls(
                index,
                forward_scope,
                call_node.lineno,
                call_node.col_offset,
                ast_branches._call_visible_name(call_node),
            )
        assert walks == [], "The warm index must not re-walk on the first query round."
    finally:
        ast.walk = real_walk  # type: ignore[assignment]

    invalidate_cache()
    cold_index = get_file_index(str(path))
    assert cold_index is not None
    cold_forward = next(scope for scope in cold_index.scopes if scope.qualname == "forward")
    cold_helper = next(scope for scope in cold_index.scopes if scope.qualname == "helper")
    cold_heavy = cold_index._heavy
    assert cold_heavy is not None

    try:
        ast.walk = _counting_walk  # type: ignore[assignment]
        for call_node in forward_calls:
            ast_branches._find_candidate_calls(
                cold_index,
                cold_forward,
                call_node.lineno,
                call_node.col_offset,
                ast_branches._call_visible_name(call_node),
            )
        assert walks == [cold_heavy.scope_nodes[cold_forward.index]]
        ast_branches._find_candidate_calls(cold_index, cold_helper, 1, 0, None)
        assert walks == [
            cold_heavy.scope_nodes[cold_forward.index],
            cold_heavy.scope_nodes[cold_helper.index],
        ]
    finally:
        ast.walk = real_walk  # type: ignore[assignment]

    assert index.scope_calls(forward_scope) is first_entries
    assert helper_scope is not None


def _forward_source_frame(path: Path, index: ast_branches.FileIndex, col: bool) -> FuncCallLocation:
    """Build a source-loading frame at the fixture's ``torch.relu`` call.

    Parameters
    ----------
    path:
        Fixture source file.
    index:
        Cached index for the file (supplies the ``forward`` scope identity).
    col:
        Whether to carry the column offset (point matching) or drop it.

    Returns
    -------
    FuncCallLocation
        Frame resolving to the ``hidden = torch.relu(...)`` call site.
    """

    scope = next(scope for scope in index.scopes if scope.qualname == "forward")
    line, col_offset = _find_token(path.read_text(encoding="utf-8"), "torch.relu")
    return FuncCallLocation(
        file=str(path),
        line_number=line,
        func_name="forward",
        code_firstlineno=scope.code_firstlineno,
        func_qualname=scope.qualname,
        col_offset=col_offset if col else None,
        source_loading_enabled=True,
    )


# Kept by shrink-r4 (per-id subset rule): sole coverage of _postprocess/ast_branches.py.
def test_released_index_fails_closed_when_retained_source_is_corrupt(tmp_path: Path) -> None:
    """Refuse resolution (empty, no raise) when the re-parse cannot align."""

    path = _write_source(tmp_path, "release_corrupt.py", _CALL_INDEX_SOURCE)
    index = get_file_index(str(path))
    assert index is not None
    frame = _forward_source_frame(path, index, col=True)
    ast_branches.release_parsed_asts()

    index.source = "def broken(:"
    assert ast_branches.resolve_var_names([frame], "relu") == []
    assert ast_branches.resolve_arg_expressions([frame], "relu") == []
    assert index._heavy is None

    # Structurally valid but misaligned source also fails closed.
    index.source = "def other():\n    return 1\n"
    assert ast_branches.resolve_var_names([frame], "relu") == []
    assert index._heavy is None


# Kept by shrink-r4 (per-id subset rule): sole coverage of _postprocess/ast_branches.py.
def test_classification_and_attribution_never_need_the_hot_tier(tmp_path: Path) -> None:
    """Answer classify/attribute queries on a released index without re-parsing."""

    path = _write_source(
        tmp_path,
        "release_classify.py",
        """
        def forward(x):
            gate = bool(x.sum() > 0)
            if gate:
                y = torch.relu(x)
            else:
                y = torch.sigmoid(x)
            return y
        """,
    )
    index = get_file_index(str(path))
    assert index is not None
    source = path.read_text(encoding="utf-8")
    gate_line, gate_col = _find_token(source, "x.sum() > 0")
    relu_line, relu_col = _find_token(source, "torch.relu")
    scope = next(scope for scope in index.scopes if scope.qualname == "forward")

    before = classify_bool(str(path), gate_line, gate_col)
    frame = FuncCallLocation(
        file=str(path),
        line_number=relu_line,
        func_name="forward",
        code_firstlineno=scope.code_firstlineno,
        func_qualname=scope.qualname,
        col_offset=relu_col,
        source_loading_enabled=True,
    )
    before_arms = attribute_op([frame])
    assert before_arms, "The fixture op must attribute to the taken arm."

    ast_branches.release_parsed_asts()
    assert classify_bool(str(path), gate_line, gate_col) == before
    assert attribute_op([frame]) == before_arms
    assert index._heavy is None, "Span-only queries must not rebuild the hot tier."
