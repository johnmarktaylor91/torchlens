"""Unit tests for ``torchlens._postprocess.ast_branches``."""

from __future__ import annotations

import ast
import os
import warnings
from collections.abc import Iterator
from dataclasses import replace
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


def _make_frame(
    path: Path,
    line: int,
    col: int | None,
    func_name: str,
    code_firstlineno: int,
    func_qualname: str | None,
) -> FuncCallLocation:
    """Build a manual ``FuncCallLocation`` for attribution tests.

    Parameters
    ----------
    path:
        Source file path.
    line:
        Frame line number.
    col:
        Frame column offset.
    func_name:
        Simple function name.
    code_firstlineno:
        Code object first line number.
    func_qualname:
        Qualified function name, if available.

    Returns
    -------
    FuncCallLocation
        Constructed frame location.
    """

    return FuncCallLocation(
        file=str(path),
        line_number=line,
        func_name=func_name,
        code_firstlineno=code_firstlineno,
        func_qualname=func_qualname,
        col_offset=col,
        source_loading_enabled=False,
    )


def _classify_at_token(path: Path, token: str, occurrence: int = 1) -> BoolClassification:
    """Classify a bool consumer at a located token.

    Parameters
    ----------
    path:
        Source file path.
    token:
        Token whose location should be classified.
    occurrence:
        One-based occurrence count for the token.

    Returns
    -------
    BoolClassification
        Classification result at the token location.
    """

    source = _load_source(path)
    line, col = _find_token(source, token, occurrence)
    return classify_bool(str(path), line, col)


@pytest.mark.parametrize(
    ("filename", "source", "token", "expected_kind"),
    [
        (
            "if_test_case.py",
            """
            def forward():
                if cond_if:
                    return 1
                return 0
            """,
            "cond_if",
            "if_test",
        ),
        (
            "elif_test_case.py",
            """
            def forward():
                if cond_a:
                    return 1
                elif cond_elif:
                    return 2
                return 3
            """,
            "cond_elif",
            "elif_test",
        ),
        (
            "ifexp_case.py",
            """
            def forward():
                result = left_value if cond_ifexp else right_value
                return result
            """,
            "cond_ifexp",
            "ifexp",
        ),
        (
            "assert_case.py",
            """
            def forward():
                assert cond_assert
                return 1
            """,
            "cond_assert",
            "assert",
        ),
        (
            "bool_cast_case.py",
            """
            def forward():
                value = bool(cond_cast)
                return value
            """,
            "cond_cast",
            "bool_cast",
        ),
        (
            "while_case.py",
            """
            def forward():
                while cond_while:
                    return 1
                return 0
            """,
            "cond_while",
            "while",
        ),
        (
            "comp_case.py",
            """
            def forward(xs):
                return [value for value in xs if cond_comp]
            """,
            "cond_comp",
            "comprehension_filter",
        ),
        (
            "match_case.py",
            """
            def forward(value):
                match value:
                    case candidate if cond_guard:
                        return candidate
                return None
            """,
            "cond_guard",
            "match_guard",
        ),
    ],
)
def test_classify_bool_consumer_kinds(
    tmp_path: Path, filename: str, source: str, token: str, expected_kind: str
) -> None:
    """Classify every requested bool-consumer kind from synthetic source."""

    path = _write_source(tmp_path, filename, source)
    classification = _classify_at_token(path, token)

    assert classification.kind == expected_kind
    if expected_kind in {"if_test", "elif_test", "ifexp"}:
        assert classification.conditional_key is not None
    else:
        assert classification.conditional_key is None


def test_classify_bool_cast_inside_if_test_reports_wrapper(tmp_path: Path) -> None:
    """Prefer the outer ``if`` test when a direct ``bool(...)`` wraps it."""

    path = _write_source(
        tmp_path,
        "wrapped_if.py",
        """
        def forward():
            if bool(cond_wrapped):
                return 1
            return 0
        """,
    )

    classification = _classify_at_token(path, "cond_wrapped")

    assert classification.kind == "if_test"
    assert classification.wrapper_kind == "bool_cast"
    assert classification.branch_test_kind == "then"
    assert classification.conditional_key is not None


def test_line_only_arm_body_bool_cast_fails_closed(tmp_path: Path) -> None:
    """Refuse line-only branch classification when the arm body consumes a bool.

    ``if c: keep = bool(d)`` puts TWO bool consumption sites on one line: the
    ``if`` test and the arm-body ``bool(...)`` cast. Line-only evidence cannot
    tell which one consumed a given bool, so classifying either as the branch
    TEST would let the arm-body bool ``d`` cross-wire into the conditional's
    public record (deep-hunt C1). Both must fail closed to ``unknown``.
    """

    path = _write_source(
        tmp_path,
        "arm_body_cast.py",
        """
        def forward():
            if cond_test: keep = bool(other_flag)
            return keep
        """,
    )
    source = _load_source(path)
    line, _col = _find_token(source, "cond_test")

    classification = classify_bool(str(path), line, None)

    assert classification == BoolClassification("unknown", None, None, None)


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


def test_line_only_bool_cast_inside_test_still_classifies(tmp_path: Path) -> None:
    """Keep line-only classification when every consumer nests in the test.

    ``if bool(c):`` has a cast consumer INSIDE the test span: whichever site
    consumed the bool, the branch classification is identical, so the C1
    fail-close guard must not fire.
    """

    path = _write_source(
        tmp_path,
        "wrapped_if_line_only.py",
        """
        def forward():
            if bool(cond_only):
                return 1
            return 0
        """,
    )
    source = _load_source(path)
    line, _col = _find_token(source, "cond_only")

    classification = classify_bool(str(path), line, None)

    assert classification.kind == "if_test"
    assert classification.wrapper_kind == "bool_cast"
    assert classification.branch_test_kind == "then"


def test_classify_unknown_when_no_bool_consumer_contains_point(tmp_path: Path) -> None:
    """Return ``unknown`` when no indexed bool consumer contains the point."""

    path = _write_source(
        tmp_path,
        "unknown_case.py",
        """
        def forward():
            value = plain_expression
            return value
        """,
    )
    source = _load_source(path)
    line, col = _find_token(source, "plain_expression")

    classification = classify_bool(str(path), line, col)

    assert classification == BoolClassification("unknown", None, None, None)


def test_multiline_if_test_resolves_correct_conditional(tmp_path: Path) -> None:
    """Resolve a multiline ``if`` predicate to the owning conditional record."""

    path = _write_source(
        tmp_path,
        "multiline_if.py",
        """
        def forward():
            if (
                cond_multiline
            ):
                return 1
            return 0
        """,
    )
    index = get_file_index(str(path))
    assert index is not None
    assert len(index.conditionals) == 1

    classification = _classify_at_token(path, "cond_multiline")

    assert classification.kind == "if_test"
    assert classification.conditional_key == index.conditionals[0].key
    assert classification.branch_test_kind == "then"


def test_nested_ifs_attribute_operation_with_full_branch_stack(tmp_path: Path) -> None:
    """Attribute an inner operation to both outer and inner branch arms."""

    path = _write_source(
        tmp_path,
        "nested_ifs.py",
        """
        def forward():
            if outer_cond:
                if inner_cond:
                    nested_value = inner_then
                    return nested_value
            return 0
        """,
    )
    index = get_file_index(str(path))
    assert index is not None
    forward_scope = next(scope for scope in index.scopes if scope.qualname == "forward")
    source = _load_source(path)
    line, col = _find_token(source, "inner_then")

    stack = attribute_op(
        [
            _make_frame(
                path=path,
                line=line,
                col=col,
                func_name="forward",
                code_firstlineno=forward_scope.code_firstlineno,
                func_qualname=forward_scope.qualname,
            )
        ]
    )

    assert stack == [
        (index.conditionals[0].key, "then"),
        (index.conditionals[1].key, "then"),
    ]


def test_ternary_then_and_else_attribution_uses_column_offsets(tmp_path: Path) -> None:
    """Attribute same-line ternary arms using the frame column offset."""

    path = _write_source(
        tmp_path,
        "ternary_attr.py",
        """
        def forward():
            result = then_expr if cond_ternary else else_expr
            return result
        """,
    )
    index = get_file_index(str(path))
    assert index is not None
    forward_scope = next(scope for scope in index.scopes if scope.qualname == "forward")
    source = _load_source(path)
    then_line, then_col = _find_token(source, "then_expr")
    else_line, else_col = _find_token(source, "else_expr")
    conditional_key = index.conditionals[0].key

    then_stack = attribute_op(
        [
            _make_frame(
                path,
                then_line,
                then_col,
                "forward",
                forward_scope.code_firstlineno,
                forward_scope.qualname,
            )
        ]
    )
    else_stack = attribute_op(
        [
            _make_frame(
                path,
                else_line,
                else_col,
                "forward",
                forward_scope.code_firstlineno,
                forward_scope.qualname,
            )
        ]
    )

    assert then_stack == [(conditional_key, "then")]
    assert else_stack == [(conditional_key, "else")]


def test_ternary_same_line_without_column_fails_closed(tmp_path: Path) -> None:
    """Drop ambiguous ternary attribution in degraded line-only mode."""

    path = _write_source(
        tmp_path,
        "ternary_fail_closed.py",
        """
        def forward():
            result = then_side if cond_inline else else_side
            return result
        """,
    )
    index = get_file_index(str(path))
    assert index is not None
    forward_scope = next(scope for scope in index.scopes if scope.qualname == "forward")
    source = _load_source(path)
    line, _ = _find_token(source, "then_side")

    stack = attribute_op(
        [
            _make_frame(
                path,
                line,
                None,
                "forward",
                forward_scope.code_firstlineno,
                forward_scope.qualname,
            )
        ]
    )

    assert stack == []


def test_elif_chains_flatten_to_single_conditional_record(tmp_path: Path) -> None:
    """Flatten synthetic ``elif`` nodes into one conditional record."""

    path = _write_source(
        tmp_path,
        "flatten_elif.py",
        """
        def forward():
            if cond_a:
                return "a"
            elif cond_b:
                return "b"
            elif cond_c:
                return "c"
            else:
                return "d"
        """,
    )
    index = get_file_index(str(path))
    assert index is not None

    assert len(index.conditionals) == 1
    record = index.conditionals[0]
    assert record.kind == "if_chain"
    assert set(record.branch_ranges) == {"then", "elif_1", "elif_2", "else"}
    assert set(record.branch_test_spans) == {"then", "elif_1", "elif_2"}


def test_indented_if_inside_else_is_not_flattened_as_elif(tmp_path: Path) -> None:
    """Keep an indented nested ``if`` as a child conditional of the outer ``else``."""

    path = _write_source(
        tmp_path,
        "nested_if_in_else.py",
        """
        def forward():
            if outer_cond:
                return "outer"
            else:
                if inner_cond:
                    return "inner"
        """,
    )
    index = get_file_index(str(path))
    assert index is not None

    assert len(index.conditionals) == 2
    outer, inner = index.conditionals
    assert set(outer.branch_ranges) == {"then", "else"}
    assert inner.parent_conditional_key == outer.key
    assert inner.parent_branch_kind == "else"


def test_scope_resolution_prefers_code_firstlineno_for_same_function_name(tmp_path: Path) -> None:
    """Resolve same-named nested helpers by ``code_firstlineno``."""

    path = _write_source(
        tmp_path,
        "same_name_helpers.py",
        """
        def outer_one():
            def helper():
                if cond_one:
                    return helper_one_value
                return 0
            return helper()

        def outer_two():
            def helper():
                if cond_two:
                    return helper_two_value
                return 0
            return helper()
        """,
    )
    source = _load_source(path)
    index = get_file_index(str(path))
    assert index is not None

    helper_one_scope = next(
        scope for scope in index.scopes if scope.qualname == "outer_one.<locals>.helper"
    )
    helper_two_scope = next(
        scope for scope in index.scopes if scope.qualname == "outer_two.<locals>.helper"
    )
    line_one, col_one = _find_token(source, "helper_one_value")
    line_two, col_two = _find_token(source, "helper_two_value")

    stack_one = attribute_op(
        [
            _make_frame(
                path,
                line_one,
                col_one,
                "helper",
                helper_one_scope.code_firstlineno,
                None,
            )
        ]
    )
    stack_two = attribute_op(
        [
            _make_frame(
                path,
                line_two,
                col_two,
                "helper",
                helper_two_scope.code_firstlineno,
                None,
            )
        ]
    )

    assert stack_one == [(index.conditionals[0].key, "then")]
    assert stack_two == [(index.conditionals[1].key, "then")]


def test_scope_resolution_fails_closed_when_name_match_is_ambiguous(tmp_path: Path) -> None:
    """Skip a frame when fallback scope resolution has multiple candidates."""

    path = _write_source(
        tmp_path,
        "ambiguous_scope.py",
        """
        def outer():
            def helper():
                if cond_ambiguous:
                    return ambiguous_value
                return 0
            return helper()
        """,
    )
    index = get_file_index(str(path))
    assert index is not None

    helper_scope = next(
        scope for scope in index.scopes if scope.qualname == "outer.<locals>.helper"
    )
    index.scopes.append(replace(helper_scope, qualname="shadow.<locals>.helper"))

    source = _load_source(path)
    line, col = _find_token(source, "ambiguous_value")
    stack = attribute_op(
        [
            _make_frame(
                path,
                line,
                col,
                "helper",
                helper_scope.code_firstlineno,
                None,
            )
        ]
    )

    assert stack == []


def test_get_file_index_never_swaps_a_live_index_for_changed_disk_content(
    tmp_path: Path,
) -> None:
    """Keep serving the first-read index when the file changes on disk.

    Deep-hunt C4 (REVIEWED REBASELINE of the old reparse-on-mtime pin): the
    runtime line numbers being attributed come from code objects loaded
    BEFORE the on-disk edit, so re-parsing the new content re-attributes
    ops/bools to whatever now occupies those lines -- silently WRONG, not
    stale. A live index is pinned to its first-read content for the process
    lifetime; ``invalidate_cache`` is the explicit re-index opt-out.
    """

    path = _write_source(
        tmp_path,
        "cache_case.py",
        """
        def forward():
            if cond_before:
                return before_value
            return 0
        """,
    )

    first_index = get_file_index(str(path))
    assert first_index is not None
    assert len(first_index.conditionals) == 1

    path.write_text(
        dedent(
            """
            def forward():
                result = before_value if cond_after else after_value
                return result
            """
        ).lstrip("\n"),
        encoding="utf-8",
    )
    updated_ns = first_index.mtime_ns + 1_000_000
    os.utime(path, ns=(updated_ns, updated_ns))

    second_index = get_file_index(str(path))
    assert second_index is first_index
    assert second_index.conditionals[0].kind == "if_chain"

    invalidate_cache(str(path))
    reindexed = get_file_index(str(path))
    assert reindexed is not None
    assert reindexed is not first_index
    assert reindexed.conditionals[0].kind == "ifexp"


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


def test_invalidate_cache_clears_specific_file_and_global_cache(tmp_path: Path) -> None:
    """Create fresh indexes after explicit cache invalidation."""

    first_path = _write_source(
        tmp_path,
        "invalidate_one.py",
        """
        def forward():
            if cond_first:
                return 1
            return 0
        """,
    )
    second_path = _write_source(
        tmp_path,
        "invalidate_two.py",
        """
        def forward():
            if cond_second:
                return 1
            return 0
        """,
    )

    first_index = get_file_index(str(first_path))
    second_index = get_file_index(str(second_path))
    assert first_index is not None
    assert second_index is not None

    invalidate_cache(str(first_path))
    first_reparsed = get_file_index(str(first_path))
    second_cached = get_file_index(str(second_path))
    assert first_reparsed is not None
    assert second_cached is not None

    assert first_reparsed is not first_index
    assert second_cached is second_index

    invalidate_cache()
    first_global = get_file_index(str(first_path))
    second_global = get_file_index(str(second_path))
    assert first_global is not None
    assert second_global is not None

    assert first_global is not first_reparsed
    assert second_global is not second_cached


def _naive_candidate_calls(
    scope_node: ast.AST, line: int, col: int | None, func_name: str | None
) -> list[ast.Call]:
    """Resolve candidate calls by re-walking the scope, as a reference oracle.

    This mirrors the pre-index implementation exactly: walk the whole scope per
    query, filter by visible name and span containment, stable-sort by span
    width, then reject equal-span ties.

    Parameters
    ----------
    scope_node:
        Function scope to search.
    line:
        Query line number.
    col:
        Query column offset, or ``None`` for line-only matching.
    func_name:
        Captured function name, or ``None`` to accept any callee.

    Returns
    -------
    list[ast.Call]
        Candidate call nodes, innermost first.
    """

    matches = [
        node
        for node in ast.walk(scope_node)
        if isinstance(node, ast.Call)
        and (func_name is None or ast_branches._call_visible_name(node) == func_name)
        and (
            ast_branches._range_contains_line(ast_branches._node_span(node), line)
            if col is None
            else ast_branches._range_contains_point(ast_branches._node_span(node), line, col)
        )
    ]
    if not matches:
        return []
    matches.sort(key=lambda node: ast_branches._source_range_width(ast_branches._node_span(node)))
    if len(matches) > 1 and ast_branches._node_span(matches[0]) == ast_branches._node_span(
        matches[1]
    ):
        return []
    return matches


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


def test_scope_call_index_matches_a_full_walk_in_order(tmp_path: Path) -> None:
    """Project every scope call in ``ast.walk`` order with correct span metadata."""

    path = _write_source(tmp_path, "call_index_order.py", _CALL_INDEX_SOURCE)
    index = get_file_index(str(path))
    assert index is not None
    scope = next(scope for scope in index.scopes if scope.qualname == "forward")

    entries = index.scope_calls(scope)
    reference_node = _reference_scope_node(path, "forward")
    expected = [node for node in ast.walk(reference_node) if isinstance(node, ast.Call)]

    assert [entry.span for entry in entries] == [ast_branches._node_span(node) for node in expected]
    assert [entry.visible_name for entry in entries] == [
        ast_branches._call_visible_name(node) for node in expected
    ]
    assert entries, "The fixture scope must contain call nodes."


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


def test_scope_call_index_matches_naive_rewalk_at_every_call_site(tmp_path: Path) -> None:
    """Return the same candidate nodes a per-query re-walk would return."""

    path = _write_source(tmp_path, "call_index_equivalence.py", _CALL_INDEX_SOURCE)
    index = get_file_index(str(path))
    assert index is not None

    checked = 0
    for scope in index.scopes:
        reference_node = _reference_scope_node(path, scope.qualname)
        call_nodes = [node for node in ast.walk(reference_node) if isinstance(node, ast.Call)]
        queries: list[tuple[int, int | None, str | None]] = []
        for call_node in call_nodes:
            visible_name = ast_branches._call_visible_name(call_node)
            queries.append((call_node.lineno, call_node.col_offset, visible_name))
            queries.append((call_node.lineno, call_node.col_offset, None))
            queries.append((call_node.lineno, None, visible_name))
            queries.append((call_node.lineno, None, None))
            queries.append((call_node.lineno, call_node.col_offset, "not_a_real_callee"))
            queries.append((call_node.end_lineno or call_node.lineno, 0, visible_name))
        for line, col, func_name in queries:
            indexed = ast_branches._find_candidate_calls(index, scope, line, col, func_name)
            naive = _naive_candidate_calls(reference_node, line, col, func_name)
            assert [entry.span for entry in indexed] == [
                ast_branches._node_span(node) for node in naive
            ]
            assert [entry.visible_name for entry in indexed] == [
                ast_branches._call_visible_name(node) for node in naive
            ]
            checked += 1

    assert checked > 50, "Equivalence sweep must cover every fixture call site."


def test_scope_call_index_is_rebuilt_after_source_changes(tmp_path: Path) -> None:
    """Drop the per-scope call index with the file index on explicit re-index.

    REVIEWED REBASELINE (deep-hunt C4): implicit mtime-driven reparse is now
    fail-closed (a live index is pinned to its first-read content), so the
    sanctioned path to a changed file is ``invalidate_cache``. The invariant
    under test is unchanged: the projected per-scope call entries never
    outlive their owning ``FileIndex``.
    """

    path = _write_source(
        tmp_path,
        "call_index_reparse.py",
        """
        def forward(x):
            first = torch.relu(x)
            return first
        """,
    )
    first_index = get_file_index(str(path))
    assert first_index is not None
    first_scope = next(scope for scope in first_index.scopes if scope.qualname == "forward")
    assert [entry.visible_name for entry in first_index.scope_calls(first_scope)] == ["relu"]

    path.write_text(
        dedent(
            """
            def forward(x):
                first = torch.sigmoid(torch.abs(x))
                return first
            """
        ).lstrip("\n"),
        encoding="utf-8",
    )
    invalidate_cache(str(path))

    second_index = get_file_index(str(path))
    assert second_index is not None
    assert second_index is not first_index
    second_scope = next(scope for scope in second_index.scopes if scope.qualname == "forward")
    assert [entry.visible_name for entry in second_index.scope_calls(second_scope)] == [
        "sigmoid",
        "abs",
    ]


def test_resolved_var_names_survive_repeated_scope_queries(tmp_path: Path) -> None:
    """Resolve real assignment names identically on cold and warm call indexes."""

    path = _write_source(tmp_path, "call_index_var_names.py", _CALL_INDEX_SOURCE)
    index = get_file_index(str(path))
    assert index is not None
    scope = next(scope for scope in index.scopes if scope.qualname == "forward")
    source = _load_source(path)

    hidden_line, hidden_col = _find_token(source, "torch.relu")
    tie_line, tie_col = _find_token(source, "torch.add")

    def _source_frame(line: int, col: int) -> FuncCallLocation:
        """Build a source-loading-enabled frame inside the fixture scope."""

        return FuncCallLocation(
            file=str(path),
            line_number=line,
            func_name="forward",
            code_firstlineno=scope.code_firstlineno,
            func_qualname=scope.qualname,
            col_offset=col,
            source_loading_enabled=True,
        )

    hidden_frame = _source_frame(hidden_line, hidden_col)
    tie_frame = _source_frame(tie_line, tie_col)

    assert ast_branches.resolve_var_names([hidden_frame], "relu") == ["hidden"]
    assert ast_branches.resolve_var_names([hidden_frame], "relu") == ["hidden"]
    assert ast_branches.resolve_var_names([hidden_frame], "sigmoid") == []
    assert ast_branches.resolve_arg_expressions([hidden_frame], "relu") == ["self.fc1(x)"]
    assert ast_branches.resolve_arg_expressions([hidden_frame], "relu") == ["self.fc1(x)"]
    assert ast_branches.resolve_var_names([tie_frame], "add") == []


def _cached_index_retains_ast_nodes(index: ast_branches.FileIndex) -> bool:
    """Return whether any ``ast.AST`` object is reachable from a cached index.

    Parameters
    ----------
    index:
        Cached file index to sweep.

    Returns
    -------
    bool
        ``True`` when the index's object graph still holds any ast node.
    """

    seen: set[int] = set()
    stack: list[object] = [index]
    while stack:
        obj = stack.pop()
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        if isinstance(obj, ast.AST):
            return True
        if isinstance(obj, dict):
            stack.extend(obj.keys())
            stack.extend(obj.values())
        elif isinstance(obj, (list, tuple, set, frozenset)):
            stack.extend(obj)
        else:
            for klass in type(obj).__mro__:
                for slot in getattr(klass, "__slots__", ()):
                    try:
                        stack.append(getattr(obj, slot))
                    except AttributeError:
                        continue
            attrs = getattr(obj, "__dict__", None)
            if isinstance(attrs, dict):
                stack.append(attrs)
    return False


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


def test_release_parsed_asts_drops_every_ast_node_but_keeps_resolution(tmp_path: Path) -> None:
    """Release the hot tier; projected resolution answers stay identical."""

    path = _write_source(tmp_path, "release_lifecycle.py", _CALL_INDEX_SOURCE)
    index = get_file_index(str(path))
    assert index is not None
    frame = _forward_source_frame(path, index, col=True)

    before_names = ast_branches.resolve_var_names([frame], "relu")
    before_args = ast_branches.resolve_arg_expressions([frame], "relu")
    assert before_names == ["hidden"]
    assert before_args == ["self.fc1(x)"]

    ast_branches.release_parsed_asts()
    assert index._heavy is None
    assert index._source_lines is None
    assert not _cached_index_retains_ast_nodes(index)

    # Projected scope: identical answers with NO hot tier rebuild needed for
    # the var-name path; the arg path only re-derives the line split.
    assert ast_branches.resolve_var_names([frame], "relu") == before_names
    assert index._heavy is None
    assert ast_branches.resolve_arg_expressions([frame], "relu") == before_args


def test_released_index_reprojects_new_scope_from_retained_source_not_disk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Re-parse an unprojected scope from retained source, never the file."""

    path = _write_source(tmp_path, "release_reparse.py", _CALL_INDEX_SOURCE)
    index = get_file_index(str(path))
    assert index is not None
    frame = _forward_source_frame(path, index, col=True)
    ast_branches.release_parsed_asts()
    assert index._heavy is None

    def _no_disk_read(filename: str) -> str | None:
        raise AssertionError("hot-tier rebuild must not read the file from disk")

    monkeypatch.setattr(ast_branches, "_read_source_file", _no_disk_read)
    assert ast_branches.resolve_var_names([frame], "relu") == ["hidden"]
    assert index._heavy is not None, "The unprojected scope must rebuild the hot tier."

    # And the rebuilt tier releases again cleanly.
    ast_branches.release_parsed_asts()
    assert index._heavy is None
    assert not _cached_index_retains_ast_nodes(index)


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
