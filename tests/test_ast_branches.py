"""Unit tests for ``torchlens.postprocess.ast_branches``."""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path
from textwrap import dedent
from typing import Iterator, Optional, Tuple

import pytest

from torchlens.data_classes.func_call_location import FuncCallLocation
from torchlens.postprocess.ast_branches import (
    BoolClassification,
    ConditionalKey,
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


def _find_token(source: str, token: str, occurrence: int = 1) -> Tuple[int, int]:
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
    col: Optional[int],
    func_name: str,
    code_firstlineno: int,
    code_qualname: Optional[str],
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
    code_qualname:
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
        code_qualname=code_qualname,
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
                code_qualname=forward_scope.qualname,
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


def test_get_file_index_reparses_when_file_mtime_changes(tmp_path: Path) -> None:
    """Reparse a file when its cached modification time no longer matches."""

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
    assert second_index is not None

    assert second_index is not first_index
    assert second_index.mtime_ns == updated_ns
    assert second_index.conditionals[0].kind == "ifexp"


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
