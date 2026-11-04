"""Frame-level conditional branch attribution for one op's call stack.

Split out of ``control_flow`` (phase 5e): these helpers read only the
captured ``FuncCallLocation`` frames and the AST branch index, never an op.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from . import ast_branches

if TYPE_CHECKING:
    from .._data_classes.func_call_location import FuncCallLocation
    from .._data_classes.trace import ConditionalEvent


def _code_context_position_key(
    code_context: list[FuncCallLocation],
) -> tuple[tuple[Any, ...], ...]:
    """Return the frame fields branch attribution reads, as a hashable stack key.

    ``ast_branches.attribute_op`` and the decorator fallback read exactly
    these six fields of each frame, so two stacks with equal keys attribute
    identically within one pass.
    """

    return tuple(
        (
            frame.file,
            frame.code_firstlineno,
            frame.func_name,
            frame.func_qualname,
            frame.line_number,
            frame.col_offset,
        )
        for frame in code_context
    )


def _translate_conditional_stack(
    code_context: list[FuncCallLocation],
    events_by_key: dict[ast_branches.ConditionalKey, ConditionalEvent],
) -> list[tuple[int, str]]:
    """Translate a structural AST branch stack into dense conditional IDs.

    Parameters
    ----------
    code_context:
        Captured runtime call stack for one operation.
    events_by_key:
        Structural-to-dense conditional event lookup created in phase 5c.

    Returns
    -------
    List[Tuple[int, str]]
        Dense ``(conditional_id, branch_kind)`` pairs ordered outer-to-inner.
        Structural keys that were never materialized are dropped.
    """

    translated_stack: list[tuple[int, str]] = []
    for conditional_key, branch_kind in _attribute_op_with_scope_fallback(code_context):
        if conditional_key not in events_by_key:
            continue
        translated_stack.append((events_by_key[conditional_key].id, branch_kind))
    return translated_stack


def _attribute_op_with_scope_fallback(
    code_context: list[FuncCallLocation],
) -> list[tuple[ast_branches.ConditionalKey, str]]:
    """Attribute an op, retrying decorated-function scope resolution when needed.

    Parameters
    ----------
    code_context:
        Captured runtime call stack for one operation.

    Returns
    -------
    List[Tuple[ast_branches.ConditionalKey, str]]
        Structural ``(conditional_key, branch_kind)`` pairs ordered
        outer-to-inner.
    """

    branch_stack = ast_branches.attribute_op(code_context)
    if branch_stack:
        return branch_stack

    fallback_branch_stack: list[tuple[ast_branches.ConditionalKey, str]] = []
    for frame in code_context:
        file_index = ast_branches.get_file_index(frame.file)
        if file_index is None:
            continue

        scope = _resolve_scope_with_decorator_fallback(file_index, frame)
        if scope is None:
            continue

        for conditional_key, branch_kind, _depth in scope.query_intervals(
            frame.line_number,
            frame.col_offset,
        ):
            entry = (conditional_key, branch_kind)
            if not fallback_branch_stack or fallback_branch_stack[-1] != entry:
                fallback_branch_stack.append(entry)

    return fallback_branch_stack


def _resolve_scope_with_decorator_fallback(
    file_index: ast_branches.FileIndex,
    frame: FuncCallLocation,
) -> ast_branches.ScopeEntry | None:
    """Resolve a frame, tolerating decorator-line ``co_firstlineno`` offsets.

    Parameters
    ----------
    file_index:
        AST index for the frame's source file.
    frame:
        Runtime frame metadata captured in ``FuncCallLocation`` form.

    Returns
    -------
    Optional[ast_branches.ScopeEntry]
        Resolved scope entry, or ``None`` when the fallback still fails closed.
    """

    resolved_scope = file_index.resolve_scope(
        code_firstlineno=frame.code_firstlineno,
        func_name=frame.func_name,
        func_qualname=frame.func_qualname,
    )
    if resolved_scope is not None:
        return resolved_scope

    candidate_firstlineno = frame.code_firstlineno + 1
    if frame.func_qualname is not None:
        qualname_matches = [
            scope
            for scope in file_index.scopes
            if scope.qualname == frame.func_qualname
            and scope.code_firstlineno == candidate_firstlineno
        ]
        if len(qualname_matches) == 1:
            return qualname_matches[0]
        return None

    name_matches = [
        scope
        for scope in file_index.scopes
        if scope.func_name == frame.func_name and scope.code_firstlineno == candidate_firstlineno
    ]
    if len(name_matches) == 1:
        return name_matches[0]
    return None
