"""AST indexing for boolean-context classification and branch attribution.

This module parses Python source files into a lightweight, cached index that can:

1. Classify where a captured scalar-bool operation occurred (``if`` test,
   ``elif`` test, ternary test, ``assert``, ``while``, comprehension filter,
   ``match`` guard, or standalone ``bool(...)`` cast).
2. Attribute an arbitrary operation to the enclosing conditional branch arms
   for each stack frame in ``FuncCallLocation``.

The index is structural and process-local. Dense conditional IDs are assigned
later by postprocess integration; this module works only with ``ConditionalKey``.
"""

from __future__ import annotations

import ast
import os
import tokenize
from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional, Sequence, Tuple, TypeAlias, cast

from torchlens.data_classes.func_call_location import FuncCallLocation

ConditionalKey: TypeAlias = Tuple[str, int, int, int]
SourceRange: TypeAlias = Tuple[int, int, int, int]
LineSpan: TypeAlias = Tuple[int, int]
FunctionNode: TypeAlias = ast.FunctionDef | ast.AsyncFunctionDef

_BRANCH_CONSUMER_KINDS = {"if_test", "elif_test", "ifexp"}
_file_cache: Dict[str, "FileIndex"] = {}


@dataclass(frozen=True)
class BoolClassification:
    """Classification result for a terminal boolean operation.

    Attributes
    ----------
    kind:
        The enclosing boolean consumer kind, or ``"unknown"``.
    wrapper_kind:
        Wrapper classification when a direct ``bool(...)`` cast is nested inside
        a branch-participating consumer. Otherwise ``None``.
    conditional_key:
        Structural key for the owning conditional when ``kind`` is
        branch-participating. Otherwise ``None``.
    branch_test_kind:
        Branch discriminator for branch-participating kinds. ``"then"`` is used
        for top-level ``if`` tests and ternary tests; flattened ``elif`` tests
        use ``"elif_N"``.
    """

    kind: str
    wrapper_kind: Optional[str]
    conditional_key: Optional[ConditionalKey]
    branch_test_kind: Optional[str]


@dataclass(frozen=True)
class ConditionalRecord:
    """Structural representation of a conditional found in source code.

    Attributes
    ----------
    key:
        Structural conditional key ``(file, func_firstlineno, if_lineno, if_col)``.
    kind:
        Conditional kind. Either ``"if_chain"`` or ``"ifexp"``.
    source_file:
        Source file that owns the conditional.
    function_span:
        Inclusive line span for the owning function scope.
    if_stmt_span:
        Inclusive line span for the full ``if`` chain or ternary node.
    test_span:
        Full source range for the conditional test expression.
    branch_ranges:
        Mapping from branch kind to full source range for that arm.
    branch_test_spans:
        Mapping from branch kind to the test expression range relevant for that
        arm. ``if`` chains use ``"then"`` and any flattened ``"elif_N"`` keys.
        Ternaries use ``"then"`` only.
    nesting_depth:
        Lexical conditional nesting depth within the owning function scope.
    parent_conditional_key:
        Structural key of the lexically enclosing conditional, if any.
    parent_branch_kind:
        Branch kind inside the parent conditional that contains this record.
    """

    key: ConditionalKey
    kind: Literal["if_chain", "ifexp"]
    source_file: str
    function_span: LineSpan
    if_stmt_span: LineSpan
    test_span: SourceRange
    branch_ranges: Dict[str, SourceRange]
    branch_test_spans: Dict[str, SourceRange]
    nesting_depth: int
    parent_conditional_key: Optional[ConditionalKey]
    parent_branch_kind: Optional[str]


@dataclass(frozen=True)
class BoolConsumer:
    """AST node that consumes a truthy/falsy value.

    Attributes
    ----------
    kind:
        Consumer kind.
    span:
        Source range that contains the consumed boolean expression.
    depth:
        AST ancestor depth for innermost-first ordering.
    conditional_key:
        Structural conditional key when this consumer is tied to an ``if`` chain
        or ternary test. Otherwise ``None``.
    branch_test_kind:
        Branch discriminator for branch test consumers. Otherwise ``None``.
    """

    kind: str
    span: SourceRange
    depth: int
    conditional_key: Optional[ConditionalKey]
    branch_test_kind: Optional[str]


@dataclass(frozen=True)
class BranchInterval:
    """Branch-arm interval used by point queries inside a function scope.

    Attributes
    ----------
    conditional_key:
        Structural key of the owning conditional.
    branch_kind:
        Branch kind for this interval.
    nesting_depth:
        Lexical conditional depth of the owning conditional.
    span:
        Full source range for the branch arm.
    """

    conditional_key: ConditionalKey
    branch_kind: str
    nesting_depth: int
    span: SourceRange


@dataclass
class ScopeEntry:
    """Single function scope indexed within a source file.

    Attributes
    ----------
    code_firstlineno:
        First line number for the function code object.
    func_name:
        Simple function name.
    qualname:
        Qualified name matching ``code.co_qualname`` semantics where possible.
    node:
        Owning AST function node.
    span:
        Inclusive line span for the function body.
    conditionals:
        Conditional records defined inside the scope.
    branch_intervals:
        Branch-arm intervals used for operation attribution.
    """

    code_firstlineno: int
    func_name: str
    qualname: str
    node: FunctionNode
    span: LineSpan
    conditionals: List[ConditionalRecord] = field(default_factory=list)
    branch_intervals: List[BranchInterval] = field(default_factory=list)

    def query_intervals(
        self, line: int, col: Optional[int]
    ) -> List[Tuple[ConditionalKey, str, int]]:
        """Return all branch arms containing a point in this scope.

        Parameters
        ----------
        line:
            Source line number for the query point.
        col:
            Source column for the query point. ``None`` activates degraded
            line-only matching.

        Returns
        -------
        List[Tuple[ConditionalKey, str, int]]
            Matching ``(conditional_key, branch_kind, nesting_depth)`` tuples
            sorted outermost-to-innermost. Ambiguous same-line matches for a
            single conditional are dropped in degraded mode.
        """

        matches: List[BranchInterval] = []
        for interval in self.branch_intervals:
            if col is None:
                if _range_contains_line(interval.span, line):
                    matches.append(interval)
            elif _range_contains_point(interval.span, line, col):
                matches.append(interval)

        if col is None:
            grouped: Dict[ConditionalKey, List[BranchInterval]] = {}
            for interval in matches:
                grouped.setdefault(interval.conditional_key, []).append(interval)

            filtered: List[BranchInterval] = []
            for intervals in grouped.values():
                if len(intervals) == 1:
                    filtered.extend(intervals)
            matches = filtered

        matches.sort(key=lambda item: item.nesting_depth)
        return [
            (interval.conditional_key, interval.branch_kind, interval.nesting_depth)
            for interval in matches
        ]


@dataclass
class FileIndex:
    """Parsed AST index for one source file.

    Attributes
    ----------
    filename:
        Source filename for the parsed module.
    mtime_ns:
        File modification timestamp used for cache invalidation.
    module:
        Parsed AST module.
    scopes:
        All function scopes discovered in the file.
    conditionals:
        Flattened list of conditionals found across all scopes.
    bool_consumers:
        Flattened list of all boolean consumers found across all scopes.
    """

    filename: str
    mtime_ns: int
    module: ast.Module
    scopes: List[ScopeEntry]
    conditionals: List[ConditionalRecord]
    bool_consumers: List[BoolConsumer]

    def resolve_scope(
        self, code_firstlineno: int, func_name: str, code_qualname: Optional[str]
    ) -> Optional[ScopeEntry]:
        """Resolve a runtime frame to a single indexed function scope.

        Parameters
        ----------
        code_firstlineno:
            ``co_firstlineno`` from the runtime frame.
        func_name:
            Simple function name from the runtime frame.
        code_qualname:
            Qualified function name from the runtime frame, when available.

        Returns
        -------
        Optional[ScopeEntry]
            Resolved scope entry, or ``None`` when the D14 fail-closed rules
            require the frame to be skipped.
        """

        if code_qualname is not None:
            for scope in self.scopes:
                if scope.code_firstlineno == code_firstlineno and scope.qualname == code_qualname:
                    return scope
            return None

        candidates = [
            scope
            for scope in self.scopes
            if scope.code_firstlineno == code_firstlineno and scope.func_name == func_name
        ]
        if len(candidates) == 1:
            return candidates[0]
        return None


def get_file_index(filename: str) -> Optional[FileIndex]:
    """Return a cached AST index for ``filename``.

    Parameters
    ----------
    filename:
        Source file to parse and index.

    Returns
    -------
    Optional[FileIndex]
        Cached or newly parsed file index, or ``None`` when the file cannot be
        read, parsed, or stated.
    """

    try:
        mtime_ns = os.stat(filename).st_mtime_ns
    except OSError:
        return None

    cached = _file_cache.get(filename)
    if cached is not None and cached.mtime_ns == mtime_ns:
        return cached

    source = _read_source_file(filename)
    if source is None:
        return None

    try:
        module = ast.parse(source, filename=filename)
    except SyntaxError:
        return None

    parent_map = _build_parent_map(module)
    scopes = _collect_scopes(module)
    conditionals: List[ConditionalRecord] = []
    bool_consumers: List[BoolConsumer] = []

    for scope in scopes:
        indexer = _ScopeIndexer(
            filename=filename,
            scope=scope,
            parent_map=parent_map,
            all_conditionals=conditionals,
            all_bool_consumers=bool_consumers,
        )
        indexer.index_scope()

    file_index = FileIndex(
        filename=filename,
        mtime_ns=mtime_ns,
        module=module,
        scopes=scopes,
        conditionals=conditionals,
        bool_consumers=bool_consumers,
    )
    _file_cache[filename] = file_index
    return file_index


def classify_bool(filename: str, line: int, col: Optional[int] = None) -> BoolClassification:
    """Classify a scalar-bool operation by its enclosing AST consumer.

    Parameters
    ----------
    filename:
        Source file containing the operation.
    line:
        Source line number for the operation.
    col:
        Source column number for the operation. ``None`` activates degraded
        line-only matching.

    Returns
    -------
    BoolClassification
        Classification result following the D3/D18 rules.
    """

    file_index = get_file_index(filename)
    if file_index is None:
        return BoolClassification("unknown", None, None, None)

    consumers: List[BoolConsumer] = []
    for consumer in file_index.bool_consumers:
        if col is None:
            if _range_contains_line(consumer.span, line):
                consumers.append(consumer)
        elif _range_contains_point(consumer.span, line, col):
            consumers.append(consumer)

    consumers.sort(
        key=lambda item: (item.depth, 1 if item.kind == "bool_cast" else 0),
        reverse=True,
    )

    saw_bool_cast = False
    for consumer in consumers:
        if consumer.kind == "bool_cast":
            saw_bool_cast = True
            continue
        if consumer.kind in _BRANCH_CONSUMER_KINDS:
            return BoolClassification(
                kind=consumer.kind,
                wrapper_kind="bool_cast" if saw_bool_cast else None,
                conditional_key=consumer.conditional_key,
                branch_test_kind=consumer.branch_test_kind,
            )
        return BoolClassification(consumer.kind, None, None, None)

    if saw_bool_cast:
        return BoolClassification("bool_cast", None, None, None)
    return BoolClassification("unknown", None, None, None)


def attribute_op(func_call_stack: List[FuncCallLocation]) -> List[Tuple[ConditionalKey, str]]:
    """Attribute an operation to enclosing conditional branch arms.

    Parameters
    ----------
    func_call_stack:
        Runtime call stack, ordered shallowest-to-deepest.

    Returns
    -------
    List[Tuple[ConditionalKey, str]]
        Concatenated branch stack across frames, with adjacent duplicate entries
        removed. Each tuple is ``(conditional_key, branch_kind)``.
    """

    branch_stack: List[Tuple[ConditionalKey, str]] = []
    for frame in func_call_stack:
        file_index = get_file_index(frame.file)
        if file_index is None:
            continue

        scope = file_index.resolve_scope(
            code_firstlineno=frame.code_firstlineno,
            func_name=frame.func_name,
            code_qualname=frame.code_qualname,
        )
        if scope is None:
            continue

        for conditional_key, branch_kind, _depth in scope.query_intervals(
            frame.line_number, frame.col_offset
        ):
            entry = (conditional_key, branch_kind)
            if not branch_stack or branch_stack[-1] != entry:
                branch_stack.append(entry)

    return branch_stack


def invalidate_cache(filename: Optional[str] = None) -> None:
    """Invalidate cached AST indexes.

    Parameters
    ----------
    filename:
        Specific filename to invalidate. When ``None``, the entire cache is
        cleared.
    """

    if filename is None:
        _file_cache.clear()
    else:
        _file_cache.pop(filename, None)


def _read_source_file(filename: str) -> Optional[str]:
    """Read a source file using its declared encoding.

    Parameters
    ----------
    filename:
        Source file to read.

    Returns
    -------
    Optional[str]
        File contents, or ``None`` if the file cannot be read.
    """

    try:
        with tokenize.open(filename) as handle:
            return handle.read()
    except OSError:
        return None


def _build_parent_map(module: ast.Module) -> Dict[ast.AST, ast.AST]:
    """Build a parent map for every AST node in a module.

    Parameters
    ----------
    module:
        Parsed module to index.

    Returns
    -------
    Dict[ast.AST, ast.AST]
        Mapping from child node to direct parent node.
    """

    parent_map: Dict[ast.AST, ast.AST] = {}
    for parent in ast.walk(module):
        for child in ast.iter_child_nodes(parent):
            parent_map[child] = parent
    return parent_map


def _collect_scopes(module: ast.Module) -> List[ScopeEntry]:
    """Collect all function scopes in a module with runtime-style qualnames.

    Parameters
    ----------
    module:
        Parsed module to inspect.

    Returns
    -------
    List[ScopeEntry]
        Collected function scopes in source order.
    """

    scopes: List[ScopeEntry] = []
    _collect_scopes_from_node(module, None, "module", scopes)
    return scopes


def _collect_scopes_from_node(
    node: ast.AST,
    qualname_prefix: Optional[str],
    container_kind: Literal["module", "class", "function"],
    scopes: List[ScopeEntry],
) -> None:
    """Recursively collect function scopes from a node.

    Parameters
    ----------
    node:
        Current AST node.
    qualname_prefix:
        Qualname prefix for child definitions.
    container_kind:
        Container type for qualname composition.
    scopes:
        Output list to populate.
    """

    for child in ast.iter_child_nodes(node):
        if isinstance(child, ast.ClassDef):
            child_prefix = _compose_child_qualname(
                qualname_prefix=qualname_prefix,
                container_kind=container_kind,
                child_name=child.name,
            )
            _collect_scopes_from_node(child, child_prefix, "class", scopes)
            continue

        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            qualname = _compose_child_qualname(
                qualname_prefix=qualname_prefix,
                container_kind=container_kind,
                child_name=child.name,
            )
            scopes.append(
                ScopeEntry(
                    code_firstlineno=child.lineno,
                    func_name=child.name,
                    qualname=qualname,
                    node=child,
                    span=(child.lineno, _end_lineno(child)),
                )
            )
            _collect_scopes_from_node(child, qualname, "function", scopes)
            continue

        _collect_scopes_from_node(child, qualname_prefix, container_kind, scopes)


def _compose_child_qualname(
    qualname_prefix: Optional[str],
    container_kind: Literal["module", "class", "function"],
    child_name: str,
) -> str:
    """Compose a child qualname using Python code-object conventions.

    Parameters
    ----------
    qualname_prefix:
        Existing qualname prefix, if any.
    container_kind:
        Type of container holding the child definition.
    child_name:
        Child function or class name.

    Returns
    -------
    str
        Composed qualified name.
    """

    if qualname_prefix is None:
        return child_name
    if container_kind == "function":
        return f"{qualname_prefix}.<locals>.{child_name}"
    return f"{qualname_prefix}.{child_name}"


def _range_contains_point(span: SourceRange, line: int, col: int) -> bool:
    """Return whether a source range contains a point.

    Parameters
    ----------
    span:
        Source range ``(start_line, start_col, end_line, end_col)``.
    line:
        Query line number.
    col:
        Query column number.

    Returns
    -------
    bool
        ``True`` when the point falls within the span.
    """

    return (span[0], span[1]) <= (line, col) <= (span[2], span[3])


def _range_contains_line(span: SourceRange, line: int) -> bool:
    """Return whether a source range contains a line in degraded mode.

    Parameters
    ----------
    span:
        Source range ``(start_line, start_col, end_line, end_col)``.
    line:
        Query line number.

    Returns
    -------
    bool
        ``True`` when the query line falls within the span's line bounds.
    """

    return span[0] <= line <= span[2]


def _node_span(node: ast.AST) -> SourceRange:
    """Return the source span for an AST node.

    Parameters
    ----------
    node:
        AST node with location information.

    Returns
    -------
    SourceRange
        Full source range for the node.
    """

    return (_lineno(node), _col_offset(node), _end_lineno(node), _end_col_offset(node))


def _statement_list_span(statements: List[ast.stmt]) -> SourceRange:
    """Return the bounding source range for a non-empty statement list.

    Parameters
    ----------
    statements:
        Statement list to span.

    Returns
    -------
    SourceRange
        Bounding range from the first statement start to the last statement end.
    """

    first = statements[0]
    last = statements[-1]
    return (_lineno(first), _col_offset(first), _end_lineno(last), _end_col_offset(last))


def _lineno(node: ast.AST) -> int:
    """Return a node's starting line number.

    Parameters
    ----------
    node:
        AST node with location information.

    Returns
    -------
    int
        Starting line number.
    """

    return cast(int, getattr(node, "lineno"))


def _col_offset(node: ast.AST) -> int:
    """Return a node's starting column offset.

    Parameters
    ----------
    node:
        AST node with location information.

    Returns
    -------
    int
        Starting column offset.
    """

    return cast(int, getattr(node, "col_offset"))


def _end_lineno(node: ast.AST) -> int:
    """Return a node's ending line number.

    Parameters
    ----------
    node:
        AST node with location information.

    Returns
    -------
    int
        Ending line number, falling back to ``lineno`` when absent.
    """

    end_lineno = getattr(node, "end_lineno", None)
    if end_lineno is None:
        return _lineno(node)
    return cast(int, end_lineno)


def _end_col_offset(node: ast.AST) -> int:
    """Return a node's ending column offset.

    Parameters
    ----------
    node:
        AST node with location information.

    Returns
    -------
    int
        Ending column offset, falling back to ``col_offset`` when absent.
    """

    end_col_offset = getattr(node, "end_col_offset", None)
    if end_col_offset is None:
        return _col_offset(node)
    return cast(int, end_col_offset)


def _ast_depth(node: ast.AST, parent_map: Dict[ast.AST, ast.AST]) -> int:
    """Return AST ancestor depth for ordering nested consumers.

    Parameters
    ----------
    node:
        Node to measure.
    parent_map:
        Full parent map for the parsed module.

    Returns
    -------
    int
        Number of ancestors between ``node`` and the module root.
    """

    depth = 0
    current = node
    while current in parent_map:
        current = parent_map[current]
        depth += 1
    return depth


def _is_direct_bool_call(node: ast.AST) -> bool:
    """Return whether a node is a direct ``bool(...)`` cast call.

    Parameters
    ----------
    node:
        AST node to inspect.

    Returns
    -------
    bool
        ``True`` when the node is a direct call to the built-in ``bool``.
    """

    return (
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "bool"
        and len(node.args) == 1
        and not node.keywords
    )


def _is_match_case_node(node: ast.AST) -> bool:
    """Return whether a node is a structural pattern-matching case node.

    Parameters
    ----------
    node:
        AST node to inspect.

    Returns
    -------
    bool
        ``True`` when the node is a ``match_case`` object.
    """

    return type(node).__name__ == "match_case"


class _ScopeIndexer:
    """Per-scope conditional and bool-consumer index builder."""

    def __init__(
        self,
        filename: str,
        scope: ScopeEntry,
        parent_map: Dict[ast.AST, ast.AST],
        all_conditionals: List[ConditionalRecord],
        all_bool_consumers: List[BoolConsumer],
    ) -> None:
        """Initialize the scope indexer.

        Parameters
        ----------
        filename:
            Source filename for the owning module.
        scope:
            Function scope being indexed.
        parent_map:
            Parent map for the whole parsed module.
        all_conditionals:
            Shared output list for flattened conditionals.
        all_bool_consumers:
            Shared output list for flattened bool consumers.
        """

        self.filename = filename
        self.scope = scope
        self.parent_map = parent_map
        self.all_conditionals = all_conditionals
        self.all_bool_consumers = all_bool_consumers

    def index_scope(self) -> None:
        """Index all conditionals and bool consumers for the scope."""

        self._walk_nodes(self.scope.node.body, None, None, 0)

    def _walk_nodes(
        self,
        nodes: Sequence[ast.AST],
        parent_conditional_key: Optional[ConditionalKey],
        parent_branch_kind: Optional[str],
        nesting_depth: int,
    ) -> None:
        """Walk a list of AST nodes under shared conditional context.

        Parameters
        ----------
        nodes:
            Nodes to traverse.
        parent_conditional_key:
            Structural key of the enclosing conditional, if any.
        parent_branch_kind:
            Enclosing branch kind, if any.
        nesting_depth:
            Current conditional nesting depth within the scope.
        """

        for node in nodes:
            self._walk_node(node, parent_conditional_key, parent_branch_kind, nesting_depth)

    def _walk_node(
        self,
        node: ast.AST,
        parent_conditional_key: Optional[ConditionalKey],
        parent_branch_kind: Optional[str],
        nesting_depth: int,
    ) -> None:
        """Walk one AST node under shared conditional context.

        Parameters
        ----------
        node:
            AST node to traverse.
        parent_conditional_key:
            Structural key of the enclosing conditional, if any.
        parent_branch_kind:
            Enclosing branch kind, if any.
        nesting_depth:
            Current conditional nesting depth within the scope.
        """

        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            return

        if isinstance(node, ast.If):
            self._handle_if(node, parent_conditional_key, parent_branch_kind, nesting_depth)
            return

        if isinstance(node, ast.IfExp):
            self._handle_ifexp(node, parent_conditional_key, parent_branch_kind, nesting_depth)
            return

        if isinstance(node, ast.While):
            self._add_bool_consumer("while", node.test, None, None)

        if isinstance(node, ast.Assert):
            self._add_bool_consumer("assert", node.test, None, None)

        if isinstance(node, ast.comprehension):
            for if_expr in node.ifs:
                self._add_bool_consumer("comprehension_filter", if_expr, None, None)

        if _is_match_case_node(node):
            guard = getattr(node, "guard", None)
            if isinstance(guard, ast.AST):
                self._add_bool_consumer("match_guard", guard, None, None)

        if _is_direct_bool_call(node):
            self._add_bool_consumer("bool_cast", node, None, None)

        for child in ast.iter_child_nodes(node):
            self._walk_node(child, parent_conditional_key, parent_branch_kind, nesting_depth)

    def _handle_if(
        self,
        node: ast.If,
        parent_conditional_key: Optional[ConditionalKey],
        parent_branch_kind: Optional[str],
        nesting_depth: int,
    ) -> None:
        """Index a flattened ``if``/``elif``/``else`` chain.

        Parameters
        ----------
        node:
            Top-level ``if`` node.
        parent_conditional_key:
            Structural key of the enclosing conditional, if any.
        parent_branch_kind:
            Enclosing branch kind, if any.
        nesting_depth:
            Current conditional nesting depth within the scope.
        """

        if self._is_synthetic_elif(node):
            self._walk_node(node.test, parent_conditional_key, parent_branch_kind, nesting_depth)
            self._walk_nodes(node.body, parent_conditional_key, parent_branch_kind, nesting_depth)
            self._walk_nodes(node.orelse, parent_conditional_key, parent_branch_kind, nesting_depth)
            return

        flattened_elifs, terminal_else = self._flatten_elif_chain(node)
        record = self._build_if_record(
            node=node,
            flattened_elifs=flattened_elifs,
            terminal_else=terminal_else,
            parent_conditional_key=parent_conditional_key,
            parent_branch_kind=parent_branch_kind,
            nesting_depth=nesting_depth,
        )
        self._register_conditional(record)

        self._add_bool_consumer("if_test", node.test, record.key, "then")
        self._walk_node(node.test, parent_conditional_key, parent_branch_kind, nesting_depth)
        self._walk_nodes(node.body, record.key, "then", nesting_depth + 1)

        for index, elif_node in enumerate(flattened_elifs, start=1):
            branch_kind = f"elif_{index}"
            self._add_bool_consumer("elif_test", elif_node.test, record.key, branch_kind)
            self._walk_node(
                elif_node.test, parent_conditional_key, parent_branch_kind, nesting_depth
            )
            self._walk_nodes(elif_node.body, record.key, branch_kind, nesting_depth + 1)

        if terminal_else:
            self._walk_nodes(terminal_else, record.key, "else", nesting_depth + 1)

    def _handle_ifexp(
        self,
        node: ast.IfExp,
        parent_conditional_key: Optional[ConditionalKey],
        parent_branch_kind: Optional[str],
        nesting_depth: int,
    ) -> None:
        """Index a ternary conditional expression.

        Parameters
        ----------
        node:
            Ternary expression node.
        parent_conditional_key:
            Structural key of the enclosing conditional, if any.
        parent_branch_kind:
            Enclosing branch kind, if any.
        nesting_depth:
            Current conditional nesting depth within the scope.
        """

        key: ConditionalKey = (
            self.filename,
            self.scope.code_firstlineno,
            node.lineno,
            node.col_offset,
        )
        record = ConditionalRecord(
            key=key,
            kind="ifexp",
            source_file=self.filename,
            function_span=self.scope.span,
            if_stmt_span=(node.lineno, _end_lineno(node)),
            test_span=_node_span(node.test),
            branch_ranges={
                "then": _node_span(node.body),
                "else": _node_span(node.orelse),
            },
            branch_test_spans={"then": _node_span(node.test)},
            nesting_depth=nesting_depth,
            parent_conditional_key=parent_conditional_key,
            parent_branch_kind=parent_branch_kind,
        )
        self._register_conditional(record)

        self._add_bool_consumer("ifexp", node.test, key, "then")
        self._walk_node(node.test, parent_conditional_key, parent_branch_kind, nesting_depth)
        self._walk_node(node.body, key, "then", nesting_depth + 1)
        self._walk_node(node.orelse, key, "else", nesting_depth + 1)

    def _flatten_elif_chain(self, node: ast.If) -> Tuple[List[ast.If], List[ast.stmt]]:
        """Flatten a synthetic ``elif`` chain rooted at ``node``.

        Parameters
        ----------
        node:
            Top-level ``if`` node.

        Returns
        -------
        Tuple[List[ast.If], List[ast.stmt]]
            Flattened synthetic ``elif`` nodes and terminal ``else`` statements.
        """

        flattened_elifs: List[ast.If] = []
        terminal_else: List[ast.stmt] = []
        current = node
        while current.orelse:
            if len(current.orelse) == 1 and isinstance(current.orelse[0], ast.If):
                next_if = current.orelse[0]
                flattened_elifs.append(next_if)
                current = next_if
                continue
            terminal_else = list(current.orelse)
            break
        return flattened_elifs, terminal_else

    def _build_if_record(
        self,
        node: ast.If,
        flattened_elifs: List[ast.If],
        terminal_else: List[ast.stmt],
        parent_conditional_key: Optional[ConditionalKey],
        parent_branch_kind: Optional[str],
        nesting_depth: int,
    ) -> ConditionalRecord:
        """Build the conditional record for a flattened ``if`` chain.

        Parameters
        ----------
        node:
            Top-level ``if`` node.
        flattened_elifs:
            Synthetic ``elif`` nodes flattened into the record.
        terminal_else:
            Terminal ``else`` statement list, if any.
        parent_conditional_key:
            Structural key of the enclosing conditional, if any.
        parent_branch_kind:
            Enclosing branch kind, if any.
        nesting_depth:
            Current conditional nesting depth within the scope.

        Returns
        -------
        ConditionalRecord
            Built conditional record.
        """

        key: ConditionalKey = (
            self.filename,
            self.scope.code_firstlineno,
            node.lineno,
            node.col_offset,
        )
        branch_ranges: Dict[str, SourceRange] = {"then": _statement_list_span(node.body)}
        branch_test_spans: Dict[str, SourceRange] = {"then": _node_span(node.test)}

        for index, elif_node in enumerate(flattened_elifs, start=1):
            branch_kind = f"elif_{index}"
            branch_ranges[branch_kind] = _statement_list_span(elif_node.body)
            branch_test_spans[branch_kind] = _node_span(elif_node.test)

        if terminal_else:
            branch_ranges["else"] = _statement_list_span(terminal_else)

        return ConditionalRecord(
            key=key,
            kind="if_chain",
            source_file=self.filename,
            function_span=self.scope.span,
            if_stmt_span=(node.lineno, _end_lineno(node)),
            test_span=_node_span(node.test),
            branch_ranges=branch_ranges,
            branch_test_spans=branch_test_spans,
            nesting_depth=nesting_depth,
            parent_conditional_key=parent_conditional_key,
            parent_branch_kind=parent_branch_kind,
        )

    def _register_conditional(self, record: ConditionalRecord) -> None:
        """Register a conditional record with scope-level interval data.

        Parameters
        ----------
        record:
            Conditional record to register.
        """

        self.scope.conditionals.append(record)
        self.all_conditionals.append(record)
        for branch_kind, span in record.branch_ranges.items():
            self.scope.branch_intervals.append(
                BranchInterval(
                    conditional_key=record.key,
                    branch_kind=branch_kind,
                    nesting_depth=record.nesting_depth,
                    span=span,
                )
            )

    def _add_bool_consumer(
        self,
        kind: str,
        node: ast.AST,
        conditional_key: Optional[ConditionalKey],
        branch_test_kind: Optional[str],
    ) -> None:
        """Add a bool consumer record to the flattened file index.

        Parameters
        ----------
        kind:
            Consumer kind.
        node:
            AST node spanning the consumed boolean expression.
        conditional_key:
            Structural conditional key for branch consumers.
        branch_test_kind:
            Branch discriminator for branch consumers.
        """

        consumer = BoolConsumer(
            kind=kind,
            span=_node_span(node),
            depth=_ast_depth(node, self.parent_map),
            conditional_key=conditional_key,
            branch_test_kind=branch_test_kind,
        )
        self.all_bool_consumers.append(consumer)

    def _is_synthetic_elif(self, node: ast.If) -> bool:
        """Return whether an ``ast.If`` is the synthetic child of an ``elif``.

        Parameters
        ----------
        node:
            ``ast.If`` node to inspect.

        Returns
        -------
        bool
            ``True`` when ``node`` is the sole statement in its parent's
            ``orelse`` list.
        """

        parent = self.parent_map.get(node)
        return isinstance(parent, ast.If) and len(parent.orelse) == 1 and parent.orelse[0] is node


__all__ = [
    "BoolClassification",
    "ConditionalKey",
    "ConditionalRecord",
    "FileIndex",
    "ScopeEntry",
    "attribute_op",
    "classify_bool",
    "get_file_index",
    "invalidate_cache",
]
