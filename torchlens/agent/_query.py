"""query_sites: structured site discovery over persisted facts (memo 3.5).

The wire query is a versioned, CLOSED JSON AST that serializes the public
selector algebra -- never a parallel language. v1 leaves are persisted facts
only; value-dependent predicates, callables, import paths, and regex REFUSE
typed, naming the supported set and the Python path that serves them. LISTING
semantics: no ``max_fanout`` cap ever -- resolution runs against the full
persisted population, then pages (the fanout cap stays on intervention
resolution, where ambiguity is a real safety property).

Combinator semantics over the executed dataflow graph (documented here, and
matching the shipped capture-time algebra): ``followed_by`` matches ops with
at least one DOWNSTREAM op (transitive dataflow descendant) matching the
inner query; ``preceded_by`` mirrors it upstream.
"""

from __future__ import annotations

from typing import Any

from .._errors import InvalidArgumentError

#: Versioned query-AST identifier.
QUERY_AST_VERSION = "torchlens.agent_query.v1"

#: Result schema id.
QUERY_SITES_SCHEMA = "torchlens.agent.query_sites.v1"

#: Decoder ceilings, enforced BEFORE resolution (configuration, fuzz-tested).
MAX_QUERY_DEPTH = 16
MAX_QUERY_NODES = 128
MAX_QUERY_STRING = 512

#: v1 leaf vocabulary: op token -> (value type, matcher builder).
_LEAF_OPS = (
    "label",
    "func",
    "in_module",
    "contains",
    "glob",
    "payload_state",
    "saved",
    "dtype",
    "pass_index",
    "is_output",
    "is_input",
)

#: Combinator vocabulary.
_COMBINATORS = ("and", "or", "not", "followed_by", "preceded_by")

#: Deliberately refused query kinds -> the Python spelling that serves them.
_REFUSED_KINDS = {
    "regex": "glob serves discovery; regex is a backtracking-DoS surface",
    "where": "value predicates run in Python: tl.trace(..., save=tl.where(...))",
    "changed": "cross-run value selection runs in Python: tl.changed(reference)",
    "callable": "callables run in Python: pass them to tl.trace(save=...)",
    "import_path": "import paths never execute here; call the object in Python",
}


def _check_ceilings(nodes: int, depth: int) -> None:
    """Refuse over-ceiling queries BEFORE any resolution work (DoS bounds)."""

    if nodes > MAX_QUERY_NODES:
        raise InvalidArgumentError(
            f"query exceeds the {MAX_QUERY_NODES}-node ceiling",
            code="agent_query_ceiling",
            remedy="split the query; ceilings are DoS bounds, not semantics",
        )
    if depth > MAX_QUERY_DEPTH:
        raise InvalidArgumentError(
            f"query exceeds the depth-{MAX_QUERY_DEPTH} ceiling",
            code="agent_query_ceiling",
            remedy="flatten nested combinators; ceilings are DoS bounds",
        )


def validate_query(node: Any, *, _depth: int = 1, _counter: list[int] | None = None) -> None:
    """Validate one query AST against the closed v1 grammar and ceilings.

    Parameters
    ----------
    node:
        Candidate AST node (``{"op": ..., ...}`` mapping).
    _depth:
        Recursion depth (internal).
    _counter:
        Node counter (internal).

    Raises
    ------
    InvalidArgumentError
        ``agent_query_invalid`` for grammar violations;
        ``agent_query_ceiling`` when depth/node/string ceilings trip.
    """

    counter = _counter if _counter is not None else [0]
    counter[0] += 1
    _check_ceilings(counter[0], _depth)
    if not isinstance(node, dict) or "op" not in node:
        raise _query_invalid(
            f"query node {node!r} is not an {{'op': ...}} mapping",
            code="agent_query_invalid",
            remedy=f"use leaves {', '.join(_LEAF_OPS)} and combinators {', '.join(_COMBINATORS)}",
        )
    op = node["op"]
    if op in _REFUSED_KINDS:
        raise _query_invalid(
            f"query kind {op!r} is deliberately not served",
            code="agent_query_invalid",
            remedy=_REFUSED_KINDS[op],
            refused_kind=str(op),
        )
    if op in _LEAF_OPS:
        _validate_leaf(op, node)
        return
    _validate_combinator(op, node, _depth, counter)


def _validate_combinator(op: str, node: dict[str, Any], depth: int, counter: list[int]) -> None:
    """Validate one combinator node's children (the recursive arm)."""

    if op in ("and", "or"):
        items = node.get("items")
        if not isinstance(items, list) or not items:
            raise _query_invalid(
                f"{op!r} needs a non-empty 'items' list",
                code="agent_query_invalid",
                remedy="pass items=[<node>, ...]",
            )
        for item in items:
            validate_query(item, _depth=depth + 1, _counter=counter)
        return
    if op in ("not", "followed_by", "preceded_by"):
        if "item" not in node:
            raise _query_invalid(
                f"{op!r} needs an 'item' node",
                code="agent_query_invalid",
                remedy="pass item=<node>",
            )
        validate_query(node["item"], _depth=depth + 1, _counter=counter)
        return
    raise _query_invalid(
        f"unknown query op {op!r}",
        code="agent_query_invalid",
        remedy=f"supported leaves: {', '.join(_LEAF_OPS)}; combinators: {', '.join(_COMBINATORS)}",
        unknown_op=str(op),
    )


def _validate_leaf(op: str, node: dict[str, Any]) -> None:
    """Validate one leaf node's value type and string ceiling."""

    value = node.get("value")
    if op in ("saved", "is_output", "is_input"):
        if not isinstance(value, bool):
            raise _query_invalid(
                f"{op!r} takes a boolean value",
                code="agent_query_invalid",
                remedy="pass value=true or value=false",
            )
        return
    if op == "pass_index":
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise _query_invalid(
                f"{op!r} takes a 1-based integer pass",
                code="agent_query_invalid",
                remedy="pass value=<positive int>",
            )
        return
    if not isinstance(value, str) or not value:
        raise _query_invalid(
            f"{op!r} takes a non-empty string value",
            code="agent_query_invalid",
            remedy="pass value=<string>",
        )
    if len(value) > MAX_QUERY_STRING:
        raise InvalidArgumentError(
            f"leaf string exceeds the {MAX_QUERY_STRING}-char ceiling",
            code="agent_query_ceiling",
            remedy="shorten the value; ceilings are DoS bounds, not semantics",
        )
    if op == "payload_state" and value not in ("present", "lazy", "unsaved"):
        raise _query_invalid(
            f"payload_state value {value!r} is outside the closed vocabulary",
            code="agent_query_invalid",
            remedy="pass one of present, lazy, unsaved",
        )


def _in_module_matches(value: str, row: dict[str, Any]) -> bool:
    """Whether one row's call stack sits at or under a module address."""

    addresses = [entry.rsplit(":", 1)[0] for entry in row["module_call_stack"]]
    return any(address == value or address.startswith(value + ".") for address in addresses)


def _glob_matches(value: str, row: dict[str, Any]) -> bool:
    """Case-sensitive glob over the pass-qualified and layer labels."""

    from fnmatch import fnmatchcase

    return fnmatchcase(row["label"], value) or fnmatchcase(row["layer_label"], value)


#: Leaf op -> row predicate (the ONE evaluation table; validation already
#: guaranteed the value's type).
_LEAF_MATCHERS: dict[str, Any] = {
    "label": lambda value, row: value in (row["label"], row["layer_label"]),
    "func": lambda value, row: row["func_name"] == value,
    "in_module": _in_module_matches,
    "contains": lambda value, row: value in row["label"] or value in row["layer_label"],
    "glob": _glob_matches,
    "payload_state": lambda value, row: row["payload_state"] == value,
    "saved": lambda value, row: row["saved"] is value,
    "dtype": lambda value, row: row["dtype"] == value,
    "pass_index": lambda value, row: row["pass_index"] == value,
    "is_output": lambda value, row: row["is_output"] is value,
    "is_input": lambda value, row: row["is_input"] is value,
}


def _leaf_matches(op: str, value: Any, row: dict[str, Any]) -> bool:
    """Evaluate one leaf against one op row (persisted facts only)."""

    return bool(_LEAF_MATCHERS[op](value, row))


def _eval_query(node: dict[str, Any], rows: list[dict[str, Any]]) -> list[bool]:
    """Evaluate one validated AST over all rows, returning a match mask.

    ``followed_by``/``preceded_by`` compute transitive dataflow closure in
    one reverse/forward pass over execution order (children always execute
    after parents in a forward), so evaluation stays O(V+E) per combinator.

    Parameters
    ----------
    node:
        Validated AST node.
    rows:
        Execution-ordered op rows with graph edges.

    Returns
    -------
    list[bool]
        Per-row match mask.
    """

    op = node["op"]
    if op in ("and", "or"):
        masks = [_eval_query(item, rows) for item in node["items"]]
        combine = all if op == "and" else any
        return [combine(mask[i] for mask in masks) for i in range(len(rows))]
    if op == "not":
        inner = _eval_query(node["item"], rows)
        return [not flag for flag in inner]
    if op in ("followed_by", "preceded_by"):
        inner = _eval_query(node["item"], rows)
        # Graph edges carry LAYER labels (rolled), so the index maps both the
        # pass-qualified label and the layer label (all passes) per row.
        index_of: dict[str, list[int]] = {}
        for i, row in enumerate(rows):
            index_of.setdefault(row["label"], []).append(i)
            index_of.setdefault(row["layer_label"], []).append(i)
        closure = [False] * len(rows)
        order = range(len(rows) - 1, -1, -1) if op == "followed_by" else range(len(rows))
        edge_key = "children" if op == "followed_by" else "parents"
        for i in order:
            for neighbor in rows[i][edge_key]:
                hit = any(inner[j] or closure[j] for j in index_of.get(neighbor, []) if j != i)
                if hit:
                    closure[i] = True
                    break
        return closure
    return [_leaf_matches(op, node.get("value"), row) for row in rows]


def _match_reasons(node: dict[str, Any], row: dict[str, Any]) -> list[dict[str, Any]]:
    """Collect the satisfied LEAF nodes for one matched row (FF2 closure).

    The disclosed reason plus the row's own fields must suffice to reproduce
    the match locally; closure combinators disclose themselves as reasons
    since their truth lives on other rows.

    Parameters
    ----------
    node:
        Validated AST node.
    row:
        One matched op row.

    Returns
    -------
    list[dict[str, Any]]
        Satisfied leaves (bounded by the query-node ceiling by construction).
    """

    op = node["op"]
    if op in ("and", "or"):
        reasons: list[dict[str, Any]] = []
        for item in node["items"]:
            reasons.extend(_match_reasons(item, row))
        return reasons
    if op == "not":
        return [{"op": "not", "item_op": node["item"]["op"]}]
    if op in ("followed_by", "preceded_by"):
        return [{"op": op, "item_op": node["item"]["op"], "basis": "graph_closure"}]
    if _leaf_matches(op, node.get("value"), row):
        return [{"op": op, "value": node.get("value")}]
    return []


def _leaf_condition(op: str, value: Any) -> str:
    """Render one leaf as a Python condition over the loop variable ``op``.

    Every rendered condition is comment-free and runnable verbatim inside a
    comprehension over ``log.layer_list`` -- the handoff test executes it.

    Parameters
    ----------
    op:
        Leaf op token.
    value:
        Leaf value.

    Returns
    -------
    str
        Python boolean expression.
    """

    conditions = {
        "label": f"str(op.label) == {value!r} or str(op.layer_label) == {value!r}",
        "func": f"op.func_name == {value!r}",
        "contains": f"{value!r} in str(op.label) or {value!r} in str(op.layer_label)",
        "glob": (
            f"__import__('fnmatch').fnmatchcase(str(op.label), {value!r}) or "
            f"__import__('fnmatch').fnmatchcase(str(op.layer_label), {value!r})"
        ),
        "in_module": (
            f"any(e.rsplit(':', 1)[0] == {value!r} or "
            f"e.rsplit(':', 1)[0].startswith({value!r} + '.') "
            "for e in op.module_call_stack)"
        ),
        "dtype": f"str(op.dtype) == {value!r}",
        "pass_index": f"int(op.pass_index or 1) == {value!r}",
        "saved": f"bool(op.has_saved_activation) is {value!r}",
        "payload_state": {
            "present": ("getattr(op, 'out_ref', None) is None and bool(op.has_saved_activation)"),
            "lazy": "getattr(op, 'out_ref', None) is not None",
            "unsaved": (
                "getattr(op, 'out_ref', None) is None and not bool(op.has_saved_activation)"
            ),
        }.get(str(value), "False"),
        "is_output": (f"(str(op.label) in {{str(o.label) for o in log.output_ops}}) is {value!r}"),
        "is_input": (f"(str(op.label) in {{str(o.label) for o in log.input_ops}}) is {value!r}"),
    }
    return conditions[op]


def _contains_closure(node: dict[str, Any]) -> bool:
    """Whether the AST uses a graph-closure combinator."""

    op = node["op"]
    if op in ("followed_by", "preceded_by"):
        return True
    if op in ("and", "or"):
        return any(_contains_closure(item) for item in node["items"])
    if op == "not":
        return _contains_closure(node["item"])
    return False


def _condition_expr(node: dict[str, Any]) -> str:
    """Render one closure-free AST as a Python boolean expression."""

    op = node["op"]
    if op == "and":
        return "(" + " and ".join(_condition_expr(item) for item in node["items"]) + ")"
    if op == "or":
        return "(" + " or ".join(_condition_expr(item) for item in node["items"]) + ")"
    if op == "not":
        return f"not {_condition_expr(node['item'])}"
    return "(" + _leaf_condition(op, node.get("value")) + ")"


def _python_handoff(node: dict[str, Any]) -> str:
    """Render the exact runnable Python reproduction of one query.

    Closure-free queries hand off a one-line comprehension; closure queries
    hand off a short verbatim-runnable snippet computing the transitive
    dataflow reach. Both assume ``log`` (the loaded trace) in scope -- the
    spelling every result's envelope teaches.

    Parameters
    ----------
    node:
        Validated AST node.

    Returns
    -------
    str
        Runnable Python reproducing the matched labels.
    """

    if not _contains_closure(node):
        return f"[str(op.label) for op in log.layer_list if {_condition_expr(node)}]"
    if node["op"] in ("followed_by", "preceded_by") and not _contains_closure(node["item"]):
        # A literal mirror of ``_eval_query``'s closure pass: edges spell
        # single-pass neighbours BARE and multi-pass neighbours pass-
        # qualified, so the index maps BOTH spellings per row and the
        # sweep runs in the tool's order. A layer_label-keyed dict collapsed
        # the passes of a recurrent layer and missed every pass-qualified
        # edge, so the handoff disagreed with the tool on multi-pass traces
        # (AUD-CODE 3.11d).
        edge = "children" if node["op"] == "followed_by" else "parents"
        order = (
            "range(len(rows) - 1, -1, -1)" if node["op"] == "followed_by" else "range(len(rows))"
        )
        return (
            "rows = list(log.layer_list)\n"
            f"inner = [bool({_condition_expr(node['item'])}) for op in rows]\n"
            "index = {}\n"
            "for i, op in enumerate(rows):\n"
            "    index.setdefault(str(op.label), []).append(i)\n"
            "    index.setdefault(str(op.layer_label), []).append(i)\n"
            "closure = [False] * len(rows)\n"
            f"for i in {order}:\n"
            f"    for neighbor in rows[i].{edge}:\n"
            "        if any(inner[j] or closure[j] for j in index.get(str(neighbor), []) if j != i):\n"
            "            closure[i] = True\n"
            "            break\n"
            "matched = [str(op.label) for op, flag in zip(rows, closure) if flag]"
        )
    return (
        "matched = call_tool('torchlens_query_sites', "
        "{'path': path, 'query': query})['data']['rows']"
    )


def op_row(op: Any, is_input: bool, is_output: bool) -> dict[str, Any]:
    """Project one op record into the query row shape (persisted facts)."""

    from ..report._agent_json import _op_entry

    entry = _op_entry(op)
    entry["is_input"] = is_input
    entry["is_output"] = is_output
    return entry


def query_sites(log: Any, query: Any | None) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    """Resolve one query over the full persisted op population.

    Parameters
    ----------
    log:
        Loaded ``Trace``.
    query:
        Query AST mapping, or ``None`` to list every site.

    Returns
    -------
    tuple[list[dict], dict, str]
        Execution-ordered matched rows (each with ``matched`` reasons), the
        header (``population_total``/``matches_total``), and the runnable
        Python handoff spelling.
    """

    if query is not None:
        validate_query(query)
    ops = list(getattr(log, "layer_list", []) or [])
    input_labels = {
        str(getattr(item, "label", item)) for item in getattr(log, "input_ops", []) or []
    }
    output_labels = {
        str(getattr(item, "label", item)) for item in getattr(log, "output_ops", []) or []
    }
    rows = [
        op_row(
            op,
            str(getattr(op, "label", "")) in input_labels,
            str(getattr(op, "label", "")) in output_labels,
        )
        for op in ops
    ]
    if query is None:
        matched = rows
        for row in matched:
            row["matched"] = [{"op": "all"}]
        handoff = "[str(op.label) for op in log.layer_list]"
    else:
        mask = _eval_query(query, rows)
        matched = []
        for row, flag in zip(rows, mask, strict=True):
            if flag:
                row["matched"] = _match_reasons(query, row)
                matched.append(row)
        handoff = _python_handoff(query)
    header = {
        "population_total": len(rows),
        "matches_total": len(matched),
        "query_ast_version": QUERY_AST_VERSION,
    }
    return matched, header, handoff


def _query_invalid(
    problem: str,
    *,
    code: str,
    remedy: str,
    **context: Any,
) -> InvalidArgumentError:
    """Build one typed query-grammar refusal.

    The ``code`` is passed at every raise site (site-visible, so the S-17
    census records the site as coded through the factory -- the C01 registry
    precedent).

    Parameters
    ----------
    problem:
        What the query violated.
    code:
        Stable refusal code (``agent_query_invalid``).
    remedy:
        Concrete caller fix.
    **context:
        Structured diagnostic context.

    Returns
    -------
    InvalidArgumentError
        The typed refusal for the caller to raise.
    """

    return InvalidArgumentError(problem, code=code, remedy=remedy, **context)
