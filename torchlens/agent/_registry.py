"""The ONE tool registry: names, schemas, budgets, handlers, transports.

Registry law (agent memo 3.1): this module owns every tool's name,
description, input/output schema, read-only/idempotent annotations, default
budgets, CLI verb mapping, and handler. Python (:func:`call_tool`), MCP
(``torchlens.bridge.mcp``), and the CLI (``torchlens.agent.cli``) are thin
adapters over it; transports may override DEFAULTS, never semantics. The
field-level parity test (served ``list_tools`` == registry) is a release
gate -- the structural fix for declared-vs-served schema drift.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .._errors import InvalidArgumentError
from . import _budgets
from ._envelope import build_envelope, build_error_envelope, canonical_dumps, json_safe

#: Closed tool-class vocabulary driving default budgets.
TOOL_CLASSES = ("environment", "orientation", "rows", "payload")


@dataclass(frozen=True)
class AgentToolSpec:
    """One registered agent tool: the single source of every served fact."""

    name: str
    description: str
    input_schema: dict[str, Any]
    output_schema: str
    tool_class: str
    cli_verb: str | None
    handler: Callable[[dict[str, Any]], dict[str, Any]]
    read_only: bool = True
    idempotent: bool = True
    default_limits: dict[str, int] = field(default_factory=dict)
    #: True when the HANDLER already enforces the token budget inside its
    #: data (explain's report engine prunes sections with its own honesty
    #: floor); the envelope backstop then skips re-flooring the result.
    budgets_in_handler: bool = False


def _path_property() -> dict[str, Any]:
    """The shared ``path`` input property."""

    return {"type": "string", "description": "Path to a .tlspec artifact."}


def _validate_args(spec: AgentToolSpec, args: dict[str, Any]) -> None:
    """Validate request arguments against the declared input schema.

    Covers the schema subset the registry declares: required keys, unknown
    keys (``additionalProperties: false`` everywhere), primitive types, and
    enums. Full JSON Schema validation runs in CI lockstep tests.

    Parameters
    ----------
    spec:
        Tool spec.
    args:
        Request arguments.

    Raises
    ------
    InvalidArgumentError
        ``agent_argument_invalid`` naming the offending key.
    """

    properties = spec.input_schema.get("properties", {})
    for key in spec.input_schema.get("required", []):
        if key not in args:
            raise InvalidArgumentError(
                f"{spec.name} requires {key!r}",
                code="agent_argument_invalid",
                remedy=f"pass {key}=<{properties.get(key, {}).get('type', 'value')}>",
                tool=spec.name,
            )
    for key, value in args.items():
        if key not in properties:
            raise InvalidArgumentError(
                f"{spec.name} does not take {key!r}",
                code="agent_argument_invalid",
                remedy=f"supported arguments: {', '.join(sorted(properties)) or '(none)'}",
                tool=spec.name,
            )
        declared = properties[key]
        enum = declared.get("enum")
        if enum is not None and value not in enum:
            raise InvalidArgumentError(
                f"{spec.name} {key}={value!r} is outside the closed vocabulary",
                code="agent_argument_invalid",
                remedy=f"pass one of {', '.join(map(str, enum))}",
                tool=spec.name,
            )
        expected = declared.get("type")
        if expected is not None and not _type_ok(value, expected):
            raise InvalidArgumentError(
                f"{spec.name} {key} must be {expected}",
                code="agent_argument_invalid",
                remedy=f"pass a {expected} value for {key}",
                tool=spec.name,
            )


def _type_ok(value: Any, expected: str | list[str]) -> bool:
    """Check one value against a JSON-Schema primitive type declaration."""

    kinds = expected if isinstance(expected, list) else [expected]
    checks = {
        "string": lambda v: isinstance(v, str),
        "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
        "number": lambda v: isinstance(v, int | float) and not isinstance(v, bool),
        "boolean": lambda v: isinstance(v, bool),
        "object": lambda v: isinstance(v, dict),
        "array": lambda v: isinstance(v, list),
        "null": lambda v: v is None,
    }
    return any(checks.get(kind, lambda v: True)(value) for kind in kinds)


def _fit_token_budget(
    envelope: dict[str, Any],
    max_tokens: int,
    *,
    row_key: str | tuple[str, ...] | None,
) -> dict[str, Any]:
    """Enforce the response-token budget on one built envelope.

    Row caps applied FIRST by the handlers; this is the genuine backstop.
    When rows are dropped on a PAGING tool the continuation struct is
    re-minted from the trimmed row count, so the next page starts at the
    first dropped row (AUD-CODE 2.7b: trimming after the handler minted
    ``data.next`` left the omitted rows unreachable). When the non-droppable
    floor alone exceeds the budget, the floor returns with
    ``budget_floor_exceeded`` -- carrying the record's ``capture`` honesty
    block when it has one, never a plausible fragment.

    Parameters
    ----------
    envelope:
        Built result envelope.
    max_tokens:
        Effective token budget.
    row_key:
        Key (or candidate keys, first present list wins) inside ``data``
        holding droppable rows (``None`` = nothing to drop).

    Returns
    -------
    dict[str, Any]
        Envelope within budget (rows dropped with disclosure) or the floor.
    """

    if _budgets.estimate_tokens(canonical_dumps(envelope)) <= max_tokens:
        return envelope
    data = envelope.get("data") or {}
    key = _present_row_key(data, row_key)
    rows = data.get(key) if key else None
    if isinstance(rows, list) and rows:
        pages = "next" in data and isinstance(data.get("offset"), int)
        keep = len(rows)
        while keep > 0:
            keep = min(keep - 1, keep * 3 // 4)
            trial = dict(envelope)
            trial_data = dict(data)
            trial_data[key] = rows[:keep]
            if pages:
                trial_data["next"] = _budgets.continuation_struct(
                    offset=int(data["offset"]) + keep,
                    artifact_id=str((envelope.get("artifact") or {}).get("id")),
                    request=dict(envelope.get("request") or {}),
                    schema=str(envelope.get("schema")),
                )
            trial["data"] = trial_data
            omitted_total = (envelope.get("truncation") or {}).get("omitted", 0)
            trial["truncation"] = {
                "included": keep,
                "omitted": omitted_total + (len(rows) - keep),
                "policy": "token budget backstop dropped trailing rows after the row cap",
                "how_to_get_more": (
                    "echo data.next as the continuation argument (the next page starts "
                    "at the first dropped row), or raise max_tokens within the ceiling"
                    if pages
                    else "raise max_tokens (within the ceiling) or narrow the site set "
                    "with labels=/query=; this tool does not page"
                ),
            }
            if _budgets.estimate_tokens(canonical_dumps(trial)) <= max_tokens:
                return trial
        envelope = trial  # floor candidate: zero rows, disclosure intact
    if _budgets.estimate_tokens(canonical_dumps(envelope)) <= max_tokens:
        return envelope
    floor = dict(envelope)
    floor_data: dict[str, Any] = {"budget_floor_exceeded": True}
    if isinstance(data.get("capture"), dict):
        floor_data["capture"] = data["capture"]
    floor["data"] = floor_data
    floor["status"] = "budget_floor_exceeded"
    floor["truncation"] = {
        "included": 0,
        "omitted": -1,
        "policy": "the non-droppable floor alone exceeded max_tokens",
        "how_to_get_more": (
            "raise max_tokens within the served ceiling"
            + (
                " or lower max_rows so a page fits"
                if "max_rows" in (envelope.get("limits") or {})
                else ""
            )
            + "; the floor carries only the capture honesty facts (when the record has them)"
        ),
    }
    return floor


def _present_row_key(data: dict[str, Any], row_key: str | tuple[str, ...] | None) -> str | None:
    """Return the first candidate row key whose value is a list in ``data``."""

    if row_key is None:
        return None
    candidates = (row_key,) if isinstance(row_key, str) else row_key
    for candidate in candidates:
        if isinstance(data.get(candidate), list):
            return candidate
    return None


# --------------------------------------------------------------------------
# Handlers (each returns a COMPLETE envelope).


def _handle_doctor(args: dict[str, Any]) -> dict[str, Any]:
    """Environment health report."""

    from ..utils import doctor

    report = doctor()
    data = {
        "checks": [
            {"name": check.name, "status": check.status, "detail": check.detail}
            for check in report.checks
        ]
    }
    return build_envelope(schema="torchlens.agent.doctor.v1", data=data, request=args)


def _handle_api_map(args: dict[str, Any]) -> dict[str, Any]:
    """Public-surface index (compact rows or one per-name detail record)."""

    from ._apimap import (
        API_MAP_SCHEMA,
        capabilities_block,
        compact_rows,
        name_detail,
        submodule_map,
    )

    name = args.get("name")
    data: dict[str, Any] = {
        "names": compact_rows() if name is None else [],
        "detail": name_detail(name) if name is not None else None,
        "capabilities": capabilities_block(),
        "submodules_not_in_all": submodule_map(),
    }
    return build_envelope(schema=API_MAP_SCHEMA, data=data, request=args)


def _handle_overview(args: dict[str, Any]) -> dict[str, Any]:
    """Manifest-mode (torch-free) or folded-mode structural overview."""

    from ._artifacts import (
        artifact_block,
        load_trace,
        read_manifest,
        resolve_artifact_path,
        resolve_digest,
    )
    from ._overview import (
        OVERVIEW_FOLDED_SCHEMA,
        OVERVIEW_MANIFEST_SCHEMA,
        folded_overview,
        manifest_overview,
    )

    mode = args.get("mode", "folded")
    if mode == "manifest":
        path = resolve_artifact_path(args["path"])
        digest, manifest_bytes = resolve_digest(path)
        manifest, _ = read_manifest(path)
        return build_envelope(
            schema=OVERVIEW_MANIFEST_SCHEMA,
            data=manifest_overview(manifest),
            artifact=artifact_block(path, digest, manifest),
            request=args,
        )
    log, plan, block = load_trace(args["path"])
    data = folded_overview(log)
    data["load_plan"] = plan
    return build_envelope(schema=OVERVIEW_FOLDED_SCHEMA, data=data, artifact=block, request=args)


def _handle_dump(args: dict[str, Any]) -> dict[str, Any]:
    """Paged dump views over one loaded trace."""

    from ._artifacts import load_trace
    from ._overview import DUMP_SCHEMA, dump_view

    view = args.get("view", "overview")
    max_rows = _budgets.resolve_limit(
        args.get("max_rows"), _budgets.DEFAULT_MAX_ROWS, name="max_rows"
    )
    log, plan, block = load_trace(args["path"])
    request_echo = {key: value for key, value in args.items() if key != "continuation"}
    offset = _budgets.check_continuation(
        args.get("continuation"),
        artifact_id=block["id"],
        request=request_echo,
        schema=DUMP_SCHEMA,
    )
    data, included, omitted = dump_view(
        log, view=view, max_rows=max_rows, offset=offset, class_id=args.get("class_id")
    )
    data["view"] = view
    data["load_plan"] = plan
    if view != "overview":
        # Every gateable record carries the honesty blocks (AUD-CODE 3.11a):
        # the full view's own capture block is the same source, so setdefault.
        from ._overview import honesty_blocks

        blocks = honesty_blocks(log)
        data.setdefault("capture", blocks["capture"])
        data["audit"] = blocks["audit"]
    truncation = None
    if view in ("graph", "full"):
        data["next"] = (
            _budgets.continuation_struct(
                offset=offset + included,
                artifact_id=block["id"],
                request=request_echo,
                schema=DUMP_SCHEMA,
            )
            if omitted
            else None
        )
        truncation = _budgets.truncation_block(
            included=included,
            omitted=omitted,
            policy="execution-ordered rows paged at max_rows",
            how_to_get_more="echo data.next as the continuation argument",
        )
    return build_envelope(
        schema=DUMP_SCHEMA,
        data=data,
        artifact=block,
        request=request_echo,
        limits={"max_rows": max_rows},
        truncation=truncation,
    )


def _handle_explain(args: dict[str, Any]) -> dict[str, Any]:
    """Budgeted plain-language report.

    ``max_tokens`` defaults to the tool's declared orientation budget (the
    registry's ``default_limits``; AUD-CODE 4.8 -- the handler used to pass
    ``None``, an unbudgeted report under a declared default). The record
    carries the same ``capture``/``audit`` honesty blocks the overview
    serves, so ``--fail-on unverified|incomplete|nonfinite`` READ them here
    instead of being dead gates (AUD-CODE 3.11a).
    """

    from ..report import explain
    from ._artifacts import load_trace
    from ._overview import honesty_blocks

    log, plan, block = load_trace(args["path"])
    max_tokens = args.get("max_tokens", _budgets.ORIENTATION_MAX_TOKENS)
    report = explain(
        log,
        audience=args.get("audience", "auto"),
        max_tokens=max_tokens,
    )
    return build_envelope(
        schema="torchlens.agent.explain.v1",
        data={"report": str(report), "load_plan": plan, **honesty_blocks(log)},
        artifact=block,
        request=args,
        limits={"max_tokens": max_tokens, "token_estimator": _budgets.TOKEN_ESTIMATOR},
    )


def _handle_query_sites(args: dict[str, Any]) -> dict[str, Any]:
    """Structured site discovery over the full persisted population."""

    from ._artifacts import load_trace
    from ._query import QUERY_SITES_SCHEMA, query_sites

    max_rows = _budgets.resolve_limit(
        args.get("max_rows"), _budgets.DEFAULT_MAX_ROWS, name="max_rows"
    )
    log, plan, block = load_trace(args["path"])
    request_echo = {key: value for key, value in args.items() if key != "continuation"}
    offset = _budgets.check_continuation(
        args.get("continuation"),
        artifact_id=block["id"],
        request=request_echo,
        schema=QUERY_SITES_SCHEMA,
    )
    matched, header, handoff = query_sites(log, args.get("query"))
    page = matched[offset : offset + max_rows]
    omitted = max(0, len(matched) - offset - len(page))
    data = {
        "header": header,
        "rows": page,
        "offset": offset,
        "handoff": handoff,
        "load_plan": plan,
        "next": (
            _budgets.continuation_struct(
                offset=offset + len(page),
                artifact_id=block["id"],
                request=request_echo,
                schema=QUERY_SITES_SCHEMA,
            )
            if omitted
            else None
        ),
    }
    return build_envelope(
        schema=QUERY_SITES_SCHEMA,
        data=data,
        artifact=block,
        request=request_echo,
        limits={"max_rows": max_rows},
        truncation=_budgets.truncation_block(
            included=len(page),
            omitted=omitted,
            policy="execution-ordered matches paged at max_rows",
            how_to_get_more="echo data.next as the continuation argument",
        ),
    )


def _handle_payload_stats(args: dict[str, Any]) -> dict[str, Any]:
    """Bounded per-site statistics over saved payloads."""

    from ._artifacts import load_trace
    from ._stats import PAYLOAD_STATS_SCHEMA, payload_stats_rows

    max_rows = _budgets.resolve_limit(
        args.get("max_rows"), _budgets.DEFAULT_MAX_ROWS, name="max_rows"
    )
    log, plan, block = load_trace(args["path"])
    rows, header = payload_stats_rows(
        log,
        labels=args.get("labels"),
        query=args.get("query"),
        target=args.get("target", "out"),
        metrics=args.get("metrics"),
        reduction=args.get("reduction"),
        k=args.get("k"),
        max_rows=max_rows,
        rank_by=args.get("rank_by"),
        max_blob_bytes=_budgets.MAX_BLOB_BYTES,
        max_call_bytes=_budgets.MAX_CALL_BYTES,
    )
    unsaved = [row["label"] for row in rows if row.get("status") == "unsaved"]
    warnings: list[str] = []
    requested = args.get("labels")
    if requested:
        served = {row["label"] for row in rows} | {row["label"].rsplit(":", 1)[0] for row in rows}
        unmatched = [str(label) for label in requested if str(label) not in served]
        if unmatched:
            # An unknown label used to vanish into ``status ok, rows=[]`` (AUD-CODE
            # 4.8); name it so a typo never reads as "no data".
            warnings.append(
                f"labels not found in this artifact (no row served): {', '.join(unmatched)}; "
                "discover spellings via torchlens_query_sites"
            )
    handoff = (
        "tl.trace(model, x, save="
        + (
            " | ".join(f"tl.label({label.rsplit(':', 1)[0]!r})" for label in unsaved[:4])
            if unsaved
            else "tl.func('relu')"
        )
        + ")"
    )
    data = {"header": header, "rows": rows, "handoff": handoff, "load_plan": plan}
    return build_envelope(
        schema=PAYLOAD_STATS_SCHEMA,
        data=data,
        artifact=block,
        request=args,
        limits={
            "max_rows": max_rows,
            "max_blob_bytes": _budgets.MAX_BLOB_BYTES,
            "max_call_bytes": _budgets.MAX_CALL_BYTES,
        },
        warnings=warnings,
        truncation=_budgets.truncation_block(
            included=len(rows),
            omitted=max(0, header["population_total"] - len(rows)),
            policy="site rows capped at max_rows before any payload read",
            how_to_get_more="pass labels= for the specific sites, or page with a narrower query",
        ),
    )


def _handle_compare(args: dict[str, Any]) -> dict[str, Any]:
    """Two-artifact structural + value comparison."""

    from ._artifacts import load_trace
    from ._compare import COMPARE_SCHEMA, compare_traces

    max_rows = _budgets.resolve_limit(
        args.get("max_rows"), _budgets.DEFAULT_MAX_ROWS, name="max_rows"
    )
    tolerances = {"rtol": float(args.get("rtol", 1e-5)), "atol": float(args.get("atol", 1e-8))}
    for name, value in tolerances.items():
        if not (value >= 0) or value == float("inf"):  # NaN fails the >= comparison
            # torch.allclose raises an untyped RuntimeError on a negative
            # tolerance (AUD-CODE 4.8); refuse typed at the boundary instead.
            raise InvalidArgumentError(
                f"{name}={value!r} is not a finite non-negative tolerance",
                code="agent_argument_invalid",
                remedy=f"pass {name} >= 0 (allclose tolerances are magnitudes)",
                tool="torchlens_compare",
            )
    reference, _, ref_block = load_trace(args["reference"])
    subject, _, sub_block = load_trace(args["subject"])
    data = compare_traces(
        reference,
        subject,
        ref_block=ref_block,
        sub_block=sub_block,
        rtol=tolerances["rtol"],
        atol=tolerances["atol"],
        max_rows=max_rows,
    )
    data["handoff"] = (
        "ref = tl.load(reference); sub = tl.load(subject); "
        "sel = tl.changed(ref).resolve(sub)  # every element that moved"
    )
    # Per-side honesty blocks so the CLI's --fail-on gates READ this record
    # (AUD-CODE 3.11a): a diff over an unverified or NaN-poisoned side must
    # be gateable without a second tool call.
    from ._overview import honesty_blocks

    reference_blocks = honesty_blocks(reference)
    subject_blocks = honesty_blocks(subject)
    data["capture"] = {
        "reference": reference_blocks["capture"],
        "subject": subject_blocks["capture"],
    }
    data["audit"] = {"reference": reference_blocks["audit"], "subject": subject_blocks["audit"]}
    envelope = build_envelope(
        schema=COMPARE_SCHEMA,
        data=data,
        artifact=None,
        request=args,
        limits={"max_rows": max_rows},
        truncation=_budgets.truncation_block(
            included=len(data["rows"]),
            omitted=max(0, data["rows_total"] - len(data["rows"])),
            policy="changed/asymmetric rows rank first under max_rows",
            how_to_get_more="raise max_rows (within the ceiling) or narrow via query_sites first",
        ),
    )
    envelope["artifacts"] = {"reference": json_safe(ref_block), "subject": json_safe(sub_block)}
    return envelope


def _handle_schema(args: dict[str, Any]) -> dict[str, Any]:
    """The machine-readable schema index, or one shipped document."""

    from ._schemas import SCHEMA_TOOL_SCHEMA, load_schema, schema_index

    name = args.get("name")
    data = {
        "index": schema_index(),
        "document": load_schema(name) if name is not None else None,
    }
    return build_envelope(schema=SCHEMA_TOOL_SCHEMA, data=data, request=args)


# --------------------------------------------------------------------------
# The registry.


def _continuation_property() -> dict[str, Any]:
    """The shared ``continuation`` input property."""

    return {
        "type": "object",
        "description": "The data.next struct echoed verbatim from the prior page.",
    }


def _max_tokens_property(ceiling: int) -> dict[str, Any]:
    """The shared ``max_tokens`` input property (requests lower, never raise)."""

    return {
        "type": "integer",
        "minimum": 1,
        "maximum": ceiling,
        "description": f"Response token budget (~4 chars/token; served ceiling {ceiling}).",
    }


def _max_rows_property() -> dict[str, Any]:
    """The shared ``max_rows`` input property."""

    return {
        "type": "integer",
        "minimum": 1,
        "maximum": _budgets.DEFAULT_MAX_ROWS,
        "description": f"Row cap (served default and ceiling {_budgets.DEFAULT_MAX_ROWS}).",
    }


_NOT_SERVED = (
    " Not served here by design: raw tensors or base64 payloads, anything "
    "needing a live model, anything that writes, directory enumeration."
)


def tool_specs() -> tuple[AgentToolSpec, ...]:
    """Return the full tool registry (the ONE source every transport reads)."""

    return (
        AgentToolSpec(
            name="torchlens_doctor",
            description=(
                "Run the TorchLens environment health check (PyTorch/CUDA/"
                "Graphviz/extras/capability flags). Call this first when "
                "captures misbehave; each failing row names what is missing."
            ),
            input_schema={"type": "object", "properties": {}, "additionalProperties": False},
            output_schema="torchlens.agent.doctor.v1",
            tool_class="environment",
            cli_verb="doctor",
            handler=_handle_doctor,
        ),
        AgentToolSpec(
            name="torchlens_api_map",
            description=(
                "Machine-readable index of the public torchlens surface from "
                "the ONE export registry: compact rows plus the capabilities "
                "block (tools, CLI verbs, schema ids, budget names, "
                "writes:false, executes_user_code:false). Pass name= for one "
                "detailed record with the full normalized signature."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Optional root name for detail mode.",
                    },
                    # Measured (D4 rule): 151 compact rows serialize to ~9.1k
                    # tokens, so the orientation default rises to the smallest
                    # passing round value for THIS tool.
                    "max_tokens": _max_tokens_property(12_000),
                },
                "additionalProperties": False,
            },
            output_schema="torchlens.agent.api_map.v2",
            tool_class="orientation",
            cli_verb="api-map",
            handler=_handle_api_map,
            default_limits={"max_tokens": 12_000},
        ),
        AgentToolSpec(
            name="torchlens_overview",
            description=(
                "Structural overview of a saved .tlspec artifact. "
                'mode="manifest" is the torch-free preflight: a bounded digest '
                "read from validated manifest JSON only -- it NEVER unpickles, "
                "so it is the safe first look at an artifact you did not "
                'produce. mode="folded" (default) loads the trace and returns '
                "the recurrence-folded orientation view: repeated blocks state "
                "once with n_instances, splits disclosed, plus capture "
                "honesty, coverage, audit, and next operations." + _NOT_SERVED
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "path": _path_property(),
                    "mode": {
                        "type": "string",
                        "enum": ["manifest", "folded"],
                        "description": "Overview mode; the two modes return DIFFERENT schemas.",
                    },
                    "max_tokens": _max_tokens_property(_budgets.ORIENTATION_MAX_TOKENS),
                },
                "required": ["path"],
                "additionalProperties": False,
            },
            output_schema="torchlens.agent.overview_folded.v1",
            tool_class="orientation",
            cli_verb="overview",
            handler=_handle_overview,
            default_limits={"max_tokens": _budgets.ORIENTATION_MAX_TOKENS},
        ),
        AgentToolSpec(
            name="torchlens_dump",
            description=(
                "Structural dump views over a saved Trace. view=overview is "
                "the folded orientation; view=graph pages execution-ordered "
                "op rows with edges (class_id= drills into one fold class); "
                "view=full is every agent-safe structural block with its op "
                "rows paged the same way (echo data.next). Payload values are "
                "never inlined."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "path": _path_property(),
                    "view": {"type": "string", "enum": ["overview", "graph", "full"]},
                    "max_rows": _max_rows_property(),
                    "class_id": {
                        "type": "string",
                        "description": "Fold-class filter (graph view).",
                    },
                    "max_tokens": _max_tokens_property(_budgets.ROW_TOOL_MAX_TOKENS),
                    "continuation": _continuation_property(),
                },
                "required": ["path"],
                "additionalProperties": False,
            },
            output_schema="torchlens.agent.dump.v1",
            tool_class="rows",
            cli_verb="dump",
            handler=_handle_dump,
            default_limits={
                "max_rows": _budgets.DEFAULT_MAX_ROWS,
                "max_tokens": _budgets.ROW_TOOL_MAX_TOKENS,
            },
        ),
        AgentToolSpec(
            name="torchlens_explain",
            description=(
                "Plain-language report over a saved Trace, budgeted: "
                "max_tokens drops whole sections low-value-first and "
                "discloses every drop; capture-status honesty facts never drop."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "path": _path_property(),
                    "max_tokens": {
                        "type": "integer",
                        "minimum": 1,
                        "description": (
                            "Report token budget (~4 chars/token); default "
                            f"{_budgets.ORIENTATION_MAX_TOKENS}. Whole sections drop "
                            "low-value-first; the capture-status floor never drops."
                        ),
                    },
                    "audience": {"type": "string", "enum": ["researcher", "practitioner", "auto"]},
                },
                "required": ["path"],
                "additionalProperties": False,
            },
            output_schema="torchlens.agent.explain.v1",
            tool_class="orientation",
            cli_verb="explain",
            handler=_handle_explain,
            default_limits={"max_tokens": _budgets.ORIENTATION_MAX_TOKENS},
            budgets_in_handler=True,
        ),
        AgentToolSpec(
            name="torchlens_query_sites",
            description=(
                "Structured site discovery over the FULL persisted op "
                "population (no fanout cap on listing; results page). The "
                "query is a closed JSON AST over persisted facts -- leaves "
                "label/func/in_module/contains/glob/payload_state/saved/"
                "dtype/pass_index/is_output/is_input, combinators and/or/not/"
                "followed_by/preceded_by. Value-dependent predicates, regex, "
                "callables, and import paths refuse typed, naming the Python "
                "path. Every result carries the runnable Python spelling."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "path": _path_property(),
                    "query": {"type": "object", "description": "torchlens.agent_query.v1 AST."},
                    "max_rows": _max_rows_property(),
                    "max_tokens": _max_tokens_property(_budgets.ROW_TOOL_MAX_TOKENS),
                    "continuation": _continuation_property(),
                },
                "required": ["path"],
                "additionalProperties": False,
            },
            output_schema="torchlens.agent.query_sites.v1",
            tool_class="rows",
            cli_verb="query",
            handler=_handle_query_sites,
            default_limits={
                "max_rows": _budgets.DEFAULT_MAX_ROWS,
                "max_tokens": _budgets.ROW_TOOL_MAX_TOKENS,
            },
        ),
        AgentToolSpec(
            name="torchlens_payload_stats",
            description=(
                "Bounded, deterministic numbers over saved tensor payloads -- "
                "never the tensors. Metrics: counts, min/max/mean/std "
                "(float64, correction=0), L1/L2 norms, top/bottom-k with full "
                "coordinates. reduction={'retain_dim': i} keeps ONE explicit "
                "dimension INDEX (axis roles are never guessed). Byte budgets "
                "refuse BEFORE materialization from declared manifest bytes; "
                "an unsaved site is a per-row typed status with the exact "
                "save= recapture remedy, never a batch failure."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "path": _path_property(),
                    "labels": {"type": "array", "items": {"type": "string"}},
                    "query": {"type": "object", "description": "torchlens.agent_query.v1 AST."},
                    "target": {"type": "string", "enum": ["out", "grad"]},
                    "metrics": {"type": "array", "items": {"type": "string"}},
                    "reduction": {"type": "object", "description": '{"retain_dim": <int>}.'},
                    "k": {"type": "integer", "minimum": 1, "maximum": 64},
                    "rank_by": {"type": "string"},
                    "max_rows": _max_rows_property(),
                    "max_tokens": _max_tokens_property(_budgets.ROW_TOOL_MAX_TOKENS),
                },
                "required": ["path"],
                "additionalProperties": False,
            },
            output_schema="torchlens.agent.payload_stats.v1",
            tool_class="payload",
            cli_verb="stats",
            handler=_handle_payload_stats,
            default_limits={
                "max_rows": _budgets.DEFAULT_MAX_ROWS,
                "max_tokens": _budgets.ROW_TOOL_MAX_TOKENS,
                "max_blob_bytes": _budgets.MAX_BLOB_BYTES,
                "max_call_bytes": _budgets.MAX_CALL_BYTES,
            },
        ),
        AgentToolSpec(
            name="torchlens_compare",
            description=(
                "Structural + value diff of two saved artifacts; direction is "
                "always subject - reference. Fingerprints compare FIRST (a "
                "mismatch is loud relationship evidence, never a silent "
                "numeric diff of different models); matching keys on the "
                "structural site_key with the legacy fallback disclosed; the "
                "coverage header is non-droppable -- 0 mismatches never means "
                "'no differences' when nothing comparable was saved."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "reference": _path_property(),
                    "subject": _path_property(),
                    "rtol": {"type": "number"},
                    "atol": {"type": "number"},
                    "max_rows": _max_rows_property(),
                    "max_tokens": _max_tokens_property(_budgets.ROW_TOOL_MAX_TOKENS),
                },
                "required": ["reference", "subject"],
                "additionalProperties": False,
            },
            output_schema="torchlens.agent.compare.v1",
            tool_class="payload",
            cli_verb="diff",
            handler=_handle_compare,
            default_limits={
                "max_rows": _budgets.DEFAULT_MAX_ROWS,
                "max_tokens": _budgets.ROW_TOOL_MAX_TOKENS,
                "max_blob_bytes": _budgets.MAX_BLOB_BYTES,
                "max_call_bytes": _budgets.MAX_CALL_BYTES,
            },
        ),
        AgentToolSpec(
            name="torchlens_schema",
            description=(
                "Fetch the machine-readable schema contract at runtime: the "
                "index of served schema ids, or one shipped JSON Schema "
                "(Draft 2020-12) document by id."
            ),
            input_schema={
                "type": "object",
                "properties": {"name": {"type": "string", "description": "Optional schema id."}},
                "additionalProperties": False,
            },
            output_schema="torchlens.agent.schema.v1",
            tool_class="environment",
            cli_verb="schema",
            handler=_handle_schema,
        ),
    )


#: Row keys per tool for the token-budget backstop (droppable rows).
_ROW_KEYS: dict[str, str | tuple[str, ...]] = {
    "torchlens_dump": ("rows", "ops"),
    "torchlens_query_sites": "rows",
    "torchlens_payload_stats": "rows",
    "torchlens_compare": "rows",
    "torchlens_overview": "classes",
    "torchlens_api_map": "names",
}


def spec_for(name: str) -> AgentToolSpec:
    """Return one tool spec by name.

    Raises
    ------
    InvalidArgumentError
        ``agent_tool_unknown`` naming the served roster.
    """

    for spec in tool_specs():
        if spec.name == name:
            return spec
    raise InvalidArgumentError(
        f"unknown tool {name!r}",
        code="agent_tool_unknown",
        remedy=f"served tools: {', '.join(spec.name for spec in tool_specs())}",
    )


def call_tool(name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    """Dispatch one tool call: validate, run, budget, return the envelope.

    Parameters
    ----------
    name:
        Registered tool name.
    arguments:
        JSON arguments matching the tool's declared input schema.

    Returns
    -------
    dict[str, Any]
        The result envelope (canonical-serializable).

    Raises
    ------
    torchlens.errors.TorchLensError
        Typed refusals (Python callers branch on ``exc.fields['code']``);
        transports that need an error ENVELOPE use
        :func:`call_tool_envelope`.
    """

    spec = spec_for(name)
    args = dict(arguments or {})
    _validate_args(spec, args)
    envelope = spec.handler(args)
    if spec.budgets_in_handler:
        return json_safe(envelope)
    max_tokens = _budgets.resolve_limit(
        args.get("max_tokens") if "max_tokens" in spec.input_schema.get("properties", {}) else None,
        spec.default_limits.get("max_tokens", _budgets.ROW_TOOL_MAX_TOKENS),
        name="max_tokens",
    )
    envelope = _fit_token_budget(envelope, max_tokens, row_key=_ROW_KEYS.get(name))
    limits = dict(envelope.get("limits") or {})
    limits.setdefault("max_tokens", max_tokens)
    limits.setdefault("token_estimator", _budgets.TOKEN_ESTIMATOR)
    envelope["limits"] = limits
    return json_safe(envelope)


def call_tool_envelope(name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    """Transport-facing dispatch: typed failures become error envelopes.

    Parameters
    ----------
    name:
        Registered tool name.
    arguments:
        JSON arguments.

    Returns
    -------
    dict[str, Any]
        Result envelope, or the ``torchlens.agent.error.v1`` envelope.
    """

    try:
        return call_tool(name, arguments)
    except Exception as exc:  # noqa: BLE001 - transport boundary: hosts get an error ENVELOPE, never a crash
        return build_error_envelope(exc)
