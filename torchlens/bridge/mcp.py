"""Model Context Protocol (MCP) bridge: serve TorchLens over stdio.

DOCUMENTED-UNSTABLE surface (naming ratification pending). The server exposes
read-only tools over SAVED ``.tlspec`` artifacts and the runtime environment,
wrapping the SAME public surface a human drives (``tl.load``,
``Trace.summary``, ``Trace.to_agent_json``, ``tl.report.explain``,
``tl.utils.doctor``) -- never a parallel API. No tool executes user code, and
no tool mutates anything: live capture stays a Python-process concern.

Run it as ``python -m torchlens.bridge.mcp`` (stdio transport; requires the
``mcp`` extra: ``pip install torchlens[mcp]``). The pure tool layer
(``TOOL_SPECS`` / ``call_tool``) has no ``mcp`` dependency so hosts and tests
can drive it directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

#: JSON Schema tool declarations served verbatim over MCP ``list_tools``.
TOOL_SPECS: tuple[dict[str, Any], ...] = (
    {
        "name": "torchlens_doctor",
        "description": (
            "Run the TorchLens environment health check (PyTorch/CUDA/"
            "Graphviz/extras/capability flags). Call this first when captures "
            "misbehave; each failing row names what is missing."
        ),
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "torchlens_api_map",
        "description": (
            "Machine-readable index of the public torchlens surface: every "
            "name in torchlens.__all__ with its kind and first docstring "
            "line. Use it to discover the exact spelling to write in Python."
        ),
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "torchlens_load_overview",
        "description": (
            "Load a saved .tlspec Trace artifact (analysis-only, no code "
            "execution) and return a bounded text summary plus "
            "capture-honesty facts and the disclosed load_plan (manifest-"
            "first eager/lazy/refuse decision)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to a .tlspec artifact."}
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    },
    {
        "name": "torchlens_agent_dump",
        "description": (
            "Return the torchlens.agent_trace.v1 machine-readable dump of a "
            "saved .tlspec Trace: capture facts, counts, pass-qualified op "
            "rows with graph edges, module hierarchy, and a navigation guide. "
            "Bounded by default (op rows cap at 2,000 when max_ops is "
            "omitted; the truncation block discloses omissions)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to a .tlspec artifact."},
                "max_ops": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10_000,
                    "description": (
                        "Cap on op rows (default 2,000; served ceiling "
                        "10,000); omissions are disclosed."
                    ),
                },
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    },
    {
        "name": "torchlens_explain",
        "description": (
            "Plain-language report over a saved .tlspec Trace, optionally "
            "budgeted: max_tokens drops whole sections low-value-first and "
            "discloses every drop; capture-status honesty facts never drop."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to a .tlspec artifact."},
                "max_tokens": {
                    "type": "integer",
                    "minimum": 1,
                    "description": "Optional token budget (~4 chars/token).",
                },
                "audience": {
                    "type": "string",
                    "enum": ["researcher", "practitioner", "auto"],
                    "description": "Report style; defaults to 'auto'.",
                },
            },
            "required": ["path"],
            "additionalProperties": False,
        },
    },
)

#: Manifest-declared payload bytes above which the server loads LAZILY
#: (payload blobs stay on disk; reads sha-verify one blob at a time).
EAGER_LOAD_MAX_BYTES = 256 * 1024 * 1024

#: Declared payload-entry count above which the load is REFUSED: even the
#: metadata materialization of such an artifact is not a safe single tool
#: call. The bound is disclosed in the refusal.
LOAD_MAX_PAYLOAD_ENTRIES = 100_000

#: Server-side default cap on agent-dump op rows when the caller passes no
#: ``max_ops`` (the dump then carries an explicit ``truncation`` block).
DUMP_DEFAULT_MAX_OPS = 2_000

#: Hard ceiling on caller-requested ``max_ops``; larger requests refuse with
#: the bound named rather than materializing an unbounded response.
DUMP_MAX_OPS_CEILING = 10_000

#: Content-digest-keyed cache of loaded artifacts (AG stage-0 item 3): the
#: artifact's identity is the sha256 of its manifest bytes, never the path.
_TRACE_CACHE: dict[str, tuple[Any, dict[str, Any]]] = {}
_TRACE_CACHE_MAX = 4

#: (path, dev, ino, size, mtime_ns) -> digest. A VERIFIED cache hint only:
#: the full stat identity must match to skip re-hashing (rename-replace
#: changes st_ino, an in-place rewrite changes size/mtime_ns -- the same
#: identity contract the lazy blob reader trusts); the digest stays the one
#: true cache key.
_DIGEST_HINTS: dict[tuple[str, int, int, int, int], str] = {}
_DIGEST_HINTS_MAX = 16


def _artifact_digest(path: Path) -> tuple[str, bytes | None]:
    """Return the artifact's content digest plus raw manifest bytes.

    Parameters
    ----------
    path:
        Artifact path (a ``.tlspec`` directory or a single file).

    Returns
    -------
    tuple[str, bytes | None]
        Hex digest identifying the artifact content, and the manifest bytes
        when the artifact has a readable ``manifest.json`` (``None`` for
        single-file artifacts, which are stream-hashed).
    """

    from hashlib import sha256

    manifest_path = path / "manifest.json" if path.is_dir() else None
    if manifest_path is not None and manifest_path.is_file():
        manifest_bytes = manifest_path.read_bytes()
        return sha256(manifest_bytes).hexdigest(), manifest_bytes
    from torchlens._io.manifest import sha256_of_file

    return sha256_of_file(path), None


def _stat_identity(path: Path) -> tuple[str, int, int, int, int] | None:
    """Return the (path, dev, ino, size, mtime_ns) identity for the hint map."""

    probe = path / "manifest.json" if path.is_dir() else path
    try:
        stat_result = probe.stat()
    except OSError:
        return None
    return (
        str(path),
        stat_result.st_dev,
        stat_result.st_ino,
        stat_result.st_size,
        stat_result.st_mtime_ns,
    )


def _build_load_plan(manifest_bytes: bytes | None) -> dict[str, Any]:
    """Choose eager/lazy/refuse from manifest-DECLARED numbers, torch-free.

    Parameters
    ----------
    manifest_bytes:
        Raw ``manifest.json`` bytes, or ``None`` when the artifact has none.

    Returns
    -------
    dict[str, Any]
        The disclosed ``load_plan``: mode, declared payload bytes/count, the
        thresholds applied, and the reason.
    """

    import json as json_module

    declared_bytes = 0
    declared_count = 0
    if manifest_bytes is not None:
        from torchlens._io.payload_reader import declared_payload_bytes

        try:
            manifest = json_module.loads(manifest_bytes)
        except (ValueError, UnicodeDecodeError):
            manifest = {}
        if isinstance(manifest, dict):
            declared_bytes = declared_payload_bytes(manifest)
            body_index = manifest.get("body_index")
            declared_count = len(body_index) if isinstance(body_index, list) else 0
    if declared_count > LOAD_MAX_PAYLOAD_ENTRIES:
        mode = "refuse"
        reason = (
            f"declared payload entries ({declared_count:,}) exceed the "
            f"single-tool-call bound ({LOAD_MAX_PAYLOAD_ENTRIES:,})"
        )
    elif declared_bytes > EAGER_LOAD_MAX_BYTES:
        mode = "lazy"
        reason = (
            f"declared payload bytes ({declared_bytes:,}) exceed the eager "
            f"threshold ({EAGER_LOAD_MAX_BYTES:,}); payloads stay on disk"
        )
    else:
        mode = "eager"
        reason = "declared payload bytes fit the eager threshold"
    return {
        "mode": mode,
        "declared_payload_bytes": declared_bytes,
        "declared_payload_count": declared_count,
        "eager_threshold_bytes": EAGER_LOAD_MAX_BYTES,
        "max_payload_entries": LOAD_MAX_PAYLOAD_ENTRIES,
        "reason": reason,
    }


def _resolve_digest(path: Path) -> tuple[str, bytes | None]:
    """Resolve the artifact content digest through the stat-identity hint map.

    ``(path, dev, ino, size, mtime_ns)`` survives only as a hint that skips
    re-hashing when the full file identity matches; the digest itself is the
    cache authority. A hint hit returns ``None`` manifest bytes (nothing was
    read); a fresh hash returns whatever ``_artifact_digest`` read.

    Parameters
    ----------
    path:
        Artifact path (a ``.tlspec`` directory or a single file).

    Returns
    -------
    tuple[str, bytes | None]
        Content digest, and the manifest bytes when hashing read them.
    """

    identity = _stat_identity(path)
    if identity is not None:
        hinted = _DIGEST_HINTS.get(identity)
        if hinted is not None:
            return hinted, None
    digest, manifest_bytes = _artifact_digest(path)
    if identity is not None:
        if len(_DIGEST_HINTS) >= _DIGEST_HINTS_MAX:
            _DIGEST_HINTS.pop(next(iter(_DIGEST_HINTS)))
        _DIGEST_HINTS[identity] = digest
    return digest, manifest_bytes


def _load_trace(path_arg: str) -> tuple[Any, dict[str, Any]]:
    """Load a saved Trace artifact for read-only inspection, with caching.

    The load is MANIFEST-FIRST (AG stage-0 item 2): declared payload bytes
    decide eager/lazy/refuse BEFORE any blob is opened, so one tool call can
    never materialize an unbounded artifact. The cache key is the artifact's
    content digest (AG stage-0 item 3), resolved through ``_resolve_digest``'s
    stat-identity hint map.

    Parameters
    ----------
    path_arg:
        Filesystem path to a ``.tlspec`` artifact.

    Returns
    -------
    tuple[Any, dict[str, Any]]
        Loaded ``Trace`` and the disclosed ``load_plan``.

    Raises
    ------
    ValueError
        If the path does not exist, the artifact exceeds the load bound, or
        it is not a single Trace (bundles and intervention specs are out of
        the v1 tool contract).
    """

    import torchlens as tl

    path = Path(path_arg).expanduser()
    if not path.exists():
        raise ValueError(
            f"No file at {str(path)!r}. Pass the path of an artifact saved "
            "with tl.save(trace, path)."
        )
    digest, manifest_bytes = _resolve_digest(path)
    cached = _TRACE_CACHE.get(digest)
    if cached is not None:
        return cached
    if manifest_bytes is None and path.is_dir():
        manifest_path = path / "manifest.json"
        if manifest_path.is_file():
            manifest_bytes = manifest_path.read_bytes()
    plan = _build_load_plan(manifest_bytes)
    if plan["mode"] == "refuse":
        raise ValueError(
            f"Refusing to load {str(path)!r}: {plan['reason']}. Load it in "
            "Python via tl.load(path, lazy=True) where you control the "
            "process budget."
        )
    loaded = tl.load(path, lazy=plan["mode"] == "lazy")
    if not isinstance(loaded, tl.Trace):
        raise ValueError(
            f"{str(path)!r} loaded as {type(loaded).__name__}, not a Trace. "
            "The v1 MCP tools cover single-Trace artifacts; load bundles or "
            "intervention specs in Python via tl.load(...)."
        )
    if len(_TRACE_CACHE) >= _TRACE_CACHE_MAX:
        _TRACE_CACHE.pop(next(iter(_TRACE_CACHE)))
    result = (loaded, plan)
    _TRACE_CACHE[digest] = result
    return result


def _tool_doctor() -> dict[str, Any]:
    """Run the environment health check.

    Returns
    -------
    dict[str, Any]
        Doctor rows as ``{"checks": [{"name", "status", "detail"}, ...]}``.
    """

    from ..utils import doctor

    report = doctor()
    return {
        "checks": [
            {"name": check.name, "status": check.status, "detail": check.detail}
            for check in report.checks
        ]
    }


def _api_map_entry(module: Any, name: str) -> dict[str, Any]:
    """Describe one public name for the API map.

    Parameters
    ----------
    module:
        Module owning the name (``torchlens``).
    name:
        Public attribute name.

    Returns
    -------
    dict[str, Any]
        ``{"name", "kind", "summary"}`` row.
    """

    value = getattr(module, name, None)
    if isinstance(value, type):
        kind = "class"
    elif callable(value):
        kind = "function"
    else:
        kind = type(value).__name__
    doc = (getattr(value, "__doc__", None) or "").strip()
    summary = doc.splitlines()[0] if doc else ""
    return {"name": name, "kind": kind, "summary": summary}


def _tool_api_map() -> dict[str, Any]:
    """Build the machine-readable public-surface index.

    Returns
    -------
    dict[str, Any]
        Every ``torchlens.__all__`` name with kind and first docstring line,
        plus the deliberately-unlisted submodules an agent should know about.
    """

    import torchlens as tl

    return {
        "schema": "torchlens.api_map.v1",
        "names": [_api_map_entry(tl, name) for name in sorted(tl.__all__)],
        "submodules_not_in_all": {
            "tl.report": "explain(), TraceProfile/build_profile, log_value",
            "tl.compat": "compat.report(model, x): capture-compatibility findings",
            "tl.debug": "power-user diagnostics (bisect_nan, hot_path, ...)",
            "tl.receptive_field": "lazy influence-geometry submodule",
            "tl.bridge": "optional external-tool adapters (captum, shap, mcp, ...)",
        },
        "docs": "docs/for-ai-agents.md is the agent-facing map of this surface.",
    }


def _tool_load_overview(path: str) -> dict[str, Any]:
    """Summarize a saved Trace artifact.

    Parameters
    ----------
    path:
        Filesystem path to a ``.tlspec`` artifact.

    Returns
    -------
    dict[str, Any]
        Text summary plus the capture block of the agent dump.
    """

    trace, load_plan = _load_trace(path)
    dump = trace.to_agent_json(max_ops=1)
    if int(getattr(trace, "num_ops", 0) or 0) > DUMP_DEFAULT_MAX_OPS:
        # summary() emits one line per layer, so a huge trace would make this
        # "overview" tool the unbounded response; the budgeted explain report
        # is the bounded stand-in and disclosed as such.
        from torchlens.report import explain

        summary_text = str(explain(trace, max_tokens=2000))
        summary_form = "budgeted_explain"
    else:
        summary_text = trace.summary()
        summary_form = "full_summary"
    return {
        "summary": summary_text,
        "summary_form": summary_form,
        "capture": dump["capture"],
        "counts": dump["counts"],
        "load_plan": load_plan,
    }


def _tool_agent_dump(path: str, max_ops: int | None = None) -> dict[str, Any]:
    """Return the agent dump of a saved Trace artifact.

    Parameters
    ----------
    path:
        Filesystem path to a ``.tlspec`` artifact.
    max_ops:
        Optional cap on emitted op rows.

    Returns
    -------
    dict[str, Any]
        ``torchlens.agent_trace.v1`` dump.
    """

    if max_ops is not None and max_ops > DUMP_MAX_OPS_CEILING:
        raise ValueError(
            f"max_ops={max_ops:,} exceeds the served ceiling "
            f"({DUMP_MAX_OPS_CEILING:,}): one tool call must stay bounded. "
            "Page through the graph with smaller dumps, or load the artifact "
            "in Python via tl.load(...).to_agent_json()."
        )
    trace, load_plan = _load_trace(path)
    # Bounded by DEFAULT (WT1 A-V row 25): an omitted max_ops used to dump
    # every op row; the default cap keeps one call bounded and the dump's
    # truncation block discloses exactly what was omitted.
    result = dict(trace.to_agent_json(max_ops=max_ops or DUMP_DEFAULT_MAX_OPS))
    result["load_plan"] = load_plan
    return result


def _tool_explain(
    path: str,
    max_tokens: int | None = None,
    audience: str = "auto",
) -> dict[str, Any]:
    """Return the plain-language report of a saved Trace artifact.

    Parameters
    ----------
    path:
        Filesystem path to a ``.tlspec`` artifact.
    max_tokens:
        Optional token budget forwarded to ``tl.report.explain``.
    audience:
        Report style forwarded to ``tl.report.explain``.

    Returns
    -------
    dict[str, Any]
        ``{"report": <text>}``.
    """

    from ..report import explain

    trace, load_plan = _load_trace(path)
    report = explain(trace, audience=audience, max_tokens=max_tokens)  # type: ignore[arg-type]
    return {"report": report, "load_plan": load_plan}


def call_tool(name: str, arguments: dict[str, Any] | None = None) -> dict[str, Any]:
    """Dispatch one MCP tool call to its handler.

    Parameters
    ----------
    name:
        Tool name from :data:`TOOL_SPECS`.
    arguments:
        JSON arguments matching the tool's ``input_schema``.

    Returns
    -------
    dict[str, Any]
        JSON-serializable tool result.

    Raises
    ------
    ValueError
        If the tool name is unknown or the arguments are invalid; the message
        names the valid tools or the fix.
    """

    args = dict(arguments or {})
    if name == "torchlens_doctor":
        return _tool_doctor()
    if name == "torchlens_api_map":
        return _tool_api_map()
    if name == "torchlens_load_overview":
        return _tool_load_overview(_required_path(args))
    if name == "torchlens_agent_dump":
        return _tool_agent_dump(_required_path(args), max_ops=args.get("max_ops"))
    if name == "torchlens_explain":
        return _tool_explain(
            _required_path(args),
            max_tokens=args.get("max_tokens"),
            audience=args.get("audience", "auto"),
        )
    known = ", ".join(spec["name"] for spec in TOOL_SPECS)
    raise ValueError(f"Unknown tool {name!r}. Known tools: {known}.")


#: Served-schema parity registry (AG stage-0 item 5): tool name -> the pure
#: handler whose signature the declared input_schema must match field-for-
#: field. tests/test_report_honesty_mcp.py enforces the parity in both
#: directions, so a schema/handler drift is a red test, not a runtime
#: surprise for the agent reading the served schema.
TOOL_HANDLERS: dict[str, Any] = {
    "torchlens_doctor": _tool_doctor,
    "torchlens_api_map": _tool_api_map,
    "torchlens_load_overview": _tool_load_overview,
    "torchlens_agent_dump": _tool_agent_dump,
    "torchlens_explain": _tool_explain,
}


def _required_path(args: dict[str, Any]) -> str:
    """Extract the required ``path`` argument.

    Parameters
    ----------
    args:
        Tool arguments.

    Returns
    -------
    str
        The ``path`` value.

    Raises
    ------
    ValueError
        If ``path`` is missing or not a string.
    """

    path = args.get("path")
    if not isinstance(path, str) or not path:
        raise ValueError("This tool requires a 'path' string naming a .tlspec artifact.")
    return path


def _spec_description(name: str) -> str:
    """Return the declared description for one tool.

    Parameters
    ----------
    name:
        Tool name from :data:`TOOL_SPECS`.

    Returns
    -------
    str
        Declared tool description.
    """

    return next(str(spec["description"]) for spec in TOOL_SPECS if spec["name"] == name)


def _build_server() -> Any:
    """Build the wired MCP server serving :data:`TOOL_SPECS` over ``call_tool``.

    Returns
    -------
    Any
        ``mcp.server.MCPServer`` instance (mcp>=2.0 high-level API).

    Raises
    ------
    ImportError
        If the ``mcp`` package (>=2.0) is unavailable.
    """

    from mcp.server import MCPServer

    server = MCPServer(
        name="torchlens",
        instructions=(
            "Read-only TorchLens tools over saved .tlspec artifacts and the "
            "runtime environment. Live capture stays in Python: write "
            "tl.trace(...) there and tl.save(...) the result for these tools."
        ),
    )

    @server.tool(name="torchlens_doctor", description=_spec_description("torchlens_doctor"))
    def _doctor() -> dict[str, Any]:
        """Run the TorchLens environment health check."""

        return call_tool("torchlens_doctor")

    @server.tool(name="torchlens_api_map", description=_spec_description("torchlens_api_map"))
    def _api_map() -> dict[str, Any]:
        """Index the public torchlens surface."""

        return call_tool("torchlens_api_map")

    @server.tool(
        name="torchlens_load_overview",
        description=_spec_description("torchlens_load_overview"),
    )
    def _load_overview(path: str) -> dict[str, Any]:
        """Summarize a saved .tlspec Trace artifact."""

        return call_tool("torchlens_load_overview", {"path": path})

    @server.tool(
        name="torchlens_agent_dump",
        description=_spec_description("torchlens_agent_dump"),
    )
    def _agent_dump(path: str, max_ops: int | None = None) -> dict[str, Any]:
        """Dump a saved .tlspec Trace in torchlens.agent_trace.v1 form."""

        arguments: dict[str, Any] = {"path": path}
        if max_ops is not None:
            arguments["max_ops"] = max_ops
        return call_tool("torchlens_agent_dump", arguments)

    @server.tool(name="torchlens_explain", description=_spec_description("torchlens_explain"))
    def _explain(
        path: str,
        max_tokens: int | None = None,
        audience: str = "auto",
    ) -> dict[str, Any]:
        """Report on a saved .tlspec Trace in plain language."""

        arguments: dict[str, Any] = {"path": path, "audience": audience}
        if max_tokens is not None:
            arguments["max_tokens"] = max_tokens
        return call_tool("torchlens_explain", arguments)

    return server


async def _serve_stdio() -> None:
    """Run the MCP stdio server until the host disconnects.

    Raises
    ------
    ImportError
        If the ``mcp`` package is unavailable.
    """

    await _build_server().run_stdio_async()


def main() -> None:
    """Entry point for ``python -m torchlens.bridge.mcp``.

    Raises
    ------
    ImportError
        If the ``mcp`` package is unavailable; the message names the extra.
    """

    try:
        from mcp.server import MCPServer  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "The MCP bridge requires the `mcp` extra (mcp>=2.0): install "
            "torchlens[mcp] (or `pip install 'mcp>=2.0'`)."
        ) from exc
    import asyncio

    asyncio.run(_serve_stdio())


if __name__ == "__main__":
    main()
