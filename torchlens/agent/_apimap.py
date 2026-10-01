"""api_map v2: compact rows from ONE authoritative export registry (memo 3.3).

Rows are generated from the C01 surface table (``torchlens._surface``), never
hand-listed, so the map can never advertise a spelling the facade does not
serve. Signatures are NOT inlined per row (measured 2.8x token tax); the
per-name detail mode serves ONE full record instead. The capabilities block
declares tool names, CLI verbs, schema ids, budget parameter names, and the
explicit ``writes: false`` / ``executes_user_code: false`` facts.
"""

from __future__ import annotations

import inspect
from typing import Any

#: Result schema id (v2: compact rows + capabilities + detail mode).
API_MAP_SCHEMA = "torchlens.agent.api_map.v2"

#: Elision ceiling on required_params entries per compact row.
_REQUIRED_PARAMS_MAX = 6

#: Curated canonical examples for the flagship entry points.
_EXAMPLES: dict[str, str] = {
    "trace": "log = tl.trace(model, x, save=tl.func('relu'))",
    "load": "log = tl.load('run.tlspec')",
    "save": "tl.save(log, 'run.tlspec')",
    "record": "rec = tl.record(model, x, save=tl.func('relu'))",
    "func": "tl.trace(model, x, save=tl.func('relu'))",
    "in_module": "tl.trace(model, x, save=tl.in_module('encoder'))",
    "when": "tl.trace(model, x, intervene=tl.when(tl.func('relu'), tl.zero_ablate()))",
}


def _resolve(name: str) -> Any:
    """Resolve one root name through the live facade (import side effects OK here)."""

    import torchlens

    try:
        return getattr(torchlens, name)
    except AttributeError:
        return None


def _kind_of(value: Any) -> str:
    """Classify one resolved object for the map row."""

    import types

    if value is None:
        return "unresolvable"
    if isinstance(value, type):
        return "class"
    if isinstance(value, types.ModuleType):
        return "module"
    if callable(value):
        return "function"
    return type(value).__name__


def _required_params(value: Any) -> tuple[list[str], bool]:
    """Return required (no-default) parameter names, elided past the ceiling."""

    if not callable(value) or isinstance(value, type):
        target = getattr(value, "__init__", None) if isinstance(value, type) else None
        if target is None:
            return [], False
        value = target
    try:
        signature = inspect.signature(value)
    except (TypeError, ValueError):
        return [], False
    required = [
        parameter.name
        for parameter in signature.parameters.values()
        if parameter.default is inspect.Parameter.empty
        and parameter.kind
        in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        and parameter.name not in ("self", "cls")
    ]
    if len(required) > _REQUIRED_PARAMS_MAX:
        return required[:_REQUIRED_PARAMS_MAX], True
    return required, False


def _summary_of(value: Any) -> str:
    """First docstring line of one resolved object."""

    doc = (getattr(value, "__doc__", None) or "").strip()
    return doc.splitlines()[0] if doc else ""


def compact_rows() -> list[dict[str, Any]]:
    """Build the compact api_map rows from the surface registry.

    Returns
    -------
    list[dict[str, Any]]
        One row per root surface name: name, kind, canonical module, one-line
        summary, deprecation flag, required_params (elision disclosed), arity.
    """

    from .._surface import surface_rows

    rows: list[dict[str, Any]] = []
    for surface_row in surface_rows():
        name = surface_row.canonical_path.rsplit(".", 1)[1]
        value = _resolve(name)
        required, elided = _required_params(value)
        rows.append(
            {
                "name": name,
                "kind": _kind_of(value),
                "module": surface_row.home_module,
                "summary": _summary_of(value),
                "in_all": surface_row.in_all,
                "deprecated": False,
                "required_params": required,
                "required_params_elided": elided,
            }
        )
    return rows


def name_detail(name: str) -> dict[str, Any]:
    """Build the ONE detailed record for a named public spelling.

    Parameters
    ----------
    name:
        Root surface name (``"trace"``).

    Returns
    -------
    dict[str, Any]
        Full normalized signature, docstring, canonical example, doc anchor.

    Raises
    ------
    InvalidArgumentError
        ``agent_name_unknown`` when the name is not on the surface.
    """

    from .._errors import InvalidArgumentError

    value = _resolve(name)
    if value is None:
        raise InvalidArgumentError(
            f"{name!r} is not a torchlens root surface name",
            code="agent_name_unknown",
            remedy="list the surface with api_map first; submodule members are documented on their module",
        )
    signature_text: str | None
    try:
        signature_text = f"{name}{inspect.signature(value)}"
    except (TypeError, ValueError):
        signature_text = None
    doc = inspect.getdoc(value) or ""
    return {
        "name": name,
        "kind": _kind_of(value),
        "module": getattr(value, "__module__", None) or "torchlens",
        "signature": signature_text,
        "doc": doc[:4000],
        "doc_truncated": len(doc) > 4000,
        "example": _EXAMPLES.get(name),
        # The SHIPPED resource, never a repo-relative path a wheel install
        # lacks (the v1 map's dead docs pointer, agent memo 3.10 item 1).
        "doc_anchor": "torchlens.agent.guide()",
    }


def capabilities_block() -> dict[str, Any]:
    """Declare the agent surface's own capabilities (memo 3.3 item 2).

    Returns
    -------
    dict[str, Any]
        Tool names, CLI verbs, schema ids, budget parameter names, and the
        explicit write/execute facts.
    """

    from . import _registry
    from ._schemas import schema_index

    specs = _registry.tool_specs()
    return {
        "tools": [spec.name for spec in specs],
        "cli_verbs": sorted({spec.cli_verb for spec in specs if spec.cli_verb}),
        "schemas": sorted(schema_index()),
        "budget_parameters": ["max_tokens", "max_rows", "max_blob_bytes", "max_call_bytes"],
        "token_estimator": "approx_chars_div_4",  # gitleaks:allow (estimator name)
        "path_root_ceiling": None,
        "writes": False,
        "executes_user_code": False,
    }


def submodule_map() -> dict[str, str]:
    """The deliberately-unlisted power submodules an agent should know about."""

    return {
        "tl.report": "explain(), TraceProfile/build_profile, log_value",
        "tl.compat": "compat.report(model, x): capture-compatibility findings",
        "tl.debug": "power-user diagnostics (bisect_nan, hot_path, ...)",
        "tl.receptive_field": "lazy influence-geometry submodule",
        "tl.bridge": "optional external-tool adapters (captum, shap, mcp, ...)",
        "tl.agent": "this agent surface (call_tool, guide, registry)",
    }
