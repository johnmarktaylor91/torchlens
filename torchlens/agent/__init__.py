"""The read-only agent inspection surface: one tool registry exposed through Python (:func:`call_tool`), the MCP server (``torchlens.bridge.mcp``) and the CLI (``python -m torchlens``).

Every tool is read-only and idempotent; nothing here executes user code, writes files, or mutates artifacts. Reach it as ``tl.agent``; it is deliberately not in ``torchlens.__all__``."""

from __future__ import annotations

from typing import Any

from ._envelope import build_envelope, build_error_envelope, canonical_dumps, json_safe
from ._guide_text import AGENT_GUIDE, guide
from ._registry import AgentToolSpec, call_tool, call_tool_envelope, spec_for, tool_specs

__all__ = [
    "AGENT_GUIDE",
    "AgentToolSpec",
    "build_envelope",
    "build_error_envelope",
    "call_tool",
    "call_tool_envelope",
    "canonical_dumps",
    "guide",
    "json_safe",
    "list_tools",
    "spec_for",
    "tool_specs",
]


def list_tools() -> list[dict[str, Any]]:
    """Return the served tool declarations exactly as MCP ``list_tools`` serves them.

    Returns
    -------
    list[dict[str, Any]]
        One declaration per tool: name, description, input/output schema,
        annotations, CLI verb, and default limits -- the field-parity gate
        compares real served output against these rows.

    Examples
    --------
    >>> tools = list_tools()
    >>> "torchlens_doctor" in [tool["name"] for tool in tools]
    True
    >>> tools[0]["annotations"]["readOnlyHint"]
    True
    """

    return [
        {
            "name": spec.name,
            "description": spec.description,
            "input_schema": spec.input_schema,
            "output_schema": spec.output_schema,
            "annotations": {"readOnlyHint": spec.read_only, "idempotentHint": spec.idempotent},
            "tool_class": spec.tool_class,
            "cli_verb": spec.cli_verb,
            "default_limits": dict(spec.default_limits),
        }
        for spec in tool_specs()
    ]
