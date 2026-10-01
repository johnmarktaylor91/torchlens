"""torchlens.agent -- the read-only agent inspection surface (agent memo 3.1).

One inspection core, one registry, three transports: Python
(:func:`call_tool`), the MCP server (``torchlens.bridge.mcp``), and the CLI
(``python -m torchlens``) are thin adapters over the SAME tool registry.
Transports may override DEFAULTS, never semantics. Every tool is read-only
and idempotent; nothing here executes user code, writes files, or mutates
artifacts -- unconditionally (memo 3.12).

DOCUMENTED-UNSTABLE spellings pending naming ratification. Deliberately NOT
in ``torchlens.__all__`` (frozen root budget): reach it as ``tl.agent``.
"""

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
