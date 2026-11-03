"""Compatibility door for the Dagua bridge, which lives in the visualization package.

The bridge body is :mod:`torchlens._visualization._dagua_bridge`; this module keeps the
``torchlens.experimental.dagua._bridge`` spelling importable for the opt-in package.
"""

from __future__ import annotations

from ..._visualization._dagua_bridge import (
    MODELLOG_FIELD_USAGE,
    TorchLensRenderAudit,
    _container_semantic_attrs,
    build_render_audit,
    build_torchlens_caption,
    render_trace_with_dagua,
    trace_to_dagua_graph,
)

__all__ = [
    "MODELLOG_FIELD_USAGE",
    "TorchLensRenderAudit",
    "build_render_audit",
    "build_torchlens_caption",
    "render_trace_with_dagua",
    "trace_to_dagua_graph",
]
_ = _container_semantic_attrs
