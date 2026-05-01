"""Private compatibility shim for moved summary internals."""

from ..visualization._summary_internal import (
    format_discoverability_summary,
    format_model_repr,
    render_model_summary,
)

__all__ = ["format_discoverability_summary", "format_model_repr", "render_model_summary"]
