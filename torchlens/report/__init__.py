"""Reporting helpers for TorchLens observer metadata."""

from __future__ import annotations

# ``log_value``'s canonical home is ``torchlens.observers`` (capture-time
# annotation WRITERS live with the observers; reporting surfaces are the
# READERS). This import is the kept compatibility alias for the historical
# ``tl.report.log_value`` spelling.
from ..observers import log_value
from ._explain import explain
from ._profile import TraceProfile, build_profile

__all__ = ["TraceProfile", "build_profile", "explain", "log_value"]
