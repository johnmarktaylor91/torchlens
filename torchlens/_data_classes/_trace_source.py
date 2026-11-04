"""Captured forward-source annotations, independent of live source files."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from .._errors import InvalidArgumentError

if TYPE_CHECKING:
    from .trace import Trace


def _annotated_source(trace: Trace) -> str:
    """Join captured op frames to the captured root forward's source lines."""
    source = trace._source_code_blob.get("forward", "")
    filename = getattr(source, "file_path", None) or trace.source.forward_file
    first = getattr(source, "line_number", None) or trace.source.forward_line
    if not source or not filename or not isinstance(first, int):
        raise InvalidArgumentError(
            "This trace has no captured forward source to annotate.",
            code="source_unavailable",
            remedy="re-capture a model defined in a source file; save with include_source=True",
        )
    lines = source.splitlines()
    labels: defaultdict[int, list[str]] = defaultdict(list)
    for op in trace.ops:
        if op.is_input or op.is_output or op.is_buffer:
            continue
        # Frames include callers: a child module's ops belong beside the root
        # forward line that invoked it, but a helper's own lines do not.
        for frame in op.code_context:
            if (
                frame.file == filename
                and frame.func_name == "forward"
                and first <= frame.line_number < first + len(lines)
            ):
                labels[frame.line_number].append(op.label)
                break
    width = len(str(first + len(lines) - 1))
    return "\n".join(
        f"{number:>{width}} | {line}"
        + ("  # ops: " + ", ".join(labels[number]) if number in labels else "")
        for number, line in enumerate(lines, first)
    )
