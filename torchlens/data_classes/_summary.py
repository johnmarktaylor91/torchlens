"""Summary helpers for compact formatting of captured call arguments."""

from typing import Any

import torch

DISPLAY_MAX_DEPTH = 50
"""Display-walk nesting ceiling (r-b4 R27-5).

Display paths must be TOTAL on arbitrary user-shaped data: a self-referential
or absurdly deep captured argument renders a bounded ``<cycle>`` /
``<max-depth>`` marker instead of crashing ``repr``/``summary`` with a raw
``RecursionError``. 50 is far beyond any legible nesting.
"""


def format_call_arg(value: Any, _depth: int = 0, _in_progress: set[int] | None = None) -> str:
    """Render a compact recursive summary for a captured call argument.

    Parameters
    ----------
    value:
        Arbitrary Python value captured from a module's positional or keyword
        arguments.
    _depth:
        Internal recursion depth (callers must not supply this).
    _in_progress:
        Internal path-scoped container-id set used to render container cycles
        as ``<cycle>`` markers (callers must not supply this).

    Returns
    -------
    str
        Recursive string summary using TorchLens' compact call-argument format.
    """
    if isinstance(value, torch.Tensor):
        dtype_name = str(value.dtype).removeprefix("torch.")
        return f"Tensor(shape={tuple(value.shape)}, dtype={dtype_name})"
    from ..ir.workspaces import ReleasedTensorStub  # deferred: layer lint (no eager upward import)

    if isinstance(value, ReleasedTensorStub):
        # F20 W1a: a released module-arg payload renders exactly like the
        # live tensor it replaced -- the summary is shape/dtype-only.
        return f"Tensor(shape={tuple(value.shape)}, dtype={value.dtype})"
    if isinstance(value, (bool, int, float, str)) or value is None:
        return repr(value)
    if isinstance(value, (list, tuple, dict)):
        if _depth >= DISPLAY_MAX_DEPTH:
            return "<max-depth>"
        if _in_progress is None:
            _in_progress = set()
        value_id = id(value)
        if value_id in _in_progress:
            return "<cycle>"
        _in_progress.add(value_id)
        try:
            if isinstance(value, (list, tuple)):
                return (
                    "["
                    + ", ".join(format_call_arg(v, _depth + 1, _in_progress) for v in value)
                    + "]"
                )
            return (
                "{"
                + ", ".join(
                    f"{k}: {format_call_arg(v, _depth + 1, _in_progress)}" for k, v in value.items()
                )
                + "}"
            )
        finally:
            _in_progress.discard(value_id)
    return f"<{type(value).__name__}>"
