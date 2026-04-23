"""Summary helpers for compact formatting of captured call arguments."""

from typing import Any

import torch


def format_call_arg(value: Any) -> str:
    """Render a compact recursive summary for a captured call argument.

    Parameters
    ----------
    value:
        Arbitrary Python value captured from a module's positional or keyword
        arguments.

    Returns
    -------
    str
        Recursive string summary following the IO sprint Fork C rules.
    """
    if isinstance(value, torch.Tensor):
        dtype_name = str(value.dtype).removeprefix("torch.")
        return f"Tensor(shape={tuple(value.shape)}, dtype={dtype_name})"
    if isinstance(value, (bool, int, float, str)) or value is None:
        return repr(value)
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(format_call_arg(v) for v in value) + "]"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{k}: {format_call_arg(v)}" for k, v in value.items()) + "}"
    return f"<{type(value).__name__}>"
