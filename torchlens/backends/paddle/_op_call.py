"""The wrapped Paddle call a backend's ``emit_paddle_operation`` records.

A leaf module (no torchlens imports) so the wrappers, the intervention
replacement path and both backends (capture and ``tl.record``) share one call
shape without an import cycle.
"""

from __future__ import annotations

from typing import Any, NamedTuple


class PaddleOpCall(NamedTuple):
    """One wrapped Paddle call: the callable, its arguments and what it returned.

    Attributes
    ----------
    func
        The original (unwrapped) callable, re-executed by replay validation.
    args
        Positional arguments of the call.
    kwargs
        Keyword arguments of the call.
    output
        The call's raw output object (tensor or container of tensors).
    """

    func: object
    args: tuple[Any, ...]
    kwargs: dict[str, Any]
    output: object
