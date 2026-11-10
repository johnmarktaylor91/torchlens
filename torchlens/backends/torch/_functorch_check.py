"""The functorch wrapped-tensor check, with torch's checker resolved once per capture run.

``is_functorch_wrapped_tensor`` resolves ``torch._C._functorch.is_functorch_wrapped_tensor``
through the capability probe on every call. A module entry checks each of its input
tensors, so the resolution ran once per input tensor per module call. The checker this
module returns gives the same answer for a tensor, resolving torch's function at the
run's first module entry and reusing it for the rest of that run.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..._utils._torch_compat import get_functorch_wrapped_tensor_checker

__all__ = ["functorch_wrapped_check"]

_resolved: tuple[int | None, Callable[[Any], bool] | None] = (None, None)
"""``(run nonce, check)`` for the run that resolved last; swapped as one tuple."""


def functorch_wrapped_check(run_nonce: int) -> Callable[[Any], bool] | None:
    """Return ``is_functorch_wrapped_tensor`` for tensors of one capture run.

    Args:
        run_nonce: The capture run's ``CaptureEvents.run_nonce``.

    Returns:
        A function answering ``is_functorch_wrapped_tensor(t)`` for a tensor ``t``, or
        ``None`` when torch has no functorch wrapped-tensor API (every answer is then
        ``False``).
    """

    global _resolved
    nonce, check = _resolved
    if nonce == run_nonce:
        return check
    checker = get_functorch_wrapped_tensor_checker()
    check = None if checker is None else _make_check(checker)
    _resolved = (run_nonce, check)
    return check


def _make_check(checker: Callable[[Any], Any]) -> Callable[[Any], bool]:
    """Wrap torch's checker as ``is_functorch_wrapped_tensor`` does past its resolution."""

    def check(value: Any) -> bool:
        """Return whether torch reports ``value`` as a functorch wrapper tensor."""
        try:
            return bool(checker(value))
        except RuntimeError:
            return False

    return check
