"""Which fused normalization calls write the running statistics they are handed.

``batch_norm`` and its native variants (and ``instance_norm``) receive a
module's ``running_mean`` / ``running_var`` buffers on every call, but update
them only when the call's own ``training`` (``use_input_stats``) argument is
true. An eval-mode call reads the statistics and writes nothing, so journaling a
write for it invents two buffer-version layers per normalization layer that the
eager computation does not have. The decision reads the declared flag, never a
value comparison, so a weights-free (meta) capture decides it identically.
``native_group_norm`` is not a mutator: GroupNorm keeps no running statistics
and the call only reads its input, weight and bias, so a registered buffer
handed to it is never journaled as written.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

__all__ = ["_FUSED_MUTATOR_NAMES", "_fused_update_mode", "_is_fused_mutator", "fused_call_writes"]

_FUSED_MUTATOR_NAMES = frozenset(
    {
        "batch_norm",
        "native_batch_norm",
        "_native_batch_norm_legit",
        "_native_batch_norm_legit_no_training",
        "cudnn_batch_norm",
        "instance_norm",
    }
)

# Position and keyword of the argument that turns the running-statistic update
# on. Every listed signature is (input, weight, bias, running_mean,
# running_var, <flag>, ...).
_UPDATE_FLAG_ARGUMENT: dict[str, tuple[int, str]] = {
    "batch_norm": (5, "training"),
    "native_batch_norm": (5, "training"),
    "_native_batch_norm_legit": (5, "training"),
    "cudnn_batch_norm": (5, "training"),
    "instance_norm": (5, "use_input_stats"),
}

_NEVER_WRITES = frozenset({"_native_batch_norm_legit_no_training"})


def _is_fused_mutator(func_name: str) -> bool:
    """Return whether a torch function is a known fused/native buffer mutator."""

    return func_name in _FUSED_MUTATOR_NAMES


def _fused_update_mode(func_name: str) -> bool:
    """Return whether a known fused mutator should emit op-execution write events."""

    return func_name in _FUSED_MUTATOR_NAMES


def fused_call_writes(
    func_name: str, args: Sequence[Any] | None, kwargs: Mapping[str, Any] | None
) -> bool:
    """Return whether this call of a fused mutator can write the buffers it receives.

    Parameters
    ----------
    func_name:
        Wrapped torch function name.
    args:
        Positional arguments of the call, or ``None`` when unknown.
    kwargs:
        Keyword arguments of the call, or ``None`` when unknown.

    Returns
    -------
    bool
        ``False`` for a non-mutator, for the no-training variant, and for a call
        whose update flag is the literal ``False``; ``True`` otherwise, including
        whenever the flag cannot be read (an unknown call keeps its write).

    Examples
    --------
    >>> fused_call_writes("batch_norm", (None,) * 5 + (False, 0.1, 1e-5, True), {})
    False
    >>> fused_call_writes("batch_norm", (None,) * 5 + (True, 0.1, 1e-5, True), {})
    True
    >>> fused_call_writes("relu", (None,), {})
    False
    """

    if func_name not in _FUSED_MUTATOR_NAMES or func_name in _NEVER_WRITES:
        return False
    position_and_keyword = _UPDATE_FLAG_ARGUMENT.get(func_name)
    if position_and_keyword is None:
        return True
    position, keyword = position_and_keyword
    if kwargs is not None and keyword in kwargs:
        flag = kwargs[keyword]
    elif args is not None and len(args) > position:
        flag = args[position]
    else:
        return True
    return flag is not False
