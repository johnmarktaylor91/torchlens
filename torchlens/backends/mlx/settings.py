"""Backend-local MLX settings (no public trace option grows for one backend).

The one setting today is the unwrapped-op policy: what a capture does when
the forward produced an output or intermediate that no wrapped operation made
(an ``mlx.core`` function outside the wrapped surface, an array constructed
inside the forward, an unwrapped submodule function). ``"flag"`` keeps the
trace and marks it ``capture_verified=False`` with reason
``unwrapped_op_in_capture``; ``"refuse"`` raises ``BackendUnsupportedError``
with that code instead. The spelling ``unwrapped_policy`` is coded here once.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Literal

from .._entry_refusals import refuse_mlx_unwrapped_policy

UnwrappedPolicy = Literal["flag", "refuse"]

#: The capture outcomes a caller may choose for an unwrapped op.
UNWRAPPED_POLICIES: frozenset[str] = frozenset({"flag", "refuse"})

#: Refusal code and ``capture_verification_reason`` for the unwrapped-op gap.
UNWRAPPED_OP_REASON = "unwrapped_op_in_capture"

_unwrapped_policy: str = "flag"


def get_unwrapped_policy() -> str:
    """The current unwrapped-op policy, ``"flag"`` (default) or ``"refuse"``."""

    return _unwrapped_policy


def set_unwrapped_policy(policy: str) -> str:
    """Set the unwrapped-op policy and return the previous one.

    Parameters
    ----------
    policy:
        ``"flag"`` to keep a flagged trace, ``"refuse"`` to raise instead.

    Returns
    -------
    str
        The policy in force before this call.

    Raises
    ------
    ValueError
        If ``policy`` is not one of :data:`UNWRAPPED_POLICIES`.
    """

    global _unwrapped_policy
    if policy not in UNWRAPPED_POLICIES:
        refuse_mlx_unwrapped_policy(policy, UNWRAPPED_POLICIES)
    previous = _unwrapped_policy
    _unwrapped_policy = policy
    return previous


@contextmanager
def unwrapped_policy(policy: str) -> Iterator[None]:
    """Apply an unwrapped-op policy for the duration of a block.

    Parameters
    ----------
    policy:
        ``"flag"`` or ``"refuse"``.
    """

    previous = set_unwrapped_policy(policy)
    try:
        yield
    finally:
        set_unwrapped_policy(previous)


__all__ = [
    "UNWRAPPED_OP_REASON",
    "UNWRAPPED_POLICIES",
    "UnwrappedPolicy",
    "get_unwrapped_policy",
    "set_unwrapped_policy",
    "unwrapped_policy",
]
