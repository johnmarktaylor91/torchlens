"""Nested wrapped-call bookkeeping for the guarded fast live run.

Capture records the innermost wrapped call that produced a tensor: an outer
wrapped call that hands back a nested call's output adds no op of its own,
whatever its name. ``F.relu(x, inplace=True)`` runs ``torch.relu_`` inside the
wrapped ``F.relu``, so capture records one ``relu_`` op and the outer ``relu``
call vanishes. Capture tells the two apart with a per-call barcode that any
nested wrapped call overwrites; this mixin keeps the same marker for the live
session so the outer event is dropped exactly when capture dropped it, instead
of being read as a separate ``relu`` event that a later plain ``relu`` plan
claims (a false arm divergence).
"""

from __future__ import annotations

import threading
from typing import Any

import torch

__all__ = ["_NestedEventTracker", "_output_tensor_leaves", "returns_nested_output"]


def _output_tensor_leaves(output: Any) -> tuple[Any, ...]:
    """Return the tensor leaves of one live call output (one container level)."""

    if isinstance(output, torch.Tensor):
        return (output,)
    if isinstance(output, (tuple, list)):
        return tuple(item for item in output if isinstance(item, torch.Tensor))
    return ()


def returns_nested_output(nested_output: Any, output: Any) -> bool:
    """Return whether every tensor an outer call returns came from its nested call.

    Parameters
    ----------
    nested_output:
        Output of the last wrapped call that finished inside the outer call.
    output:
        Output of the outer call.
    """

    if nested_output is None:
        return False
    nested_leaves = _output_tensor_leaves(nested_output)
    if isinstance(output, torch.Tensor):
        return output is nested_output or any(output is leaf for leaf in nested_leaves)
    leaves = _output_tensor_leaves(output)
    return bool(leaves) and all(any(leaf is kept for kept in nested_leaves) for leaf in leaves)


class _NestedEventTracker:
    """Mirror capture's nesting barcode over the owner thread's wrapped calls.

    The host class supplies ``active``, ``collection_paused``,
    ``owner_thread_id`` and ``capture_function``, which skips the function
    names no plan wants.
    """

    active: bool
    collection_paused: bool
    owner_thread_id: int | None
    _event_serial: int = 0
    _event_marker: int = 0
    _last_event_output: Any = None
    # The last collected functional event: its name, output object and tensor
    # leaves, held strongly so a later same-name event is compared by object
    # identity, never by a reusable ``id()``.
    last_function_output: tuple[str, Any, tuple[Any, ...]] | None = None

    def reset_event_nesting(self) -> None:
        """Forget the previous run's nesting marker and outputs."""

        self._event_serial = 0
        self._event_marker = 0
        self._last_event_output = None
        self.last_function_output = None

    def enter_event(self, transparent: bool) -> tuple[int, int, bool] | None:
        """Open one wrapped call; ``None`` when the session does not observe it.

        Parameters
        ----------
        transparent:
            Whether capture restores the enclosing call's barcode after this
            call (``as_subclass`` return conversion and storage accessors).
        """

        if (
            not self.active
            or self.collection_paused
            or threading.get_ident() != self.owner_thread_id
        ):
            return None
        self._event_serial += 1
        prior = self._event_marker
        self._event_marker = self._event_serial
        return (self._event_serial, prior, transparent)

    def exit_event(
        self,
        token: tuple[int, int, bool],
        func_name: str,
        output: Any,
        args: tuple[Any, ...],
        property_accessor: str | None,
    ) -> None:
        """Close one wrapped call and hand it to collection unless capture dropped it."""

        serial, prior, transparent = token
        nested = self._event_marker != serial
        if transparent:
            self._event_marker = prior
        nested_output = self._last_event_output
        self._last_event_output = output
        if nested and returns_nested_output(nested_output, output):
            return
        host: Any = self
        host.capture_function(func_name, output, args, property_accessor)
