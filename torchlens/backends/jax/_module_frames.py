"""Event-layer module frames for JAX operation events.

Module identity reaches JAX equations only through decoded ``named_scope``
labels; these helpers turn a decoded stack into the ``ModuleFrame`` tuple an
operation event (and an intervention predicate context) carries.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..._ir.events import ModuleFrame


def jax_event_module_frames(
    module_stack: Sequence[str], module_call_stack: Sequence[tuple[str, int]]
) -> tuple[tuple[tuple[str, int], ...], tuple[ModuleFrame, ...]]:
    """Return the event module calls and frames for one decoded JAX stack."""

    module_calls = _jax_event_module_call_stack(
        _jax_event_module_stack(module_stack), module_call_stack
    )
    return module_calls, tuple(_jax_module_frame(addr, index) for addr, index in module_calls)


def _jax_event_module_stack(module_stack: Sequence[str]) -> tuple[str, ...]:
    """Normalize decoded JAX module scopes for TorchLens Op containment.

    Root-only equations are attributed to ``self`` so non-function-root
    invariants have an owning module. Nested equations omit ``self`` because
    Torch module stacks historically contain only submodule calls.

    Parameters
    ----------
    module_stack
        Decoded module scopes in outer-to-inner order.

    Returns
    -------
    tuple[str, ...]
        Module stack to store on JAX operation events.
    """

    stack = tuple(module_stack)
    if len(stack) > 1 and stack[0] == "self":
        return stack[1:]
    return stack


def _jax_event_module_call_stack(
    module_addresses: Sequence[str],
    module_call_stack: Sequence[tuple[str, int]],
) -> tuple[tuple[str, int], ...]:
    """Normalize decoded JAX module-call scopes for event emission.

    Parameters
    ----------
    module_addresses
        Address-only module stack after root elision.
    module_call_stack
        Decoded module-call stack in outer-to-inner order.

    Returns
    -------
    tuple[tuple[str, int], ...]
        Module calls aligned with ``module_addresses``.
    """

    call_by_address = dict(module_call_stack)
    return tuple((address, call_by_address.get(address, 1)) for address in module_addresses)


def _jax_module_frame(address: str, call_index: int) -> ModuleFrame:
    """Return an event-layer module frame for one JAX module address.

    Parameters
    ----------
    address
        TorchLens module address.
    call_index
        One-based module call index.

    Returns
    -------
    ModuleFrame
        Module frame.
    """

    return ModuleFrame(
        address=address,
        address_normalized=address,
        module_type="pytree_module",
        call_index=call_index,
        fx_qualpath=None,
        entry_argnames=(),
    )
