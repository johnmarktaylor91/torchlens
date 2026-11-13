"""The ``tlm_``/``tlc_`` module-scope codec and the module stacks read off JAX equations.

Nested frames (jit calls, taken cond branches, scan and while bodies) inherit the module
scope of the equation that runs them; :func:`_frame_stacks` composes the two. Imports
nothing from the backend's import cycle.
"""

from __future__ import annotations

import base64
from collections.abc import Callable, Sequence
from typing import Any

MODULE_SCOPE_PREFIX = "tlm_"
MODULE_CALL_SCOPE_PREFIX = "tlc_"


def encode_module_scope(address: str) -> str:
    """Return a reversible ``jax.named_scope`` marker for a module address.

    Parameters
    ----------
    address
        TorchLens module address.

    Returns
    -------
    str
        Named-scope-safe marker string.
    """

    encoded = base64.urlsafe_b64encode(address.encode("utf-8")).decode("ascii")
    return MODULE_SCOPE_PREFIX + encoded.rstrip("=")


def encode_module_call_scope(address: str, call_index: int) -> str:
    """Return a reversible ``jax.named_scope`` marker for a module call.

    Parameters
    ----------
    address
        TorchLens module address.
    call_index
        One-based module call index.

    Returns
    -------
    str
        Named-scope-safe marker string.
    """

    payload = f"{address}\0{call_index}"
    encoded = base64.urlsafe_b64encode(payload.encode("utf-8")).decode("ascii")
    return MODULE_CALL_SCOPE_PREFIX + encoded.rstrip("=")


def decode_module_scope(scope_name: str) -> str | None:
    """Decode a TorchLens module named-scope marker.

    Parameters
    ----------
    scope_name
        Name-stack component.

    Returns
    -------
    str | None
        Decoded module address, or ``None`` for non-TorchLens scopes.
    """

    if not scope_name.startswith(MODULE_SCOPE_PREFIX):
        return None
    payload = scope_name.removeprefix(MODULE_SCOPE_PREFIX)
    padded = payload + "=" * (-len(payload) % 4)
    try:
        return base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return None


def decode_module_call_scope(scope_name: str) -> tuple[str, int] | None:
    """Decode a TorchLens module-call named-scope marker.

    Parameters
    ----------
    scope_name
        Name-stack component.

    Returns
    -------
    tuple[str, int] | None
        Decoded ``(module_address, call_index)``, or ``None`` for non-call
        TorchLens scopes.
    """

    if not scope_name.startswith(MODULE_CALL_SCOPE_PREFIX):
        return None
    payload = scope_name.removeprefix(MODULE_CALL_SCOPE_PREFIX)
    padded = payload + "=" * (-len(payload) % 4)
    try:
        decoded = base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return None
    address, separator, call_index_text = decoded.partition("\0")
    if not separator:
        return None
    try:
        call_index = int(call_index_text)
    except ValueError:
        return None
    if call_index < 1:
        return None
    return address, call_index


def module_stack_from_eqn(eqn: Any) -> tuple[str, ...]:
    """Return the TorchLens module stack encoded on one JAX equation.

    Parameters
    ----------
    eqn
        JAX equation whose source metadata may carry a name stack.

    Returns
    -------
    tuple[str, ...]
        Decoded module addresses in outer-to-inner order.
    """

    source_info = getattr(eqn, "source_info", None)
    return module_stack_from_name_stack(getattr(source_info, "name_stack", None))


def module_call_stack_from_eqn(eqn: Any) -> tuple[tuple[str, int], ...]:
    """Return the TorchLens module-call stack encoded on one JAX equation.

    Parameters
    ----------
    eqn
        JAX equation whose source metadata may carry a name stack.

    Returns
    -------
    tuple[tuple[str, int], ...]
        Decoded module calls in outer-to-inner order.
    """

    source_info = getattr(eqn, "source_info", None)
    return module_call_stack_from_name_stack(getattr(source_info, "name_stack", None))


def module_stack_from_name_stack(name_stack: Any) -> tuple[str, ...]:
    """Decode TorchLens module markers from a JAX ``NameStack``.

    Parameters
    ----------
    name_stack
        JAX source-info name stack, or ``None``.

    Returns
    -------
    tuple[str, ...]
        Module addresses in outer-to-inner order.
    """

    stack = getattr(name_stack, "stack", ())
    decoded: list[str] = []
    for component in stack:
        name = getattr(component, "name", str(component))
        address = decode_module_scope(str(name))
        if address is not None and (not decoded or decoded[-1] != address):
            decoded.append(address)
    return tuple(decoded)


def module_call_stack_from_name_stack(name_stack: Any) -> tuple[tuple[str, int], ...]:
    """Decode TorchLens module-call markers from a JAX ``NameStack``.

    Parameters
    ----------
    name_stack
        JAX source-info name stack, or ``None``.

    Returns
    -------
    tuple[tuple[str, int], ...]
        Module-call pairs in outer-to-inner order.
    """

    stack = getattr(name_stack, "stack", ())
    decoded: list[tuple[str, int]] = []
    for component in stack:
        name = getattr(component, "name", str(component))
        call = decode_module_call_scope(str(name))
        if call is not None and (not decoded or decoded[-1] != call):
            decoded.append(call)
    return tuple(decoded)


def _compose_stack(inherited: tuple[Any, ...], own: tuple[Any, ...]) -> tuple[Any, ...]:
    """Join a nested frame's module stack onto the enclosing equation's stack.

    Equations inside a nested jaxpr may carry the full stack (the enclosing scope is a
    prefix) or only the scopes opened inside the nested function; both compose to the
    full outer-to-inner stack.
    """

    if not own or own[: len(inherited)] == inherited:
        return own or inherited
    return (*inherited, *own)


def _frame_stacks(
    eqn: Any,
    inherited_module_stack: tuple[str, ...],
    inherited_module_call_stack: tuple[tuple[str, int], ...],
) -> tuple[tuple[str, ...], tuple[tuple[str, int], ...]]:
    """Return ``eqn``'s module and module-call stacks composed with the inherited ones."""

    return (
        _compose_stack(inherited_module_stack, module_stack_from_eqn(eqn)),
        _compose_stack(inherited_module_call_stack, module_call_stack_from_eqn(eqn)),
    )


def _scoped_inner(
    interpret_inner: Callable[..., tuple[Any, ...]],
    stacks: tuple[tuple[str, ...], tuple[tuple[str, int], ...]],
) -> Callable[..., tuple[Any, ...]]:
    """Return ``interpret_inner`` with the enclosing equation's module stacks bound.

    Taken cond branches, scan bodies and while bodies inherit the module scope of the
    control-flow equation that runs them.
    """

    def scoped(
        inner: Any,
        args: Sequence[Any],
        path: tuple[str, ...],
        inlined_depth: int,
        control_capture_indices: tuple[int, ...] = (),
    ) -> tuple[Any, ...]:
        """Interpret one nested frame under the enclosing module scope."""

        return interpret_inner(inner, args, path, inlined_depth, control_capture_indices, *stacks)

    return scoped
