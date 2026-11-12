"""Structural argument hash for the equivalence class of a parameter-free op.

Moved out of ``tensor_tracking`` unchanged, then split into an observation and a
resolver (lane FQ). :func:`observe_arg_inputs` records each top-level argument at the
call: an immutable value (an exact tensor's shape, an exact ``int``, ``float``,
``bool``, ``None``, ``str``, dtype or device) by reference, and anything else as
today's tokens computed at the call, so a mutable argument changed after the call
keeps its call-time class. :func:`resolve_arg_hash` is a pure function of that record
and may run after the call. The digest memo is keyed on the record: two equal records
give the same tokens, so they have the same digest.
"""

from math import copysign
from typing import Any

import torch

from ..._utils.hashing import make_short_barcode_from_input
from ..._utils.tensor_utils import _is_uninitialized_lazy_tensor as _is_uninitialized_param

__all__ = ("_get_hash_from_args", "observe_arg_inputs", "resolve_arg_hash")


def _get_hash_from_args(args: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    """Compute a structural hash of non-tensor arguments for equivalence fingerprinting.

    Tensor arguments are excluded (they define graph edges, not structural identity).
    Parameters are also excluded (they have their own barcode system).
    The hash preserves positional indices and kwarg keys to avoid collisions
    between e.g. ``f(a=1, b=2)`` and ``f(a=2, b=1)``.

    Returns:
        A short deterministic hash string, or ``"no_args"`` if no non-tensor
        arguments are present.
    """
    return resolve_arg_hash(observe_arg_inputs(args, kwargs))


# Record entry kinds. Every entry starts with its kind and its argument's prefix, so
# entries of different kinds never compare equal and the record is a sound memo key.
_SHAPE = 0  # (kind, prefix, shape): an exact tensor
_PLAIN = 1  # (kind, prefix, value, type): an exact int, bool or None
_FLOAT = 2  # (kind, prefix, value, sign): an exact float; the sign tells 0.0 from -0.0
_LEAF = 3  # (kind, prefix, value, type): an exact str, dtype or device
_TOKENS = 4  # (kind, token, ...): tokens computed at the call

_PLAIN_TYPES = frozenset((int, bool, type(None)))
"""Exact leaf types whose token is ``f"{prefix}_{arg}"``; equal values of one type print alike."""

_LEAF_TYPES = frozenset((str, torch.dtype, torch.device))
"""Exact immutable types whose token is :func:`_leaf_arg_token`; equal values print alike."""

_POS_PREFIXES = tuple(f"pos{index}" for index in range(32))


def observe_arg_inputs(args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[Any, ...]:
    """Record the call's arguments for :func:`resolve_arg_hash`, in argument order.

    Immutable values are held by reference; every other value is held as the tokens
    :func:`_append_arg_hash` computes for it now. A value that yields no token (a
    parameter) adds no entry, so the record is empty exactly when the tokens are.
    """

    record: list[Any] = []
    for index, arg in enumerate(args):
        prefix = _POS_PREFIXES[index] if index < 32 else f"pos{index}"
        _observe_top_arg(arg, prefix, record)
    for key, arg in kwargs.items():
        _observe_top_arg(arg, f"kw_{key}", record)
    return tuple(record)


def _observe_top_arg(arg: Any, prefix: str, record: list[Any]) -> None:
    """Append the record entry of one top-level argument (none for a parameter)."""

    arg_type = type(arg)
    if arg_type is torch.Tensor:
        # An exact tensor is never a lazy one: torch's lazy test is an isinstance check
        # against the uninitialized mixin, which only its subclasses carry, and a
        # materialized one has become an initialized tensor.
        record.append((_SHAPE, prefix, arg.shape))
    elif arg_type in _PLAIN_TYPES:
        record.append((_PLAIN, prefix, arg, arg_type))
    elif arg_type is float:
        record.append((_FLOAT, prefix, arg, copysign(1.0, arg)))
    elif arg_type in _LEAF_TYPES:
        record.append((_LEAF, prefix, arg, arg_type))
    else:
        tokens: list[Any] = [_TOKENS]
        _append_arg_hash(arg, prefix, tokens)
        if len(tokens) > 1:
            record.append(tuple(tokens))


def _record_tokens(record: tuple[Any, ...]) -> list[str]:
    """Return the hash tokens of a record: the tokens the generic walk appends."""

    tokens: list[str] = []
    for entry in record:
        kind = entry[0]
        if kind == _SHAPE:
            tokens.append(f"{entry[1]}_tensor{entry[2]}")
        elif kind in (_PLAIN, _FLOAT):
            tokens.append(f"{entry[1]}_{entry[2]}")
        elif kind == _LEAF:
            tokens.append(_leaf_arg_token(entry[2], entry[1]))
        else:
            tokens.extend(entry[1:])
    return tokens


def resolve_arg_hash(record: tuple[Any, ...]) -> str:
    """Return the argument hash of a record from :func:`observe_arg_inputs`.

    A pure function of the record, so it gives the call-time hash whenever it runs.
    """

    if not record:
        return "no_args"
    barcode = _ARG_HASH_DIGESTS.get(record)
    if barcode is None:
        if len(_ARG_HASH_DIGESTS) >= _ARG_HASH_DIGESTS_MAX:
            _ARG_HASH_DIGESTS.clear()
        barcode = _ARG_HASH_DIGESTS[record] = make_short_barcode_from_input(_record_tokens(record))
    return barcode


def _append_top_arg_hash(arg: Any, prefix: str, args_to_hash: list[Any]) -> None:
    """Append one top-level argument's tokens, as :func:`_append_arg_hash` appends them.

    The tokens come from the argument's record entry, so this is the observation and
    the resolver applied to one argument.
    """

    record: list[Any] = []
    _observe_top_arg(arg, prefix, record)
    args_to_hash.extend(_record_tokens(tuple(record)))


_ARG_HASH_DIGESTS: dict[tuple[Any, ...], str] = {}
"""Argument records to their barcode (a pure function of the record), bounded."""

_ARG_HASH_DIGESTS_MAX = 4096


def _tensor_hash_token(prefix: str, arg: torch.Tensor) -> str:
    """Return the barcode hash token for one tensor argument.

    Uses shape/dtype only — formatting a tensor can trigger wrapped methods
    (item, __format__) which re-enter logging and cause infinite recursion.
    Lazy-buffer completion: a pending UninitializedBuffer is a plain
    Tensor subclass (unlike UninitializedParameter, which rides the
    Parameter carve-out at the caller) and raises on any shape read until
    torch's lazy pre-hook materializes it, so it hashes as a fixed token.
    """

    if _is_uninitialized_param(arg):
        return f"{prefix}_tensor_uninitialized"
    return f"{prefix}_tensor{arg.shape}"


def _append_arg_hash(arg: Any, prefix: str, args_to_hash: list[Any], _depth: int = 0) -> None:
    """Append structural fingerprint tokens for a single argument to the accumulator list.

    Builds an ``equivalence_class`` -- a structural fingerprint of the operation's
    argument types and shapes (not a content hash). This fingerprint is used by loop
    detection to identify operations that are structurally identical across ops.

    For tensors, only shape and dtype are recorded (not values). Containers (dicts, lists,
    tuples, sets) are recursed into with depth-limited traversal. Parameters are excluded.

    Args:
        arg: The argument value to fingerprint.
        prefix: String prefix encoding the argument's position/key path.
        args_to_hash: Accumulator list that fingerprint tokens are appended to.
        _depth: Recursion depth guard; stops at 10 to prevent infinite recursion.
    """
    if _depth > 10:
        args_to_hash.append(f"{prefix}_deep")
        return
    if isinstance(arg, torch.nn.Parameter):
        pass  # exclude parameters from hash — must check before Tensor (Parameter is a subclass)
    elif isinstance(arg, torch.Tensor):
        args_to_hash.append(_tensor_hash_token(prefix, arg))
    elif isinstance(arg, (torch.TypedStorage, torch.UntypedStorage)):
        # Same hazard as torch.Tensor above: str()/repr() on a Storage walks
        # every element via wrapped __getitem__ (and even constructs a fresh
        # wrapped tensor per element on some torch builds), re-entering
        # logging and causing infinite/runaway recursion. Use size/dtype only.
        dtype = getattr(arg, "dtype", None)
        args_to_hash.append(f"{prefix}_storage{arg.size()}_{dtype}")
    elif isinstance(arg, dict):
        for k, v in arg.items():
            _append_arg_hash(v, f"{prefix}_dk{k}", args_to_hash, _depth + 1)
    elif isinstance(arg, (list, tuple)):
        for i, elem in enumerate(arg):
            _append_arg_hash(elem, f"{prefix}_i{i}", args_to_hash, _depth + 1)
    elif isinstance(arg, (set, frozenset)):
        _append_set_arg_hash(arg, prefix, args_to_hash, _depth)
    else:
        args_to_hash.append(_leaf_arg_token(arg, prefix))


def _leaf_arg_token(arg: Any, prefix: str) -> str:
    """Return the fingerprint token for one non-container leaf argument.

    The default object repr IS the memory address,
    so ``str(arg)`` here persisted a process-local address into
    ``equivalence_class`` -- cross-process keys diverged for any op holding
    an object arg (``torch.randn(..., generator=g)``), the SAME key could
    collide for two distinct objects after address reuse, and recurrence
    grouping split when a semantically-identical fresh object was passed per
    call. Same rule as ``_capture_fingerprint``: never repr an address; use
    the address-free type token.
    """

    arg_type = type(arg)
    # mypy sees bound-descriptor types diverge here; the identity comparison
    # against the object slots is exactly the intended check.
    default_repr = arg_type.__repr__ is object.__repr__  # type: ignore[comparison-overlap]
    default_str = arg_type.__str__ is object.__str__  # type: ignore[comparison-overlap]
    if default_repr and default_str:
        return f"{prefix}_obj:{arg_type.__module__}.{arg_type.__qualname__}"
    token = f"{prefix}_{arg}"
    if " at 0x" in token:
        # Custom reprs that still embed a memory address (C types,
        # functools.partial interiors) are equally id()-derived.
        return f"{prefix}_obj:{arg_type.__module__}.{arg_type.__qualname__}"
    return token


def _append_set_arg_hash(
    arg: "set[Any] | frozenset[Any]", prefix: str, args_to_hash: list[Any], _depth: int
) -> None:
    """Fingerprint one set/frozenset arg in seed-independent member order.

    Sets iterate in hash order, so the old
    position-encoded ``_i{i}`` prefixes made the persisted fingerprint
    PYTHONHASHSEED-dependent (and frozenset fell to the str() tail).
    Fingerprint each member independently, then fold the members in sorted
    token order -- deterministic for any seed and identical for
    set/frozenset of equal members.
    """

    member_tokens: list[tuple[str, ...]] = []
    for elem in arg:
        elem_tokens: list[Any] = []
        _append_arg_hash(elem, "", elem_tokens, _depth + 1)
        member_tokens.append(tuple(str(token) for token in elem_tokens))
    member_tokens.sort()
    for i, tokens in enumerate(member_tokens):
        args_to_hash.extend(f"{prefix}_s{i}{token}" for token in tokens)
