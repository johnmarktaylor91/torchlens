"""Structural argument hash for the equivalence class of a parameter-free op.

Moved out of ``tensor_tracking`` unchanged, plus a bounded memo of the final digest:
the tokens are exact strings, so equal token tuples have the same digest.
"""

from typing import Any

import torch

from ..._utils.hashing import make_short_barcode_from_input
from ..._utils.tensor_utils import _is_uninitialized_lazy_tensor as _is_uninitialized_param

__all__ = ("_get_hash_from_args",)


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
    args_to_hash: list[Any] = []
    for a, arg in enumerate(args):
        _append_arg_hash(arg, f"pos{a}", args_to_hash)
    for key, arg in kwargs.items():
        _append_arg_hash(arg, f"kw_{key}", args_to_hash)

    if len(args_to_hash) == 0:
        return "no_args"
    # Every token is an f-string (an exact ``str``), so equal token tuples encode to the
    # same JSON payload and the same digest: the memo answers the digest of the tokens.
    key = tuple(args_to_hash)
    barcode = _ARG_HASH_DIGESTS.get(key)
    if barcode is None:
        if len(_ARG_HASH_DIGESTS) >= _ARG_HASH_DIGESTS_MAX:
            _ARG_HASH_DIGESTS.clear()
        barcode = _ARG_HASH_DIGESTS[key] = make_short_barcode_from_input(args_to_hash)
    return barcode


_ARG_HASH_DIGESTS: dict[tuple[str, ...], str] = {}
"""Argument-hash tokens to their barcode (a pure function of the tokens), bounded."""

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
