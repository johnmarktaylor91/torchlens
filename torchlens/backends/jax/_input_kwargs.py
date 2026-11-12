"""Keyword inputs for JAX capture: array kwargs trace as pytree leaves.

``jax.make_jaxpr`` traces keyword values as pytrees, so a keyword input needs no
special capture path. Keyword values whose leaves are all arrays become one
trailing positional dict argument (dynamic jaxpr leaves with saved payloads and
input source ops, exactly like positional inputs); every other keyword value
(Python flags, strings, ``None``) is bound statically into the callable, the way
torch records non-tensor arguments as constants of the call.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

_KWARG_TARGET_ATTR = "__torchlens_kwarg_target__"


def _kwarg_bound(
    fn: Callable[..., Any], static_kwargs: Mapping[str, Any], dynamic_keys: tuple[str, ...]
) -> Callable[..., Any]:
    """Return a plain function forwarding a trailing positional dict as keywords.

    A function (not a callable object) keeps ``inspect.getclosurevars`` usable
    by the derived-gradient host-state check; :func:`unwrap_kwarg_bound`
    returns ``fn`` so that check still inspects the user callable itself.
    """

    bound_static = dict(static_kwargs)

    def kwarg_bound(*args: Any) -> Any:
        """Call ``fn`` with the trailing dict (when present) spread as keywords."""

        if not dynamic_keys:
            return fn(*args, **bound_static)
        *positional, dynamic = args
        return fn(*positional, **bound_static, **dynamic)

    setattr(kwarg_bound, _KWARG_TARGET_ATTR, fn)
    return kwarg_bound


def _all_array_leaves(value: Any) -> bool:
    """Return whether ``value`` is a non-empty pytree of array leaves."""

    import jax
    import numpy as np

    leaves = jax.tree.leaves(value)
    return bool(leaves) and all(isinstance(leaf, (jax.Array, np.ndarray)) for leaf in leaves)


def bind_input_kwargs(
    model: Callable[..., Any], args: Sequence[Any], input_kwargs: Mapping[Any, Any] | None
) -> tuple[Callable[..., Any], list[Any]]:
    """Fold keyword inputs into the positional capture convention.

    Parameters
    ----------
    model:
        The user callable.
    args:
        Normalized positional arguments.
    input_kwargs:
        Public keyword inputs, or ``None``.

    Returns
    -------
    tuple[Callable[..., Any], list[Any]]
        The callable to trace and the positional arguments to trace it with.
        Without keyword inputs both are returned unchanged.
    """

    if not input_kwargs:
        return model, list(args)
    dynamic = {str(k): v for k, v in input_kwargs.items() if _all_array_leaves(v)}
    static = {str(k): v for k, v in input_kwargs.items() if str(k) not in dynamic}
    bound = _kwarg_bound(model, static, tuple(dynamic))
    return bound, [*args, dynamic] if dynamic else list(args)


def unwrap_kwarg_bound(model: Any) -> Any:
    """Return the user callable behind a keyword-bound wrapper (else ``model``)."""

    return getattr(model, _KWARG_TARGET_ATTR, model)
