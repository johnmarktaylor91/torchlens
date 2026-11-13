"""JAX-native application of capture-time intervention helpers and hook results.

The torch helper factories build hooks over ``torch.*`` calls that reject JAX
arrays, so each supported helper is re-expressed here with ``jax.numpy`` from
its portable args and kwargs. A helper outside the table refuses typed rather
than half-working. Every replacement is checked against the edited equation's
output aval (a jaxpr binds downstream equations with fixed shapes and dtypes)
and copied, so the edited value is a fresh array object: parent edges resolve
by value identity.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..registry import BackendUnsupportedError

#: Helper names with a JAX-native application, sorted.
JAX_SUPPORTED_HELPER_NAMES = (
    "add",
    "clamp",
    "mean_ablate",
    "replace_with",
    "scale",
    "zero_ablate",
)

_OPTION_CODE = "trace_option_unsupported"


def _jax_helper_refusal(message: str, *, code: str) -> BackendUnsupportedError:
    """Return the typed refusal for an intervention helper shape JAX cannot honor."""

    return BackendUnsupportedError(message, code=code, option="intervene")


def _reject_foreign_value(value: Any, helper_name: str) -> None:
    """Refuse a torch tensor passed to a JAX intervention helper."""

    if type(value).__module__.split(".", maxsplit=1)[0] == "torch":
        raise _jax_helper_refusal(
            f"JAX intervention helper {helper_name!r} received a torch.Tensor argument; "
            "pass a jax.Array, a NumPy array or a Python scalar.",
            code=_OPTION_CODE,
        )


def _mean_ablate(spec: Any) -> Callable[[Any], Any]:
    """Return the self-form ``mean_ablate`` application (``dim`` None, int or tuple)."""

    import jax.numpy as jnp

    kwargs = dict(spec.kwargs)
    source = spec.args[0] if spec.args else kwargs.get("source")
    if source is not None or "population_identity" in kwargs:
        raise _jax_helper_refusal(
            "JAX tl.mean_ablate(...) interventions support the self form (source=None); "
            "tensor, op and population sources are torch-side semantics.",
            code=_OPTION_CODE,
        )
    dim = kwargs.get("dim")
    if dim is not None and not isinstance(dim, (int, tuple)):
        raise _jax_helper_refusal(
            f"JAX tl.mean_ablate(dim={dim!r}) is unsupported; pass None, an int or a tuple.",
            code=_OPTION_CODE,
        )

    def _apply(out: Any) -> Any:
        """Fill ``out`` with its mean over ``dim`` (every axis when ``None``)."""

        mean = jnp.mean(out, axis=dim, keepdims=dim is not None)
        return jnp.broadcast_to(mean, out.shape).astype(out.dtype)

    return _apply


def _replace_with(spec: Any) -> Callable[[Any], Any]:
    """Return the ``replace_with`` application (a value or a zero-argument callable)."""

    import jax.numpy as jnp

    value = spec.args[0] if spec.args else dict(spec.kwargs).get("value")
    if not callable(value):
        _reject_foreign_value(value, "replace_with")

    def _apply(out: Any) -> Any:
        """Return the replacement, resolving a callable at fire time, in ``out``'s dtype."""

        replacement = value() if callable(value) else value
        _reject_foreign_value(replacement, "replace_with")
        return jnp.asarray(replacement).astype(out.dtype)

    return _apply


def resolve_helper_applier(spec: Any) -> Callable[[Any], Any]:
    """Return the JAX-native application of one builtin helper spec.

    Parameters
    ----------
    spec
        Builtin helper spec (``tl.zero_ablate()`` and friends) exposing
        ``name``, ``args`` and ``kwargs``.

    Returns
    -------
    Callable[[Any], Any]
        Function mapping the pre-hook output array to its replacement.
    """

    import jax.numpy as jnp

    name = spec.name
    args = tuple(spec.args)
    kwargs = dict(spec.kwargs)
    if name == "zero_ablate":
        return jnp.zeros_like
    if name == "scale":
        factor = args[0] if args else kwargs["factor"]
        return lambda out: (out * factor).astype(out.dtype)
    if name == "add":
        delta = args[0] if args else kwargs["delta"]
        _reject_foreign_value(delta, name)
        return lambda out: (out + delta).astype(out.dtype)
    if name == "clamp":
        low, high = kwargs.get("low"), kwargs.get("high")
        return lambda out: jnp.clip(out, low, high).astype(out.dtype)
    if name == "mean_ablate":
        return _mean_ablate(spec)
    if name == "replace_with":
        return _replace_with(spec)
    raise _jax_helper_refusal(
        f"JAX backend has no native application for intervention helper {name!r}; "
        f"supported helpers are {', '.join(JAX_SUPPORTED_HELPER_NAMES)}, plus callable "
        "hooks written against jax.Array. The torch helper factories build torch-only "
        "hooks that reject JAX arrays.",
        code=_OPTION_CODE,
    )


def checked_replacement(site_label: str, original: Any, replacement: Any) -> Any:
    """Validate a hook result against the edited output and return a fresh copy.

    Parameters
    ----------
    site_label
        Raw label of the edited site, for diagnostics.
    original
        Pre-hook output array.
    replacement
        Hook result.

    Returns
    -------
    Any
        ``original`` itself when the hook returned it unchanged, else a fresh
        copy of ``replacement`` (same shape and dtype as ``original``).

    Raises
    ------
    BackendUnsupportedError
        ``intervention_result_invalid`` for a non-array result or a shape or
        dtype change the downstream jaxpr equations cannot bind.
    """

    import jax
    import jax.numpy as jnp
    import numpy as np

    if replacement is original:
        return original
    if not isinstance(replacement, (jax.Array, np.ndarray)):
        raise BackendUnsupportedError(
            f"JAX trace(intervene=...) hook for site {site_label!r} returned "
            f"{type(replacement).__name__}; hooks must return a jax.Array.",
            code="intervention_result_invalid",
        )
    if tuple(replacement.shape) != tuple(original.shape) or replacement.dtype != original.dtype:
        raise BackendUnsupportedError(
            f"JAX trace(intervene=...) hook for site {site_label!r} changed the output "
            f"from {original.dtype}{list(original.shape)} to "
            f"{replacement.dtype}{list(replacement.shape)}; the downstream jaxpr "
            "equations are bound to the original shape and dtype.",
            code="intervention_result_invalid",
        )
    return jnp.array(replacement, copy=True)


__all__ = [
    "JAX_SUPPORTED_HELPER_NAMES",
    "checked_replacement",
    "resolve_helper_applier",
]
