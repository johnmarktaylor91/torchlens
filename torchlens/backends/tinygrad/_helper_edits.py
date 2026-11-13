"""Built-in edit helpers applied to tinygrad Tensors.

One table serves both edit paths: a live ``trace(intervene=tl.when(...))``
(``_edit_plan``) and an edit of a finished trace (``trace.fork().do``), which
reaches it through the replay adapter's ``apply_helper``. Every application is
a lazy tinygrad expression over the site's value, so a live edit still
realizes once. The helpers are the set the MLX backend applies:
``zero_ablate``, ``scale``, ``add``, ``clamp``, ``mean_ablate`` and
``replace_with``. Helpers that draw random numbers or need torch-side state
(``add_noise``, ``steer`` and the rest) refuse typed; a callable over tinygrad
Tensors covers them.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..registry import BackendUnsupportedError

__tl_layer__ = "L2"

SUPPORTED_HELPERS = ("zero_ablate", "scale", "add", "clamp", "mean_ablate", "replace_with")
HELPER_REFUSAL = "interventions_unsupported"


def _refuse_helper(message: str, *, code: str) -> BackendUnsupportedError:
    """A typed refusal of a helper form the tinygrad backend does not apply."""

    return BackendUnsupportedError(f"tinygrad backend: {message}", code=code)


def _is_tinygrad_tensor(value: Any) -> bool:
    """Whether ``value`` is a tinygrad Tensor (checked by type module, no import)."""

    return type(value).__module__.split(".", 1)[0] == "tinygrad"


def _mean_source(source: Any) -> Any | None:
    """The tinygrad Tensor a ``mean_ablate`` source averages, or None for the site itself."""

    if source is None:
        return None
    value = getattr(source, "out", source)
    if not _is_tinygrad_tensor(value):
        raise _refuse_helper(
            "tl.mean_ablate(source) needs the site itself, a tinygrad Tensor or a tinygrad "
            f"op as its source; got {type(source).__name__} (populations are torch-side).",
            code=HELPER_REFUSAL,
        )
    return value


def _mean_fill(out: Any, source: Any | None, dim: Any) -> Any:
    """``out``-shaped fill with the mean of ``source`` (``out`` when None) over ``dim``."""

    values = out if source is None else source.cast(out.dtype)
    if dim is None:
        return out.zeros_like() + values.mean()
    axes = (dim,) if isinstance(dim, int) else tuple(dim)
    resolved = tuple(sorted({axis % values.ndim for axis in axes}))
    reduced = values.mean(axis=resolved, keepdim=True)
    return out.zeros_like() + reduced.expand(out.shape)


def _mean_ablate(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Callable[[Any], Any]:
    """The lazy application of ``tl.mean_ablate(source, dim=...)``."""

    source = _mean_source(args[0] if args else None)
    dim = kwargs.get("dim")
    return lambda out: _mean_fill(out, source, dim)


def _replace_with(args: tuple[Any, ...]) -> Callable[[Any], Any]:
    """The application of ``tl.replace_with(value)``; a zero-argument callable runs at apply time."""

    value = args[0]
    return lambda out: value() if callable(value) else value


def helper_application(spec: Any) -> Callable[[Any], Any]:
    """Resolve one built-in helper spec to its lazy tinygrad application.

    Parameters
    ----------
    spec
        The ``tl.<helper>(...)`` ``Edit`` (read through ``name``, ``args`` and
        ``kwargs``).

    Returns
    -------
    Callable[[Any], Any]
        Maps the site's tinygrad Tensor to its replacement.

    Raises
    ------
    BackendUnsupportedError
        ``interventions_unsupported`` for a helper or helper form outside
        :data:`SUPPORTED_HELPERS`.
    """

    name = str(spec.name)
    args = tuple(spec.args)
    kwargs = dict(spec.kwargs)
    simple: dict[str, Callable[[Any], Any]] = {
        "zero_ablate": lambda out: out.zeros_like(),
        "scale": lambda out: out * args[0],
        "add": lambda out: out + args[0],
        "clamp": lambda out: out.clip(kwargs.get("low"), kwargs.get("high")),
    }
    if name in simple:
        return simple[name]
    if name == "mean_ablate":
        return _mean_ablate(args, kwargs)
    if name == "replace_with":
        return _replace_with(args)
    raise _refuse_helper(
        f"no tinygrad application for helper {name!r}; supported helpers are "
        f"{', '.join(SUPPORTED_HELPERS)}, plus callable transforms over tinygrad Tensors.",
        code=HELPER_REFUSAL,
    )


def apply_helper(trace: Any, helper: Any, value: Any) -> Any:
    """Apply a built-in helper to one value of a finished tinygrad trace.

    The replay adapter's ``apply_helper`` delegates here, so ``trace.fork().do``
    runs the same helper table as a live ``intervene=``.

    Parameters
    ----------
    trace
        The finished trace being edited (unused; part of the adapter contract).
    helper
        The helper ``Edit``.
    value
        The site's tinygrad Tensor.

    Returns
    -------
    Any
        The edited tinygrad Tensor, realized so the replay reads a concrete value.

    Raises
    ------
    BackendUnsupportedError
        ``interventions_unsupported`` for a helper outside :data:`SUPPORTED_HELPERS`.
    """

    del trace
    result = helper_application(helper)(value)
    return result.realize() if _is_tinygrad_tensor(result) else result
