"""Paddle-native applications of the built-in intervention helpers.

The built-in helper factories (``tl.zero_ablate()`` and friends) build hooks
over ``torch.*`` calls that reject Paddle tensors, so each supported helper is
re-expressed here with Paddle ops from its portable ``args`` and ``kwargs``.
Live ``trace(intervene=...)`` captures and finished-trace edits
(``trace.fork().do(...)``, through the Paddle replay adapter's
``apply_helper``) share these applications, so one helper means the same
thing on both paths. Anything outside the table refuses typed.

This module imports without Paddle installed; ``paddle`` is passed in or
imported at apply time.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .._protocols.paddle import PADDLE_REFUSAL_FOREIGN, PADDLE_REFUSAL_OPTION
from ..registry import BackendUnsupportedError

PADDLE_SUPPORTED_HELPER_NAMES = (
    "add",
    "clamp",
    "mean_ablate",
    "replace_with",
    "scale",
    "zero_ablate",
)
"""Built-in helper names with a Paddle-native application, sorted."""


def _refusal(message: str, *, code: str = PADDLE_REFUSAL_OPTION) -> BackendUnsupportedError:
    """Return the typed refusal for a helper shape Paddle cannot honor."""

    if code == PADDLE_REFUSAL_FOREIGN:
        return BackendUnsupportedError(message, code=code)
    return BackendUnsupportedError(message, code=code, option="intervene")


def reject_foreign_tensor(value: Any, *, helper_name: str) -> None:
    """Refuse a torch tensor passed to a Paddle intervention helper.

    Parameters
    ----------
    value
        Helper argument value.
    helper_name
        Helper name for the diagnostic.
    """

    if type(value).__module__.split(".", maxsplit=1)[0] == "torch":
        raise _refusal(
            f"paddle backend intervention helper {helper_name!r} received a "
            "torch.Tensor argument; pass a paddle.Tensor or Python scalar.",
            code=PADDLE_REFUSAL_FOREIGN,
        )


def _first(helper: Any, key: str) -> Any:
    """The helper's first positional argument, else its ``key`` keyword."""

    args = tuple(helper.args)
    return args[0] if args else dict(helper.kwargs).get(key)


def _mean_axes(helper: Any) -> tuple[int, ...] | None:
    """Axes of a self-form ``mean_ablate`` (``None`` for the global mean); refuse other forms."""

    kwargs = dict(helper.kwargs)
    if _first(helper, "source") is not None or "population_identity" in kwargs:
        raise _refusal(
            "paddle tl.mean_ablate(...) interventions support the self form "
            "(source=None); tensor, op and population sources are torch-side semantics."
        )
    dim = kwargs.get("dim")
    if dim is None:
        return None
    axes = (dim,) if isinstance(dim, int) else dim
    if not isinstance(axes, tuple) or not all(isinstance(axis, int) for axis in axes):
        raise _refusal(
            f"paddle tl.mean_ablate(dim={dim!r}) is unsupported; pass None, an int or a "
            "tuple of ints."
        )
    return axes


def _mean_ablate(helper: Any, paddle: Any) -> Callable[[Any], Any]:
    """Fill the site with its own mean (over ``dim`` with keepdim, or over every axis)."""

    axes = _mean_axes(helper)

    def _apply(out: Any) -> Any:
        """Return ``out``-shaped values filled with the mean of ``out``."""

        if axes is None:
            return paddle.zeros_like(out) + paddle.mean(out)
        return paddle.zeros_like(out) + paddle.mean(out, axis=list(axes), keepdim=True)

    return _apply


def _replace_with(helper: Any, paddle: Any) -> Callable[[Any], Any]:
    """Return the replacement (a value or a zero-argument callable) in ``out``'s dtype."""

    value = _first(helper, "value")
    if not callable(value):
        reject_foreign_tensor(value, helper_name="replace_with")

    def _apply(out: Any) -> Any:
        """Resolve the replacement at fire time and cast a Paddle tensor to ``out``'s dtype."""

        replacement = value() if callable(value) else value
        reject_foreign_tensor(replacement, helper_name="replace_with")
        if not isinstance(replacement, paddle.Tensor):
            return replacement
        return replacement.astype(out.dtype)

    return _apply


def _arithmetic(helper: Any, paddle: Any) -> Callable[[Any], Any] | None:
    """The elementwise helpers (``zero_ablate``, ``scale``, ``add``, ``clamp``), else ``None``."""

    name = helper.name
    if name == "zero_ablate":
        return paddle.zeros_like
    if name == "scale":
        factor = _first(helper, "factor")
        return lambda out: out * factor
    if name == "add":
        delta = _first(helper, "delta")
        reject_foreign_tensor(delta, helper_name=name)
        return lambda out: out + delta
    if name == "clamp":
        kwargs = dict(helper.kwargs)
        low, high = kwargs.get("low"), kwargs.get("high")
        return lambda out: paddle.clip(out, min=low, max=high)
    return None


def resolve_paddle_helper(helper: Any, paddle: Any) -> Callable[[Any], Any]:
    """Return the Paddle-native application of one built-in helper spec.

    Parameters
    ----------
    helper
        Built-in helper spec exposing ``name``, ``args`` and ``kwargs``.
    paddle
        Imported ``paddle`` module.

    Returns
    -------
    Callable[[Any], Any]
        Function mapping the pre-edit output tensor to its replacement.
    """

    applier = _arithmetic(helper, paddle)
    if applier is not None:
        return applier
    if helper.name == "mean_ablate":
        return _mean_ablate(helper, paddle)
    if helper.name == "replace_with":
        return _replace_with(helper, paddle)
    raise _refusal(
        f"the paddle backend has no adapter for intervention helper {helper.name!r}; "
        f"supported builtin helpers: {', '.join(PADDLE_SUPPORTED_HELPER_NAMES)}. Pass a "
        "callable hook operating on paddle.Tensor values, or use the PyTorch backend."
    )


def apply_paddle_helper(trace: Any, helper: Any, value: Any) -> Any:
    """Apply one built-in helper to a Paddle value of a finished trace (fork edits).

    Parameters
    ----------
    trace
        Finished Paddle trace being edited (unused; kept for the adapter signature).
    helper
        Built-in helper spec.
    value
        Current Paddle value at the edited site.

    Returns
    -------
    Any
        The edited Paddle value.
    """

    del trace
    import paddle

    return resolve_paddle_helper(helper, paddle)(value)


def _replace_leaves(output: Any, replacements: dict[Any, Any], path: tuple[Any, ...] = ()) -> Any:
    """Rebuild the supported output containers with edited leaves."""

    if path in replacements:
        return replacements[path]
    if isinstance(output, tuple):
        return tuple(
            _replace_leaves(item, replacements, (*path, i)) for i, item in enumerate(output)
        )
    if isinstance(output, list):
        return [_replace_leaves(item, replacements, (*path, i)) for i, item in enumerate(output)]
    if isinstance(output, dict):
        return {
            key: _replace_leaves(item, replacements, (*path, key)) for key, item in output.items()
        }
    return output


__all__ = (
    "PADDLE_SUPPORTED_HELPER_NAMES",
    "apply_paddle_helper",
    "reject_foreign_tensor",
    "resolve_paddle_helper",
)
