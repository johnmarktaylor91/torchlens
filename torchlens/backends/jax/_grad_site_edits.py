"""JAX ``tl.grad`` sites at capture: edits on the cotangent of an equation output.

A ``tl.when(tl.grad(selector), edit)`` clause traced with ``grad_options`` edits
the gradient that flows into each equation output ``selector`` matches. The
edited program (``_edited_grads``) wraps that output in an identity whose VJP
applies the edit, so the derived gradients of parameters and inputs are
``jax.vjp`` of the program with the edit applied to the site's cotangent.
The forward values are unchanged, so no op is marked edited.

Intervention-layer names arrive through ``api`` (the edit runtime's one door
to that layer), which keeps this backend module free of upward imports.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ..registry import BackendUnsupportedError


def _grad_site_refusal(message: str, *, code: str) -> BackendUnsupportedError:
    """Return the typed refusal of a ``tl.grad`` clause this capture cannot honor."""

    return BackendUnsupportedError(
        f"JAX backend trace(intervene=tl.grad(...)): {message}", code=code
    )


def gradient_universes(selector: Any, api: Any) -> frozenset[str]:
    """Return the site universes (``activation``, ``gradient``, ``node``) of ``selector``."""

    return frozenset(universe for universe, _branch in api.split_site_branches(selector))


def gradient_predicates(selector: Any, action: Any, grad_options: Any, api: Any) -> tuple[Any, ...]:
    """Lower a gradient-universe clause to one forward predicate per ``tl.grad`` leaf.

    Returns ``()`` when ``selector`` addresses activations only. Each predicate
    is a one-clause recipe over the leaf's forward selector, so it matches the
    equation contexts the activation path sees.

    Raises
    ------
    BackendUnsupportedError
        When the clause mixes activation and gradient sites, addresses
        autograd nodes, has no ``grad_options`` to differentiate, or names a
        backward pass other than the first.
    """

    universes = gradient_universes(selector, api)
    if universes <= {"activation"}:
        return ()
    if universes != {"gradient"}:
        raise _grad_site_refusal(
            f"the site {selector!r} mixes {sorted(universes)} sites; the JAX capture edits "
            "either equation outputs or their gradients in one clause. Split it into two "
            "captures, or use tl.grad(...) sites only.",
            code="backward_capture_unsupported",
        )
    if grad_options is None:
        raise _grad_site_refusal(
            f"the site {selector!r} edits a gradient, and JAX computes gradients only as "
            "derived gradients. Pass capture=tl.options.CaptureOptions("
            "grad_options=GradOptions(...)).",
            code="backward_capture_unsupported",
        )
    targets = api.grad_site_targets(selector, api.target_operands)
    if any(pass_index not in (None, 1) for _inner, pass_index in targets):
        raise _grad_site_refusal(
            "derived gradients run one backward pass; tl.in_backward_pass(n) with n > 1 "
            "has no pass to edit.",
            code="backward_capture_unsupported",
        )
    return tuple(api.when(inner, action) for inner, _pass in targets)


def gradient_identity(apply: Callable[[Any], Any], site_label: str) -> Callable[[Any], Any]:
    """Return an identity on ``site_label``'s output whose VJP applies ``apply``.

    The edit must keep the cotangent's shape and dtype; a change refuses typed
    while the gradient program is traced.
    """

    import jax

    @jax.custom_vjp
    def identity(value: Any) -> Any:
        """Return ``value`` unchanged (the forward pass is not edited)."""

        return value

    def forward(value: Any) -> tuple[Any, None]:
        """Save nothing: the backward edit reads only the cotangent."""

        return value, None

    def backward(_residual: None, cotangent: Any) -> tuple[Any]:
        """Apply the edit to the cotangent flowing into the site."""

        edited = apply(cotangent)
        shape, dtype = getattr(edited, "shape", None), getattr(edited, "dtype", None)
        if shape != cotangent.shape or dtype != cotangent.dtype:
            raise _grad_site_refusal(
                f"the gradient edit at {site_label!r} returned shape {shape} and dtype "
                f"{dtype}; it must keep the gradient's {cotangent.shape} and {cotangent.dtype}.",
                code="backward_capture_unsupported",
            )
        return (edited,)

    identity.defvjp(forward, backward)
    return identity
