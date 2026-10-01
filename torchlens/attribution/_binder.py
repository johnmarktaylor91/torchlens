"""The attribution wrapping contract: primitive + sugar, one private binder.

Attrib memo D1-D3. The kit's method-of-methods contract has exactly two
public routes and NO public registry or explainer object:

* the PRIMITIVE: a closed callable ``attribute(inputs, input_kwargs) ->
  AttributionResult`` -- no keyword is forwarded to it, so no kwarg can
  collide; users bind settings with ``functools.partial``;
* the SUGAR: ``method=`` any kit-contract callable (signature
  ``method(model, inputs, input_kwargs=None, *, target, **settings)``) plus
  ONE explicit ``method_kwargs=dict(...)`` mapping, echoed verbatim into the
  wrapper result's ``extra["method_kwargs"]``.

The keyword splat is dead: a wrapper and its child can both own ``n_samples``
and ``seed`` and no splat spelling can say which is which; the explicit
mapping kills the ambiguity and IS the disclosure record.

Both routes normalize to the private, export-ready ``_BoundMethod``. A public
``explainer()`` export is [UI-SPRINT]; the binder is written so exposure is a
one-line decision later (memo D2).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from torch.nn import Module

from torchlens.attribution._result import AttributionError, AttributionResult

# Registry of the kit's OWN methods: name -> (kind, known_stochastic,
# names of the settings whose presence freezes the randomness). This is a
# private disclosure table, deliberately NOT a public method registry
# (memo: "no public method registry or explainer object").
_KNOWN_METHODS: dict[str, tuple[str, bool, tuple[str, ...]]] = {
    "saliency": ("input", False, ()),
    "input_x_grad": ("input", False, ()),
    "integrated_gradients": ("input", False, ()),
    "smoothgrad": ("input", True, ("seed",)),
    "noise_tunnel": ("input", True, ("seed", "noise_bank")),
    "gradient_shap": ("input", True, ("seed", "draw_bank")),
    "guided_backprop": ("input", False, ()),
    "deconvolution": ("input", False, ()),
    "occlusion_map": ("input", False, ()),
    "grad_cam": ("layer", False, ()),
    "layer_attribution": ("layer", False, ()),
    "layer_integrated_gradients": ("layer", False, ()),
    "layer_conductance": ("layer", False, ()),
    "occlusion": ("trace", False, ()),
    "activation_occlusion": ("trace", False, ()),
    "text": ("input", False, ()),
}


@dataclass(frozen=True)
class _BoundMethod:
    """One normalized attribution method ready to call on (inputs, input_kwargs).

    Attributes
    ----------
    call
        Closed callable ``(inputs, input_kwargs) -> AttributionResult``.
    kind
        Method kind for typed composition refusals: ``"input"``, ``"layer"``,
        ``"trace"``, or ``"opaque"`` (an ``attribute=`` callable whose kind
        cannot be inspected).
    method_name
        Best-known method name for disclosure (``"<closed callable>"`` when
        opaque).
    settings
        The verbatim ``method_kwargs`` mapping (sugar route) or ``{}``.
        Wrapper provenance echoes this mapping; it never forwards blindly.
    known_stochastic
        Whether the bound method is known to consume randomness.
    frozen_randomness
        Whether the bound settings carry a randomness-freezing marker (a seed
        or a stored bank). Only meaningful when ``known_stochastic``.
    model_identity
        Model class name for provenance on the sugar route; ``None`` on the
        primitive route (a wrapper handed a closed callable cannot inspect a
        model, memo D3).
    """

    call: Callable[[Any, dict[str, Any] | None], AttributionResult]
    kind: str
    method_name: str
    settings: dict[str, Any]
    known_stochastic: bool
    frozen_randomness: bool
    model_identity: str | None


def _method_display_name(method: Callable[..., Any]) -> str:
    """Return the best-known display name for a kit-contract callable.

    Parameters
    ----------
    method
        Kit-contract callable passed through ``method=``.

    Returns
    -------
    str
        ``__name__`` when present, else the ``repr``.
    """

    name = getattr(method, "__name__", None)
    return name if isinstance(name, str) else repr(method)


def _bind_method(
    *,
    attribute: Callable[..., AttributionResult] | None = None,
    method: Callable[..., AttributionResult] | None = None,
    method_kwargs: dict[str, Any] | None = None,
    model: Module | None = None,
    target: Any = None,
) -> _BoundMethod:
    """Normalize the two public wrapping routes into one private binder.

    Parameters
    ----------
    attribute
        Closed callable ``(inputs, input_kwargs) -> AttributionResult``.
        Exclusive with ``method``.
    method
        Kit-contract callable ``(model, inputs, input_kwargs=None, *, target,
        **settings)``. Exclusive with ``attribute``.
    method_kwargs
        Explicit settings mapping for ``method``; legal ONLY with ``method``.
    model
        Model bound on the sugar route; required with ``method``, forbidden
        with ``attribute`` (a closed callable is already fully bound).
    target
        Target bound on the sugar route; required with ``method``, forbidden
        with ``attribute``.

    Returns
    -------
    _BoundMethod
        The normalized binder.

    Raises
    ------
    AttributionError
        If the route selection or its arguments violate the D1 contract.
    """

    if (attribute is None) == (method is None):
        raise AttributionError(
            "exactly one of attribute= (a closed callable taking (inputs, "
            "input_kwargs)) or method= (a kit-contract callable) must be "
            "passed. Remedy: pass attribute=functools.partial(...) for a "
            "fully bound callable, or method=<kit function> with model= and "
            "target=.",
            code="attribution_method_binding_invalid",
        )
    if attribute is not None:
        if method_kwargs is not None or model is not None or target is not None:
            raise AttributionError(
                "attribute= is a CLOSED callable: model=, target=, and "
                "method_kwargs= cannot accompany it because nothing is "
                "forwarded to a closed callable. Remedy: bind those into the "
                "callable with functools.partial, or use method= instead.",
                code="attribution_method_binding_invalid",
            )
        if not callable(attribute):
            raise AttributionError(
                "attribute= must be callable. Remedy: pass a callable taking "
                "(inputs, input_kwargs) and returning an AttributionResult.",
                code="attribution_method_binding_invalid",
            )

        def _call_closed(inputs: Any, input_kwargs: dict[str, Any] | None) -> AttributionResult:
            """Invoke the user's closed callable and validate its result type."""

            result = attribute(inputs, input_kwargs)
            if not isinstance(result, AttributionResult):
                raise AttributionError(
                    "the closed attribute= callable must return an "
                    f"AttributionResult; got {type(result).__name__}. "
                    "Remedy: return the child method's AttributionResult "
                    "unchanged.",
                    code="attribution_method_binding_invalid",
                )
            return result

        return _BoundMethod(
            call=_call_closed,
            kind="opaque",
            method_name="<closed callable>",
            settings={},
            known_stochastic=False,
            frozen_randomness=False,
            model_identity=None,
        )

    if not callable(method):
        raise AttributionError(
            "method= must be a kit-contract callable. Remedy: pass one of the "
            "torchlens.attribution methods or a callable with the same "
            "(model, inputs, input_kwargs=None, *, target, **settings) shape.",
            code="attribution_method_binding_invalid",
        )
    if model is None or target is None:
        raise AttributionError(
            "method= binds the sugar route, which requires model= and "
            "target=. Remedy: pass both, or use attribute= with a closed "
            "callable.",
            code="attribution_method_binding_invalid",
        )
    if method_kwargs is None:
        settings: dict[str, Any] = {}
    elif isinstance(method_kwargs, dict):
        settings = dict(method_kwargs)
    else:
        raise AttributionError(
            "method_kwargs must be a dict mapping the child method's keyword "
            "names to values. Remedy: pass method_kwargs=dict(...).",
            code="attribution_method_binding_invalid",
        )

    name = _method_display_name(method)
    kind, known_stochastic, freeze_markers = _KNOWN_METHODS.get(name, ("input", False, ()))
    frozen = any(settings.get(marker) is not None for marker in freeze_markers)

    def _call_sugar(inputs: Any, input_kwargs: dict[str, Any] | None) -> AttributionResult:
        """Invoke the kit-contract callable with the bound model/target/settings."""

        result = method(model, inputs, input_kwargs, target=target, **settings)
        if not isinstance(result, AttributionResult):
            raise AttributionError(
                f"method= callable {name!r} must return an AttributionResult; "
                f"got {type(result).__name__}. Remedy: satisfy the kit "
                "contract or wrap the value in an AttributionResult.",
                code="attribution_method_binding_invalid",
            )
        return result

    return _BoundMethod(
        call=_call_sugar,
        kind=kind,
        method_name=name,
        settings=settings,
        known_stochastic=known_stochastic,
        frozen_randomness=frozen,
        # The sugar layer may add model identity because it SAW the model; a
        # primitive-route user can write the same fact into their own child
        # extra (memo D3 keeps the two routes' disclosure equally rich).
        model_identity=type(model).__name__,
    )


__all__ = ["_KNOWN_METHODS", "_BoundMethod", "_bind_method"]
