"""Backend-neutral post-processing plan for non-torch ``tl.trace`` captures.

A finished non-torch capture holds every op's realized payload, its parents and
its execution order. The options below therefore need nothing from the
framework at capture time and run on the finished trace instead:

- value-dependent ``save=`` predicates (plain callables and ``tl.where``),
- lookback selectors (``tl.preceded_by`` and ``candidate & tl.followed_by``)
  with their ``lookback`` window,
- save shaping: ``activation_transform`` (with ``save_raw_activations``),
  ``output_device``, ``detach_saved_activations``, ``layer_visualizers`` and
  ``save_visualizations``.

:func:`split_post_options` removes those options from the dispatch bundle
before the backend runs (the backend then performs a full-save capture) and
returns the plan; ``_selective_save.apply_backend_post_plan`` applies it to the
returned trace with the same semantics torch applies at capture time.
"""

from __future__ import annotations

__tl_layer__ = "L3"

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Final

from ..options import _CAPTURE_DEFAULTS, CaptureOptions

_TORCH_BACKEND: Final = "torch"

#: Capture fields applied after a full-save non-torch capture.
_SHAPING_FIELDS: Final[tuple[str, ...]] = (
    "activation_transform",
    "output_device",
    "detach_saved_activations",
    "layer_visualizers",
    "save_visualizations",
    "lookback",
    "lookback_payload_policy",
)

#: Flat dispatch keys that mirror a capture field, with their inert value.
_FLAT_INERT: Final[dict[str, Any]] = {"lookback": 0, "lookback_payload_policy": "metadata_only"}


@dataclass(frozen=True)
class BackendPostPlan:
    """Options a non-torch trace applies after capture.

    Parameters
    ----------
    save:
        Value-dependent or lookback ``save=`` predicate, or ``None``.
    lookback:
        Event window for ``tl.preceded_by`` / ``tl.followed_by``.
    lookback_payload_policy:
        Lookback payload policy (``"metadata_only"`` keeps no candidates).
    activation_transform:
        Transform applied to every saved payload.
    save_raw_activations:
        Whether the raw payload is kept beside a transformed one.
    output_device:
        Device every saved payload is moved to (``"same"`` keeps it).
    detach_saved_activations:
        Whether saved payloads are detached from autograd history.
    layer_visualizers:
        Selector to visualizer mapping rendered after capture.
    save_visualizations:
        Whether rendered visualizations persist in portable bundles.
    """

    save: Any = None
    lookback: int = 0
    lookback_payload_policy: str = "metadata_only"
    activation_transform: Callable[[Any], Any] | None = None
    save_raw_activations: bool = True
    output_device: str = "same"
    detach_saved_activations: bool = False
    layer_visualizers: Mapping[Any, Callable[..., Any]] | None = None
    save_visualizations: bool = False


def split_post_options(
    kwargs: dict[str, Any], backend_name: str, *, post_save: Any
) -> BackendPostPlan | None:
    """Move post-processable options out of a non-torch dispatch bundle.

    Parameters
    ----------
    kwargs:
        Mutable public dispatch bundle. Post-processed options are replaced by
        their inert values so the backend performs a full-save capture.
    backend_name:
        Resolved backend name; torch keeps every option at capture time.
    post_save:
        The ``save=`` value when it is a value-dependent or lookback predicate,
        else ``None``.

    Returns
    -------
    BackendPostPlan | None
        The plan to apply to the returned trace, or ``None`` when nothing moved.
    """

    if backend_name == _TORCH_BACKEND:
        return None
    capture = kwargs.get("capture")
    values = _explicit_shaping_values(capture)
    if not values and post_save is None:
        return None
    if values and isinstance(capture, CaptureOptions):
        kwargs["capture"] = _without_fields(capture, tuple(values))
    for name, inert in _FLAT_INERT.items():
        if name in kwargs:
            kwargs[name] = inert
    if post_save is not None:
        kwargs["save"] = None
    return BackendPostPlan(save=post_save, **values)


def _explicit_shaping_values(capture: Any) -> dict[str, Any]:
    """Return the explicit, non-default shaping fields of ``capture``."""

    if not isinstance(capture, CaptureOptions):
        return {}
    values = {
        name: getattr(capture, name)
        for name in _SHAPING_FIELDS
        if capture._is_field_explicit(name) and getattr(capture, name) != _CAPTURE_DEFAULTS[name]
    }
    # ``save_raw_activations`` only has a torch meaning beside a transform
    # (torch keeps the raw payload whenever no transform is set); alone it stays
    # with the backend's own capture-policy seam.
    if "activation_transform" in values and capture._is_field_explicit("save_raw_activations"):
        values["save_raw_activations"] = bool(capture.save_raw_activations)
    return values


def _without_fields(capture: CaptureOptions, names: tuple[str, ...]) -> CaptureOptions:
    """Return ``capture`` with ``names`` reset to their defaults and unspecified."""

    values = capture.as_dict()
    for name in names:
        values[name] = _CAPTURE_DEFAULTS[name]
    return CaptureOptions._from_values(
        values,
        frozenset(capture._specified_fields) - frozenset(names),
        _layers_to_save=capture._layers_to_save,
        _layers_to_save_explicit=capture._layers_to_save_explicit,
    )
