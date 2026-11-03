"""Held torch-function reference normalization for ``release_model``.

A plain module attribute holding a torch function (``self.act = F.relu``)
captures whichever object -- pristine original or wrap-epoch wrapper -- the
name resolved to at construction time. Pickle serializes functions by
reference with a strict identity check against the CURRENT module attribute,
so a held ref from the other wrap state fails ``pickle``/``torch.save`` with
``PicklingError`` in both directions. ``release_model`` normalizes each held
ref to the value currently live at its public name so the released model is
serializable NOW, and registers the model so every later wrap-state flip
(``wrap_torch()``/``unwrap_torch()``) re-normalizes it: released stays
serializable in EVERY epoch, never a one-way trip (split out of
``model_prep.py`` under the R43 file-size ratchet -- this is the release-time
serializability seam, not preparation).

Container coverage is type-exact by design: one level of exact builtin
``list``/``tuple``/``dict``/``set``/``frozenset`` plus namedtuples (rebuilt
through ``_make`` so the runtime type survives). Other builtin subclasses are
left untouched -- a subclass can carry instance state, custom item protocols,
or constructor signatures a blind rebuild would corrupt -- and stay part of
the disclosed closures/partials/custom-container residual.
"""

import weakref
from typing import Any

from torch import nn

from ... import _state
from .wrappers import get_optional_torch_namespace

__all__ = [
    "normalize_held_torch_function_refs",
    "register_released_model",
    "renormalize_released_models",
]

# Models the user explicitly released. A wrap-state flip re-points their held
# refs at the newly-live values so ``release_model``'s serializability promise
# survives ``unwrap_torch()``/re-wrap instead of silently inverting into the
# unrecoverable pickle shape (the released model held the transient epoch's
# wrapper, which no later namespace state ever matches again).
_RELEASED_MODELS: "weakref.WeakSet[nn.Module]" = weakref.WeakSet()


def register_released_model(model: nn.Module) -> None:
    """Track a released model for re-normalization on wrap-state flips."""

    _RELEASED_MODELS.add(model)


def renormalize_released_models() -> None:
    """Re-point held refs on every released model at the now-live values.

    Called by ``wrap_torch()``/``unwrap_torch()`` after the namespace flip.
    Walks each registered model's CURRENT tree; the sweep is the same bounded
    normalization ``release_model`` ran, so a model released while wrapped
    comes back to pristine originals at unwrap (and vice versa).
    """

    for model in tuple(_RELEASED_MODELS):
        for module in model.modules():
            normalize_held_torch_function_refs(module)


def _live_counterpart(value: Any) -> Any | None:
    """Resolve the value currently LIVE at a held torch function's public name.

    ``value`` is a plain instance attribute captured from a model tree. When
    it is a ledgered torch-function wrapper OR a ledgered original whose
    public torch name currently resolves to its exact wrap-epoch counterpart,
    return that live counterpart; otherwise return ``None`` (leave the
    attribute alone). The swap is ledger-fenced BOTH ways: a foreign patch or
    an unstamped (class-namespace) wrapper never matches and is never
    touched.
    """

    decorated_to_orig, orig_to_decorated = _state.wrap_epoch_ledgers()
    wrapper: Any = None
    if callable(value):
        if id(value) in decorated_to_orig:
            wrapper = value
        elif id(value) in orig_to_decorated:
            wrapper = orig_to_decorated[id(value)]
    if wrapper is None:
        return None
    namespace_name = getattr(wrapper, "__module__", None)
    func_name = getattr(wrapper, "__qualname__", None)
    if not namespace_name or not func_name or "." in func_name:
        return None

    namespace_obj = get_optional_torch_namespace(namespace_name)
    if namespace_obj is None or isinstance(namespace_obj, type):
        return None
    current = getattr(namespace_obj, func_name, None)
    if current is None or current is value:
        return None
    # Only the exact wrap-epoch counterpart qualifies: the held wrapper's own
    # original, or the wrapper installed over the held original.
    if (
        decorated_to_orig.get(id(current)) is decorated_to_orig.get(id(value), value)
        or decorated_to_orig.get(id(value)) is current
    ):
        return current
    return None


def _is_namedtuple_instance(value: Any) -> bool:
    """Return True for namedtuple instances (``_fields`` + ``_make``)."""

    return (
        isinstance(value, tuple)
        and isinstance(getattr(type(value), "_fields", None), tuple)
        and callable(getattr(type(value), "_make", None))
    )


def normalize_held_torch_function_refs(module: nn.Module) -> None:
    """Re-point epoch-mismatched torch-function attrs at their live values.

    A plain module attribute holding a torch function (``self.act = F.relu``)
    captures whichever object -- pristine original or wrap-epoch wrapper --
    the name resolved to at construction time. Pickle serializes functions
    by reference with a strict identity check against the CURRENT module
    attribute, so a held ref from the other wrap state fails
    ``pickle``/``torch.save`` with ``PicklingError`` in both directions.
    Normalizing each held ref to the value currently live at its public name
    makes the released model serializable NOW (and a fresh-process load
    resolves the public torch name either way). The sweep is bounded and
    type-exact: direct ``__dict__`` values plus one level of exact builtin
    ``list``/``tuple``/``dict``/``set``/``frozenset`` containers and
    namedtuples (rebuilt through ``_make``, preserving the runtime type).
    Dict KEYS are swapped as well as values (functions key by identity, so a
    ``{F.relu: cfg}`` table is exactly the pickle-failing shape). Refs inside
    closures, partials, other builtin subclasses, or custom objects are a
    disclosed residual, as are bare references held outside the model.
    """

    for attr_name, attr_value in tuple(module.__dict__.items()):
        replacement = _live_counterpart(attr_value)
        if replacement is not None:
            module.__dict__[attr_name] = replacement
        else:
            rebuilt = _normalized_container(attr_value)
            if rebuilt is not None:
                module.__dict__[attr_name] = rebuilt


def _normalized_container(attr_value: Any) -> Any | None:
    """Normalize one container attr; return a rebuild or ``None`` (no rebind).

    Type-exact by design: mutable exact builtins normalize in place (always
    ``None``), exact tuples and namedtuples rebuild (namedtuples through
    ``_make``, preserving the runtime type), exact frozensets rebuild only
    when no cardinality collapse would drop a user member. Every other
    subclass is left untouched (disclosed custom-container residual).
    """

    value_type = type(attr_value)
    if value_type is list or value_type is dict:
        _swap_live_refs_inplace(attr_value)
    elif value_type is set:
        _swap_live_refs_in_set(attr_value)
    elif value_type is tuple:
        return _live_swapped_tuple(attr_value)
    elif value_type is frozenset:
        swapped_members = _live_swapped_tuple(tuple(attr_value))
        if swapped_members is not None:
            rebuilt = frozenset(swapped_members)
            # A cardinality change means both epochs' objects were held;
            # preserving the user's members beats serializability.
            if len(rebuilt) == len(attr_value):
                return rebuilt
    elif _is_namedtuple_instance(attr_value):
        swapped = _live_swapped_tuple(tuple(attr_value))
        if swapped is not None:
            try:
                return value_type._make(swapped)
            except (TypeError, ValueError):
                # Exotic _make override: leave the user's object alone
                # (disclosed custom-container residual).
                return None
    return None


def _swap_live_refs_inplace(container: list[Any] | dict[Any, Any]) -> None:
    """Re-point epoch-mismatched refs inside one list/dict, preserving identity."""

    if isinstance(container, list):
        for index, item in enumerate(container):
            live = _live_counterpart(item)
            if live is not None:
                container[index] = live
        return
    for key, item in tuple(container.items()):
        live = _live_counterpart(item)
        if live is not None:
            container[key] = live
    # Dict KEYS fail pickle's by-reference identity check exactly like values
    # (functions hash and compare by identity). Pop/reinsert; on a collision
    # (both epochs' objects present as keys) keep both entries untouched --
    # preserving the user's data beats serializability.
    for key in tuple(container.keys()):
        live_key = _live_counterpart(key)
        if live_key is not None and live_key not in container:
            container[live_key] = container.pop(key)


def _live_swapped_tuple(values: tuple[Any, ...]) -> tuple[Any, ...] | None:
    """Return a live-ref rebuild of one tuple, or ``None`` when nothing changed."""

    swapped = tuple(_live_counterpart(item) or item for item in values)
    if any(new is not old for new, old in zip(swapped, values, strict=True)):
        return swapped
    return None


def _swap_live_refs_in_set(container: set[Any]) -> None:
    """Re-point epoch-mismatched refs inside one set, preserving identity.

    Add-before-discard so an interrupt mid-swap can only leave BOTH members
    present, never neither; a pre-existing counterpart member (both epochs'
    objects held) skips the swap entirely so no user member is ever lost.
    """

    for item in tuple(container):
        live = _live_counterpart(item)
        if live is None or live in container:
            continue
        container.add(live)
        container.discard(item)
