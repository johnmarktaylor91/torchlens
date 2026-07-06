"""Internal predicate and selector helpers for public trace capture."""

from __future__ import annotations

import collections.abc
from collections.abc import Iterable
from typing import Any, cast

from .backends.registry import PUBLIC_OPTION_SPINE_TRACE_OPTIONS
from .fastlog.exceptions import PredicateError
from .fastlog.options import PredicateFn
from .fastlog.types import RecordContext
from .intervention.selectors import BaseSelector
from .options import SaveOptions


def _split_save_options_and_predicate(
    save_value: SaveOptions | PredicateFn | BaseSelector | None,
) -> tuple[SaveOptions | None, PredicateFn | None]:
    """Separate grouped save options from ``trace(save=predicate)`` sugar.

    Parameters
    ----------
    save_value:
        Value supplied to the public ``save`` parameter.

    Returns
    -------
    tuple[SaveOptions | None, PredicateFn | None]
        Grouped save options and optional selective-save predicate.
    """

    if save_value is None:
        return None, None
    if isinstance(save_value, SaveOptions):
        return save_value, None
    if isinstance(save_value, BaseSelector):
        return None, cast(PredicateFn, save_value)
    if callable(save_value):
        return None, cast(PredicateFn, save_value)
    raise TypeError("save must be a SaveOptions instance, predicate callable, selector, or None")


def _is_selective_label_save(value: object) -> bool:
    """Return whether ``layers_to_save`` needs predicate-time label matching.

    Parameters
    ----------
    value
        Normalized public ``layers_to_save`` value.

    Returns
    -------
    bool
        ``True`` for selective layer lists or selectors.
    """

    return value not in ("all", "none", None, [])


def _label_save_candidates(ctx: RecordContext) -> set[str]:
    """Return legacy lookup-key spellings that can identify ``ctx``.

    Parameters
    ----------
    ctx
        Predicate context for one capture event.

    Returns
    -------
    set[str]
        Candidate labels, types, and function names available before postprocess.
    """

    candidates = {ctx.label}
    if ctx.raw_label is not None:
        candidates.add(ctx.raw_label)
        if ctx.raw_label.endswith("_raw"):
            candidates.add(ctx.raw_label[: -len("_raw")])
    if ctx.layer_type is not None:
        candidates.add(ctx.layer_type)
        if ctx.type_index is not None:
            candidates.add(f"{ctx.layer_type}_{ctx.type_index}")
            candidates.add(f"{ctx.layer_type}_{ctx.type_index}_{ctx.pass_index}")
            pass_index = ctx.pass_index
            indexed_candidates = {f"{ctx.layer_type}_{ctx.type_index}_{ctx.pass_index}"}
            if ctx.raw_index is not None:
                indexed_candidates.add(f"{ctx.layer_type}_{ctx.type_index}_{ctx.raw_index}")
                if ctx.raw_index > 0:
                    indexed_candidates.add(f"{ctx.layer_type}_{ctx.type_index}_{ctx.raw_index - 1}")
            for candidate in indexed_candidates:
                candidates.add(candidate)
                candidates.add(f"{candidate}:{pass_index}")
    if ctx.func_name is not None:
        candidates.add(ctx.func_name)
        candidates.add(ctx.func_name.lower().replace("_", ""))
    if ctx.address:
        candidates.add(ctx.address)
        if ctx.module_pass_index is not None:
            candidates.add(f"{ctx.address}:{ctx.module_pass_index}")
        candidates.add(ctx.address.rsplit(".", 1)[-1])
        if ctx.module_pass_index is not None:
            candidates.add(f"{ctx.address.rsplit('.', 1)[-1]}:{ctx.module_pass_index}")
    return candidates


def _make_layers_to_save_predicate(layers_to_save: object) -> PredicateFn:
    """Translate selective ``layers_to_save`` values into a save predicate.

    Parameters
    ----------
    layers_to_save
        Public layer selection list or selector-like callable.

    Returns
    -------
    PredicateFn
        Predicate evaluated by the unified capture spine.
    """

    if isinstance(layers_to_save, BaseSelector):
        return cast(PredicateFn, layers_to_save)
    if callable(layers_to_save):
        return cast(PredicateFn, layers_to_save)
    requested = (
        {layers_to_save}
        if isinstance(layers_to_save, str)
        else set(cast(Iterable[Any], layers_to_save))
    )
    requested_ints = {int(item) for item in requested if isinstance(item, int)}
    requested_strings = {str(item) for item in requested if not isinstance(item, int)}
    requested_string_bases = {item for item in requested_strings if ":" not in item}

    def predicate(ctx: RecordContext) -> bool:
        """Return whether this event matches the absorbed layer selection."""

        if ctx.kind != "op":
            return False
        if ctx.raw_index in requested_ints:
            return True
        candidates = _label_save_candidates(ctx)
        if candidates & requested_strings:
            return True
        return bool(candidates & requested_string_bases)

    return cast(PredicateFn, predicate)


def _layers_to_save_mentions_output(layers_to_save: object) -> bool:
    """Return whether a selection names a postprocess output wrapper.

    Parameters
    ----------
    layers_to_save
        Public ``layers_to_save`` selection.

    Returns
    -------
    bool
        ``True`` when any token appears to target an output layer.
    """

    if isinstance(layers_to_save, str):
        values = (layers_to_save,)
    elif isinstance(layers_to_save, collections.abc.Iterable):
        values = tuple(layers_to_save)
    else:
        return False
    return any(str(value).startswith("output") for value in values)


def _layers_to_save_has_negative_index(layers_to_save: object) -> bool:
    """Return whether ``layers_to_save`` contains a negative op index.

    Parameters
    ----------
    layers_to_save
        Public ``layers_to_save`` selection.

    Returns
    -------
    bool
        ``True`` when the selection needs final graph cardinality to resolve.
    """

    if isinstance(layers_to_save, bool):
        return False
    if isinstance(layers_to_save, int):
        return layers_to_save < 0
    if isinstance(layers_to_save, collections.abc.Iterable) and not isinstance(
        layers_to_save,
        str,
    ):
        return any(
            isinstance(value, int) and not isinstance(value, bool) and value < 0
            for value in layers_to_save
        )
    return False


def _layers_to_save_has_integer_selector(layers_to_save: object) -> bool:
    """Return whether ``layers_to_save`` contains a legacy integer selector.

    Parameters
    ----------
    layers_to_save
        Public ``layers_to_save`` selection.

    Returns
    -------
    bool
        ``True`` when the selection contains an integer layer-list index.
    """

    if isinstance(layers_to_save, bool):
        return False
    if isinstance(layers_to_save, int):
        return True
    if isinstance(layers_to_save, collections.abc.Iterable) and not isinstance(
        layers_to_save,
        str,
    ):
        return any(
            isinstance(value, int) and not isinstance(value, bool) for value in layers_to_save
        )
    return False


def _layers_to_save_mentions_identity(layers_to_save: object) -> bool:
    """Return whether ``layers_to_save`` targets pass-through identity layers.

    Parameters
    ----------
    layers_to_save
        Public ``layers_to_save`` selection.

    Returns
    -------
    bool
        ``True`` when matching needs the legacy module pass-through resolver.
    """

    if isinstance(layers_to_save, str):
        values = (layers_to_save,)
    elif isinstance(layers_to_save, collections.abc.Iterable):
        values = tuple(layers_to_save)
    else:
        return False
    return any(str(value).startswith("identity") for value in values)


def _combine_save_predicates(
    first: PredicateFn | None,
    second: PredicateFn,
) -> PredicateFn:
    """Return a union predicate for public ``save=`` and ``layers_to_save``.

    Parameters
    ----------
    first
        Existing public save predicate, if supplied.
    second
        Predicate generated from ``layers_to_save``.

    Returns
    -------
    PredicateFn
        Predicate that saves when either input predicate matches.
    """

    if first is None:
        return second

    def predicate(ctx: RecordContext) -> bool:
        """Return whether either constituent save predicate matches."""

        return bool(first(ctx) or second(ctx))

    return cast(PredicateFn, predicate)


_TRACE_OPTION_FILTERED_NAMES = (
    *PUBLIC_OPTION_SPINE_TRACE_OPTIONS,
    "jax_static_argnums",
    "grad_options",
)
