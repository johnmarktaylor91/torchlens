"""Internal predicate and selector helpers for public trace capture."""

from __future__ import annotations

import functools
import hashlib
import types
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any, cast

from . import options as _options
from ._capture.arg_positions import _normalize_func_name
from ._deprecations import MISSING
from ._errors import AmbiguousOpLookupError, ArgumentTypeError
from ._selection.selectors import BaseSelector
from ._trace_selection_shape import _selector_component_is_final_only
from .backends import registry as _backend_registry
from .recording.options import PredicateFn
from .recording.types import RecordContext

if TYPE_CHECKING:
    from .options import CaptureOptions


def _split_save_options_and_predicate(
    save_value: _options._SaveTransforms | PredicateFn | BaseSelector | None,
) -> tuple[_options._SaveTransforms | None, PredicateFn | None]:
    """Separate grouped save options from ``trace(save=predicate)`` sugar.

    Parameters
    ----------
    save_value:
        Value supplied to the public ``save`` parameter.

    Returns
    -------
    tuple[_SaveTransforms | None, PredicateFn | None]
        Grouped save options and optional selective-save predicate.
    """

    if save_value is None:
        return None, None
    if isinstance(save_value, _options._SaveTransforms):
        return save_value, None
    from ._selection.selectors import FacetSelector
    from .intervention._facet_save import _facet_save_predicate

    if isinstance(save_value, FacetSelector):
        # Save-only seam: intervene=/bind/hooks= evaluate the facet selector
        # itself, which refuses at capture, so no facet edit lands on a whole
        # home tensor; save= gets the ordinary selector for the facet's homes.
        save_value = _facet_save_predicate(save_value)
    if isinstance(save_value, BaseSelector):
        return None, cast(PredicateFn, save_value)
    if callable(save_value):
        return None, cast(PredicateFn, save_value)
    raise ArgumentTypeError(
        f"save must be a layer list or string, predicate callable, "
        f"selector, or None; "
        f"received {type(save_value).__name__}",
        code="save_predicate_type_invalid",
        remedy=(
            "pass a list of layer names or module addresses such as save=['layer1'], "
            "a selector such as save=tl.func('relu'), save='all', or a a reduce-only save"
        ),
        argument="save",
        received_type=type(save_value).__name__,
    )


def _resolve_save_transforms(
    capture_options: CaptureOptions,
    *,
    activation_transform: Any = MISSING,
    grad_transform: Any = MISSING,
    save_raw_activations: Any = MISSING,
    save_raw_grads: Any = MISSING,
) -> _options._SaveTransforms:
    """Return the save transforms: the merged capture fields, flat values winning."""

    base = _options._save_transforms_from_capture(capture_options)
    return _options._SaveTransforms(
        activation_transform=(
            base.activation_transform if activation_transform is MISSING else activation_transform
        ),
        grad_transform=base.grad_transform if grad_transform is MISSING else grad_transform,
        save_raw_activations=(
            base.save_raw_activations if save_raw_activations is MISSING else save_raw_activations
        ),
        save_raw_grads=(base.save_raw_gradients if save_raw_grads is MISSING else save_raw_grads),
    )


def _is_save_label_selection(save_value: object) -> bool:
    """Return whether ``save=`` holds the documented label/module/index list form.

    Parameters
    ----------
    save_value
        Value supplied to the public ``save`` parameter.

    Returns
    -------
    bool
        ``True`` for a string or a list/tuple of strings and integers.
    """

    if isinstance(save_value, str):
        return True
    if isinstance(save_value, (list, tuple)):
        return all(
            isinstance(item, (str, int)) and not isinstance(item, bool) for item in save_value
        )
    return False


def _route_save_label_selection(
    save_value: object,
    capture: Any,
    capture_options: CaptureOptions,
    model: Any = None,
) -> tuple[Any, CaptureOptions]:
    """Route ``save=[...]`` / ``save='...'`` into the ``layers_to_save`` selection.

    The list form documented on ``tl.trace`` (layer names, pass-qualified labels,
    module addresses, integer indexes, substrings, ``'all'``/``'none'``) selects
    the same internal label-selection machinery;
    ``save='stats'`` becomes the reduce-and-discard ``_SaveTransforms``.

    Parameters
    ----------
    save_value
        Value supplied to the public ``save`` parameter.
    capture
        The caller's ``capture=`` object, or ``None``.
    capture_options
        Merged capture options for this call.
    model
        The model being traced, used to map dot-flattened stage names.

    Returns
    -------
    tuple[Any, CaptureOptions]
        ``(save value left for predicate/_SaveTransforms splitting, capture options)``.

    """

    if isinstance(save_value, str) and save_value.lower() == "stats":
        # Reduce-and-discard save level: one Spine per op. deferred: cycle stats (layering follow-up)
        from .stats._capture_stats import stats_save_options

        return stats_save_options(), _flatten_stage_names_in_options(capture_options, model)
    if not _is_save_label_selection(save_value):
        return save_value, _flatten_stage_names_in_options(capture_options, model)
    from .options import merge_capture_options

    selection = save_value if isinstance(save_value, str) else list(cast(Iterable[Any], save_value))
    merged = merge_capture_options(capture=capture_options, _layers_to_save=selection)
    return None, _flatten_stage_names_in_options(merged, model)


def _flatten_stage_names_in_options(capture_options: CaptureOptions, model: Any) -> CaptureOptions:
    """Rewrite dotted stage names registered with dots flattened to their module address.

    timm ``FeatureListNet`` registers stage ``"stages.0"`` as module ``"stages_0"``;
    a selection naming ``"stages.0"`` matches no address as a substring, so it is
    rewritten to ``"stages_0"`` before capture. Only dotted strings that are no
    substring of any module address and whose flattened form is an exact address
    change; everything else is left as written.

    Parameters
    ----------
    capture_options
        Merged capture options for this call.
    model
        The model being traced (anything without ``named_modules`` is skipped).

    Returns
    -------
    CaptureOptions
        ``capture_options``, or a copy with the rewritten ``layers_to_save``.
    """

    selection = capture_options._layers_to_save
    named_modules = getattr(model, "named_modules", None)
    if not callable(named_modules) or not isinstance(selection, (str, list, tuple)):
        return capture_options
    addresses = [name for name, _module in named_modules() if name]
    items = [selection] if isinstance(selection, str) else list(selection)
    rewritten = [_flattened_stage_name(item, addresses) for item in items]
    if rewritten == items:
        return capture_options
    from ._option_receipt import _explicit_fields

    values = capture_options.as_dict()
    selection = rewritten[0] if isinstance(selection, str) else rewritten
    return cast(
        "CaptureOptions",
        type(capture_options)._from_values(
            values,
            _explicit_fields(capture_options),
            _layers_to_save=selection,
            _layers_to_save_explicit=capture_options._layers_to_save_explicit,
        ),
    )


def _flattened_stage_name(item: Any, addresses: list[str]) -> Any:
    """Return the flattened module address a dotted stage name means, else ``item``."""

    if not isinstance(item, str) or "." not in item:
        return item
    base, call_suffix = _split_call_suffix(item)
    if any(base in address for address in addresses):
        return item
    flattened = base.replace(".", "_")
    return f"{flattened}{call_suffix}" if flattened in addresses else item


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
            # Numeric-indexed candidates carry capture-time numbering only.
            # Requests that target FINAL numbering (any ``_<digit>`` or
            # ``:<digit>`` component) never reach this predicate: they route
            # through the deferred post-postprocess resolution instead, so
            # no raw-index "prediction" of final ordinals happens here.
            candidates.add(f"{ctx.layer_type}_{ctx.type_index}")
            indexed_candidate = f"{ctx.layer_type}_{ctx.type_index}_{ctx.pass_number}"
            candidates.add(indexed_candidate)
            candidates.add(f"{indexed_candidate}:{ctx.pass_number}")
    if ctx.func_name is not None:
        candidates.add(ctx.func_name)
        candidates.add(_normalize_func_name(ctx.func_name))
    if ctx.address:
        candidates.add(ctx.address)
        if ctx.module_pass_number is not None:
            candidates.add(f"{ctx.address}:{ctx.module_pass_number}")
        candidates.add(ctx.address.rsplit(".", 1)[-1])
        if ctx.module_pass_number is not None:
            candidates.add(f"{ctx.address.rsplit('.', 1)[-1]}:{ctx.module_pass_number}")
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
    final_only_components = sorted(
        (item for item in requested if _selector_component_is_final_only(item)),
        key=repr,
    )
    if final_only_components:
        # Fail closed: integer ordinals and final-label-shaped strings are
        # defined against FINAL layer numbering, which is unknowable during
        # the forward (postprocess orphan removal renumbers ordinals, type
        # indexes, and passes). Routing must send them through the deferred
        # post-postprocess resolution; matching them here silently saves the
        # wrong layer's data.
        raise RuntimeError(
            "internal invariant violated: final-numbering layers_to_save components "
            f"{final_only_components!r} reached the capture-time save predicate; "
            "they must resolve on the deferred post-postprocess path"
        )
    requested_strings = {str(item) for item in requested}
    cache_key = (
        "layers_to_save",
        tuple(sorted(requested_strings)),
    )

    def string_matches(requested_string: str, ctx: RecordContext, candidates: set[str]) -> bool:
        """Return whether a legacy string selector matches candidate labels."""

        requested_base = requested_string
        if ":" in requested_string:
            requested_base, pass_index_text = requested_string.rsplit(":", 1)
            try:
                requested_pass_index = int(pass_index_text)
            except ValueError:
                requested_base = requested_string
            else:
                if ctx.pass_number != requested_pass_index:
                    return False

        if requested_base == "":
            return False
        return any(requested_base in candidate for candidate in candidates)

    def predicate(ctx: RecordContext) -> bool:
        """Return whether this event matches the absorbed layer selection."""

        if ctx.kind != "op":
            return False
        candidates = _label_save_candidates(ctx)
        return any(
            string_matches(requested_string, ctx, candidates)
            for requested_string in requested_strings
        )

    setattr(predicate, "__torchlens_cache_key__", cache_key)
    return cast(PredicateFn, predicate)


def _predicate_cache_key(predicate: object) -> object:
    """Return a stable cache-key fragment for predicate-like objects.

    Parameters
    ----------
    predicate:
        Predicate, selector, or ``None``.

    Returns
    -------
    object
        JSON-serializable identity fragment that avoids memory-address reprs.
    """

    if predicate is None:
        return None
    explicit_key = getattr(predicate, "__torchlens_cache_key__", None)
    if explicit_key is not None:
        return explicit_key
    if isinstance(predicate, BaseSelector):
        return (
            "selector",
            predicate.selector_kind,
            _stable_cache_fragment(predicate.selector_value),
        )
    module = getattr(predicate, "__module__", None)
    qualname = getattr(predicate, "__qualname__", None)
    code_digest = _callable_code_digest(predicate)
    # The callable lane engages whenever a code digest exists, not only when
    # __module__/__qualname__ are set: two module-less (exec-built) predicates
    # used to fall through to the object lane and COLLIDE on
    # ("object", "builtins", "function"). Callables without a code object
    # (builtins, callable instances) keep the type-identity object lane.
    if callable(predicate) and (
        code_digest is not None or (module is not None and qualname is not None)
    ):
        defaults = getattr(predicate, "__defaults__", None)
        # Keyword-only defaults are a separate attribute: ``def p(ctx, *,
        # thr=0.5)`` redefined with ``thr=0.9`` has identical co_code, an
        # empty closure, and ``__defaults__ is None`` -- omitting
        # ``__kwdefaults__`` collided the two on one key (stale hit).
        kwdefaults = getattr(predicate, "__kwdefaults__", None)
        closure = getattr(predicate, "__closure__", None)
        closure_values: tuple[object, ...] = ()
        if closure:
            closure_values = tuple(_stable_cache_fragment(cell.cell_contents) for cell in closure)
        return (
            "callable",
            str(module),
            str(qualname),
            code_digest,
            _stable_cache_fragment(defaults),
            _stable_cache_fragment(kwdefaults),
            closure_values,
        )
    return ("object", type(predicate).__module__, type(predicate).__qualname__)


def _stable_cache_fragment(value: object) -> object:
    """Return a stable primitive fragment for nested predicate state.

    Parameters
    ----------
    value:
        Candidate value captured by a predicate or selector.

    Returns
    -------
    object
        JSON-friendly primitive, list, tuple, or type identity.
    """

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, BaseSelector):
        return ("selector", value.selector_kind, _stable_cache_fragment(value.selector_value))
    if isinstance(value, dict):
        return tuple(
            sorted(
                (
                    _stable_cache_fragment(key),
                    _stable_cache_fragment(item_value),
                )
                for key, item_value in value.items()
            )
        )
    if isinstance(value, (list, tuple)):
        return tuple(_stable_cache_fragment(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((_stable_cache_fragment(item) for item in value), key=repr))
    if isinstance(value, functools.partial):
        return (
            "partial",
            _stable_cache_fragment(value.func),
            _stable_cache_fragment(value.args),
            _stable_cache_fragment(value.keywords),
        )
    if callable(value):
        module = getattr(value, "__module__", None)
        qualname = getattr(value, "__qualname__", None)
        code_digest = _callable_code_digest(value)
        if code_digest is not None or (module is not None and qualname is not None):
            # Both default families participate: the code digest alone cannot
            # distinguish two definitions differing only in default values.
            return (
                "callable",
                str(module),
                str(qualname),
                code_digest,
                _stable_cache_fragment(getattr(value, "__defaults__", None)),
                _stable_cache_fragment(getattr(value, "__kwdefaults__", None)),
            )
    return ("object", type(value).__module__, type(value).__qualname__)


def _callable_code_digest(value: object) -> str | None:
    """Return a stable digest of a Python callable's executable code.

    Parameters
    ----------
    value:
        Callable candidate.

    Returns
    -------
    str | None
        SHA-256 digest for Python code objects, otherwise ``None``.
    """

    code = getattr(value, "__code__", None)
    if not isinstance(code, types.CodeType):
        return None
    hasher = hashlib.sha256()

    def update_code(current: types.CodeType) -> None:
        """Add one code object and its nested constants to ``hasher``."""

        hasher.update(current.co_code)
        hasher.update(repr(current.co_names).encode("utf-8"))
        hasher.update(repr(current.co_varnames).encode("utf-8"))
        for constant in current.co_consts:
            if isinstance(constant, types.CodeType):
                update_code(constant)
            else:
                hasher.update(repr((type(constant).__qualname__, constant)).encode("utf-8"))

    update_code(code)
    return hasher.hexdigest()


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

    setattr(
        predicate,
        "__torchlens_cache_key__",
        ("or", _predicate_cache_key(first), _predicate_cache_key(second)),
    )
    return cast(PredicateFn, predicate)


_TRACE_OPTION_FILTERED_NAMES: tuple[str, ...] = _backend_registry._trace_option_names(
    c for c in _backend_registry._BACKEND_CAPABILITIES if not c.gated
)
"""Every option of every non-gated capability record: the dispatcher's strip/refuse list."""


def _trace_module_addresses(trace: Any) -> list[str]:
    """Return the module addresses recorded on ``trace`` (root excluded)."""

    try:
        keys = list(trace.modules.keys())
    except (AttributeError, TypeError):
        return []
    return [key for key in keys if isinstance(key, str) and key != "self"]


def canonical_module_site_key(trace: Any, name: object) -> str | None:
    """Map a module or stage name that is not a lookup key to the address it names.

    A module name resolves to that module's output at every selector door, and
    to the Module itself at ``trace[...]`` (DOCUMENTED-UNSTABLE rule). Exact
    module addresses are already lookup keys;
    this adds the two spellings wrappers teach their users: a name relative to
    a wrapped inner model (timm ``FeatureGetterNet`` reports ``"blocks.9"`` for
    the module at ``"model.blocks.9"``) and a dotted name whose module was
    registered with dots flattened to underscores (timm ``FeatureListNet``
    registers ``"stages.0"`` as ``"stages_0"``). A ``":<call>"`` suffix is kept.

    Parameters
    ----------
    trace
        Finished trace whose module addresses and lookup keys are searched.
    name
        Requested selector string.

    Returns
    -------
    str | None
        The canonical lookup key (``"<address>"`` or ``"<address>:<call>"``), or
        ``None`` when ``name`` is already a key or names no unique module.

    Raises
    ------
    AmbiguousOpLookupError
        When a relative name ends more than one module address; the message
        lists every candidate address.
    """

    if not isinstance(name, str) or not name or name.startswith("."):
        return None
    try:
        keys = trace._layer_dict_all_keys or {}
        ambiguous = trace._ambiguous_lookup_keys or {}
    except AttributeError:
        return None
    if name in keys or name in ambiguous:
        return None
    base, call_suffix = _split_call_suffix(name)
    addresses = _trace_module_addresses(trace)
    if not addresses or base in addresses:
        return None
    address = _unique_module_for_name(name, base, addresses)
    if address is None:
        return None
    key = f"{address}{call_suffix}"
    return key if key in keys or key in ambiguous else None


def _split_call_suffix(name: str) -> tuple[str, str]:
    """Split ``"address:3"`` into ``("address", ":3")``; other names keep an empty suffix."""

    base, separator, call_index = name.rpartition(":")
    if separator and call_index.isdigit():
        return base, f":{call_index}"
    return name, ""


def _unique_module_for_name(name: str, base: str, addresses: list[str]) -> str | None:
    """Return the one module address a relative or dot-flattened ``base`` names.

    Raises
    ------
    AmbiguousOpLookupError
        When ``base`` ends more than one module address.
    """

    candidates = [address for address in addresses if address.endswith(f".{base}")]
    flattened = base.replace(".", "_")
    if not candidates and flattened != base and flattened in addresses:
        candidates = [flattened]
    if len(candidates) > 1:
        listed = ", ".join(repr(address) for address in candidates[:10])
        more = f" (and {len(candidates) - 10} more)" if len(candidates) > 10 else ""
        raise AmbiguousOpLookupError(
            f"Module name {name!r} ends {len(candidates)} module addresses: {listed}{more}. "
            "Use the full module address."
        )
    return candidates[0] if candidates else None
