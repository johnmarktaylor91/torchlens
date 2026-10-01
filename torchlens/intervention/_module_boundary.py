"""Module-boundary live-hook application (the ``tl.module(...)`` door).

Split out of ``runtime.py`` (T98 size ratchet): this module owns the live
hook application at REAL module forward boundaries -- walking a module's
output structure, minting a boundary site proxy per tensor leaf, firing the
active hook plan plus any module-kind predicate ``intervene=`` rule against
it, and rebuilding the output with replacements -- together with the
tensor-attached evidence channels (fire results, replaced-parent labels)
the module-exit consumer in ``backends/torch/model_prep.py`` reads back.

Everything here is private; ``runtime.py`` re-exports the names its
historical importers reach for, so the import surface is unchanged.
"""

from __future__ import annotations

import warnings
import weakref
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

import torch

from .. import _state
from .._state import pause_logging
from ..backends.torch._tl import clear_tensor_label, get_tensor_label
from ..ir.intervention import FireResult
from .hooks import make_live_site_proxy


@dataclass(frozen=True)
class _BoundaryCall:
    """One module forward boundary as the live hook door sees it.

    Attributes
    ----------
    trace:
        The active trace (``None`` outside a capture).
    call_args:
        Original module positional inputs.
    call_kwargs:
        Original module keyword inputs.
    predicate:
        ``(predicate_options, selector)`` when a module-kind predicate
        ``intervene=`` rule is armed on the trace, else ``None``.
    """

    trace: Any
    call_args: tuple[Any, ...]
    call_kwargs: dict[str, Any]
    predicate: tuple[Any, Any] | None


def _resolve_boundary_predicate(trace: Any) -> tuple[Any, Any] | None:
    """Return ``(predicate_options, selector)`` for an armed module-kind rule.

    Only a ``BaseSelector`` that contains a ``module`` term can match a
    boundary site; anything else (no predicate options, an op-only selector,
    an opaque callable) yields ``None`` and the boundary door fires the hook
    plan alone.
    """

    if trace is None:
        return None
    predicate_options = getattr(trace, "_predicate_save_options", None)
    predicate_intervene = getattr(predicate_options, "intervene", None)
    predicate_selector = getattr(predicate_intervene, "selector", None)
    if predicate_selector is None:
        return None
    from ..ir.selector_eval import selector_contains_kind
    from .selectors import BaseSelector

    if not isinstance(predicate_selector, BaseSelector):
        return None
    if not selector_contains_kind(predicate_selector, "module"):
        return None
    return predicate_options, predicate_selector


def _make_boundary_site(
    out: torch.Tensor,
    container_path: tuple[Any, ...],
    *,
    module_address: str,
    module_call_index: int,
    module_type: str,
) -> Any:
    """Mint the live site proxy for one tensor leaf of a module output."""

    module_call = (module_address, module_call_index)
    site = make_live_site_proxy(
        _layer_label_raw=f"{module_address}:{module_call_index}",
        func_name=module_type,
        layer_type=module_type.lower(),
        tensor=out,
        func_call_id=0,
        container_path=container_path,
        fields={
            "raw_index": 0,
            "module": module_call,
            "modules": (module_call,),
            "output_of_module_calls": (module_call,),
            "_tl_module_boundary": True,
        },
    )
    setattr(site, "_tl_module_boundary", True)
    return site


def _fire_boundary_predicate_hooks(
    hooked: Any,
    *,
    boundary: _BoundaryCall,
    site: Any,
    container_path: tuple[Any, ...],
) -> tuple[Any, tuple[FireResult, ...]]:
    """Fire the armed module-kind predicate rule against one boundary site.

    Returns the (possibly replaced) value and the fire results; an empty
    tuple means the predicate declined this site.
    """

    from ..backends.torch.ops import _record_predicate_intervention_spec
    from ..capture.predicates import _evaluate_intervene_op
    from .hooks import normalize_hook_plan
    from .runtime import (
        _apply_live_hooks,
        _armed_injection_state,
        _current_injection_rule,
        active_intervention_context,
    )

    if boundary.predicate is None:
        return hooked, ()
    predicate_options, predicate_selector = boundary.predicate
    trace = boundary.trace
    decision = _evaluate_intervene_op(site, predicate_options)
    if decision is None:
        return hooked, ()
    _record_predicate_intervention_spec(trace, site, decision)
    hook_entries = normalize_hook_plan(
        decision.hook,
        default_site_target=predicate_selector,
        direction=decision.direction,
    )
    # F01 log_injections: anchor this boundary firing's injected ops to the
    # PERSISTED rule id (the public spec threads it through the decision),
    # exactly as the op door does; without it the recorder fell back to
    # ``adhoc:<helper>`` for every tl.module(...) rule (AUD-CODE 2.3b).
    with (
        _current_injection_rule(_armed_injection_state(trace), getattr(decision, "rule_id", None)),
        active_intervention_context(
            intervention_spec=getattr(trace, "_intervention_spec", None),
            hook_plan=hook_entries,
        ),
    ):
        hooked, fire_results = _apply_live_hooks(
            hooked,
            site=site,
            container_path=container_path,
            call_args=boundary.call_args,
            call_kwargs=boundary.call_kwargs,
        )
    if fire_results:
        trace._tl_intervene_selector_fire_count = int(
            getattr(trace, "_tl_intervene_selector_fire_count", 0)
        ) + len(fire_results)
    return hooked, tuple(fire_results)


def _attach_boundary_fire_evidence(
    out: torch.Tensor,
    hooked: Any,
    fire_results: tuple[FireResult, ...],
    trace: Any,
) -> None:
    """Attach fire results (and replaced-parent labels) to the hooked value."""

    if hooked is not out:
        parent_label = get_tensor_label(out)
        if parent_label is not None:
            _record_module_intervention_parent_labels(hooked, (parent_label,), trace)
        clear_tensor_label(hooked)
    _record_tensor_live_fire_results(hooked, fire_results)


def _apply_module_boundary_live_hooks(
    out_orig: Any,
    *,
    module_address: str,
    module_call_index: int,
    module_type: str,
    call_args: tuple[Any, ...],
    call_kwargs: dict[str, Any],
) -> Any:
    """Apply module-boundary live hooks to module forward outputs.

    Parameters
    ----------
    out_orig:
        Raw module forward output.
    module_address:
        TorchLens module address.
    module_call_index:
        One-based module call index.
    module_type:
        Module type name.
    call_args:
        Original module positional inputs.
    call_kwargs:
        Original module keyword inputs.

    Returns
    -------
    Any
        Module output with any tensor replacements applied.
    """

    from .runtime import _apply_live_hooks

    trace = _state._active_trace
    boundary = _BoundaryCall(
        trace=trace,
        call_args=call_args,
        call_kwargs=call_kwargs,
        predicate=_resolve_boundary_predicate(trace),
    )
    if not _state._active_hook_plan and boundary.predicate is None:
        return out_orig
    replacements: dict[tuple[Any, ...], torch.Tensor] = {}
    for out, container_path in _iter_tensor_outputs(out_orig):
        site = _make_boundary_site(
            out,
            container_path,
            module_address=module_address,
            module_call_index=module_call_index,
            module_type=module_type,
        )
        hooked, plan_fire_results = _apply_live_hooks(
            out,
            site=site,
            container_path=container_path,
            call_args=call_args,
            call_kwargs=call_kwargs,
        )
        hooked, predicate_fire_results = _fire_boundary_predicate_hooks(
            hooked,
            boundary=boundary,
            site=site,
            container_path=container_path,
        )
        fire_results = (*plan_fire_results, *predicate_fire_results)
        if fire_results:
            _attach_boundary_fire_evidence(out, hooked, fire_results, trace)
        if hooked is not out:
            replacements[container_path] = hooked
    if not replacements:
        return out_orig
    return _replace_tensor_outputs(out_orig, replacements)


_MODULE_INTERVENTION_PARENTS_ATTR = "_tl_module_intervention_parent_labels"
_MODULE_INTERVENTION_PARENTS_TABLE = "_tl_module_intervention_parents_by_id"


def _record_tensor_live_fire_results(
    tensor: torch.Tensor, fire_results: tuple[FireResult, ...]
) -> None:
    """Attach module-boundary fire results to a tensor, never dropping them silently.

    A replacement tensor that rejects dynamic attributes used to swallow the
    evidence (bare ``except: pass``), so the module exit reran with no
    intervention provenance and the fresh value was misclassified as an
    ``internal_source``. Delegate to the op-level setter, which falls back to
    the storage-owned side table and raises a typed ``CompatibilityError``
    only when NEITHER channel is writable.

    Parameters
    ----------
    tensor:
        Tensor that received a live module-boundary hook.
    fire_results:
        Fire results emitted by the live hook dispatcher.
    """

    from ..backends.torch._ops_interventions import _set_tensor_live_fire_results

    _set_tensor_live_fire_results(tensor, fire_results)


def _peek_tensor_live_fire_results(tensor: torch.Tensor) -> tuple[FireResult, ...]:
    """Return (without consuming) live fire results attached to ``tensor``.

    Checks the plain attribute first, then the storage-owned side table the
    robust setter falls back to for attr-rejecting replacement tensors. The
    module-exit consumer gates its replacement-vs-internal-source
    classification on this peek, so it must see both channels.

    Parameters
    ----------
    tensor:
        Tensor about to be classified at a module exit.
    """

    try:
        fire_results = tuple(getattr(tensor, "_tl_live_fire_results", ()) or ())
    except Exception:
        fire_results = ()
    if fire_results:
        return fire_results
    from ..backends.torch import _ops_interventions as intervention_state

    try:
        with pause_logging():
            storage = tensor.untyped_storage()
        records = getattr(storage, intervention_state._LIVE_FIRE_RESULTS_STORAGE_ATTR, None)
    except Exception:
        return ()
    if not isinstance(records, dict):
        return ()
    entry = records.get(id(tensor))
    if entry is not None and entry[0]() is tensor:
        return tuple(entry[1])
    return ()


def _record_module_intervention_parent_labels(
    tensor: torch.Tensor,
    parent_labels: tuple[str, ...],
    trace: Any,
) -> None:
    """Record the replaced-parent labels for a module-boundary replacement.

    Falls back to a trace-scoped identity-keyed table (weakly guarded against
    id reuse, dying with the capture) when the replacement tensor rejects
    dynamic attributes, so the boundary op minted at module exit keeps its
    dataflow parents instead of silently losing them.

    Parameters
    ----------
    tensor:
        Replacement tensor produced by a live module-boundary hook.
    parent_labels:
        Raw labels of the replaced module-output tensors.
    trace:
        Active trace owning the fallback table (``None`` tolerated; the loss
        is then disclosed with a warning rather than swallowed).
    """

    labels = tuple(parent_labels)
    try:
        setattr(tensor, _MODULE_INTERVENTION_PARENTS_ATTR, labels)
        return
    except Exception:
        pass
    if trace is None:
        warnings.warn(
            "TorchLens could not record intervention parent provenance for a "
            "module-boundary replacement tensor (dynamic attributes rejected and "
            "no active trace); the replacement op will carry no parents.",
            stacklevel=2,
        )
        return
    table = trace.__dict__.setdefault(_MODULE_INTERVENTION_PARENTS_TABLE, {})
    table[id(tensor)] = (weakref.ref(tensor), labels)


def _peek_module_intervention_parent_labels(tensor: torch.Tensor, trace: Any) -> tuple[str, ...]:
    """Return the recorded replaced-parent labels for ``tensor``, if any.

    Parameters
    ----------
    tensor:
        Module-output tensor being classified at a module exit.
    trace:
        Active trace whose fallback table is consulted when the tensor
        carries no attribute.
    """

    try:
        labels = tuple(getattr(tensor, _MODULE_INTERVENTION_PARENTS_ATTR, ()) or ())
    except Exception:
        labels = ()
    if labels:
        return labels
    table = getattr(trace, _MODULE_INTERVENTION_PARENTS_TABLE, None) if trace is not None else None
    if not isinstance(table, dict):
        return ()
    entry = table.get(id(tensor))
    if entry is not None and entry[0]() is tensor:
        return tuple(entry[1])
    return ()


def _iter_tensor_outputs(
    value: Any, path: tuple[Any, ...] = ()
) -> Iterator[tuple[torch.Tensor, tuple[Any, ...]]]:
    """Yield tensor leaves and their paths from a module output structure.

    Parameters
    ----------
    value:
        Module output value to traverse.
    path:
        Current container path prefix.

    Yields
    ------
    tuple[torch.Tensor, tuple[Any, ...]]
        Tensor leaf and stable path.
    """

    if isinstance(value, torch.Tensor):
        yield value, path
        return
    if isinstance(value, tuple):
        for index, item in enumerate(value):
            yield from _iter_tensor_outputs(item, (*path, index))
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            yield from _iter_tensor_outputs(item, (*path, index))
        return
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _iter_tensor_outputs(item, (*path, key))


def _replace_tensor_outputs(value: Any, replacements: dict[tuple[Any, ...], torch.Tensor]) -> Any:
    """Return ``value`` with tensor leaves replaced by path.

    Parameters
    ----------
    value:
        Original module output structure.
    replacements:
        Mapping from tensor leaf path to replacement tensor.

    Returns
    -------
    Any
        Output structure with replacements applied.
    """

    if () in replacements:
        return replacements[()]
    if isinstance(value, tuple):
        rebuilt_items = tuple(
            _replace_tensor_outputs_by_child(item, replacements, (index,))
            for index, item in enumerate(value)
        )
        if _is_namedtuple_instance(value):
            return tuple.__new__(type(value), rebuilt_items)
        return type(value)(rebuilt_items)
    if isinstance(value, list):
        return [
            _replace_tensor_outputs_by_child(item, replacements, (index,))
            for index, item in enumerate(value)
        ]
    if isinstance(value, dict):
        return {
            key: _replace_tensor_outputs_by_child(item, replacements, (key,))
            for key, item in value.items()
        }
    return value


def _replace_tensor_outputs_by_child(
    value: Any, replacements: dict[tuple[Any, ...], torch.Tensor], prefix: tuple[Any, ...]
) -> Any:
    """Return a child value with replacements beneath ``prefix`` applied.

    Parameters
    ----------
    value:
        Child value to rebuild.
    replacements:
        Full replacement mapping.
    prefix:
        Path prefix for ``value`` within its parent.

    Returns
    -------
    Any
        Child value with matching replacements applied.
    """

    child_replacements = {
        path[len(prefix) :]: replacement
        for path, replacement in replacements.items()
        if path[: len(prefix)] == prefix
    }
    if not child_replacements:
        return value
    return _replace_tensor_outputs(value, child_replacements)


def _is_namedtuple_instance(value: Any) -> bool:
    """Return whether ``value`` is a namedtuple instance.

    Parameters
    ----------
    value:
        Candidate container.

    Returns
    -------
    bool
        Whether ``value`` is a tuple with ``_fields`` metadata.
    """

    fields = getattr(type(value), "_fields", None)
    return isinstance(value, tuple) and isinstance(fields, tuple)
