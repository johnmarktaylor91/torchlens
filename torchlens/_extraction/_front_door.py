"""The extract front door: ``tl.pluck`` and ``tl.extract`` and their helpers.

The package root re-exports ``pluck`` and ``extract`` lazily; the extraction
engine and the dataset extractor import the private helpers from here, so no
layered module needs the package root for them.
"""

from __future__ import annotations

from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import TYPE_CHECKING, Any as _Any

import torch

from .._data_classes._lookup_kinds import _record_view
from .._data_classes.trace import Trace
from .._errors import AmbiguousOpLookupError
from .._extract_fast import extract_outputs, try_module_output_pluck
from .._extract_refusals import (
    CaptureOptions,
    TorchLensError,
    refuse_activation_not_saved,
    refuse_lookup_not_found,
)
from .._selection.selectors import FacetSelector
from .._trace_selector_helpers import canonical_module_site_key
from .._user_public_impls import trace as _capture_trace
from .._utils._display_errors import candidate_keys, lookup_message
from ..options import CaptureOptions as _LazyCaptureOptions

if TYPE_CHECKING:
    from torch import nn

__tl_layer__ = "L5"


def _suggest_layer_labels(trace: _Any, query: str) -> list[str]:
    """Return the did-you-mean layer labels for a failed lookup (empty for non-traces)."""

    if not isinstance(trace, Trace):
        return []
    return candidate_keys(list(trace._suggest_layer_labels(str(query))), trace._layer_dict_all_keys)


def _did_you_mean_message(name: str, suggestions: list[str]) -> str:
    """Build a short suggestion suffix for lookup failures.

    Parameters
    ----------
    name:
        Lookup string supplied by the user.
    suggestions:
        Candidate layer labels.

    Returns
    -------
    str
        Human-readable lookup error.
    """

    if suggestions:
        return lookup_message(
            f"Layer {name!r} not found.",
            suggestions,
            "pass an in-range integer, a layer label, or a module address.",
            query=name,
        )
    return f"Layer {name!r} not found."


def _out_from_log(trace: Trace, layer: str) -> torch.Tensor:
    """Return a saved out from a layer lookup.

    Parameters
    ----------
    trace:
        Log containing saved outs.
    layer:
        Layer label, module path, pass-qualified label, or unique substring.

    Returns
    -------
    torch.Tensor
        Saved layer out.

    Raises
    ------
    InvalidArgumentError
        ``op_lookup_not_found`` if the layer cannot be resolved;
        ``activation_not_saved`` if it resolved but kept no out.
    """

    try:
        layer_log = trace[canonical_module_site_key(trace, layer) or layer]
    except TorchLensError:
        # A typed refusal (ambiguity, pass qualifier, index) already names its
        # condition; a "not found" would hide it.
        raise
    except (KeyError, ValueError) as exc:
        suggestions = _suggest_layer_labels(trace, layer)
        refuse_lookup_not_found(_did_you_mean_message(layer, suggestions), layer, cause=exc)

    out = getattr(layer_log, "out", None)
    if out is None:
        refuse_activation_not_saved(layer)
    if not isinstance(out, torch.Tensor):
        raise TypeError(f"Layer {layer!r} out is not a torch.Tensor.")
    return out


def _normalize_extract_layers(layers: _Iterable[str] | _Mapping[str, str]) -> dict[str, str]:
    """Normalize list or mapping layer specs to ``output_key -> lookup``.

    Parameters
    ----------
    layers:
        List of layer lookups or mapping from user label to layer lookup.

    Returns
    -------
    dict[str, str]
        Normalized extraction plan.
    """

    if isinstance(layers, _Mapping):
        return {str(label): str(pattern) for label, pattern in layers.items()}
    return {str(layer): str(layer) for layer in layers}


def _matching_saved_layer_labels(trace: Trace, pattern: str) -> list[str]:
    """Return saved layer labels matching an extraction pattern.

    Parameters
    ----------
    trace:
        Log containing candidate layer labels.
    pattern:
        Exact label or substring pattern.

    Returns
    -------
    list[str]
        Matching no-pass layer labels in execution order.
    """

    if pattern in trace._layer_dict_all_keys:
        return [pattern]
    if pattern in trace._layer_logs:
        return [pattern]
    if canonical_module_site_key(trace, pattern) is not None:
        # A wrapper-relative or dot-flattened module name keys by the caller's spelling.
        return [pattern]
    lower_pattern = pattern.lower()
    matches = [
        label
        for label in trace.layer_labels
        if lower_pattern in label.lower() and label in trace.saved_ops
    ]
    if matches:
        return matches
    try:
        resolved = trace[pattern]
    except AmbiguousOpLookupError:
        raise
    except (KeyError, ValueError):
        return []
    label = getattr(resolved, "layer_label", pattern)
    return [str(label)]


def pluck(
    model: nn.Module, inputs: _Any, layer: str, stop_after: _Any | None = None
) -> torch.Tensor:
    """Return the saved out for one layer.

    Parameters
    ----------
    model:
        PyTorch model to run.
    inputs:
        Positional input argument or argument container for ``model.forward``.
    layer:
        Layer label, module path, pass-qualified label, or unique substring.
        Resolves strictly to one result; ambiguous lookups refuse typed.
    stop_after:
        Stop-early site or predicate, passed to the capture as
        ``trace(stop_after=...)``: the forward halts inclusively after the
        site, and an unsupported spelling refuses typed. ``None`` reads the
        ambient ``tl.experimental.stop_after`` site, if one is active.

    Returns
    -------
    torch.Tensor
        Saved out for the requested layer.

    Raises
    ------
    InvalidArgumentError
        ``op_lookup_not_found`` if ``layer`` does not resolve;
        ``activation_not_saved`` if it produced no saved tensor.

    Notes
    -----
    When ``layer`` is an exact module address, the model is in eval mode and
    the input is one tensor or text on the model's device (the conditions of
    :func:`extract`'s fast path), pluck runs one plain forward with a hook on
    that module instead of a capture, and returns the same tensor. A
    ``stop_after`` that is an exact module address ends that forward from a
    hook right after the module's first call, so the rest of the model never
    runs. Any other request runs the full capture.

    Examples
    --------
    >>> relu_out = pluck(model, x, "relu")
    >>> tuple(relu_out.shape)
    (1, 4, 8, 8)
    """

    fast = try_module_output_pluck(model, inputs, layer, stop_after)
    if fast is not None:
        return fast
    # an explicit site rides the door's stop_after=, where the trace
    # validates it; without one the trace reads the ambient site itself.
    trace = _capture_trace(model, inputs, save=[layer], stop_after=stop_after)
    return _out_from_log(trace, layer)


def _extract_facet_plan(layers: _Any) -> dict[str, _Any]:
    """Return ``output_key -> facet selector`` for the facet selectors in an extract request.

    A list entry keys by the selector's spelling; a mapping entry by the user label.
    """

    items = layers.items() if isinstance(layers, _Mapping) else ((None, v) for v in layers)
    return {
        repr(value) if label is None else str(label): value
        for label, value in items
        if isinstance(value, FacetSelector)
    }


def _extract_with_facets(
    model: nn.Module,
    x: _Any,
    layers: _Any,
    facet_plan: dict[str, _Any],
    *,
    detach_saved_activations: bool,
) -> tuple[Trace, dict[str, torch.Tensor], dict[str, _Any]]:
    """Extract a request holding facet selectors: each facet's value, as ``trace[selector]`` reads it.

    One facet alone captures with ``save=selector`` (its homes); a mixed or
    several-facet request captures with every op saved. Other entries resolve
    as in a plain extract, from the same trace.
    """

    capture = _LazyCaptureOptions(detach_saved_activations=detach_saved_activations)
    only = next(iter(facet_plan.values())) if len(facet_plan) == len(layers) == 1 else None
    if only is not None:
        trace = _capture_trace(model, x, capture=capture, save=only)
    else:
        trace = _capture_trace(model, x, capture=capture)
    outputs: dict[str, torch.Tensor] = {}
    views: dict[str, _Any] = {}
    items = layers.items() if isinstance(layers, _Mapping) else ((None, v) for v in layers)
    for label, value in items:
        key = (
            str(label)
            if label is not None
            else (repr(value) if value in facet_plan.values() else str(value))
        )
        if key in facet_plan:
            facet = trace[facet_plan[key]]
            outputs[key], views[key] = facet.value, facet
        else:
            outputs[key] = _out_from_log(trace, str(value))
            views[key] = _record_view(trace, str(value))
    return trace, outputs, views


def _extract_layers_with_trace(
    model: nn.Module,
    x: _Any,
    layers: _Iterable[str] | _Mapping[str, str],
    *,
    detach_saved_activations: bool = True,
) -> tuple[Trace, dict[str, torch.Tensor], dict[str, _Any]]:
    """Run one selective capture and resolve the requested layers.

    Parameters
    ----------
    model:
        PyTorch model to run.
    x:
        Positional input argument or argument container for ``model.forward``.
    layers:
        Either a list of layer lookups or a mapping of ``user_label -> layer_lookup``.
    detach_saved_activations:
        Whether the returned outs are detached from the autograd graph.

    Returns
    -------
    tuple[Trace, dict[str, torch.Tensor], dict[str, Any]]
        The capture trace, the saved outs keyed as :func:`extract` keys them,
        and the resolved ``Layer`` views under the same keys.

    Raises
    ------
    InvalidArgumentError
        ``op_lookup_not_found`` (no such layer) or ``activation_not_saved`` (no saved tensor).
    """

    facet_plan = _extract_facet_plan(layers)
    if facet_plan:
        return _extract_with_facets(
            model, x, layers, facet_plan, detach_saved_activations=detach_saved_activations
        )
    layer_plan = _normalize_extract_layers(layers)
    capture = CaptureOptions(detach_saved_activations=detach_saved_activations)
    trace = _capture_trace(model, x, capture=capture, save=list(layer_plan.values()))
    outputs: dict[str, torch.Tensor] = {}
    views: dict[str, _Any] = {}
    if isinstance(layers, _Mapping):
        for label, pattern in layer_plan.items():
            outputs[label] = _out_from_log(trace, pattern)
            views[label] = _record_view(trace, pattern)  # the plain door canonicalizes the name
        return trace, outputs, views

    for pattern in layer_plan.values():
        matches = _matching_saved_layer_labels(trace, pattern)
        if not matches:
            suggestions = _suggest_layer_labels(trace, pattern)
            refuse_lookup_not_found(_did_you_mean_message(pattern, suggestions), pattern)
        for match in matches:
            outputs[match] = _out_from_log(trace, match)
            views[match] = _record_view(trace, match)
    return trace, outputs, views


def extract(
    model: nn.Module,
    inputs: _Any,
    layers: _Iterable[str] | _Mapping[str, str],
    *,
    detach_saved_activations: bool = True,
) -> dict[str, torch.Tensor]:
    """Return saved outs for many layers.

    ``tl.extract`` is the narrow door: it takes only ``model``, ``inputs``,
    ``layers`` and ``detach_saved_activations``. Every other capture option
    (``save=`` predicates, ``stop_after=``, ``capture=``, ``intervene=``,
    ``episode=``, storage) belongs to ``tl.trace``; call it and read
    ``trace[label].out`` when you need one of them. Exact module addresses on an
    eval-mode model with one tensor input take one hooked forward, not a capture.

    Parameters
    ----------
    model:
        PyTorch model to run.
    inputs:
        Positional input argument or argument container for ``model.forward``.
    layers:
        Either a list of layer lookups or a mapping of ``user_label -> layer_lookup``.
    detach_saved_activations:
        DOCUMENTED-UNSTABLE. Defaults to ``True``: the outs are detached
        copies. ``False`` returns graph-connected outs (``grad_fn`` set
        wherever the model's forward carries autograd history), so a loss
        built from them backpropagates into the model's parameters.

    Returns
    -------
    dict[str, torch.Tensor]
        Mapping from user labels to outs for mapping inputs, or from
        resolved layer labels to outs for list inputs.

    Examples
    --------
    >>> outs = extract(model, x, {"stem": "0", "head": "2"})
    >>> sorted((name, tuple(out.shape)) for name, out in outs.items())
    [('head', (1, 2, 8, 8)), ('stem', (1, 4, 8, 8))]
    """

    full = _extract_layers_with_trace  # the full-capture path the fast path falls back to
    return extract_outputs(model, inputs, layers, full, detach=detach_saved_activations)
