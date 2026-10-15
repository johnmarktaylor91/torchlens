"""Native input-attribution methods for TorchLens."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, replace
from typing import Any, TypeAlias

import torch
from torch import Tensor
from torch.nn import Module

from torchlens.attribution._entry import _detect_entry_route, _EntryRoute, _tensor_leaves
from torchlens.attribution._result import (
    AttributionError,
    AttributionResult,
    AttributionValueTree,
    _summarize_attribution_tree,
)
from torchlens.attribution._site import (
    SiteTarget,
    _input_dependent_site,
    _resolve_site,
    _select_unit,
    _site_unsupported,
    _SiteActivation,
    _SiteRoute,
    _target_spelling,
)

# Kept private and deliberately distinct from the selector-serialization
# ``torchlens.intervention.types.TargetSpec``.
_AttributionTarget: TypeAlias = int | Callable[[Any], Tensor] | SiteTarget
_DOCS_POINTER = "See docs/reference/attribution.md, section Integer inputs."
InputKwargs: TypeAlias = dict[str, Any] | None

__all__ = [
    "AttributionError",
    "AttributionResult",
    "AttributionValueTree",
    "InputKwargs",
    "_summarize_attribution_tree",
]


@dataclass(frozen=True)
class _PreparedInputs:
    """Normalized model-call inputs and attributed leaves.

    Attributes
    ----------
    positional
        Positional argument structure used for ``model(*args)``.
    kwargs
        Keyword argument structure used for ``model(**kwargs)``.
    attributed_leaves
        Floating-point or complex tensor leaves selected for attribution.
    is_single_tensor
        Whether the public ``inputs`` argument was a bare tensor.
    entry
        Integer-input route of a layer method whose inputs hold no
        floating-point tensor; ``attributed_leaves`` are then the outputs of
        its entry calls (embeddings), substituted by hooks.
    site
        Site-target route; the model call then returns the site activation.
    """

    positional: tuple[Any, ...]
    kwargs: dict[str, Any]
    attributed_leaves: tuple[Tensor, ...]
    is_single_tensor: bool
    entry: _EntryRoute | None = None
    site: _SiteRoute | None = None


@dataclass(frozen=True)
class _Caller:
    """The public call normalizing its inputs.

    Attributes
    ----------
    method
        Public method name used in refusal messages.
    model
        The model, when the method supports site targets, integer inputs or a
        concrete refusal remedy.
    target
        The public target; a :class:`SiteTarget` is resolved against ``model``.
    integer_route
        Whether integer-only inputs are accepted (layer methods): the input
        path then starts at the modules that turn them into floats.
    """

    method: str = "input attribution"
    model: Module | None = None
    target: Any = None
    integer_route: bool = False


@contextmanager
def _temporarily_eval(model: Module) -> Any:
    """Run attribution with ``model`` in eval mode, then restore prior per-module modes.

    The model is placed in eval mode for the duration of the context so that
    attribution measures the deterministic inference computation. Every
    submodule's training flag is snapshotted before the switch and restored
    individually afterward. Restoring with ``module.train(flag)`` would recurse
    and collapse an intentionally-mixed configuration (for example a frozen
    ``BatchNorm`` deliberately left in eval while the rest of the model trains)
    to the root module's single flag, so this restores each module's own flag
    directly instead.

    Gradient tracking is force-enabled for the context body because every
    attribution method needs autograd through the forward pass. Without this,
    a caller that invokes attribution inside an ambient ``torch.no_grad()``
    would build a graph-free forward and attribution would falsely raise
    ``AttributionError`` about non-differentiability. ``torch.enable_grad`` is a
    no-op when grad is already enabled, so the ordinary path is unaffected.

    Parameters
    ----------
    model
        Model whose training mode should be temporarily changed.

    Yields
    ------
    None
        Context body executes while the model is in eval mode with grad enabled.
    """

    previous_modes = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        with torch.enable_grad():
            yield
    finally:
        for module, was_training in previous_modes:
            module.training = was_training


def _is_attributed_tensor(value: Any) -> bool:
    """Return whether ``value`` is a tensor leaf selected for attribution.

    Parameters
    ----------
    value
        Candidate pytree leaf.

    Returns
    -------
    bool
        ``True`` when ``value`` is a floating-point or complex tensor.
    """

    return isinstance(value, Tensor) and (value.is_floating_point() or value.is_complex())


def _normalize_model_inputs(
    inputs: Any,
    input_kwargs: InputKwargs,
    caller: _Caller | None = None,
) -> _PreparedInputs:
    """Normalize public attribution inputs into model-call args and kwargs.

    Parameters
    ----------
    inputs
        Bare tensor or tuple/list of positional arguments.
    input_kwargs
        Optional keyword arguments for the model call.
    caller
        The public method, model and target; ``None`` for internal callers
        that support neither site targets nor integer inputs.

    Returns
    -------
    _PreparedInputs
        Normalized call structure and attributed tensor leaves.

    Raises
    ------
    AttributionError
        If the input contract is invalid, or no floating-point leaf is present
        and the method attributes the input itself.
    """

    caller = caller if caller is not None else _Caller()
    model = caller.model
    if input_kwargs is None:
        kwargs: dict[str, Any] = {}
    elif isinstance(input_kwargs, dict):
        kwargs = dict(input_kwargs)
    else:
        raise AttributionError("input_kwargs must be a dict when provided")

    is_single_tensor = isinstance(inputs, Tensor)
    if is_single_tensor:
        positional = (inputs,)
    elif isinstance(inputs, tuple | list):
        positional = tuple(inputs)
    else:
        raise AttributionError(
            "inputs must be a tensor or a tuple/list of positional model arguments"
        )

    site: _SiteRoute | None = None
    if isinstance(caller.target, SiteTarget):
        if model is None:
            raise _site_unsupported(
                caller.method, caller.target, code="attribution_site_target_unsupported"
            )
        site = _resolve_site(model, caller.target)
    attributed_leaves = tuple(_iter_attributed_tensors((positional, kwargs)))
    entry: _EntryRoute | None = None
    if not attributed_leaves:
        if not (
            caller.integer_route and model is not None and _tensor_leaves((positional, kwargs))
        ):
            raise _input_not_float(caller, positional, kwargs, code="attribution_input_not_float")
        entry, _float_outputs = _detect_entry_route(model, positional, kwargs)
        attributed_leaves = entry.values
    return _PreparedInputs(
        positional=positional,
        kwargs=kwargs,
        attributed_leaves=attributed_leaves,
        is_single_tensor=is_single_tensor,
        entry=entry,
        site=site,
    )


def _describe_tensors(positional: tuple[Any, ...], kwargs: dict[str, Any]) -> str:
    """Return a short description of the tensors a call received."""

    tensors = _tensor_leaves((positional, kwargs))
    if not tensors:
        return "no tensor at all"
    first = tensors[0]
    more = f" and {len(tensors) - 1} more tensors" if len(tensors) > 1 else ""
    return f"a {first.dtype} tensor of shape {tuple(first.shape)}{more}"


def _input_not_float(
    caller: _Caller,
    positional: tuple[Any, ...],
    kwargs: dict[str, Any],
    *,
    code: str,
) -> AttributionError:
    """Build the coded refusal for an input method given no floating-point input.

    The remedy names a layer method on the module that turns the integers
    into floats (found by one forward pass) when the caller's model is known.
    """

    layer = None
    path_method = "layer_integrated_gradients"
    if caller.model is not None and _tensor_leaves((positional, kwargs)):
        route, float_outputs = _detect_entry_route(caller.model, positional, kwargs)
        if route.sites:
            layer = route.sites[0].layer
        elif float_outputs:
            layer, path_method = float_outputs[0], "layer_attribution"
    layer_spelling = repr(layer) if layer is not None else "'<embedding layer name>'"
    return AttributionError(
        f"{caller.method} attributes the model's input, and the inputs hold no floating-point "
        f"tensor (got {_describe_tensors(positional, kwargs)}). Integer values such as "
        "token ids have no gradient. Remedy: attribute a layer's output instead, which "
        f"accepts integer inputs, e.g. tl.attribution.{path_method}(model, inputs, "
        f"{_target_spelling(caller.target)}, layer={layer_spelling}). {_DOCS_POINTER}",
        code=code,
    )


def _iter_attributed_tensors(tree: Any) -> list[Tensor]:
    """Return attributed tensor leaves in deterministic traversal order.

    Parameters
    ----------
    tree
        Pytree made from tuples, lists, dicts, and leaves.

    Returns
    -------
    list[Tensor]
        Floating-point or complex tensor leaves found in ``tree``.
    """

    if _is_attributed_tensor(tree):
        return [tree]
    if isinstance(tree, tuple | list):
        leaves: list[Tensor] = []
        for item in tree:
            leaves.extend(_iter_attributed_tensors(item))
        return leaves
    if isinstance(tree, dict):
        leaves = []
        for value in tree.values():
            leaves.extend(_iter_attributed_tensors(value))
        return leaves
    return []


def _make_input_leaf(inputs: Tensor) -> Tensor:
    """Create a detached leaf tensor for attribution gradients.

    Parameters
    ----------
    inputs
        User input tensor.

    Returns
    -------
    Tensor
        Detached clone with gradient tracking enabled.
    """

    return inputs.detach().clone().requires_grad_(True)


def _interned_by_identity(
    originals: tuple[Tensor, ...],
    make_leaf: Callable[[int], Tensor],
) -> tuple[Tensor, ...]:
    """Create one new tensor per unique original object identity, shared across repeats.

    The user's forward pass sees exactly the object topology the user built: a
    tensor passed to several input slots is ONE object there, so identity
    checks (``a is b``) and autograd accumulation treat it as one value.
    Cloning each occurrence independently would silently run a DIFFERENT
    function than the one the user called, so every occurrence of the same
    original tensor must receive the same substituted leaf.

    Parameters
    ----------
    originals
        Original attributed leaves in traversal order, possibly containing
        repeated references to the same tensor object.
    make_leaf
        Constructor invoked with the slot index of the FIRST occurrence of each
        unique original; its result is reused at every repeated occurrence.

    Returns
    -------
    tuple[Tensor, ...]
        New leaves in traversal order, with object identity mirroring
        ``originals``.
    """

    leaf_by_original_id: dict[int, Tensor] = {}
    leaves: list[Tensor] = []
    for slot, original in enumerate(originals):
        key = id(original)
        if key not in leaf_by_original_id:
            leaf_by_original_id[key] = make_leaf(slot)
        leaves.append(leaf_by_original_id[key])
    return tuple(leaves)


def _make_input_leaves(inputs: _PreparedInputs) -> tuple[Tensor, ...]:
    """Create detached leaf tensors for all attributed input leaves.

    Repeated references to the same tensor object receive the SAME new leaf at
    every occurrence, preserving the identity topology of the user's call.

    Parameters
    ----------
    inputs
        Normalized attribution inputs.

    Returns
    -------
    tuple[Tensor, ...]
        Detached clones with gradient tracking enabled.
    """

    return _interned_by_identity(
        inputs.attributed_leaves,
        lambda slot: _make_input_leaf(inputs.attributed_leaves[slot]),
    )


def _interned_path_leaves(
    inputs: _PreparedInputs,
    baseline_tensors: tuple[Tensor, ...],
    deltas: tuple[Tensor, ...],
    alpha: float,
) -> tuple[Tensor, ...]:
    """Create differentiable path leaves for one baseline-to-input path point.

    Repeated references to the same original tensor share ONE path leaf so the
    interpolated forward preserves the identity topology of the user's call.
    ``_validate_baselines`` guarantees repeated references carry identical
    baselines, so constructing from the first occurrence loses nothing.

    Parameters
    ----------
    inputs
        Normalized attribution inputs.
    baseline_tensors
        Baseline leaves in attributed-leaf traversal order.
    deltas
        Input-minus-baseline tensors in the same order.
    alpha
        Interpolation coefficient on ``[0, 1]``.

    Returns
    -------
    tuple[Tensor, ...]
        Detached differentiable path leaves.
    """

    return _interned_by_identity(
        inputs.attributed_leaves,
        lambda slot: (
            (baseline_tensors[slot] + alpha * deltas[slot]).detach().clone().requires_grad_(True)
        ),
    )


def _replace_attributed_tensors(tree: Any, replacements: list[Tensor]) -> Any:
    """Replace attributed tensor leaves in ``tree`` from ``replacements``.

    Parameters
    ----------
    tree
        Pytree made from tuples, lists, dicts, and leaves.
    replacements
        Replacement tensors consumed in traversal order.

    Returns
    -------
    Any
        Tree with attributed leaves replaced.
    """

    if _is_attributed_tensor(tree):
        return replacements.pop(0)
    if isinstance(tree, tuple):
        return tuple(_replace_attributed_tensors(item, replacements) for item in tree)
    if isinstance(tree, list):
        return [_replace_attributed_tensors(item, replacements) for item in tree]
    if isinstance(tree, dict):
        return {
            key: _replace_attributed_tensors(value, replacements) for key, value in tree.items()
        }
    return tree


def _substitute_inputs(
    inputs: _PreparedInputs,
    attributed_replacements: tuple[Tensor, ...],
) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Substitute attributed leaves into normalized model-call inputs.

    Parameters
    ----------
    inputs
        Normalized attribution inputs.
    attributed_replacements
        Replacement leaves in the same order as ``inputs.attributed_leaves``.

    Returns
    -------
    tuple[tuple[Any, ...], dict[str, Any]]
        Positional and keyword arguments ready for ``model(*args, **kwargs)``.
    """

    replacements = list(attributed_replacements)
    positional = _replace_attributed_tensors(inputs.positional, replacements)
    kwargs = _replace_attributed_tensors(inputs.kwargs, replacements)
    if replacements:
        raise AttributionError("internal error: not all attributed replacements were consumed")
    if not isinstance(positional, tuple):
        raise AttributionError("internal error: positional inputs did not remain a tuple")
    if not isinstance(kwargs, dict):
        raise AttributionError("internal error: keyword inputs did not remain a dict")
    return positional, kwargs


def _tile_unattributed_leaves(tree: Any, n_rows: int) -> Any:
    """Tile non-attributed tensor leaves along the batch axis for stacked calls.

    Step batching stacks the ATTRIBUTED leaves on dim 0 (the documented batch
    axis); every fixed tensor input that carries the batch axis (attention
    masks, token type ids, position ids) must be replicated to match, or the
    stacked forward sees inconsistent batch sizes. Attributed leaves are left
    untouched (they are replaced by the stacked leaves downstream);
    zero-dimensional tensors and non-tensor leaves pass through.

    Parameters
    ----------
    tree
        Pytree of model-call inputs.
    n_rows
        Number of stacked path points.

    Returns
    -------
    Any
        Tree with unattributed batch-carrying tensors tiled ``n_rows`` times.
    """

    if isinstance(tree, Tensor):
        if _is_attributed_tensor(tree) or tree.ndim == 0:
            return tree
        return tree.repeat(n_rows, *([1] * (tree.ndim - 1)))
    if isinstance(tree, tuple):
        return tuple(_tile_unattributed_leaves(item, n_rows) for item in tree)
    if isinstance(tree, list):
        return [_tile_unattributed_leaves(item, n_rows) for item in tree]
    if isinstance(tree, dict):
        return {key: _tile_unattributed_leaves(value, n_rows) for key, value in tree.items()}
    return tree


def _call_model(
    model: Module,
    inputs: _PreparedInputs,
    attributed_replacements: tuple[Tensor, ...],
    n_rows: int = 1,
) -> Any:
    """Call ``model`` with attributed leaves substituted into their original positions.

    With an integer-input route the replacements stand in for the entry
    calls' outputs (substituted by hooks) and the integer inputs pass through.
    With a site route the call returns the site activation, wrapped, in place
    of the model output.

    Parameters
    ----------
    model
        PyTorch module to evaluate.
    inputs
        Normalized attribution inputs.
    attributed_replacements
        Replacement leaves in the same order as ``inputs.attributed_leaves``.
    n_rows
        Number of stacked path points; when above 1, fixed batch-carrying
        tensor inputs are tiled to match the stacked batch axis.

    Returns
    -------
    Any
        Model output, or the ``_SiteActivation`` of a site target.
    """

    if inputs.entry is None:
        positional, kwargs = _substitute_inputs(inputs, attributed_replacements)
    else:
        positional, kwargs = inputs.positional, inputs.kwargs
    if n_rows > 1:
        positional = _tile_unattributed_leaves(positional, n_rows)
        kwargs = _tile_unattributed_leaves(kwargs, n_rows)
    with ExitStack() as stack:
        if inputs.entry is not None:
            stack.enter_context(inputs.entry.substituted(attributed_replacements))
        firings = stack.enter_context(inputs.site.capture()) if inputs.site is not None else None
        output = model(*positional, **kwargs)
    if inputs.site is None or firings is None:
        return output
    unsited = replace(inputs, site=None)
    return inputs.site.activation(
        firings,
        attributed_replacements,
        lambda: _input_dependent_site(
            model,
            lambda: _call_model(model, unsited, attributed_replacements, n_rows),
            attributed_replacements,
        ),
    )


def _value_tree_from_leaves(
    inputs: _PreparedInputs,
    values: tuple[Tensor, ...],
) -> AttributionValueTree:
    """Build the public attribution value structure from per-leaf tensors.

    Single attributed-leaf calls return a bare tensor for v1 compatibility.
    Multi-leaf positional-only calls return a positional structure. Multi-leaf
    calls with keyword inputs return ``{"inputs": ..., "input_kwargs": ...}``.

    Parameters
    ----------
    inputs
        Normalized attribution inputs.
    values
        Attribution tensors in attributed-leaf traversal order.

    Returns
    -------
    AttributionValueTree
        Public ``AttributionResult.values`` structure.
    """

    if len(values) == 1:
        return values[0]

    replacements = list(values)
    positional_values = _replace_unattributed_with_none(inputs.positional, replacements)
    kwargs_values = _replace_unattributed_with_none(inputs.kwargs, replacements)
    if replacements:
        raise AttributionError("internal error: not all attribution values were consumed")
    if inputs.kwargs:
        return {"inputs": positional_values, "input_kwargs": kwargs_values}
    if inputs.is_single_tensor:
        return positional_values[0]
    return positional_values


def _replace_unattributed_with_none(tree: Any, replacements: list[Tensor]) -> Any:
    """Replace attributed leaves with values and all other leaves with ``None``.

    Parameters
    ----------
    tree
        Pytree made from tuples, lists, dicts, and leaves.
    replacements
        Replacement tensors consumed in traversal order.

    Returns
    -------
    Any
        Value tree mirroring containers in ``tree``.
    """

    if _is_attributed_tensor(tree):
        return replacements.pop(0)
    if isinstance(tree, tuple):
        return tuple(_replace_unattributed_with_none(item, replacements) for item in tree)
    if isinstance(tree, list):
        return [_replace_unattributed_with_none(item, replacements) for item in tree]
    if isinstance(tree, dict):
        return {
            key: _replace_unattributed_with_none(value, replacements) for key, value in tree.items()
        }
    return None


def _target_repr(target: _AttributionTarget) -> str:
    """Return a compact target representation for result metadata.

    Parameters
    ----------
    target
        Target scalarization specification.

    Returns
    -------
    str
        Human-readable target representation.
    """

    if isinstance(target, int):
        return f"index={target}"
    name = getattr(target, "__name__", None)
    if isinstance(name, str):
        return name
    return repr(target)


def _scalarize_output(output: Any, target: _AttributionTarget) -> Tensor:
    """Convert a model output to a scalar tensor using ``target``.

    Integer targets select ``output[..., target]`` and sum all selected values to
    produce one scalar. Callable targets must return a scalar tensor and are the
    required form for ambiguous non-classification outputs.

    Parameters
    ----------
    output
        Model output.
    target
        Integer class index, callable scalarizer, or :class:`SiteTarget`
        (which requires ``output`` to be the site activation).

    Returns
    -------
    Tensor
        Scalar tensor suitable for ``torch.autograd.grad``.

    Raises
    ------
    AttributionError
        If scalarization is unsupported or ambiguous.
    """

    if isinstance(target, SiteTarget):
        if not isinstance(output, _SiteActivation):
            raise _site_unsupported(
                "this method", target, code="attribution_site_target_unsupported"
            )
        return _select_unit(output.tensor, target)
    if callable(target):
        scalar = target(output)
        if not isinstance(scalar, Tensor):
            raise AttributionError("callable target must return a scalar tensor")
        if scalar.numel() != 1:
            raise AttributionError("callable target must return a scalar tensor")
        return scalar.reshape(())

    if not isinstance(target, int):
        raise AttributionError(
            "target must be an int class index or a callable output -> scalar tensor"
        )
    if not isinstance(output, Tensor):
        raise AttributionError("int targets require tensor model outputs; use a callable target")
    if output.ndim == 0:
        raise AttributionError("int target is ambiguous for scalar outputs; use a callable target")

    try:
        selected = output[..., target]
    except IndexError as exc:
        raise AttributionError(f"target index {target} is out of bounds for output shape") from exc
    return selected.sum()


def _gradient_for_inputs(
    model: Module,
    inputs: _PreparedInputs,
    input_leaves: tuple[Tensor, ...],
    target: _AttributionTarget,
) -> tuple[tuple[Tensor, ...], Tensor]:
    """Compute gradients of a scalarized model output with respect to input leaves.

    Parameters
    ----------
    model
        Model to evaluate.
    inputs
        Normalized attribution inputs.
    input_leaves
        Differentiable leaves substituted into ``inputs``.
    target
        Integer class index or callable scalarizer.

    Returns
    -------
    tuple[tuple[Tensor, ...], Tensor]
        Gradient tensors and scalarized model output.

    Raises
    ------
    AttributionError
        If the scalar target is not differentiable with respect to the inputs.
    """

    output = _call_model(model, inputs, input_leaves)
    scalar = _scalarize_output(output, target)
    unique_index_by_id: dict[int, int] = {}
    unique_leaves: list[Tensor] = []
    for leaf in input_leaves:
        if id(leaf) not in unique_index_by_id:
            unique_index_by_id[id(leaf)] = len(unique_leaves)
            unique_leaves.append(leaf)
    try:
        raw_gradients = torch.autograd.grad(
            scalar,
            unique_leaves,
            allow_unused=True,
        )
    except RuntimeError as exc:
        raise AttributionError(
            "target scalar is not differentiable with respect to the attributed inputs"
        ) from exc
    # A leaf shared across several input slots accumulates ONE gradient over
    # every use; each public slot reports that full gradient rather than an
    # arbitrary per-occurrence split.
    gradients = tuple(
        torch.zeros_like(input_leaf)
        if raw_gradients[unique_index_by_id[id(input_leaf)]] is None
        else raw_gradients[unique_index_by_id[id(input_leaf)]]
        for input_leaf in input_leaves
    )
    return gradients, scalar.detach()


def _validate_positive_int(name: str, value: int) -> None:
    """Validate that a method count parameter is positive.

    Parameters
    ----------
    name
        Parameter name for error reporting.
    value
        Candidate integer value.

    Raises
    ------
    AttributionError
        If ``value`` is not a positive integer.
    """

    if not isinstance(value, int) or value <= 0:
        raise AttributionError(f"{name} must be a positive integer")


def _validate_baseline_tensor(input_tensor: Tensor, baseline: Any) -> Tensor:
    """Validate one baseline tensor against one attributed input tensor.

    The default zero baseline is conventional but not neutral for every problem;
    attribution users should choose a baseline that matches their domain.

    Parameters
    ----------
    input_tensor
        Attributed input tensor.
    baseline
        Candidate baseline tensor.

    Returns
    -------
    Tensor
        Detached baseline tensor matching input shape, dtype, and device.

    Raises
    ------
    AttributionError
        If the baseline does not match ``input_tensor``.
    """

    if not isinstance(baseline, Tensor):
        raise AttributionError("baseline must mirror attributed input leaves")
    if baseline.shape != input_tensor.shape:
        raise AttributionError("baseline must match input shape")
    if baseline.dtype != input_tensor.dtype:
        raise AttributionError("baseline must match input dtype")
    if baseline.device != input_tensor.device:
        raise AttributionError("baseline must match input device")
    return baseline.detach().clone()


def _validate_baseline_tree(input_tree: Any, baseline_tree: Any) -> list[Tensor]:
    """Validate a baseline pytree against an input pytree.

    Parameters
    ----------
    input_tree
        Input tree containing attributed leaves.
    baseline_tree
        Candidate baseline tree.

    Returns
    -------
    list[Tensor]
        Baseline tensors in attributed-leaf traversal order.

    Raises
    ------
    AttributionError
        If containers or attributed leaves do not mirror the input tree.
    """

    if _is_attributed_tensor(input_tree):
        return [_validate_baseline_tensor(input_tree, baseline_tree)]
    if isinstance(input_tree, tuple):
        if not isinstance(baseline_tree, tuple) or len(baseline_tree) != len(input_tree):
            raise AttributionError("baseline must mirror attributed input structure")
        baselines: list[Tensor] = []
        for input_item, baseline_item in zip(input_tree, baseline_tree, strict=True):
            baselines.extend(_validate_baseline_tree(input_item, baseline_item))
        return baselines
    if isinstance(input_tree, list):
        if not isinstance(baseline_tree, list) or len(baseline_tree) != len(input_tree):
            raise AttributionError("baseline must mirror attributed input structure")
        baselines = []
        for input_item, baseline_item in zip(input_tree, baseline_tree, strict=True):
            baselines.extend(_validate_baseline_tree(input_item, baseline_item))
        return baselines
    if isinstance(input_tree, dict):
        if not isinstance(baseline_tree, dict) or baseline_tree.keys() != input_tree.keys():
            raise AttributionError("baseline must mirror attributed input structure")
        baselines = []
        for key, input_value in input_tree.items():
            baselines.extend(_validate_baseline_tree(input_value, baseline_tree[key]))
        return baselines
    return []


def _validate_repeated_reference_baselines(
    inputs: _PreparedInputs,
    baseline_leaves: tuple[Tensor, ...],
) -> None:
    """Require identical baselines at every occurrence of a repeated-reference input.

    A tensor object passed to several input slots is ONE value along the whole
    baseline-to-input path; two different baselines for it would demand the
    shared leaf hold two values at once. Rejecting the contradiction keeps the
    interpolated forwards running the user's actual function.

    Parameters
    ----------
    inputs
        Normalized attribution inputs.
    baseline_leaves
        Baseline tensors in attributed-leaf traversal order.

    Raises
    ------
    AttributionError
        If two occurrences of the same input tensor carry different baselines.
    """

    baseline_by_original_id: dict[int, Tensor] = {}
    for original, baseline_leaf in zip(inputs.attributed_leaves, baseline_leaves, strict=True):
        key = id(original)
        seen = baseline_by_original_id.get(key)
        if seen is None:
            baseline_by_original_id[key] = baseline_leaf
        elif not torch.equal(seen, baseline_leaf):
            raise AttributionError(
                "baseline values for repeated references to the same input tensor must match"
            )


def _validate_baselines(inputs: _PreparedInputs, baseline: Any | None) -> tuple[Tensor, ...]:
    """Validate or create Integrated Gradients baselines for attributed leaves.

    The default zero baseline is conventional but not neutral for every problem;
    attribution users should choose baselines that match their domain.

    Parameters
    ----------
    inputs
        Normalized attribution inputs.
    baseline
        Optional baseline tree. A bare tensor remains valid when there is exactly
        one attributed leaf.

    Returns
    -------
    tuple[Tensor, ...]
        Baseline tensors matching attributed leaves.

    Raises
    ------
    AttributionError
        If the baseline structure or any tensor does not match.
    """

    if baseline is None:
        return tuple(torch.zeros_like(leaf) for leaf in inputs.attributed_leaves)
    if len(inputs.attributed_leaves) == 1 and isinstance(baseline, Tensor):
        return (_validate_baseline_tensor(inputs.attributed_leaves[0], baseline),)
    if inputs.kwargs:
        if not isinstance(baseline, dict):
            raise AttributionError(
                "baseline must be a dict with 'inputs' and 'input_kwargs' for kwarg inputs"
            )
        if set(baseline) != {"inputs", "input_kwargs"}:
            raise AttributionError(
                "baseline must be a dict with 'inputs' and 'input_kwargs' for kwarg inputs"
            )
        baseline_leaves = _validate_baseline_tree(
            (inputs.positional, inputs.kwargs),
            (baseline["inputs"], baseline["input_kwargs"]),
        )
    else:
        positional_baseline = tuple(baseline) if isinstance(baseline, list) else baseline
        baseline_leaves = _validate_baseline_tree(inputs.positional, positional_baseline)

    if len(baseline_leaves) != len(inputs.attributed_leaves):
        raise AttributionError("baseline must mirror attributed input leaves")
    validated = tuple(baseline_leaves)
    _validate_repeated_reference_baselines(inputs, validated)
    return validated


def saliency(
    model: Module,
    inputs: Any,
    input_kwargs: InputKwargs = None,
    *,
    target: _AttributionTarget,
) -> AttributionResult:
    """Compute absolute input gradients for a scalar target.

    Saliency is UNSIGNED: every value is ``|d target / d input|``, so it says
    how strongly the target responds to a feature, never in which direction.
    There is no ``signed=`` switch; for a signed reading use
    :func:`input_x_grad`, :func:`integrated_gradients` or
    ``guided_backprop(..., absolute=False)``. A raw gradient is a local
    reading: it is noisy, and it reads near zero for a feature whose effect
    has saturated (a ReLU in its flat region, a saturated sigmoid) even when
    removing that feature would change the output. Average over input noise
    with :func:`smoothgrad`, or integrate along a path with
    :func:`integrated_gradients`, before reading a map as evidence.

    Parameters
    ----------
    model
        PyTorch module to attribute.
    inputs
        Bare tensor for v1 behavior, or tuple/list of positional model arguments.
        Floating-point or complex tensor leaves are attributed.
    input_kwargs
        Optional keyword arguments for ``model``. Floating-point or complex
        tensor leaves are attributed.
    target
        Integer class index selecting ``output[..., target]`` and summing the
        selected values, a callable ``output -> scalar tensor``, or a
        :class:`SiteTarget` naming a hidden layer's output (and unit) to
        attribute instead of the model output.

    Returns
    -------
    AttributionResult
        Bare tensor for one attributed leaf, otherwise a mirrored value tree.
    """

    prepared_inputs = _normalize_model_inputs(
        inputs, input_kwargs, _Caller("saliency", model, target)
    )
    with _temporarily_eval(model):
        input_leaves = _make_input_leaves(prepared_inputs)
        gradients, _scalar = _gradient_for_inputs(model, prepared_inputs, input_leaves, target)
    values = tuple(gradient.detach().abs() for gradient in gradients)
    return AttributionResult(
        method="saliency",
        values=_value_tree_from_leaves(prepared_inputs, values),
        target_repr=_target_repr(target),
        extra={},
    )


def input_x_grad(
    model: Module,
    inputs: Any,
    input_kwargs: InputKwargs = None,
    *,
    target: _AttributionTarget,
) -> AttributionResult:
    """Compute gradient times input for a scalar target.

    The result is SIGNED: under the local linear approximation, a positive
    value means the feature pushes the target up. Its reference point is the
    all-zeros input, so it carries an implicit zeros baseline; when zero is
    not a neutral input (normalized images, token embeddings), use
    :func:`integrated_gradients` with an explicit ``baseline=``, which also
    reports how far the attributions fall short of the output change.

    Parameters
    ----------
    model
        PyTorch module to attribute.
    inputs
        Bare tensor for v1 behavior, or tuple/list of positional model arguments.
        Floating-point or complex tensor leaves are attributed.
    input_kwargs
        Optional keyword arguments for ``model``. Floating-point or complex
        tensor leaves are attributed.
    target
        Integer class index selecting ``output[..., target]`` and summing the
        selected values, a callable ``output -> scalar tensor``, or a
        :class:`SiteTarget` naming a hidden layer's output (and unit) to
        attribute instead of the model output.

    Returns
    -------
    AttributionResult
        Bare tensor for one attributed leaf, otherwise a mirrored value tree.
    """

    prepared_inputs = _normalize_model_inputs(
        inputs, input_kwargs, _Caller("input_x_grad", model, target)
    )
    with _temporarily_eval(model):
        input_leaves = _make_input_leaves(prepared_inputs)
        gradients, _scalar = _gradient_for_inputs(model, prepared_inputs, input_leaves, target)
    values = tuple(
        (gradient * input_leaf).detach()
        for gradient, input_leaf in zip(gradients, input_leaves, strict=True)
    )
    return AttributionResult(
        method="input_x_grad",
        values=_value_tree_from_leaves(prepared_inputs, values),
        target_repr=_target_repr(target),
        extra={},
    )


def _completeness_extra(attribution_sum: Tensor, target_delta: Tensor) -> dict[str, Any]:
    """Build the kit-wide completeness disclosure fields (attrib memo D27).

    Every result carrying a completeness residual also carries the absolute
    target delta: a relative certificate over a near-zero output change
    certifies nothing, so the denominator is always disclosed beside the
    ratio.

    Parameters
    ----------
    attribution_sum
        Detached scalar sum of all attribution values.
    target_delta
        Detached scalar ``target(input) - target(baseline)``.

    Returns
    -------
    dict[str, Any]
        ``attribution_sum``, ``target_delta``, ``completeness_residual``
        tensors plus the float ``residual_rel`` and ``target_delta_abs``.
    """

    completeness_residual = attribution_sum - target_delta
    target_delta_abs = float(target_delta.abs().item())
    residual_abs = float(completeness_residual.abs().item())
    if target_delta_abs > 0.0:
        residual_rel = residual_abs / target_delta_abs
    else:
        residual_rel = 0.0 if residual_abs == 0.0 else float("inf")
    return {
        "attribution_sum": attribution_sum.detach(),
        "target_delta": target_delta.detach(),
        "completeness_residual": completeness_residual.detach(),
        "residual_rel": residual_rel,
        "target_delta_abs": target_delta_abs,
    }


def _stacked_path_leaves(
    inputs: _PreparedInputs,
    baseline_tensors: tuple[Tensor, ...],
    deltas: tuple[Tensor, ...],
    alphas: list[float],
) -> tuple[Tensor, ...]:
    """Create one differentiable leaf per slot stacking several path points.

    The path points for every alpha in ``alphas`` are concatenated along the
    ordinary batch axis (dim 0), preserving repeated-reference identity: two
    slots holding the same original tensor share ONE stacked leaf.

    Parameters
    ----------
    inputs
        Normalized attribution inputs.
    baseline_tensors
        Baseline leaves in attributed-leaf traversal order.
    deltas
        Input-minus-baseline tensors in the same order.
    alphas
        Interpolation coefficients stacked into this chunk, in step order.

    Returns
    -------
    tuple[Tensor, ...]
        Detached differentiable stacked leaves, identity-interned.
    """

    def _make(slot: int) -> Tensor:
        """Build one stacked path leaf for the slot's unique original."""

        stacked = torch.cat(
            [baseline_tensors[slot] + alpha * deltas[slot] for alpha in alphas],
            dim=0,
        )
        return stacked.detach().clone().requires_grad_(True)

    return _interned_by_identity(inputs.attributed_leaves, _make)


def _stacked_gradients(
    model: Module,
    inputs: _PreparedInputs,
    stacked_leaves: tuple[Tensor, ...],
    target: _AttributionTarget,
    n_rows: int,
) -> list[tuple[Tensor, ...]]:
    """Compute per-path-point gradients from one stacked forward/backward.

    Parameters
    ----------
    model
        Model to evaluate.
    inputs
        Normalized attribution inputs.
    stacked_leaves
        Identity-interned stacked leaves (dim 0 carries ``n_rows`` path points).
    target
        Integer class index or callable scalarizer.
    n_rows
        Number of stacked path points.

    Returns
    -------
    list[tuple[Tensor, ...]]
        Per-path-point per-leaf gradients, in stacking order.

    Raises
    ------
    AttributionError
        If the scalar target is not differentiable with respect to the inputs,
        or a callable target's output tree does not split.
    """

    from torchlens.attribution._steps import _scalarize_stacked_output

    output = _call_model(model, inputs, stacked_leaves, n_rows=n_rows)
    scalar = _scalarize_stacked_output(output, target, n_rows, _scalarize_output)
    unique_index_by_id: dict[int, int] = {}
    unique_leaves: list[Tensor] = []
    for leaf in stacked_leaves:
        if id(leaf) not in unique_index_by_id:
            unique_index_by_id[id(leaf)] = len(unique_leaves)
            unique_leaves.append(leaf)
    try:
        raw_gradients = torch.autograd.grad(scalar, unique_leaves, allow_unused=True)
    except RuntimeError as exc:
        raise AttributionError(
            "target scalar is not differentiable with respect to the attributed "
            "inputs. Remedy: choose a target built from the model output.",
            code="attribution_target_not_differentiable",
        ) from exc
    stacked_per_slot = tuple(
        torch.zeros_like(stacked_leaf)
        if raw_gradients[unique_index_by_id[id(stacked_leaf)]] is None
        else raw_gradients[unique_index_by_id[id(stacked_leaf)]]
        for stacked_leaf in stacked_leaves
    )
    per_slot_rows = [gradient.detach().chunk(n_rows, dim=0) for gradient in stacked_per_slot]
    return [tuple(rows[row] for rows in per_slot_rows) for row in range(n_rows)]


def integrated_gradients(
    model: Module,
    inputs: Any,
    input_kwargs: InputKwargs = None,
    *,
    target: _AttributionTarget,
    n_steps: int = 50,
    baseline: Any | None = None,
    step_batch_size: int | None = None,
    step_audit: str | None = None,
    step_audit_seed: int | None = None,
) -> AttributionResult:
    """Compute Integrated Gradients along a straight baseline-to-input path.

    The default baseline is ``torch.zeros_like`` for each attributed leaf.
    Baseline choice changes the interpretation of the result, so callers should
    pass domain-meaningful baselines when zeros are not appropriate. The
    baseline actually used is returned in ``extra["baseline"]``; report it next
    to any figure made from the values, because the same input attributed
    against a different baseline answers a different question.

    Quadrature: the path integral is approximated by the midpoint Riemann rule,
    sampling the path at ``(k + 0.5) / n_steps`` for ``k`` in
    ``range(n_steps)``. Captum's default is Gauss-Legendre, so the two
    libraries give close but not identical values at the same ``n_steps``;
    compare them through the completeness fields, not element by element.
    ``extra["completeness_residual"]`` is ``attribution_sum - target_delta``
    and ``extra["residual_rel"]`` is its size relative to
    ``extra["target_delta_abs"]``: raise ``n_steps`` until ``residual_rel`` is
    small. A small relative residual over a near-zero ``target_delta_abs``
    certifies nothing, which is why the denominator is always reported.

    Parameters
    ----------
    model
        PyTorch module to attribute.
    inputs
        Bare tensor for v1 behavior, or tuple/list of positional model arguments.
        Floating-point or complex tensor leaves are attributed.
    input_kwargs
        Optional keyword arguments for ``model``. Floating-point or complex
        tensor leaves are attributed.
    target
        Integer class index selecting ``output[..., target]`` and summing the
        selected values, a callable ``output -> scalar tensor``, or a
        :class:`SiteTarget` naming a hidden layer's output (and unit) to
        attribute instead of the model output.
    n_steps
        Number of midpoint Riemann samples along the straight path.
    baseline
        Optional baseline tree matching attributed input leaves. A bare tensor is
        accepted when there is exactly one attributed leaf.
    step_batch_size
        Optional number of path points stacked on the ordinary batch axis per
        forward/backward. ``None`` (the default) and ``1`` run sequentially --
        batching is strictly opt-in and is a THROUGHPUT feature, not a memory
        feature. Batched runs are guarded by the randomized audit.
    step_audit
        Audit-ladder rung when batching is on: ``"per_call"`` (default; one
        randomized ``(chunk, row)`` recomputed sequentially), ``"per_chunk"``
        (an independent random row per chunk), or ``"off"`` (explicit expert
        choice, disclosed). The audit is a sampled test, never a proof.
    step_audit_seed
        Optional deterministic seed for the audit's own (chunk, row) draw;
        ``None`` draws a fresh disclosed seed. The seed always rides
        ``extra["step_audit"]["seed"]``.

    Returns
    -------
    AttributionResult
        Bare tensor for one attributed leaf, otherwise a mirrored value tree.
        ``extra`` carries the completeness fields, ``|target_delta|``, the
        logical path-evaluation count, the physical forward-call count, and
        the audit disclosure.
    """

    from torchlens.attribution._steps import (
        _chunk_steps,
        _midpoint_alphas,
        _StepAuditor,
        _validate_step_audit,
        _validate_step_batch_size,
    )

    _validate_positive_int("n_steps", n_steps)
    chunk_size = _validate_step_batch_size(step_batch_size)
    audit_mode = _validate_step_audit(step_audit, chunk_size)
    prepared_inputs = _normalize_model_inputs(
        inputs, input_kwargs, _Caller("integrated_gradients", model, target)
    )
    baseline_tensors = _validate_baselines(prepared_inputs, baseline)
    deltas = tuple(
        input_leaf.detach() - baseline_tensor
        for input_leaf, baseline_tensor in zip(
            prepared_inputs.attributed_leaves, baseline_tensors, strict=True
        )
    )
    alphas = _midpoint_alphas(n_steps)
    chunks = _chunk_steps(alphas, chunk_size)
    auditor = _StepAuditor(
        audit_mode, len(chunks), [len(chunk) for chunk in chunks], seed=step_audit_seed
    )
    gradients_by_step: list[tuple[Tensor, ...]] = []
    physical_calls = 2  # The two endpoint forwards below.

    with _temporarily_eval(model):
        baseline_leaves = _interned_path_leaves(prepared_inputs, baseline_tensors, deltas, 0.0)
        input_leaves = _interned_path_leaves(prepared_inputs, baseline_tensors, deltas, 1.0)
        baseline_scalar = _scalarize_output(
            _call_model(model, prepared_inputs, baseline_leaves), target
        ).detach()
        input_scalar = _scalarize_output(
            _call_model(model, prepared_inputs, input_leaves), target
        ).detach()
        for chunk_index, chunk_alphas in enumerate(chunks):
            if len(chunk_alphas) == 1:
                path_leaves = _interned_path_leaves(
                    prepared_inputs, baseline_tensors, deltas, chunk_alphas[0]
                )
                gradients, _scalar = _gradient_for_inputs(
                    model, prepared_inputs, path_leaves, target
                )
                gradients_by_step.append(tuple(gradient.detach() for gradient in gradients))
                physical_calls += 1
                continue
            stacked_leaves = _stacked_path_leaves(
                prepared_inputs, baseline_tensors, deltas, chunk_alphas
            )
            chunk_gradients = _stacked_gradients(
                model, prepared_inputs, stacked_leaves, target, len(chunk_alphas)
            )
            physical_calls += 1
            # The audit runs BEFORE later chunks are computed, so a coupled
            # first chunk fails early rather than after the full pass.
            audit_row = auditor.row_for_chunk(chunk_index)
            if audit_row is not None:
                sequential_leaves = _interned_path_leaves(
                    prepared_inputs, baseline_tensors, deltas, chunk_alphas[audit_row]
                )
                sequential_gradients, _scalar = _gradient_for_inputs(
                    model, prepared_inputs, sequential_leaves, target
                )
                physical_calls += 1
                auditor.check(
                    chunk_index,
                    audit_row,
                    chunk_gradients[audit_row],
                    tuple(gradient.detach() for gradient in sequential_gradients),
                )
            gradients_by_step.extend(chunk_gradients)

    mean_gradients = tuple(
        torch.stack([step_gradients[index] for step_gradients in gradients_by_step], dim=0).mean(
            dim=0
        )
        for index in range(len(prepared_inputs.attributed_leaves))
    )
    values = tuple(
        (delta * mean_gradient).detach()
        for delta, mean_gradient in zip(deltas, mean_gradients, strict=True)
    )
    attribution_sum = sum((value.sum() for value in values), start=torch.zeros_like(input_scalar))
    target_delta = input_scalar - baseline_scalar
    return AttributionResult(
        method="integrated_gradients",
        values=_value_tree_from_leaves(prepared_inputs, values),
        target_repr=_target_repr(target),
        extra={
            "n_steps": n_steps,
            "baseline": _value_tree_from_leaves(prepared_inputs, baseline_tensors),
            "path_evaluations_logical": n_steps,
            "physical_forward_calls": physical_calls,
            "step_batch_size": chunk_size,
            "step_audit": auditor.record().to_extra(),
            **_completeness_extra(attribution_sum, target_delta),
        },
    )


def smoothgrad(
    model: Module,
    inputs: Any,
    input_kwargs: InputKwargs = None,
    *,
    target: _AttributionTarget,
    n_samples: int = 25,
    noise_level: float = 0.1,
    seed: int | None = None,
) -> AttributionResult:
    """Average saliency over Gaussian-noised copies of an input.

    Like :func:`saliency`, the result is unsigned: each sample is
    absolute-valued before the mean. ``noise_level`` is an absolute standard
    deviation in input units, so scale it to the input's range.

    Parameters
    ----------
    model
        PyTorch module to attribute.
    inputs
        Bare tensor for v1 behavior, or tuple/list of positional model arguments.
        Floating-point or complex tensor leaves are attributed.
    input_kwargs
        Optional keyword arguments for ``model``. Floating-point or complex
        tensor leaves are attributed.
    target
        Integer class index selecting ``output[..., target]`` and summing the
        selected values, a callable ``output -> scalar tensor``, or a
        :class:`SiteTarget` naming a hidden layer's output (and unit) to
        attribute instead of the model output.
    n_samples
        Number of noised saliency samples to average.
    noise_level
        Standard deviation of Gaussian noise added to each input copy; must be
        positive (zero noise refuses ``smoothgrad_noise_level_invalid``).
    seed
        Optional random seed for deterministic noise samples without mutating
        global torch RNG state.

    Returns
    -------
    AttributionResult
        Bare tensor for one attributed leaf, otherwise a mirrored value tree.
    """

    _validate_positive_int("n_samples", n_samples)
    if noise_level <= 0:
        # zero noise averages n_samples identical saliency maps -- a
        # pointless loop dressed as SmoothGrad -- so it refuses with negatives.
        raise AttributionError(
            f"smoothgrad noise_level must be positive; got {noise_level!r}. "
            "Remedy: pass a positive Gaussian scale such as noise_level=0.1, or call "
            "torchlens.attribution.saliency(model, inputs, target=...) for the "
            "noise-free map.",
            code="smoothgrad_noise_level_invalid",
        )
    _normalize_model_inputs(inputs, input_kwargs, _Caller("smoothgrad", model, target))
    # D7: SmoothGrad is the thin alias over the sampling substrate. The
    # per-sample ABSOLUTE VALUE happens before the mean because the child is
    # saliency, which absolute-values each sample's gradients itself; the
    # noise draw order (one draw per unique leaf per sample, per-device
    # generator seeded on first use) is bit-identical to the historical
    # inline implementation.
    from torchlens.attribution._noise_tunnel import noise_tunnel

    tunneled = noise_tunnel(
        inputs,
        input_kwargs,
        method=saliency,
        model=model,
        target=target,
        n_samples=n_samples,
        stdevs=noise_level,
        seed=seed,
        aggregation="mean",
    )
    return AttributionResult(
        method="smoothgrad",
        values=tunneled.values,
        target_repr=tunneled.target_repr,
        extra={"n_samples": n_samples, "noise_level": noise_level, "seed": seed},
    )
