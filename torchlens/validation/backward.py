"""Validation helpers for backward-pass grad capture."""

from __future__ import annotations

import random
import warnings
from collections import OrderedDict
from collections.abc import Callable
from typing import Any, cast

import torch
from torch import nn

from .._capture_state_helpers import unwrap_compiled_model
from .._input_coerce import _coerce_input_args
from .._input_walk import INPUT_TREE_MAX_DEPTH, raise_input_tree_depth_refusal
from .._robustness import check_model_and_input_variants
from ..intervention.errors import AppendStateValidationWarning
from ..options import CaptureOptions
from ..utils.arg_handling import normalize_input_args
from ..utils.display import warn_parallel
from ..utils.rng import set_random_seed
from ..utils.tensor_utils import param_grad_tolerances_for_dtype

_SUM_IN_PROGRESS = object()
"""Memo sentinel: this container is on the current descent chain (a cycle)."""


def _sum_tensors(
    value: Any,
    _memo: dict[int, Any] | None = None,
    _depth: int = 0,
) -> torch.Tensor:
    """Reduce nested tensor outputs to a scalar loss.

    Depth-bounded and memoized (grind-p3 T11): the walk shares the boundary
    nesting ceiling instead of dying in a raw ``RecursionError``, a cyclic
    container refuses typed (its occurrence-weighted sum would be infinite),
    and a DAG-shaped output that reuses one sub-container under several paths
    computes that subtree's sum ONCE and adds the cached scalar per
    occurrence -- the exact value the unmemoized walk produced, in O(nodes)
    instead of O(paths) (shared substructure doubled the walk per level).

    Parameters
    ----------
    value:
        Tensor or nested container of tensors.
    _memo:
        Internal per-call cache mapping container ``id()`` to its computed
        subtree sum (or the in-progress cycle sentinel). Callers must not
        supply this.
    _depth:
        Internal recursion depth for the shared nesting ceiling. Callers must
        not supply this.

    Returns
    -------
    torch.Tensor
        Scalar sum over all tensors in ``value``, occurrence-weighted.
    """
    if isinstance(value, torch.Tensor):
        return value.sum()
    if not isinstance(value, (dict, list, tuple)):
        raise ValueError("validate_backward_pass requires the model to return at least one tensor.")
    from .._errors import InvalidArgumentError
    from ..ir.container_registry import OUTPUT_TREE_MAX_DEPTH

    if _depth >= OUTPUT_TREE_MAX_DEPTH:
        raise InvalidArgumentError(
            "Model-output tree nesting exceeds the supported output-boundary "
            f"depth ceiling ({OUTPUT_TREE_MAX_DEPTH}) in validate_backward_pass.",
            code="output_tree_depth_exceeded",
            remedy="Flatten the nested output containers before validating.",
            depth=_depth,
        )
    if _memo is None:
        _memo = {}
    value_id = id(value)
    cached = _memo.get(value_id)
    if cached is _SUM_IN_PROGRESS:
        raise InvalidArgumentError(
            "Model-output tree contains a self-referential container; its "
            "occurrence-weighted tensor sum is not defined.",
            code="output_tree_cycle",
            remedy="Remove the container reference cycle from the model output.",
        )
    if cached is not None:
        return cast(torch.Tensor, cached)
    _memo[value_id] = _SUM_IN_PROGRESS
    try:
        items = value.values() if isinstance(value, dict) else value
        tensors = [_sum_tensors(item, _memo, _depth + 1) for item in items]
    finally:
        if _memo.get(value_id) is _SUM_IN_PROGRESS:
            del _memo[value_id]
    if not tensors:
        raise ValueError("validate_backward_pass requires the model to return at least one tensor.")
    result = tensors[0]
    for tensor in tensors[1:]:
        result = result + tensor
    _memo[value_id] = result
    return result


def _clone_inputs_with_grad(
    input_args: Any,
    _memo: dict[int, Any] | None = None,
    _depth: int = 0,
) -> Any:
    """Clone tensor inputs and enable grads on floating tensors.

    Depth-bounded and memoized (grind-p3 T11), mirroring
    :func:`torchlens.utils.arg_handling.copy_arg_tree`: an over-deep nest
    refuses typed through the shared input-boundary ceiling instead of a raw
    ``RecursionError``; a mutable container is registered in the memo BEFORE
    its children so a reference cycle resolves to the in-progress copy; and a
    DAG-shaped input that reuses one sub-container under several paths is
    cloned ONCE and stays aliased in the copy (the unmemoized walk expanded
    shared substructure exponentially in depth). Tensors remain leaves cloned
    per distinct container occurrence and are never memoized.

    Parameters
    ----------
    input_args:
        User input arguments.
    _memo:
        Internal per-call cache mapping container ``id()`` to its (possibly
        in-progress) copy. Callers must not supply this.
    _depth:
        Internal recursion depth for the shared nesting ceiling. Callers must
        not supply this.

    Returns
    -------
    Any
        Input arguments with cloned floating tensors requiring grad.
    """
    if isinstance(input_args, torch.Tensor):
        cloned = input_args.detach().clone()
        if cloned.is_floating_point() or cloned.is_complex():
            cloned.requires_grad_(True)
        return cloned
    if not isinstance(input_args, (tuple, list, dict)):
        return input_args
    if _depth >= INPUT_TREE_MAX_DEPTH:
        raise_input_tree_depth_refusal(depth=_depth)
    if _memo is None:
        _memo = {}
    existing = _memo.get(id(input_args))
    if existing is not None:
        return existing
    if isinstance(input_args, tuple):
        # Immutable: any cycle passes through a registered mutable container,
        # so eager recursion is safe; memoized after construction for DAG reuse.
        copied: Any = tuple(_clone_inputs_with_grad(item, _memo, _depth + 1) for item in input_args)
        _memo[id(input_args)] = copied
        return copied
    if isinstance(input_args, list):
        copied = []
        _memo[id(input_args)] = copied
        for item in input_args:
            copied.append(_clone_inputs_with_grad(item, _memo, _depth + 1))
        return copied
    copied = {}
    _memo[id(input_args)] = copied
    for key, item in input_args.items():
        copied[key] = _clone_inputs_with_grad(item, _memo, _depth + 1)
    return copied


def _stock_param_grad_degeneracy(
    expected_param_grads: dict[str, torch.Tensor],
) -> str | None:
    """Classify a stock parameter-gradient census with zero detection power.

    Parameters
    ----------
    expected_param_grads:
        Non-empty stock autograd gradients keyed by parameter name.

    Returns
    -------
    str | None
        ``"all-nonfinite"`` when EVERY element of EVERY gradient is NaN/Inf
        (vacuous under ``equal_nan=True``), ``"all-zero"`` when every element
        is exactly zero (zero-filled buffers are indistinguishable from a
        correct capture), ``"mixed-nonfinite-zero"`` when every element is
        NaN/Inf or exactly zero but neither class alone covers the census,
        ``"element-free"`` when no gradient carries any element, and ``None``
        for a census with real comparison power. The rule is exactly the one
        the ``equal_nan=True`` comparison implies: a FINITE NONZERO element
        anywhere restores detection power and returns ``None``; every element
        that is NaN/Inf (vacuous) or zero (indistinguishable from a
        zero-filled buffer) contributes none.
    """

    saw_element = False
    saw_finite = False
    saw_nonzero = False
    for grad in expected_param_grads.values():
        if grad.numel() == 0:
            continue
        saw_element = True
        finite = torch.isfinite(grad)
        # The decisive per-element predicate: only an element that is BOTH
        # finite AND nonzero gives the comparison detection power. Tracking
        # the two properties as independent whole-census totals (the previous
        # shape) was defeated by a MIXED census -- one all-NaN grad killed the
        # all-zero arm, one all-zero grad killed the all-nonfinite arm, and a
        # census with zero finite-nonzero elements passed as "real power".
        if bool((finite & grad.ne(0)).any()):
            return None
        if bool(finite.any()):
            saw_finite = True
        if bool(grad.ne(0).any()):
            saw_nonzero = True
    if not saw_element:
        return "element-free"
    if not saw_finite:
        return "all-nonfinite"
    if not saw_nonzero:
        return "all-zero"
    return "mixed-nonfinite-zero"


def _param_grads(model: nn.Module) -> dict[str, torch.Tensor]:
    """Collect detached parameter grads by name.

    Parameters
    ----------
    model:
        Model whose parameter grads should be collected.

    Returns
    -------
    dict[str, torch.Tensor]
        Detached grad clones for parameters with grads.
    """
    return {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }


def _leaf_parameter_flags(model: nn.Module) -> tuple[tuple[nn.Parameter, bool], ...]:
    """Record each leaf Parameter with its ``requires_grad`` before a validation pass.

    Parameters
    ----------
    model:
        Model about to run the stock and candidate passes.

    Returns
    -------
    tuple[tuple[nn.Parameter, bool], ...]
        Every Parameter that is an autograd leaf now, with its current flag.
    """

    return tuple(
        (parameter, parameter.requires_grad)
        for parameter in model.parameters()
        if parameter.is_leaf
    )


def _restore_leaf_parameters(leaf_flags: tuple[tuple[nn.Parameter, bool], ...]) -> None:
    """Make Parameters that a pass turned into non-leaves leaves again, in place.

    A forward that mutates a frozen Parameter in place with a grad-requiring
    operand turns it into a non-leaf whose ``grad_fn`` chains into that pass's
    graph, exactly as eager does. ``load_state_dict`` restores its VALUE but not
    that history, so the next pass on the same object would differentiate
    through the stale graph too (a second eager forward on that object does the
    same) and its gradients could not equal a fresh model's. Validation's state
    contract is the pre-call model, so the history is cut (``detach_`` keeps the
    object and its storage) and the recorded flag is put back. (A Parameter that
    is already a non-leaf before validation is refused up front, so it never
    reaches here.)

    Parameters
    ----------
    leaf_flags:
        Output of :func:`_leaf_parameter_flags` taken before the passes.
    """

    for parameter, requires_grad in leaf_flags:
        if parameter.is_leaf:
            continue
        parameter.detach_()
        parameter.requires_grad_(requires_grad)


def _refuse_non_leaf_parameters(model: nn.Module) -> bool:
    """Warn and report True when a Parameter is already a non-leaf before validation.

    Parameters
    ----------
    model:
        Model about to be validated.

    Returns
    -------
    bool
        True when validation must return False: some Parameter carries the
        autograd history of an earlier forward, so no stock pass on this object
        describes the model the caller holds.
    """

    non_leaf_names = [name for name, parameter in model.named_parameters() if not parameter.is_leaf]
    if not non_leaf_names:
        return False
    warnings.warn(
        "validate_backward_pass cannot compare gradients: Parameter(s) "
        f"{non_leaf_names[:3]} are not autograd leaves before validation (an "
        "earlier forward mutated them in place with a grad-requiring operand, "
        "as eager allows). Their gradient history chains into that earlier "
        "pass, so stock and captured gradients would not describe the same "
        "model. Returning False; validate a fresh copy of the model instead.",
        RuntimeWarning,
        stacklevel=3,
    )
    return True


def _clone_state_dict_with_metadata(model: nn.Module) -> OrderedDict[str, torch.Tensor]:
    """Clone a module ``state_dict`` while preserving PyTorch metadata.

    Parameters
    ----------
    model:
        Model whose state should be cloned.

    Returns
    -------
    OrderedDict[str, torch.Tensor]
        Detached tensor clones with PyTorch ``state_dict`` metadata preserved.
    """

    from ..user_funcs import _clone_state_dict_with_metadata as clone_state_dict

    return clone_state_dict(model)


def _move_tensors_to_device(obj: Any, device: torch.device | str) -> Any:
    """Move nested tensors to ``device`` using the public validator helper.

    Parameters
    ----------
    obj:
        Tensor or nested container.
    device:
        Target device.

    Returns
    -------
    Any
        Object with all tensors moved to ``device``.
    """

    from ..user_funcs import _move_tensors_to_device as move_tensors

    return move_tensors(obj, device)


def _prepare_inputs_for_backward(
    input_args: Any,
    input_kwargs: dict[str, Any],
    device: torch.device | None,
) -> tuple[Any, dict[str, Any]]:
    """Clone validation inputs and move them to the model device.

    Parameters
    ----------
    input_args:
        Normalized positional model inputs.
    input_kwargs:
        Keyword model inputs.
    device:
        Target model device, if any parameters exist.

    Returns
    -------
    tuple[Any, dict[str, Any]]
        Cloned positional and keyword inputs.
    """

    cloned_args = _clone_inputs_with_grad(input_args)
    cloned_kwargs = cast(dict[str, Any], _clone_inputs_with_grad(input_kwargs))
    if device is None:
        return cloned_args, cloned_kwargs
    return _move_tensors_to_device(cloned_args, device), _move_tensors_to_device(
        cloned_kwargs, device
    )


def _restore_training_mode(model: nn.Module, training: bool) -> None:
    """Restore a module's train/eval flag.

    Parameters
    ----------
    model:
        Model whose mode should be restored.
    training:
        Original ``model.training`` value.
    """

    model.train(training)


def _reconstruct_candidate_output_for_loss(trace: Any) -> Any:
    """Rebuild the candidate model output passed to a loss function.

    Parameters
    ----------
    trace:
        Candidate TorchLens trace.

    Returns
    -------
    Any
        Single output tensor, rebuilt output container, or flat list fallback.
    """

    output_layers = [trace[layer_label].out for layer_label in trace.output_layers]
    if len(output_layers) == 1:
        return output_layers[0]
    root_call = None
    module_calls = getattr(trace, "module_calls", None)
    if module_calls is not None and "self:1" in module_calls:
        root_call = module_calls["self:1"]
    output_structure = getattr(root_call, "output_structure", None)
    if output_structure is None:
        return output_layers
    from ..ir.container import rebuild_container_from_spec

    try:
        return rebuild_container_from_spec(output_structure, output_layers)
    except ValueError:
        return output_layers


def _is_appended_trace(value: Any) -> bool:
    """Return whether a value is an appended Trace-like object.

    Parameters
    ----------
    value:
        Object passed to the backward validator.

    Returns
    -------
    bool
        True only for objects carrying the appended Trace state marker.
    """

    return bool(getattr(value, "is_appended", False))


def _warn_and_skip_appended_trace_validation(trace: Any) -> bool:
    """Warn that stacked traces cannot be freshly revalidated.

    Parameters
    ----------
    trace:
        Appended Trace-like object supplied to validation.

    Returns
    -------
    bool
        Always False because a skipped backward re-derivation is not a passing
        validation result.
    """

    warnings.warn(
        "validate_backward_pass received a stacked appended trace; fresh backward "
        "re-derivation for appended traces is not supported, so saved grads are "
        "treated as authoritative.",
        AppendStateValidationWarning,
        stacklevel=2,
    )
    return False


def validate_backward_pass(
    model: nn.Module,
    input_args: Any,
    input_kwargs: dict[str, Any] | None = None,
    loss_fn: Callable[[Any], torch.Tensor] | None = None,
    *,
    validate_metadata: bool = True,
    random_seed: int | None = None,
    atol: float | None = None,
    rtol: float | None = None,
    validate_layer_grads: bool = True,
    layer_grad_atol: float | None = None,
    layer_grad_rtol: float | None = None,
) -> bool:
    """Validate TorchLens backward capture against stock autograd parameter grads.

    Parameters
    ----------
    model:
        Model to validate.
    input_args:
        Positional input arguments for ``model``.
    input_kwargs:
        Keyword input arguments for ``model``.
    loss_fn:
        Optional callable that maps model outputs to a scalar loss. Defaults to
        summing all returned tensors.
    validate_metadata:
        If True, run metadata invariant checks on the captured backward trace.
    random_seed:
        Fixed RNG seed for stock and candidate passes. Auto-generated if None.
    atol:
        Absolute tolerance for the parameter-gradient ``torch.allclose``.
        ``None`` (default) derives the tolerance PER GRADIENT DTYPE via
        :func:`~torchlens.utils.tensor_utils.param_grad_tolerances_for_dtype`
        (fp32 resolves to the legacy
        :data:`~torchlens.utils.tensor_utils.PARAM_GRAD_VALIDATION_ATOL`;
        fp64 tightens by the eps ratio, fp16/bf16 get a few-storage-ULP
        budget). An explicit float applies to every dtype unchanged.
    rtol:
        Relative tolerance for the parameter-gradient ``torch.allclose``.
        ``None`` (default) derives per gradient dtype (fp32 row ==
        :data:`~torchlens.utils.tensor_utils.PARAM_GRAD_VALIDATION_RTOL`).
    validate_layer_grads:
        If True (default), validate captured per-module-output gradients in
        addition to parameter gradients. False preserves the legacy
        parameter-only validation path as an explicit opt-out.
    layer_grad_atol:
        Optional absolute tolerance for per-module-output gradients. ``None``
        derives per gradient dtype via
        :func:`~torchlens.utils.tensor_utils.layer_grad_tolerances_for_dtype`
        (fp32 row == :data:`~torchlens.utils.tensor_utils.LAYER_GRAD_VALIDATION_ATOL`;
        module-output grads are compared ELEMENTWISE with no cross-element
        reduction, so they earn a 10x tighter pair than parameter grads).
    layer_grad_rtol:
        Optional relative tolerance for per-module-output gradients. ``None``
        derives per gradient dtype (fp32 row ==
        :data:`~torchlens.utils.tensor_utils.LAYER_GRAD_VALIDATION_RTOL`).

    Returns
    -------
    bool
        True when captured grads match stock autograd.
    """
    from ..user_funcs import _reject_opaque_wrappers, _unwrap_data_parallel, trace as trace_fn
    from .invariants import check_metadata_invariants

    if _is_appended_trace(model):
        return _warn_and_skip_appended_trace_validation(model)

    warn_parallel()
    _reject_opaque_wrappers(model)
    model = unwrap_compiled_model(model)
    model = _unwrap_data_parallel(model)
    input_args = _coerce_input_args(model, input_args)
    check_model_and_input_variants(model, input_args, input_kwargs)
    if input_kwargs is None:
        input_kwargs = {}
    if loss_fn is None:
        loss_fn = _sum_tensors
    if random_seed is None:
        random_seed = random.randint(1, 4294967294)
    input_args = normalize_input_args(input_args, model)
    model_device = next((parameter.device for parameter in model.parameters()), None)
    if _refuse_non_leaf_parameters(model):
        return False
    state_dict = _clone_state_dict_with_metadata(model)
    leaf_flags = _leaf_parameter_flags(model)
    original_training = model.training
    trace = None
    stock_module_grads = None
    stock_identity_addresses = None

    try:
        set_random_seed(random_seed)
        stock_inputs, stock_kwargs = _prepare_inputs_for_backward(
            input_args, input_kwargs, model_device
        )
        model.zero_grad(set_to_none=True)
        # R75-1 sibling site: the "stock autograd" reference pass must run on
        # PRISTINE torch. In a wrapped process it used to run through the
        # installed pass-through wrapper shells -- the same closures the
        # candidate capture observes through -- so a wrapper-layer numeric
        # distortion corrupted stock and captured gradients IDENTICALLY and
        # the comparison passed vacuously. Refuse (fail-closed) if the
        # wrappers cannot be removed because a capture is active.
        from .._errors import CaptureContextError
        from ._pristine import pristine_torch_oracle

        try:
            with pristine_torch_oracle():
                if validate_layer_grads:
                    from ._stock_layer_grads import _stock_layer_grads

                    stock_module_grads, stock_identity_addresses = _stock_layer_grads(
                        model,
                        stock_inputs,
                        stock_kwargs,
                        loss_fn=loss_fn,
                        random_seed=random_seed,
                        state_dict_snapshot=state_dict,
                    )
                else:
                    stock_loss = loss_fn(model(*stock_inputs, **stock_kwargs))
                    stock_loss.backward()
        except CaptureContextError:
            warnings.warn(
                "validate_backward_pass could not compute pristine-torch stock "
                "gradients (a capture is active in this process, or the unwrap "
                "ledger is poisoned); the verdict would depend on the wrapper "
                "installation it is meant to check. Returning False rather "
                "than reporting unverified success.",
                RuntimeWarning,
                stacklevel=2,
            )
            return False
        expected_param_grads = _param_grads(model)

        model.load_state_dict(state_dict)
        _restore_leaf_parameters(leaf_flags)
        _restore_training_mode(model, original_training)
        set_random_seed(random_seed)
        logged_inputs, logged_kwargs = _prepare_inputs_for_backward(
            input_args, input_kwargs, model_device
        )
        model.zero_grad(set_to_none=True)
        trace = trace_fn(
            model,
            logged_inputs,
            input_kwargs=logged_kwargs,
            capture=CaptureOptions(
                layers_to_save="all",
                save_grads="all",
                random_seed=random_seed,
            ),
        )
        logged_output = _reconstruct_candidate_output_for_loss(trace)
        logged_loss = loss_fn(logged_output)
        trace.log_backward(logged_loss)
        if validate_metadata:
            check_metadata_invariants(trace)
        # Fail closed on coverage gaps: a node the walk could not observe is a
        # typed journal fact, and only a PROVEN framework-contract exclusion
        # may preserve a complete-coverage claim. No reason class has been
        # proven yet, so every recorded gap sinks the verdict rather than
        # hiding inside a tolerance.
        from ..ir.events import BackwardCoverageGap

        proven_framework_exclusions: frozenset[str] = frozenset()
        unexplained_gaps = [
            event
            for event in getattr(trace, "backward_events", ())
            if isinstance(event, BackwardCoverageGap)
            and event.reason not in proven_framework_exclusions
        ]
        if unexplained_gaps:
            warnings.warn(
                "validate_backward_pass observed "
                f"{len(unexplained_gaps)} backward coverage gap(s) "
                f"(first: {unexplained_gaps[0].reason} on "
                f"{unexplained_gaps[0].class_qualname}); coverage cannot be "
                "verified, failing closed.",
                RuntimeWarning,
                stacklevel=2,
            )
            return False
        observed_param_grads = _param_grads(model)

        if validate_layer_grads:
            from ._layer_grad_report import _compare_module_output_grads

            assert stock_module_grads is not None
            assert stock_identity_addresses is not None
            layer_report = _compare_module_output_grads(
                trace,
                stock_module_grads,
                stock_identity_addresses,
                # None flows through: the comparator derives the tolerance per
                # gradient dtype (R13: the fp32 constants applied to fp64
                # masked corruption ~4.5e11 fp64 ULPs above round-off).
                atol=layer_grad_atol,
                rtol=layer_grad_rtol,
            )
            if not bool(layer_report):
                return False
        # An empty stock parameter-gradient census is UNVERIFIABLE, and that verdict
        # cannot depend on ``validate_layer_grads``. This block used to be duplicated
        # inside the branch above with ``return True`` instead of ``return False``, so
        # the same model, the same input and the same warning text produced OPPOSITE
        # verdicts depending on a flag that defaults to True -- and the True branch
        # warned "could not verify parameter gradients" and then reported success.
        # That contradicted the policy at ``_user_public_impls.py:1300-1310``
        # ("Returning False rather than reporting unverified success") and left
        # ``test_backward_validation_zero_param_grads_is_not_pass`` red. Layer-grad
        # evidence is still collected above; it just cannot launder an unverifiable
        # parameter census into a pass.
        if not expected_param_grads:
            warnings.warn(
                "validate_backward_pass could not verify parameter gradients because "
                "stock autograd produced zero parameter gradients.",
                RuntimeWarning,
                stacklevel=2,
            )
            return False
        # Degenerate-evidence guard: an all-nonfinite census makes every
        # ``equal_nan=True`` comparison pass vacuously (a NaN loss NaNs both
        # pipelines identically), and an all-zero census cannot distinguish a
        # correct capture from zero-filled gradient buffers. Either way the
        # comparison below has ZERO detection power, so the verdict is
        # unverifiable -- never PASS (the ABORTED_NONFINITE doctrine). The
        # decisive predicate is per-element: a census with at least one FINITE
        # NONZERO element keeps its comparison power and must keep passing; a
        # census made entirely of NaN/Inf and exact-zero elements (including
        # the MIXED all-NaN-grad + all-zero-grad shape) has none.
        degeneracy = _stock_param_grad_degeneracy(expected_param_grads)
        if degeneracy is not None:
            warnings.warn(
                "validate_backward_pass could not verify gradients because the "
                f"stock autograd parameter-gradient census is degenerate "
                f"({degeneracy}): the gradient comparison has zero detection "
                "power, so the backward capture is unverifiable rather than "
                "passed.",
                RuntimeWarning,
                stacklevel=2,
            )
            return False
        if expected_param_grads.keys() != observed_param_grads.keys():
            return False
        # equal_nan follows tensor_nanequal's doctrine: an identical NaN
        # pattern in candidate and stock grads is agreement, not a mismatch
        # (NaN-vs-number still fails elementwise). Tolerances resolve PER
        # GRADIENT DTYPE when not explicitly overridden: the fp32 constants
        # applied to every dtype checked fp64 grads ~4.5e11 of their own ULPs
        # loose and false-failed fp16 grads (R13; the derivation shipped in
        # c9734a7e but had zero verdict-site consumers).
        params_passed = True
        for name in expected_param_grads:
            expected_grad = expected_param_grads[name]
            derived_rtol, derived_atol = param_grad_tolerances_for_dtype(expected_grad.dtype)
            if not torch.allclose(
                observed_param_grads[name],
                expected_grad,
                atol=atol if atol is not None else derived_atol,
                rtol=rtol if rtol is not None else derived_rtol,
                equal_nan=True,
            ):
                params_passed = False
                break
        return params_passed
    finally:
        model.load_state_dict(state_dict)
        _restore_leaf_parameters(leaf_flags)
        _restore_training_mode(model, original_training)
        model.zero_grad(set_to_none=True)
        if trace is not None:
            trace.cleanup()
