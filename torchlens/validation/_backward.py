"""Validation helpers for backward-pass grad capture."""

from __future__ import annotations

import random
import warnings
from collections import OrderedDict
from collections.abc import Callable
from typing import TYPE_CHECKING, Any, NamedTuple, cast

import torch
from torch import nn

from .._capture_state_helpers import unwrap_compiled_model
from .._errors import CaptureNotActiveError, InvalidArgumentError
from .._input_coerce import _coerce_input_args
from .._input_walk import INPUT_TREE_MAX_DEPTH, raise_input_tree_depth_refusal
from .._robustness import check_model_and_input_variants
from .._utils.arg_handling import normalize_input_args
from .._utils.display import warn_parallel
from .._utils.rng import set_random_seed
from .._utils.tensor_utils import param_grad_tolerances_for_dtype
from ..errors import TorchLensWarning as TorchLensWarning
from ..intervention.errors import AppendStateValidationWarning
from ..options import CaptureOptions
from ._backward_state import (
    _is_appended_trace as _is_appended_trace,
    _leaf_parameter_flags as _leaf_parameter_flags,
    _parameter_grad_failure_fields as _parameter_grad_failure_fields,
    _refuse_non_leaf_parameters as _refuse_non_leaf_parameters,
    _restore_leaf_parameters as _restore_leaf_parameters,
    _restore_training_mode as _restore_training_mode,
    _stock_param_grad_degeneracy as _stock_param_grad_degeneracy,
    _warn_if_no_module_output_evidence as _warn_if_no_module_output_evidence,
)
from ._carried_graph import refusing_carried_graph
from ._diagnostics import (
    CHECK_BACKWARD,
    ValidationFailure,
    record_validation_failure,
    reset_validation_failure,
)
from ._live_model_state import (
    held_parameter_grads,
    load_state_dict_restore,
    reattach_parameter_grads,
    restore_state_dict_if_changed,
)

if TYPE_CHECKING:
    from ._layer_grad_report import LayerGradReport

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
    from .._ir.container_registry import OUTPUT_TREE_MAX_DEPTH

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
    :func:`torchlens._utils.arg_handling.copy_arg_tree`: an over-deep nest
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
    # A non-leaf Parameter (a frozen one written in place with a grad-requiring
    # operand) never gets ``.grad`` unless it retains grad, and reading it warns.
    return {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if (parameter.is_leaf or parameter.retains_grad) and parameter.grad is not None
    }


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

    from .._capture_state_helpers import (
        _clone_state_dict_with_metadata as clone_state_dict,
    )

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

    from .._capture_state_helpers import _move_tensors_to_device as move_tensors

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
    from .._ir.container import rebuild_container_from_spec

    try:
        return rebuild_container_from_spec(output_structure, output_layers)
    except ValueError:
        return output_layers


def _record_backward_failure(
    trace: Any,
    *,
    reason: str,
    message: str,
    op_label: str | None = None,
    **fields: Any,
) -> None:
    """Record why ``validate_backward_pass`` is about to return False.

    One structured ``ValidationFailure`` under ``CHECK_BACKWARD`` with the
    refusal or mismatch class in ``extra["reason"]``, so a False backward verdict
    is never bare: ``last_validation_failure()`` names it like a forward one.
    """

    record_validation_failure(
        trace,
        ValidationFailure(
            check_name=CHECK_BACKWARD,
            op_label=op_label,
            message=message,
            extra={"reason": reason},
            **fields,
        ),
    )


def _record_module_output_grad_failure(trace: Any, layer_report: LayerGradReport) -> None:
    """Record a failed module-output gradient report as the backward failure."""

    gap_buckets = (
        "mismatched",
        "skipped_no_grad",
        "unresolved_output_label",
        "uncaptured_module_output",
        "missing_module_call",
    )
    first_gap = next(
        (
            (label, bucket)
            for label, bucket in layer_report.coverage.items()
            if bucket in gap_buckets
        ),
        None,
    )
    if first_gap is None and not layer_report.covered_count:
        _record_backward_failure(
            trace,
            reason="no_module_output_evidence",
            message="nothing was compared: no module output carried a gradient on both "
            f"sides ({layer_report.compared_no_grad_count} output(s) had none in stock or "
            f"capture; the other {len(layer_report.coverage) - layer_report.compared_no_grad_count}"
            " module call(s) are proven exclusions)",
        )
        return
    label, bucket = first_gap if first_gap is not None else (None, "unexpected")
    _record_backward_failure(
        trace,
        reason="module_output_grads",
        op_label=label,
        message=f"module-output gradient check failed: {bucket}"
        + (f" at {label}" if label is not None else "")
        + f" ({layer_report.mismatched_count} mismatched, "
        f"{layer_report.skipped_no_grad_count} without a captured gradient, "
        f"{layer_report.uncaptured_module_output_count} uncaptured module outputs, "
        f"{layer_report.missing_module_call_count} missing module calls, "
        f"{layer_report.unresolved_output_label_count} unresolved labels, "
        f"{layer_report.unexpected_count} unexpected)",
        max_abs_diff=layer_report.max_abs_diffs.get(label) if label is not None else None,
        max_rel_diff=layer_report.max_rel_diffs.get(label) if label is not None else None,
    )


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


class _BackwardModelState(NamedTuple):
    """Model state taken before the stock and candidate passes and restored after them."""

    state_dict: OrderedDict[str, torch.Tensor]
    leaf_flags: tuple[tuple[nn.Parameter, bool], ...]
    held_grads: tuple[tuple[nn.Parameter, Any], ...]
    original_training: bool


def _snapshot_backward_model_state(model: nn.Module) -> _BackwardModelState:
    """Snapshot model values, parameter leaf flags, held gradients and training mode."""
    state_dict = _clone_state_dict_with_metadata(model)
    leaf_flags = _leaf_parameter_flags(model)
    # The passes reset .grad; the caller's accumulated grads are reattached by
    # reference at the end, never written into.
    held_grads = held_parameter_grads(model)
    return _BackwardModelState(state_dict, leaf_flags, held_grads, model.training)


def _restore_backward_model_state(model: nn.Module, state: _BackwardModelState) -> None:
    """Restore only what a pass changed: values, leaf flags and training mode."""
    # An unconditional in-place restore moves every tensor's version counter
    # and breaks a graph the caller holds across validation.
    restore_state_dict_if_changed(model, state.state_dict, load_state_dict_restore)
    _restore_leaf_parameters(state.leaf_flags)
    _restore_training_mode(model, state.original_training)


def _run_stock_backward(
    model: nn.Module,
    stock_call: tuple[Any, dict[str, Any]],
    loss_fn: Callable[[Any], torch.Tensor],
    seed: int,
    layer_grad_snapshot: OrderedDict[str, torch.Tensor] | None,
) -> tuple[Any, Any, Any]:
    """Compute pristine stock gradients and optional module-output evidence.

    ``layer_grad_snapshot`` is the state-dict snapshot the module-output evidence
    pass restores from; ``None`` runs a plain stock backward without that evidence.
    """
    from ._pristine import pristine_torch_oracle

    stock_inputs, stock_kwargs = stock_call
    stock_module_grads = None
    stock_identity_addresses = None
    stock_reached_keys = None
    # R75-1 sibling site: the "stock autograd" reference pass must run on
    # PRISTINE torch. In a wrapped process it used to run through the
    # installed pass-through wrapper shells -- the same closures the
    # candidate capture observes through -- so a wrapper-layer numeric
    # distortion corrupted stock and captured gradients IDENTICALLY and
    # the comparison passed vacuously. Refuse (fail-closed) if the
    # wrappers cannot be removed because a capture is active.
    with pristine_torch_oracle(), refusing_carried_graph(model):
        if layer_grad_snapshot is not None:
            from ._stock_layer_grads import _stock_layer_grads

            (
                stock_module_grads,
                stock_identity_addresses,
                stock_reached_keys,
            ) = _stock_layer_grads(
                model,
                stock_call,
                loss_fn=loss_fn,
                seed=seed,
                state_dict_snapshot=layer_grad_snapshot,
            )
        else:
            stock_loss = loss_fn(model(*stock_inputs, **stock_kwargs))
            stock_loss.backward()
    return stock_module_grads, stock_identity_addresses, stock_reached_keys


def _capture_backward_candidate(
    model: nn.Module,
    inputs: tuple[Any, dict[str, Any]],
    model_device: torch.device | None,
    seed: int,
    trace_fn: Callable[..., Any],
) -> Any:
    """Capture the candidate forward pass on the restored model."""
    input_args, input_kwargs = inputs
    set_random_seed(seed)
    logged_inputs, logged_kwargs = _prepare_inputs_for_backward(
        input_args, input_kwargs, model_device
    )
    model.zero_grad(set_to_none=True)
    return trace_fn(
        model,
        logged_inputs,
        input_kwargs=logged_kwargs,
        capture=CaptureOptions(
            save_grads="all",
            seed=seed,
        ),
        save="all",
    )


def _unexplained_backward_gaps(trace: Any) -> list[Any]:
    """Collect backward coverage gaps without a proven framework exclusion."""
    # Fail closed on coverage gaps: a node the walk could not observe is a
    # typed journal fact, and only a PROVEN framework-contract exclusion
    # may preserve a complete-coverage claim. No reason class has been
    # proven yet, so every recorded gap sinks the verdict rather than
    # hiding inside a tolerance.
    from .._ir.events import BackwardCoverageGap

    proven_framework_exclusions: frozenset[str] = frozenset()
    return [
        event
        for event in getattr(trace, "backward_events", ())
        if isinstance(event, BackwardCoverageGap)
        and event.reason not in proven_framework_exclusions
    ]


def validate_backward_pass(
    model: nn.Module,
    inputs: Any,
    input_kwargs: dict[str, Any] | None = None,
    loss_fn: Callable[[Any], torch.Tensor] | None = None,
    *,
    validate_metadata: bool = True,
    seed: int | None = None,
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
    inputs:
        Positional input arguments for ``model``.
    input_kwargs:
        Keyword input arguments for ``model``.
    loss_fn:
        Optional callable that maps model outputs to a scalar loss. Defaults to
        summing all returned tensors.
    validate_metadata:
        If True, run metadata invariant checks on the captured backward trace.
    seed:
        Fixed RNG seed for stock and candidate passes. Auto-generated if None.
    atol:
        Absolute tolerance for the parameter-gradient ``torch.allclose``.
        ``None`` (default) derives the tolerance PER GRADIENT DTYPE via
        :func:`~torchlens._utils.tensor_utils.param_grad_tolerances_for_dtype`
        (fp32 resolves to the legacy
        :data:`~torchlens._utils.tensor_utils.PARAM_GRAD_VALIDATION_ATOL`;
        fp64 tightens by the eps ratio, fp16/bf16 get a few-storage-ULP
        budget). An explicit float applies to every dtype unchanged.
    rtol:
        Relative tolerance for the parameter-gradient ``torch.allclose``.
        ``None`` (default) derives per gradient dtype (fp32 row ==
        :data:`~torchlens._utils.tensor_utils.PARAM_GRAD_VALIDATION_RTOL`).
    validate_layer_grads:
        If True (default), validate captured per-module-output gradients in
        addition to parameter gradients. False preserves the legacy
        parameter-only validation path as an explicit opt-out. The root
        module call is never compared, so a model whose forward calls no child
        module has no module-output gradient to compare and returns False with
        a coded ``TorchLensWarning``
        (``backward_validation_no_module_output_evidence``).
    layer_grad_atol:
        Optional absolute tolerance for per-module-output gradients. ``None``
        derives per gradient dtype via
        :func:`~torchlens._utils.tensor_utils.layer_grad_tolerances_for_dtype`
        (fp32 row == :data:`~torchlens._utils.tensor_utils.LAYER_GRAD_VALIDATION_ATOL`;
        module-output grads are compared ELEMENTWISE with no cross-element
        reduction, so they earn a 10x tighter pair than parameter grads).
    layer_grad_rtol:
        Optional relative tolerance for per-module-output gradients. ``None``
        derives per gradient dtype (fp32 row ==
        :data:`~torchlens._utils.tensor_utils.LAYER_GRAD_VALIDATION_RTOL`).

    Returns
    -------
    bool
        True when captured grads match stock autograd.
    """
    from .._user_funcs import (
        _reject_opaque_wrappers,
        _unwrap_data_parallel,
        trace as trace_fn,
    )
    from ._invariants import check_metadata_invariants

    reset_validation_failure(None)
    if _is_appended_trace(model):
        _record_backward_failure(
            None,
            reason="appended_trace",
            message="validate_backward_pass skipped: the model is an appended Trace",
        )
        return _warn_and_skip_appended_trace_validation(model)

    warn_parallel()
    _reject_opaque_wrappers(model)
    model = unwrap_compiled_model(model)
    model = _unwrap_data_parallel(model)
    inputs = _coerce_input_args(model, inputs)
    check_model_and_input_variants(model, inputs, input_kwargs)
    if input_kwargs is None:
        input_kwargs = {}
    if loss_fn is None:
        loss_fn = _sum_tensors
    if seed is None:
        seed = random.randint(1, 4294967294)
    inputs = normalize_input_args(inputs, model)
    model_device = next((parameter.device for parameter in model.parameters()), None)
    if _refuse_non_leaf_parameters(model):
        _record_backward_failure(
            None,
            reason="non_leaf_parameter",
            message="validate_backward_pass refused: a Parameter is not an autograd leaf "
            "before validation",
        )
        return False
    model_state = _snapshot_backward_model_state(model)
    trace = None
    try:
        set_random_seed(seed)
        stock_inputs, stock_kwargs = _prepare_inputs_for_backward(
            inputs, input_kwargs, model_device
        )
        model.zero_grad(set_to_none=True)
        try:
            stock_module_grads, stock_identity_addresses, stock_reached_keys = _run_stock_backward(
                model,
                (stock_inputs, stock_kwargs),
                loss_fn,
                seed,
                model_state.state_dict if validate_layer_grads else None,
            )
        except CaptureNotActiveError as pristine_error:
            warnings.warn(
                "validate_backward_pass could not compute pristine-torch stock "
                "gradients (a capture is active in this process, or the unwrap "
                "ledger is poisoned); the verdict would depend on the wrapper "
                "installation it is meant to check. Returning False rather "
                "than reporting unverified success.",
                RuntimeWarning,
                stacklevel=2,
            )
            _record_backward_failure(
                None,
                reason="pristine_oracle_unavailable",
                message=f"pristine-torch stock gradients unavailable: {pristine_error}",
            )
            return False
        expected_param_grads = _param_grads(model)

        # Restore only what the stock pass changed before the candidate runs.
        _restore_backward_model_state(model, model_state)
        trace = _capture_backward_candidate(
            model, (inputs, input_kwargs), model_device, seed, trace_fn
        )
        logged_loss = loss_fn(_reconstruct_candidate_output_for_loss(trace))
        with refusing_carried_graph(model):
            trace.backward(logged_loss)
        if validate_metadata:
            check_metadata_invariants(trace)
        unexplained_gaps = _unexplained_backward_gaps(trace)
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
            _record_backward_failure(
                trace,
                reason="backward_coverage_gap",
                op_label=unexplained_gaps[0].class_qualname,
                message=f"{len(unexplained_gaps)} backward coverage gap(s); first: "
                f"{unexplained_gaps[0].reason}",
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
                stock_reached_keys=stock_reached_keys,
                # None flows through: the comparator derives the tolerance per
                # gradient dtype (R13: the fp32 constants applied to fp64
                # masked corruption ~4.5e11 fp64 ULPs above round-off).
                atol=layer_grad_atol,
                rtol=layer_grad_rtol,
            )
            if not bool(layer_report):
                _warn_if_no_module_output_evidence(layer_report)
                _record_module_output_grad_failure(trace, layer_report)
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
            _record_backward_failure(
                trace,
                reason="no_parameter_gradients",
                message="stock autograd produced zero parameter gradients",
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
            _record_backward_failure(
                trace,
                reason="degenerate_parameter_gradients",
                message=f"stock parameter-gradient census is degenerate ({degeneracy})",
            )
            return False
        if expected_param_grads.keys() != observed_param_grads.keys():
            missing = sorted(expected_param_grads.keys() - observed_param_grads.keys())
            unexpected = sorted(observed_param_grads.keys() - expected_param_grads.keys())
            _record_backward_failure(
                trace,
                reason="parameter_census_mismatch",
                op_label=missing[0] if missing else (unexpected[0] if unexpected else None),
                message=f"captured parameter gradients differ from stock: missing {missing!r}, "
                f"unexpected {unexpected!r}",
            )
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
                _record_backward_failure(
                    trace,
                    reason="parameter_grad_mismatch",
                    op_label=name,
                    message=f"captured gradient of parameter {name!r} differs from stock autograd",
                    **_parameter_grad_failure_fields(observed_param_grads[name], expected_grad),
                )
                break
        return params_passed
    finally:
        _restore_backward_model_state(model, model_state)
        model.zero_grad(set_to_none=True)
        reattach_parameter_grads(model_state.held_grads)
        if trace is not None:
            trace.cleanup()
