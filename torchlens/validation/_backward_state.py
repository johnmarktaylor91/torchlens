"""Model-state guards and parameter-gradient evidence for backward validation."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from ..errors import TorchLensWarning

if TYPE_CHECKING:
    from ._layer_grad_report import LayerGradReport


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
        TorchLensWarning(
            "validate_backward_pass cannot compare gradients: Parameter(s) "
            f"{non_leaf_names[:3]} are not autograd leaves before validation (an "
            "earlier forward mutated them in place with a grad-requiring operand, "
            "as eager allows). Their gradient history chains into that earlier "
            "pass, so stock and captured gradients would not describe the same "
            "model; returning False. Remedy: validate a fresh copy of the model "
            "instead",
            code="backward_validation_non_leaf_parameter",
            non_leaf_parameters=list(non_leaf_names),
        ),
        stacklevel=3,
    )
    return True


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


def _parameter_grad_failure_fields(
    observed: torch.Tensor, expected: torch.Tensor
) -> dict[str, Any]:
    """Describe one parameter-gradient mismatch for its failure record."""

    fields: dict[str, Any] = {
        "saved_shape": tuple(observed.shape),
        "recomputed_shape": tuple(expected.shape),
        "saved_dtype": str(observed.dtype).removeprefix("torch."),
        "recomputed_dtype": str(expected.dtype).removeprefix("torch."),
    }
    if observed.shape != expected.shape:
        return fields
    fields["nan_mismatch"] = bool((torch.isnan(observed) != torch.isnan(expected)).any())
    fields["inf_mismatch"] = bool((torch.isinf(observed) != torch.isinf(expected)).any())
    finite = torch.isfinite(observed) & torch.isfinite(expected)
    if bool(finite.any()):
        diff = (observed[finite].double() - expected[finite].double()).abs()
        fields["max_abs_diff"] = float(diff.max())
        scale = expected[finite].double().abs().clamp_min(torch.finfo(torch.float64).tiny)
        fields["max_rel_diff"] = float((diff / scale).max())
    return fields


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


def _warn_if_no_module_output_evidence(layer_report: LayerGradReport) -> None:
    """Disclose a module-output check that failed only for lack of evidence.

    The root module call is excluded by design (its output gradient is the
    loss gradient itself), so a model whose forward calls no child module has
    zero eligible module outputs. The exact acceptance rule then fails closed
    on ``covered_count > 0`` with no gap or mismatch recorded; without this
    warning that unverifiable verdict reads as a capture failure.

    Parameters
    ----------
    layer_report:
        Failed module-output gradient report.
    """

    failure_counts = (
        layer_report.mismatched_count,
        layer_report.skipped_no_grad_count,
        layer_report.unresolved_output_label_count,
        layer_report.uncaptured_module_output_count,
        layer_report.missing_module_call_count,
        layer_report.unexpected_count,
    )
    if layer_report.covered_count or any(failure_counts):
        return
    warnings.warn(
        TorchLensWarning(
            "validate_backward_pass found no module-output gradient to compare: "
            f"all {len(layer_report.coverage)} module call(s) are proven exclusions "
            "(the root module call always is), so the module-output gradient check "
            "has zero detection power and the backward capture is unverifiable "
            "rather than passed; returning False. A model whose forward calls no "
            "child module has no such evidence. Remedy: pass "
            "validate_layer_grads=False to check parameter gradients alone",
            code="backward_validation_no_module_output_evidence",
            excluded_module_calls=len(layer_report.coverage),
        ),
        stacklevel=3,
    )
