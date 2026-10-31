"""Layer-gradient parity report for PATH E module-output validation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import torch

from ..utils.tensor_utils import layer_grad_tolerances_for_dtype

if TYPE_CHECKING:
    from ..data_classes.trace import Trace
    from ._stock_layer_grads import ModuleOutputGradKey
else:
    ModuleOutputGradKey = tuple[str, int, int]


@dataclass
class LayerGradReport:
    """PATH E module-output gradient comparison report.

    Coverage is keyed by module-call label for single-output calls and by
    ``module:call[index]`` for multi-output calls. The classifier buckets are:

    - ``covered`` / ``mismatched``: eligible outputs that were compared;
    - ``compared_no_grad``: an output the stock forward reached that received
      no gradient in stock autograd AND none in the capture (an output the loss
      does not depend on). The two sides agree, so it never sinks the verdict,
      but it is not evidence either: it does not count toward ``covered_count``,
      so a run with only such outputs still fails as unverifiable;
    - classified LEGITIMATE exclusions (never block passing):
      ``skipped_root_module``, ``skipped_identity_output``,
      ``skipped_no_tensor_output`` (PROVEN: the module call's
      ``ModuleExitEvent`` recorded zero tensor leaves in the real output walk
      AND stock autograd observed nothing for the call — an empty
      ``output_ops`` list alone is only evidence of no CAPTURED output, the
      exact symptom of the identity-node capture-bug class), and the
      diagnostic-only ``skipped_module_less`` counter;
    - fail-closed gaps (any occurrence sinks the verdict):
      ``skipped_no_grad`` (an eligible output whose gradient was not
      captured), ``unresolved_output_label`` (a module call names an output
      layer the trace cannot resolve), ``uncaptured_module_output`` (a module
      call with no captured output ops whose exclusion could NOT be proven —
      missing/unknown exit-event leaf count, a nonzero recorded leaf count,
      or a contradicting stock observation), and ``missing_module_call``
      (stock autograd observed a module call the candidate trace has no
      module-call log for at all — the reverse census).

    The acceptance rule is EXACT: 100% of the classified-eligible denominator
    must be ``covered``. There is no coverage-ratio tolerance; a tolerance
    here could hide missing hooks, which is a disarmed tripwire.
    """

    mode: Literal["module_output"]
    overall_passed: bool
    coverage: dict[str, str]
    covered_count: int
    skipped_no_tensor_output_count: int
    uncaptured_module_output_count: int
    missing_module_call_count: int
    unresolved_output_label_count: int
    skipped_module_less_count: int
    skipped_no_grad_count: int
    skipped_identity_output_count: int
    compared_no_grad_count: int
    skipped_root_module_count: int
    mismatched_count: int
    unexpected_count: int
    candidate_grad_count: int
    # None records the default dtype-aware mode: each comparison derived its
    # (rtol, atol) from the stock gradient's dtype via
    # layer_grad_tolerances_for_dtype. A float records an explicit override.
    atol: float | None
    rtol: float | None
    mismatched_labels: tuple[str, ...] = ()
    max_abs_diffs: dict[str, float] = field(default_factory=dict)
    max_rel_diffs: dict[str, float] = field(default_factory=dict)

    def __bool__(self) -> bool:
        """Return the aggregate pass/fail result.

        Returns
        -------
        bool
            ``overall_passed``.
        """

        return self.overall_passed


def _compare_module_output_grads(
    trace: Trace,
    stock_module_grads: Mapping[ModuleOutputGradKey, torch.Tensor],
    stock_identity_addresses: set[ModuleOutputGradKey],
    *,
    atol: float | None = None,
    rtol: float | None = None,
    stock_reached_keys: set[ModuleOutputGradKey] | None = None,
) -> LayerGradReport:
    """Compare candidate module-call output grads to stock module-output grads.

    Parameters
    ----------
    trace:
        Candidate TorchLens trace with logged backward grads.
    stock_module_grads:
        Stock gradients keyed by ``(module_address, call_index, output_index)``.
    stock_identity_addresses:
        Module-output keys whose stock output is identical to input.
    atol:
        Absolute allclose tolerance. ``None`` (default) derives per stock
        gradient dtype via
        :func:`~torchlens.utils.tensor_utils.layer_grad_tolerances_for_dtype`
        (the fp32 row is the shared elementwise layer-grad pair; see the
        error model on the constants in ``torchlens.utils.tensor_utils``).
        An explicit float applies to every dtype unchanged.
    rtol:
        Relative allclose tolerance. ``None`` derives per dtype likewise.
    stock_reached_keys:
        Every module-output key the stock forward enumerated in the same run
        that produced ``stock_module_grads``. A key here with no stock gradient
        and no candidate gradient is ``compared_no_grad``; without this set a
        missing gradient on both sides stays ``skipped_no_grad`` (fail-closed).

    Returns
    -------
    LayerGradReport
        Module-output gradient comparison report.
    """

    coverage: dict[str, str] = {}
    max_abs_diffs: dict[str, float] = {}
    max_rel_diffs: dict[str, float] = {}
    mismatched: list[str] = []
    candidate_grad_count = 0
    skipped_module_less_count = 0

    # Exit-event proof source: the ModuleExitEvent leaf count is recorded from
    # the real output walk at module exit, independent of whether labeling or
    # boundary minting succeeded, so it can PROVE a no-tensor-output exclusion.
    events = getattr(trace, "_capture_events", None)
    if events is None:
        events = getattr(trace, "capture_events", None)
    exit_leaf_counts: dict[tuple[str, int], int] = {}
    for exit_event in getattr(events, "module_exit_events", ()) or ():
        exit_key = (
            str(getattr(exit_event, "address", "")),
            int(getattr(exit_event, "call_index", 0) or 0),
        )
        exit_leaf_counts[exit_key] = int(getattr(exit_event, "output_tensor_leaf_count", -1))
    stock_observed_calls = {(addr, call_index) for addr, call_index, _ in stock_module_grads}
    stock_observed_calls.update(
        (addr, call_index) for addr, call_index, _ in stock_identity_addresses
    )

    modules_map = getattr(trace, "modules", None)
    pass_dict = getattr(modules_map, "_pass_dict", {}) if modules_map is not None else {}
    candidate_calls: set[tuple[str, int]] = set()
    for call_log in list(pass_dict.values()):
        addr = getattr(call_log, "address", None)
        call_index = getattr(call_log, "call_index", None)
        if addr is None or call_index is None:
            continue
        candidate_calls.add((addr, call_index))
        call_label = f"{addr}:{call_index}"
        if addr == "self":
            coverage[call_label] = "skipped_root_module"
            continue
        output_ops = (
            getattr(call_log, "output_ops", None) or getattr(call_log, "output_layers", None) or []
        )
        if not output_ops:
            # An empty output_ops list is only evidence of no CAPTURED tensor
            # output — the symptom of the identity-node capture-bug class
            # (see CHANGELOG 055af048) — so the exclusion must be PROVEN:
            # the exit event recorded zero real tensor leaves AND stock
            # autograd observed nothing for this call. Anything else is a
            # fail-closed gap, never a classification.
            if (addr, call_index) in stock_observed_calls:
                coverage[call_label] = "uncaptured_module_output"
            elif exit_leaf_counts.get((addr, call_index)) == 0:
                coverage[call_label] = "skipped_no_tensor_output"
            else:
                coverage[call_label] = "uncaptured_module_output"
            continue
        multi_output = len(output_ops) > 1
        for output_index, output_label in enumerate(output_ops):
            coverage_label = f"{call_label}[{output_index}]" if multi_output else call_label
            try:
                cand_layer = trace[output_label]
            except (KeyError, IndexError):
                # Fail-closed gap: a module call naming an output layer the
                # trace cannot resolve is an internal inconsistency, never a
                # legitimate exclusion.
                coverage[coverage_label] = "unresolved_output_label"
                continue
            key = (addr, call_index, output_index)
            if key in stock_identity_addresses:
                coverage[coverage_label] = "skipped_identity_output"
                continue
            cand_grad = getattr(cand_layer, "grad", None)
            stock_grad = stock_module_grads.get(key)
            if cand_grad is None and stock_grad is None:
                # Agreement, not evidence: only when stock provably reached the
                # output in this run; an asymmetric None still fails below.
                reached = stock_reached_keys is not None and key in stock_reached_keys
                coverage[coverage_label] = "compared_no_grad" if reached else "skipped_no_grad"
                continue
            if cand_grad is None or stock_grad is None:
                coverage[coverage_label] = "skipped_no_grad"
                continue
            if cand_grad.shape != stock_grad.shape:
                coverage[coverage_label] = "mismatched"
                mismatched.append(coverage_label)
                continue
            abs_diff = (cand_grad - stock_grad).abs()
            max_abs_diffs[coverage_label] = abs_diff.max().item()
            max_rel_diffs[coverage_label] = (
                (abs_diff / stock_grad.abs().clamp(min=1e-30)).max().item()
            )
            # equal_nan: an identical NaN pattern in candidate and stock grads
            # is agreement (tensor_nanequal doctrine); NaN-vs-number still
            # fails elementwise. Without it a CORRECT NaN-bearing gradient
            # false-FAILED this check. Tolerances resolve per stock-grad
            # dtype unless explicitly overridden (R13 consumer wiring).
            derived_rtol, derived_atol = layer_grad_tolerances_for_dtype(stock_grad.dtype)
            if torch.allclose(
                cand_grad,
                stock_grad,
                atol=atol if atol is not None else derived_atol,
                rtol=rtol if rtol is not None else derived_rtol,
                equal_nan=True,
            ):
                coverage[coverage_label] = "covered"
            else:
                coverage[coverage_label] = "mismatched"
                mismatched.append(coverage_label)

    # Reverse census: every module call stock autograd observed must exist as
    # a candidate module-call log. A wholly absent call is invisible to the
    # forward direction (there is no output_ops list to classify), so it is
    # reconciled here as a fail-closed gap. Root addresses are excluded (the
    # root is classified skipped_root_module in the forward direction).
    for addr, call_index in sorted(stock_observed_calls - candidate_calls):
        if addr in ("", "self"):
            continue
        coverage.setdefault(f"{addr}:{call_index}", "missing_module_call")

    for layer in trace.layer_list:
        if not getattr(layer, "has_grad", False):
            continue
        candidate_grad_count += 1
        if not (getattr(layer, "modules", None) or []):
            skipped_module_less_count += 1

    covered_count = sum(value == "covered" for value in coverage.values())
    mismatched_count = sum(value == "mismatched" for value in coverage.values())
    skipped_no_tensor_output_count = sum(
        value == "skipped_no_tensor_output" for value in coverage.values()
    )
    uncaptured_module_output_count = sum(
        value == "uncaptured_module_output" for value in coverage.values()
    )
    missing_module_call_count = sum(value == "missing_module_call" for value in coverage.values())
    unresolved_output_label_count = sum(
        value == "unresolved_output_label" for value in coverage.values()
    )
    skipped_no_grad_count = sum(value == "skipped_no_grad" for value in coverage.values())
    skipped_identity_output_count = sum(
        value == "skipped_identity_output" for value in coverage.values()
    )
    skipped_root_module_count = sum(value == "skipped_root_module" for value in coverage.values())
    compared_no_grad_count = sum(value == "compared_no_grad" for value in coverage.values())
    unexpected_count = sum(value == "unexpected" for value in coverage.values())

    # Eligibility-classifier acceptance (replaces the former 0.80 coverage
    # ratio): the classified-eligible denominator is {covered, mismatched,
    # skipped_no_grad, unresolved_output_label, uncaptured_module_output,
    # missing_module_call} and 100% of it must be covered. Legitimate
    # exclusions were PROVEN out above; any unexplained gap fails closed
    # rather than hiding inside a tolerance or a classification.
    overall_passed = (
        unexpected_count == 0
        and mismatched_count == 0
        and skipped_no_grad_count == 0
        and unresolved_output_label_count == 0
        and uncaptured_module_output_count == 0
        and missing_module_call_count == 0
        and covered_count > 0
    )

    return LayerGradReport(
        mode="module_output",
        overall_passed=overall_passed,
        coverage=coverage,
        covered_count=covered_count,
        skipped_no_tensor_output_count=skipped_no_tensor_output_count,
        uncaptured_module_output_count=uncaptured_module_output_count,
        missing_module_call_count=missing_module_call_count,
        unresolved_output_label_count=unresolved_output_label_count,
        skipped_module_less_count=skipped_module_less_count,
        skipped_no_grad_count=skipped_no_grad_count,
        skipped_identity_output_count=skipped_identity_output_count,
        skipped_root_module_count=skipped_root_module_count,
        compared_no_grad_count=compared_no_grad_count,
        mismatched_count=mismatched_count,
        unexpected_count=unexpected_count,
        candidate_grad_count=candidate_grad_count,
        atol=atol,
        rtol=rtol,
        mismatched_labels=tuple(mismatched),
        max_abs_diffs=max_abs_diffs,
        max_rel_diffs=max_rel_diffs,
    )
