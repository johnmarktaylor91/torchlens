"""Layer-gradient parity report for PATH E module-output validation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal

import torch

if TYPE_CHECKING:
    from ..data_classes.trace import Trace

MIN_MODULE_OUTPUT_COVERAGE: float = 0.80


@dataclass
class LayerGradReport:
    """PATH E module-output gradient comparison report.

    Coverage is keyed by module-call label, not by operation label. The
    classifier buckets are ``covered``, ``mismatched``,
    ``skipped_no_first_leaf``, ``skipped_module_less`` (counter only),
    ``skipped_no_grad``, ``skipped_identity_output``, and
    ``skipped_root_module``.
    """

    mode: Literal["module_output"]
    overall_passed: bool
    coverage: dict[str, str]
    covered_count: int
    skipped_no_first_leaf_count: int
    skipped_module_less_count: int
    skipped_no_grad_count: int
    skipped_identity_output_count: int
    skipped_root_module_count: int
    mismatched_count: int
    unexpected_count: int
    candidate_grad_count: int
    atol: float
    rtol: float
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
    trace: "Trace",
    stock_module_grads: Mapping[tuple[str, int], torch.Tensor],
    stock_identity_addresses: set[tuple[str, int]],
    *,
    atol: float = 1e-6,
    rtol: float = 1e-5,
    min_coverage: float = MIN_MODULE_OUTPUT_COVERAGE,
) -> LayerGradReport:
    """Compare candidate module-call output grads to stock module-output grads.

    Parameters
    ----------
    trace:
        Candidate TorchLens trace with logged backward grads.
    stock_module_grads:
        Stock gradients keyed by ``(module_address, call_index)``.
    stock_identity_addresses:
        Module-call keys whose stock output is identical to input.
    atol:
        Absolute allclose tolerance.
    rtol:
        Relative allclose tolerance.
    min_coverage:
        Minimum required covered ratio.

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

    modules_map = getattr(trace, "modules", None)
    pass_dict = getattr(modules_map, "_pass_dict", {}) if modules_map is not None else {}
    for call_log in list(pass_dict.values()):
        addr = getattr(call_log, "address", None)
        call_index = getattr(call_log, "call_index", None)
        if addr is None or call_index is None:
            continue
        call_label = f"{addr}:{call_index}"
        if addr == "self":
            coverage[call_label] = "skipped_root_module"
            continue
        output_ops = (
            getattr(call_log, "output_ops", None) or getattr(call_log, "output_layers", None) or []
        )
        if not output_ops:
            coverage[call_label] = "skipped_no_first_leaf"
            continue
        try:
            cand_layer = trace[output_ops[0]]
        except (KeyError, IndexError):
            coverage[call_label] = "skipped_no_first_leaf"
            continue
        key = (addr, call_index)
        if key in stock_identity_addresses:
            coverage[call_label] = "skipped_identity_output"
            continue
        cand_grad = getattr(cand_layer, "grad", None)
        if cand_grad is None:
            coverage[call_label] = "skipped_no_grad"
            continue
        stock_grad = stock_module_grads.get(key)
        if stock_grad is None:
            coverage[call_label] = "skipped_no_grad"
            continue
        if cand_grad.shape != stock_grad.shape:
            coverage[call_label] = "mismatched"
            mismatched.append(call_label)
            continue
        abs_diff = (cand_grad - stock_grad).abs()
        max_abs_diffs[call_label] = abs_diff.max().item()
        max_rel_diffs[call_label] = (abs_diff / stock_grad.abs().clamp(min=1e-30)).max().item()
        if torch.allclose(cand_grad, stock_grad, atol=atol, rtol=rtol):
            coverage[call_label] = "covered"
        else:
            coverage[call_label] = "mismatched"
            mismatched.append(call_label)

    for layer in trace.layer_list:
        if not getattr(layer, "has_grad", False):
            continue
        candidate_grad_count += 1
        if not (getattr(layer, "modules", None) or []):
            skipped_module_less_count += 1

    covered_count = sum(value == "covered" for value in coverage.values())
    mismatched_count = sum(value == "mismatched" for value in coverage.values())
    skipped_no_first_leaf_count = sum(
        value == "skipped_no_first_leaf" for value in coverage.values()
    )
    skipped_no_grad_count = sum(value == "skipped_no_grad" for value in coverage.values())
    skipped_identity_output_count = sum(
        value == "skipped_identity_output" for value in coverage.values()
    )
    skipped_root_module_count = sum(value == "skipped_root_module" for value in coverage.values())
    unexpected_count = sum(value == "unexpected" for value in coverage.values())

    coverage_denom = (
        covered_count + mismatched_count + skipped_no_first_leaf_count + skipped_no_grad_count
    )
    coverage_ratio = covered_count / coverage_denom if coverage_denom else 0.0
    overall_passed = (
        unexpected_count == 0
        and mismatched_count == 0
        and skipped_no_grad_count == 0
        and covered_count > 0
        and coverage_ratio >= min_coverage
    )

    return LayerGradReport(
        mode="module_output",
        overall_passed=overall_passed,
        coverage=coverage,
        covered_count=covered_count,
        skipped_no_first_leaf_count=skipped_no_first_leaf_count,
        skipped_module_less_count=skipped_module_less_count,
        skipped_no_grad_count=skipped_no_grad_count,
        skipped_identity_output_count=skipped_identity_output_count,
        skipped_root_module_count=skipped_root_module_count,
        mismatched_count=mismatched_count,
        unexpected_count=unexpected_count,
        candidate_grad_count=candidate_grad_count,
        atol=atol,
        rtol=rtol,
        mismatched_labels=tuple(mismatched),
        max_abs_diffs=max_abs_diffs,
        max_rel_diffs=max_rel_diffs,
    )
