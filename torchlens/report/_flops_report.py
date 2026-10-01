"""The one-call paper number: ``flops_report`` (F09; costreport D10-D13).

Both doors -- the model door (one safe capture under the summary
execution contract: eval mode, ``no_grad``, module flags and RNG restored
bit-identically) and the retroactive trace door -- feed ONE detached
builder. The report never auto-prints, its numeric fields are plain ints
or ``None`` (T-RAW-INT), and it holds no reference to the model or trace.

The first screen, in order (D10): (1) actual-path analytic forward FLOPs
under a NAMED FMA convention, true MACs, non-MAC FLOPs, batch/token scope
when derivable; (2) coverage counts, the evidence mix over knowns, named
unknowns, and lower-bound language that "COMPLETE, not a lower bound"
must EARN; (3) the parameter contract verbatim (unique / trainable /
frozen / executed / unexecuted, ties named); (4) the always-visible
backward STATUS line; (5) the 6ND comparator facts; (6) an optional
module breakdown from the same cost-tree rows. No timing, no throughput,
no MFU in the default report.

D11: the comparator's D is the HF-as-implemented numel of the main input
-- padding included (measured: D=numel matches the executed dense body
within 0.2%; mask-based D lands tens of percent low with batch-dependent
magnitude). Applicability mirrors HF's own: ``main_input_name`` present
and a parameter count available -- otherwise the line is OMITTED, never
guessed. D13: analytic FLOP counts are "actual-path analytic", never
"measured".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from .._errors import InvalidArgumentError
from ._cost_backward import backward_status
from ._cost_tree import build_cost_tree
from ._factcore import factcore

if TYPE_CHECKING:
    from ..data_classes.trace import Trace


@dataclass(frozen=True)
class SixNDFacts:
    """The 6ND comparator block (D11): reproduces what HF Trainer logs.

    ``d_numel`` is the numel of the main input AS IMPLEMENTED by HF --
    padding included. ``two_nd`` (= 2*N*D) is the forward-body heuristic
    compared against the actual-path analytic forward
    (heuristic-vs-exact); ``six_nd`` (= 6*N*D) is the full training
    heuristic, printed as a fact, never compared against an executed
    quantity unless the hypothetical backward door was invoked.
    """

    n_params: int
    d_numel: int
    two_nd: int
    six_nd: int
    forward_delta_fraction: float | None

    def lines(self) -> list[str]:
        """Render the comparator facts."""

        delta = (
            "n/a"
            if self.forward_delta_fraction is None
            else f"{100.0 * self.forward_delta_fraction:+.3f}%"
        )
        return [
            f"6ND comparator (HF-as-implemented): N={self.n_params} params, "
            f"D={self.d_numel} main-input elements (pads counted)",
            f"  2ND forward-body heuristic = {self.two_nd} "
            f"(analytic forward vs 2ND: {delta}, heuristic-vs-exact)",
            f"  6ND training heuristic = {self.six_nd}",
        ]


@dataclass(frozen=True)
class FlopsReport:
    """The detached one-call cost report (D10). Raw ints or None only."""

    convention: str
    forward_flops: int | None
    true_macs: int | None
    non_mac_flops: int | None
    batch_size: int | None
    tokens_per_item: int | None
    coverage_known: int
    coverage_zero_by_rule: int
    coverage_not_applicable: int
    coverage_unknown: int
    unknown_op_names: tuple[str, ...]
    evidence_formula_exact: int
    evidence_estimated: int
    is_lower_bound: bool
    params_unique: int | None
    params_trainable: int | None
    params_frozen: int | None
    params_executed: int | None
    params_unexecuted: int | None
    tied_param_groups: tuple[tuple[str, ...], ...]
    backward_state: str
    backward_status_line: str
    six_nd: SixNDFacts | None
    breakdown: tuple[str, ...]

    def __repr__(self) -> str:
        """Bounded identity card (F10/D31): blocks point, never dump.

        The lower-bound honesty mark is never dropped: an incomplete
        coverage renders ``>=`` on the headline figure.
        """

        bound = ">=" if self.is_lower_bound else ""
        return (
            f"FlopsReport({self.convention}: forward_flops{bound}={self.forward_flops}, "
            f"{self.coverage_unknown} unknown op(s); print() for the report, "
            f"read .breakdown)"
        )

    def _coverage_lines(self) -> list[str]:
        """Block 2: coverage counts, evidence mix, earned completeness."""

        total_rows = (
            self.coverage_known
            + self.coverage_zero_by_rule
            + self.coverage_not_applicable
            + self.coverage_unknown
        )
        lines = [
            f"coverage: {self.coverage_known + self.coverage_zero_by_rule}/"
            f"{total_rows - self.coverage_not_applicable} compute rows covered "
            f"({self.coverage_zero_by_rule} zero-by-rule); "
            f"{self.coverage_unknown} unknown",
            f"  evidence over knowns: {self.evidence_formula_exact} formula-exact, "
            f"{self.evidence_estimated} estimated",
        ]
        if self.is_lower_bound:
            shown = ", ".join(self.unknown_op_names[:5])
            extra = (
                ""
                if len(self.unknown_op_names) <= 5
                else f" (+{len(self.unknown_op_names) - 5} more)"
            )
            lines.append(
                f"  totals are a LOWER BOUND: unknown-cost ops present ({shown}{extra}); "
                "remedy: torchlens.capture.flops.register_op_rule"
            )
        else:
            lines.append("  coverage is COMPLETE, not a lower bound (0 unknown-cost ops)")
        return lines

    def __str__(self) -> str:
        """Render the six-block first screen."""

        lines = [
            # D13: actual-path analytic, never "measured".
            f"forward compute (actual-path analytic, {self.convention} convention: "
            f"one MAC = 2 FLOPs): {self.forward_flops if self.forward_flops is not None else '-'}"
        ]
        if self.true_macs is not None:
            lines.append(f"  true MACs: {self.true_macs}")
        if self.non_mac_flops is not None:
            lines.append(f"  non-MAC FLOPs: {self.non_mac_flops}")
        if self.batch_size is not None:
            scope = f"  scope: batch={self.batch_size}"
            if self.tokens_per_item is not None:
                scope += f", tokens/item={self.tokens_per_item}"
            lines.append(scope)
        lines.extend(self._coverage_lines())
        param_bits = []
        for name, value in (
            ("unique", self.params_unique),
            ("trainable", self.params_trainable),
            ("frozen", self.params_frozen),
            ("executed", self.params_executed),
            ("unexecuted", self.params_unexecuted),
        ):
            if value is not None:
                param_bits.append(f"{value} {name}")
        lines.append("parameters: " + (", ".join(param_bits) if param_bits else "unavailable"))
        for group in self.tied_param_groups:
            lines.append(f"  tied: {' == '.join(group)}")
        lines.append(self.backward_status_line)
        if self.six_nd is not None:
            lines.extend(self.six_nd.lines())
        if self.breakdown:
            lines.append("breakdown (top module calls by exclusive known FLOPs):")
            lines.extend(f"  {line}" for line in self.breakdown)
        return "\n".join(lines)


def _input_scope(trace: Any) -> tuple[int | None, int | None]:
    """Derive (batch, tokens/item) from the first captured input, if clear."""

    for op in getattr(trace, "input_ops", ()) or ():
        shape = tuple(getattr(op, "shape", None) or ())
        if len(shape) >= 2:
            return int(shape[0]), int(shape[1])
        if len(shape) == 1:
            return int(shape[0]), None
    return None, None


def _main_input_numel(trace: Any, model: Any | None) -> int | None:
    """The 6ND D under HF's own applicability rule (D11), else None."""

    main_input_name = getattr(model, "main_input_name", None) if model is not None else None
    if not main_input_name:
        return None
    for op in getattr(trace, "input_ops", ()) or ():
        shape = getattr(op, "shape", None)
        if shape:
            numel = 1
            for dim in shape:
                numel *= int(dim)
            return numel
    return None


def _build_report(
    trace: Trace,
    *,
    model: Any | None,
    main_input_numel: int | None,
    breakdown_top_k: int,
) -> FlopsReport:
    """Assemble the detached report from the ONE aggregation (D10)."""

    core = factcore(trace)
    agg = core.compute
    macs_total = int(agg.macs_total)
    forward = int(agg.partition_total)
    non_mac = forward - 2 * macs_total if not agg.coverage.macs_unknown_split else None
    exact = sum(
        1 for row in agg.rows if row.coverage_class == "known" and row.evidence == "formula_exact"
    )
    estimated = sum(
        1 for row in agg.rows if row.coverage_class == "known" and row.evidence == "estimated"
    )
    batch, tokens = _input_scope(trace)
    d_numel = main_input_numel
    if d_numel is None:
        d_numel = _main_input_numel(trace, model)
    six_nd: SixNDFacts | None = None
    if d_numel is not None and core.params.total is not None:
        two_nd = 2 * core.params.total * d_numel
        six_nd = SixNDFacts(
            n_params=core.params.total,
            d_numel=d_numel,
            two_nd=two_nd,
            six_nd=6 * core.params.total * d_numel,
            forward_delta_fraction=((forward - two_nd) / two_nd if two_nd else None),
        )
    status = backward_status(trace)
    breakdown: tuple[str, ...] = ()
    if breakdown_top_k:
        tree = build_cost_tree(trace, top_k=breakdown_top_k, max_depth=1, aggregation=agg)
        breakdown = tuple(
            f"{row.label}: {row.self_flops if row.kind != 'root' else row.subtree_flops} FLOPs"
            for row in tree.rows
            if row.depth == 1 and (row.self_flops or 0) > 0
        )
    return FlopsReport(
        convention=agg.convention,
        forward_flops=forward,
        true_macs=macs_total,
        non_mac_flops=non_mac,
        batch_size=batch,
        tokens_per_item=tokens,
        coverage_known=agg.coverage.known,
        coverage_zero_by_rule=agg.coverage.zero_by_rule,
        coverage_not_applicable=agg.coverage.not_applicable,
        coverage_unknown=agg.coverage.unknown,
        unknown_op_names=agg.coverage.unknown_ops,
        evidence_formula_exact=exact,
        evidence_estimated=estimated,
        is_lower_bound=agg.coverage.unknown > 0,
        params_unique=core.params.total,
        params_trainable=core.params.trainable,
        params_frozen=core.params.frozen,
        params_executed=core.params.executed,
        params_unexecuted=core.params.unexecuted,
        tied_param_groups=core.params.tied_groups,
        backward_state=status.state,
        backward_status_line=status.status_line,
        six_nd=six_nd,
        breakdown=breakdown,
    )


def flops_report(
    subject: Any,
    input_args: Any = None,
    input_kwargs: dict[str, Any] | None = None,
    *,
    main_input_numel: int | None = None,
    breakdown_top_k: int = 5,
) -> FlopsReport:
    """Build the one-call cost report from a trace OR a model (D10).

    Parameters
    ----------
    subject:
        A finished :class:`~torchlens.data_classes.trace.Trace` (the
        retroactive door), or an ``nn.Module`` (the model door -- runs
        ONE capture under the summary execution contract: eval mode,
        ``torch.no_grad()``, module training flags and RNG state restored
        bit-identically afterwards).
    input_args / input_kwargs:
        Model-door inputs; refused on the trace door.
    main_input_numel:
        Explicit 6ND ``D`` for the trace door (the comparator's HF
        applicability cannot be established from a trace alone; the line
        is OMITTED without it, never guessed).
    breakdown_top_k:
        Depth-1 module rows in the optional breakdown block (0 disables).
    """

    import torch.nn as nn

    if isinstance(subject, nn.Module):
        if input_args is None:
            raise InvalidArgumentError(
                "the model door needs example inputs to run its one capture.",
                code="flops_report_inputs_required",
                remedy="Call tl.report.flops_report(model, example_inputs).",
            )
        trace = _capture_for_report(subject, input_args, input_kwargs)
        try:
            return _build_report(
                trace,
                model=subject,
                main_input_numel=main_input_numel,
                breakdown_top_k=breakdown_top_k,
            )
        finally:
            trace.cleanup()
    if input_args is not None or input_kwargs:
        raise InvalidArgumentError(
            "input_args/input_kwargs are model-door arguments; the trace door takes none.",
            code="flops_report_trace_door_inputs",
            remedy=(
                "Call tl.report.flops_report(trace) on a finished trace, or pass the model "
                "itself as the first argument to run the one-capture model door."
            ),
        )
    return _build_report(
        subject,
        model=None,
        main_input_numel=main_input_numel,
        breakdown_top_k=breakdown_top_k,
    )


def _capture_for_report(model: Any, input_args: Any, input_kwargs: dict[str, Any] | None) -> Any:
    """Run the model door's ONE capture under the summary execution contract."""

    import torch

    from ..user_funcs import trace as _trace
    from ..utils.rng import log_current_rng_states, set_rng_from_saved_states

    saved_training_flags = [(module, module.training) for module in model.modules()]
    rng_snapshot = log_current_rng_states()
    try:
        model.eval()
        with torch.no_grad():
            return _trace(model, input_args, input_kwargs or {})
    finally:
        for module, was_training in saved_training_flags:
            module.training = was_training
        set_rng_from_saved_states(rng_snapshot)
