"""One-call health reports assembled from trace-local debug diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import torch

from ._common import _compute_ops
from ._gradients import gradient_flow_audit
from ._nan import _nonfinite_kind, bisect_nan, find_nan_in_trace

if TYPE_CHECKING:
    from torchlens.data_classes.trace import Trace
    from torchlens.partial import PartialTrace


AuditSeverity = Literal["critical", "warning", "info"]
_SEVERITY_ORDER = {"critical": 0, "warning": 1, "info": 2}


@dataclass(frozen=True)
class AuditFinding:
    """One prioritized health finding from :func:`audit_trace`.

    Parameters
    ----------
    severity:
        Priority assigned to the finding.
    check:
        Name of the diagnostic that produced the finding.
    message:
        Human-readable result summary.
    ops:
        Offending operation labels.
    modules:
        Offending module addresses.
    follow_up:
        Call users can run to inspect the issue further.
    """

    severity: AuditSeverity
    check: str
    message: str
    ops: tuple[str, ...]
    modules: tuple[str, ...]
    follow_up: str


@dataclass(frozen=True)
class TraceAudit:
    """Notebook-friendly report from trace-local health diagnostics.

    Parameters
    ----------
    findings:
        Severity-ordered findings.
    checks_run:
        Names of diagnostics that ran.
    skipped:
        ``(check, reason)`` entries for diagnostics not supported by the capture.
    """

    findings: tuple[AuditFinding, ...]
    checks_run: tuple[str, ...]
    skipped: tuple[tuple[str, str], ...]

    def __repr__(self) -> str:
        """Render a compact audit suitable for notebooks.

        Returns
        -------
        str
            Health summary with findings and skipped-check reasons.
        """

        heading = (
            f"TraceAudit: {len(self.findings)} issue(s); {len(self.checks_run)} checks run, "
            f"{len(self.skipped)} skipped"
        )
        if not self.findings:
            heading = (
                f"TraceAudit: no issues found; {len(self.checks_run)} checks run, "
                f"{len(self.skipped)} skipped"
            )
        lines = [heading]
        lines.extend(
            f"- [{finding.severity}] {finding.check}: {finding.message} "
            f"Follow up: {finding.follow_up}"
            for finding in self.findings
        )
        lines.extend(f"- skipped {check}: {reason}" for check, reason in self.skipped)
        return "\n".join(lines)


def _has_full_saved_activations(trace: Trace) -> bool:
    """Return whether every compute operation retains an output payload.

    Parameters
    ----------
    trace:
        Completed trace to inspect.

    Returns
    -------
    bool
        Whether payload-dependent checks can cover the complete computation.
    """

    return all(bool(getattr(op, "has_saved_activation", False)) for op in _compute_ops(trace))


def _has_saved_gradients(trace: Trace) -> tuple[bool, str | None]:
    """Determine whether a trace supports a gradient-flow check.

    Parameters
    ----------
    trace:
        Completed trace to inspect.

    Returns
    -------
    tuple[bool, str | None]
        Support flag and a skip reason when unsupported.
    """

    try:
        if len(trace.backward_passes) == 0:
            return False, "forward-only trace; no backward pass was captured"
        if len(trace.saved_grad_ops) == 0:
            return False, "no saved gradients; re-trace with save_grads=True and log_backward()"
    except ValueError as exc:
        return False, str(exc)
    return True, None


def _audit_partial_trace(partial: PartialTrace) -> TraceAudit:
    """Audit a failed partial capture without full-trace assumptions.

    Parameters
    ----------
    partial:
        Partial capture and its original exception.

    Returns
    -------
    TraceAudit
        Evidence-backed exception/non-finite findings and explicit skipped scope.
    """

    raw_layers = partial.raw_layers
    last = raw_layers[-1] if raw_layers else None
    last_label = (
        str(getattr(last, "_label_raw", getattr(last, "_layer_label_raw", "unknown")))
        if last is not None
        else "unknown"
    )
    modules = tuple(
        str(module)
        for module in (getattr(last, "module_call_stack", ()) if last is not None else ())
    )
    exception = partial.original_exception
    findings = [
        AuditFinding(
            severity="critical",
            check="partial_capture_exception",
            message=f"{type(exception).__name__}: {exception}",
            ops=(last_label,) if last is not None else (),
            modules=modules,
            follow_up="tl.report.explain(partial, format='json')",
        )
    ]
    nonfinite = partial.first_nonfinite()
    checks_run = ["partial_capture_exception", "find_nan"]
    if not nonfinite.startswith("No non-finite"):
        nonfinite_op = next(
            (
                op
                for op in raw_layers
                if isinstance((output := getattr(op, "out", None)), torch.Tensor)
                and _nonfinite_kind(output) != "none"
            ),
            None,
        )
        nonfinite_label = (
            str(
                getattr(
                    nonfinite_op,
                    "_label_raw",
                    getattr(nonfinite_op, "_layer_label_raw", "unknown"),
                )
            )
            if nonfinite_op is not None
            else "unknown"
        )
        nonfinite_modules = tuple(
            str(module) for module in getattr(nonfinite_op, "module_call_stack", ())
        )
        findings.append(
            AuditFinding(
                severity="critical",
                check="find_nan",
                message=nonfinite,
                ops=(nonfinite_label,) if nonfinite_op is not None else (),
                modules=nonfinite_modules,
                follow_up="partial.first_nonfinite()",
            )
        )
    reason = "partial capture did not complete full-trace postprocessing"
    skipped = tuple(
        (check, reason)
        for check in (
            "bisect_nan",
            "compare",
            "dead_neurons",
            "gradient_flow_audit",
            "hot_path",
            "infer_input_shape",
            "lineage",
            "recompute_candidates",
        )
    )
    findings.sort(
        key=lambda finding: (_SEVERITY_ORDER[finding.severity], finding.check, finding.ops)
    )
    return TraceAudit(tuple(findings), tuple(checks_run), skipped)


def audit_trace(trace: Trace | PartialTrace) -> TraceAudit:
    """Run every trace-local health diagnostic supported by one capture.

    Health checks that run and can contribute findings are ``find_nan``,
    ``bisect_nan`` (full-coverage traces), ``dtype_range_audit``, and
    ``gradient_flow_audit`` (exactly one captured backward pass). Every other
    diagnostic is listed in ``skipped`` with a reason and is NOT counted as a
    check that ran: ``compare`` (a second trace), ``lineage`` (a selected start
    op), ``infer_input_shape`` (a fresh execution), ``dead_neurons`` (single-
    trace inactivity is an insufficient sample, not a per-trace health verdict),
    a multi-backward ``gradient_flow_audit`` (needs a ``bwd=`` pass selection),
    and the ``hot_path`` / ``recompute_candidates`` performance rankings (not
    health checks). Sparse traces run ``find_nan`` over their saved outputs and
    report its uncertainty zone; ``bisect_nan`` needs complete payload coverage
    and is otherwise skipped.

    Parameters
    ----------
    trace:
        Completed TorchLens trace or failed :class:`PartialTrace`.

    Returns
    -------
    TraceAudit
        Severity-ordered findings, executed checks, and honest skip reasons.
    """

    from torchlens.partial import PartialTrace

    if isinstance(trace, PartialTrace):
        return _audit_partial_trace(trace)

    findings: list[AuditFinding] = []
    checks_run: list[str] = []
    skipped: list[tuple[str, str]] = [
        ("compare", "requires a second trace"),
        ("lineage", "requires a selected starting operation"),
        ("infer_input_shape", "requires a model and a new probe execution"),
    ]
    # Op labels already reported non-finite, so a second non-finite check does
    # not double-report the same op.
    flagged_nonfinite: set[str] = set()

    result = find_nan_in_trace(trace)
    checks_run.append("find_nan")
    if result.found:
        if result.label is not None:
            flagged_nonfinite.add(result.label)
        findings.append(
            AuditFinding(
                severity="critical",
                check="find_nan",
                message=result.message,
                ops=(result.label,) if result.label is not None else (),
                modules=(result.module_address,) if result.module_address is not None else (),
                follow_up="trace.find_nan()",
            )
        )

    full_payloads = _has_full_saved_activations(trace)
    if full_payloads:
        # bisect_nan is a real health check: consult its result (do not discard
        # it) and surface a finding, deduped against find_nan by op label.
        bisect = bisect_nan(trace)
        checks_run.append("bisect_nan")
        if bisect.found and bisect.label is not None and bisect.label not in flagged_nonfinite:
            flagged_nonfinite.add(bisect.label)
            findings.append(
                AuditFinding(
                    severity="critical",
                    check="bisect_nan",
                    message=bisect.message,
                    ops=(bisect.label,),
                    modules=(),
                    follow_up="tl.debug.bisect_nan(trace)",
                )
            )
    else:
        skipped.append(
            ("bisect_nan", "selective-save trace does not retain every compute activation")
        )

    # dead_neurons over ONE trace is an insufficient-sample signal, not a
    # per-trace health verdict (see dead_neurons' own docstring). Counting it as
    # a check that ran manufactured coverage ("no issues found; N checks run")
    # even on a fully dead model; report it as skipped with an actionable reason
    # instead of a discarded result.
    skipped.append(
        (
            "dead_neurons",
            "single-trace inactivity is an insufficient-sample signal, not a per-trace health "
            "verdict; run tl.debug.dead_neurons(trace) across a representative batch",
        )
    )

    # dtype_range_audit is a trace-local health check (non-finite, range, and
    # precision hazards over saved activations). It was previously neither run
    # nor listed as skipped -- the docstring's completeness claim was a lie.
    # Deferred import: _dtype_range imports this module.
    from ._dtype_range import dtype_range_audit

    dtype_result = dtype_range_audit(trace)
    checks_run.append("dtype_range_audit")
    for dtype_finding in dtype_result.findings:
        if dtype_finding.check == "dtype_nonfinite" and set(dtype_finding.ops) & flagged_nonfinite:
            # Already reported by find_nan / bisect_nan for the same op.
            continue
        findings.append(dtype_finding)

    has_gradients, gradient_reason = _has_saved_gradients(trace)
    if not has_gradients:
        skipped.append(("gradient_flow_audit", gradient_reason or "saved gradients unavailable"))
    else:
        num_backward = len(trace.backward_passes)
        if num_backward > 1:
            # gradient_flow_audit refuses (empty frame) without a pass selection;
            # that is a skip, not a check that ran and validated health.
            skipped.append(
                (
                    "gradient_flow_audit",
                    f"{num_backward} backward passes captured; select one with "
                    "tl.debug.gradient_flow_audit(trace, bwd=...)",
                )
            )
            skipped.append(
                (
                    "bisect_nan_backward",
                    f"{num_backward} backward passes captured; select one with "
                    "tl.debug.bisect_nan_backward(trace, bwd=...)",
                )
            )
        else:
            frame = gradient_flow_audit(trace)
            checks_run.append("gradient_flow_audit")
            for _, row in frame[frame["severity"] > 0].iterrows():
                findings.append(
                    AuditFinding(
                        severity="critical" if bool(row["exploding"]) else "warning",
                        check="gradient_flow_audit",
                        message=str(row["reason"] or "gradient-flow anomaly"),
                        ops=(str(row["op"]),),
                        modules=(),
                        follow_up="tl.debug.gradient_flow_audit(trace)",
                    )
                )
            # The backward NaN bisector runs automatically exactly here: ONE
            # captured backward with saved gradient payloads is unambiguous
            # (observe item 10). Its birth headline supersedes the carrier
            # observations the flow audit reports for the same pass.
            from ._nan_backward import bisect_nan_backward

            backward_bisect = bisect_nan_backward(trace)
            checks_run.append("bisect_nan_backward")
            if backward_bisect.found:
                headline = backward_bisect.first_bad_thing
                findings.append(
                    AuditFinding(
                        severity="critical",
                        check="bisect_nan_backward",
                        message=backward_bisect.message,
                        ops=(headline.label,) if headline.label else (),
                        modules=(headline.module,) if headline.module else (),
                        follow_up="tl.debug.bisect_nan_backward(trace)",
                    )
                )

    # hot_path and recompute_candidates are performance rankings, not health
    # checks: they never establish a model-health issue, so they are honestly
    # listed as skipped rather than counted as health checks that ran.
    skipped.append(
        ("hot_path", "performance ranking, not a health check; run tl.debug.hot_path(trace)")
    )
    skipped.append(
        (
            "recompute_candidates",
            "performance ranking, not a health check; run tl.debug.recompute_candidates(trace)",
        )
    )
    findings.sort(
        key=lambda finding: (_SEVERITY_ORDER[finding.severity], finding.check, finding.ops)
    )
    return TraceAudit(tuple(findings), tuple(checks_run), tuple(skipped))
