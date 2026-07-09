"""One-call health reports assembled from trace-local debug diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from ._common import _compute_ops
from ._cost import hot_path
from ._gradients import gradient_flow_audit
from ._graph import dead_neurons
from ._nan import bisect_nan, find_nan_in_trace
from ._recompute import recompute_candidates


if TYPE_CHECKING:
    from torchlens.data_classes.trace import Trace


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


def _has_full_saved_activations(trace: "Trace") -> bool:
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


def _has_saved_gradients(trace: "Trace") -> tuple[bool, str | None]:
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


def audit_trace(trace: "Trace") -> TraceAudit:
    """Run every trace-local health diagnostic supported by one capture.

    Diagnostics requiring a second trace, a selected start operation, or a
    fresh model execution are explicitly listed as skipped. Payload-dependent
    diagnostics are skipped when selective saving cannot cover all compute
    operations; this avoids treating a partial scan as a clean bill of health.

    Parameters
    ----------
    trace:
        Completed TorchLens trace.

    Returns
    -------
    TraceAudit
        Severity-ordered findings, executed checks, and honest skip reasons.
    """

    findings: list[AuditFinding] = []
    checks_run: list[str] = []
    skipped: list[tuple[str, str]] = [
        ("compare", "requires a second trace"),
        ("lineage", "requires a selected starting operation"),
        ("infer_input_shape", "requires a model and a new probe execution"),
    ]
    full_payloads = _has_full_saved_activations(trace)
    if full_payloads:
        result = find_nan_in_trace(trace)
        checks_run.append("find_nan")
        if result.found:
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
        bisect_nan(trace)
        checks_run.append("bisect_nan")
        dead_neurons(trace)
        checks_run.append("dead_neurons")
    else:
        reason = "selective-save trace does not retain every compute activation"
        skipped.extend([("find_nan", reason), ("bisect_nan", reason), ("dead_neurons", reason)])

    has_gradients, gradient_reason = _has_saved_gradients(trace)
    if has_gradients:
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
    else:
        skipped.append(("gradient_flow_audit", gradient_reason or "saved gradients unavailable"))

    # These trace-local rankings are useful contextual diagnostics but do not
    # themselves establish a model-health issue.
    hot_path(trace, by="flops")
    checks_run.append("hot_path")
    recompute_candidates(trace)
    checks_run.append("recompute_candidates")
    findings.sort(
        key=lambda finding: (_SEVERITY_ORDER[finding.severity], finding.check, finding.ops)
    )
    return TraceAudit(tuple(findings), tuple(checks_run), tuple(skipped))
