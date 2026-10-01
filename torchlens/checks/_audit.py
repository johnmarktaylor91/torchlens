"""``audit_params`` -- the one-shot parameter/buffer-space scan (memo D15).

Mapping-only, no policy: an ``nn.Module`` or ANY name->tensor Mapping enters
through the same door (state_dicts, flattened optimizer state, EMA shadows,
loaded checkpoints, shards -- the same six-line recipe through the same
kernel). No ``grads=``, no ``optimizer=``: transient gradients are
phase-sensitive (a checkpoint verdict must not depend on WHEN you called
it), and the caller who names the mapping owns the phase label.

Knob parity with ``tl.debug.dtype_range_audit`` is pinned by test against
:mod:`torchlens.checks._constants` (one constants module, memo 4.4).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import torch

from ._constants import MAX_FRACTION_DEFAULT, SUBNORMAL_FRACTION_THRESHOLD_DEFAULT
from ._errors import CheckConfigError
from ._records import CHECK_REPORT_SCHEMA_VERSION, CheckFinding, severity_sorted
from ._scan import ScanRow, named_entries_from_target, scan_named_tensors

__tl_layer__ = "L5"


@dataclass(frozen=True)
class ParamAudit:
    """Notebook-friendly parameter-space audit (mirrors ``TraceAudit``).

    ``rows`` carries one entry per unique tensor (tied aliases preserved);
    ``skipped`` the ``(name, reason)`` pairs for unscannable tensors;
    ``coverage`` the honest scope disclosure -- verdicts are
    ``rank_local_shard``-scoped under torch.distributed, and global
    verdicts need an explicit opt-in reduce (not shipped in v1).
    """

    findings: tuple[CheckFinding, ...]
    rows: tuple[ScanRow, ...]
    checks_run: tuple[str, ...]
    skipped: tuple[tuple[str, str], ...]
    coverage: dict[str, Any] = field(default_factory=dict)
    schema_version: int = CHECK_REPORT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Return the versioned JSON-serializable audit payload."""

        return {
            "schema_version": self.schema_version,
            "findings": [finding.to_dict() for finding in self.findings],
            "rows": [row.to_dict() for row in self.rows],
            "checks_run": list(self.checks_run),
            "skipped": [list(entry) for entry in self.skipped],
            "coverage": self.coverage,
        }

    def to_json(self, *, indent: int | None = 1) -> str:
        """Serialize :meth:`to_dict` as JSON text."""

        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    def __repr__(self) -> str:
        """Render a compact audit summary suitable for notebooks."""

        heading = (
            f"ParamAudit: {len(self.findings)} finding(s) over {len(self.rows)} tensor(s); "
            f"{len(self.checks_run)} checks run, {len(self.skipped)} skipped"
        )
        if not self.findings:
            heading = (
                f"ParamAudit: no issues found over {len(self.rows)} tensor(s); "
                f"{len(self.checks_run)} checks run, {len(self.skipped)} skipped"
            )
        lines = [heading]
        lines.extend(
            f"- [{finding.severity}] {finding.check}: {finding.message}"
            for finding in self.findings
        )
        lines.extend(f"- skipped {name}: {reason}" for name, reason in self.skipped)
        return "\n".join(lines)


def _validate_fraction(name: str, value: float) -> None:
    """Refuse typed when a fraction knob leaves ``[0, 1]``."""

    if not 0.0 <= value <= 1.0:
        raise CheckConfigError(
            f"{name} must be in [0, 1], got {value!r}.",
            code="check_fraction_invalid",
            field=name,
            value=value,
            remedy=f"Pass {name} between 0.0 and 1.0.",
        )


def _validate_names(
    kwarg: str,
    names: Any,
    known: set[str],
    code: str,
) -> list[str]:
    """Validate exact qualified names against the inventory, refusing typed.

    Unknown names REFUSE (memo 4.4): a silently ignored name is the
    silent-no-op defect class the checks kit exists to kill.
    """

    requested = [str(name) for name in names]
    unknown = sorted(set(requested) - known)
    if unknown:
        sample = ", ".join(sorted(known)[:8])
        raise CheckConfigError(
            f"{kwarg} names not present in the audited target: {unknown}. "
            f"Known names include: {sample}{', ...' if len(known) > 8 else ''}.",
            code=code,
            unknown_names=unknown,
            remedy=f"Use exact qualified names from the target (fix or drop {unknown[0]!r}).",
        )
    return requested


def _validate_bounds(
    bounds: Mapping[str, tuple[float | None, float | None]],
) -> None:
    """Refuse typed when a bounds pair is malformed."""

    for name, pair in bounds.items():
        if not isinstance(pair, tuple) or len(pair) != 2:
            raise CheckConfigError(
                f"bounds[{name!r}] must be a (low, high) tuple with either side "
                f"None for one-sided bounds, got {pair!r}.",
                code="check_bounds_invalid",
                name=name,
                remedy="Pass bounds as {'layer.weight': (low, high)} with None for open sides.",
            )
        low, high = pair
        if low is not None and high is not None and low > high:
            raise CheckConfigError(
                f"bounds[{name!r}] has low={low!r} > high={high!r}.",
                code="check_bounds_invalid",
                name=name,
                remedy="Swap the bounds so low <= high.",
            )


def _row_findings(
    row: ScanRow,
    *,
    bounds: Mapping[str, tuple[float | None, float | None]] | None,
    max_fraction: float,
    subnormal_fraction_threshold: float,
) -> list[CheckFinding]:
    """Score one scanned row into audit findings (memo 4.4 severities)."""

    findings: list[CheckFinding] = []
    names = (row.name, *row.aliases)
    if row.n_nonfinite:
        findings.append(
            CheckFinding(
                check="param_nonfinite",
                code="param_value_nonfinite",
                severity="critical",
                action="collect",
                message=(
                    f"{row.name} ({row.kind}) holds nonfinite values: "
                    f"{row.n_nan} NaN, {row.n_posinf} +inf, {row.n_neginf} -inf "
                    f"of {row.numel} elements. Every later step on corrupt "
                    "state is wasted compute."
                ),
                names=names,
                evidence="scan_kernel",
                values={
                    "n_nan": float(row.n_nan),
                    "n_posinf": float(row.n_posinf),
                    "n_neginf": float(row.n_neginf),
                },
                remedy="Reload the last healthy checkpoint; arm the scheduled scan to catch the minting step.",
                follow_up="torchlens.checks session: register_nonfinite_param_check(every=1)",
            )
        )
    if bounds and row.name in bounds and ((row.bounds_below or 0) + (row.bounds_above or 0)) > 0:
        low, high = bounds[row.name]
        findings.append(
            CheckFinding(
                check="param_bounds",
                code="param_bounds_violated",
                severity="warning",
                action="collect",
                message=(
                    f"{row.name} violates declared bounds ({low}, {high}): "
                    f"{row.bounds_below} below, {row.bounds_above} above."
                ),
                names=names,
                evidence="scan_kernel",
                values={
                    "below": float(row.bounds_below or 0),
                    "above": float(row.bounds_above or 0),
                },
                remedy="Inspect the named tensor; bounds are the caller's declared semantic range.",
            )
        )
    if (
        row.abs_max is not None
        and row.dtype_max is not None
        and row.abs_max >= max_fraction * row.dtype_max
    ):
        findings.append(
            CheckFinding(
                check="dtype_headroom",
                code="param_dtype_headroom",
                severity="warning",
                action="collect",
                message=(
                    f"{row.name} abs max {row.abs_max:.6g} is within "
                    f"{(1 - max_fraction) * 100:.0f}% of the {row.dtype} ceiling "
                    f"{row.dtype_max:.6g}; the next growth step can overflow."
                ),
                names=names,
                evidence="scan_kernel",
                values={"abs_max": row.abs_max, "dtype_max": row.dtype_max},
                remedy="Rescale, clip, or move the tensor to a wider dtype.",
            )
        )
    if (
        row.subnormal_fraction is not None
        and row.subnormal_fraction > 0.0
        and row.subnormal_fraction >= subnormal_fraction_threshold
    ):
        findings.append(
            CheckFinding(
                check="param_subnormal",
                code="param_subnormal_heavy",
                severity="warning",
                action="collect",
                message=(
                    f"{row.name} has subnormal_fraction="
                    f"{row.subnormal_fraction:.6g} (threshold "
                    f"{subnormal_fraction_threshold}); values this small "
                    "lose precision and can silently flush to zero."
                ),
                names=names,
                evidence="scan_kernel",
                values={"subnormal_fraction": row.subnormal_fraction},
                remedy="Rescale the tensor or audit the upstream initialization/decay.",
            )
        )
    if row.zero_fraction == 1.0 and row.numel > 0:
        findings.append(
            CheckFinding(
                check="param_all_zero",
                code="param_all_zero",
                severity="info",
                action="collect",
                message=f"{row.name} is entirely zero ({row.numel} elements).",
                names=names,
                evidence="scan_kernel",
                values={"numel": float(row.numel)},
                remedy="Expected for fresh biases/counters; suspicious for trained weights.",
            )
        )
    elif row.all_same and row.numel > 1:
        findings.append(
            CheckFinding(
                check="param_all_same",
                code="param_all_same",
                severity="warning",
                action="collect",
                message=(
                    f"{row.name} holds one repeated value across {row.numel} "
                    "elements -- a suspicious constant for a trained tensor."
                ),
                names=names,
                evidence="scan_kernel",
                values={"numel": float(row.numel)},
                remedy="Check the initialization and whether the tensor ever receives updates.",
            )
        )
    return findings


def _coverage(rows: tuple[ScanRow, ...], skipped: int) -> dict[str, Any]:
    """Build the honest coverage disclosure for an audit."""

    scope = "process_local"
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        scope = "rank_local_shard"
    return {
        "scope": scope,
        "audited_tensors": sum(1 for row in rows if row.audited),
        "skipped_tensors": skipped,
        "audited_elements": sum(row.numel for row in rows if row.audited),
        "global_reduce": "not_applied",
    }


def audit_params(  # noqa: PLR0913 -- the memo-4.4 signature IS the designed public surface (mapping target + five audit knobs); bundling knobs would hide the dtype_range_audit parity
    target: Any,
    *,
    include_buffers: bool = True,
    within: Any | None = None,
    bounds: Mapping[str, tuple[float | None, float | None]] | None = None,
    max_fraction: float = MAX_FRACTION_DEFAULT,
    subnormal_fraction_threshold: float = SUBNORMAL_FRACTION_THRESHOLD_DEFAULT,
) -> ParamAudit:
    """Audit parameter/buffer space for numeric-health findings (memo 4.4).

    Parameters
    ----------
    target:
        ``nn.Module`` or any name->tensor ``Mapping`` (state_dict,
        flattened optimizer state, EMA shadow, loaded checkpoint).
    include_buffers:
        Whether registered buffers join a module inventory.
    within:
        Optional iterable of EXACT qualified names to audit (day-1
        spelling; selector sugar is a UI-sprint handoff). Unknown names
        refuse typed.
    bounds:
        Exact name -> ``(low, high)`` declared semantic bounds, either side
        ``None``. Unknown names refuse typed.
    max_fraction:
        Fraction of the dtype ceiling above which headroom warns (knob
        parity with ``tl.debug.dtype_range_audit``).
    subnormal_fraction_threshold:
        Minimum subnormal fraction that warns.

    Returns
    -------
    ParamAudit
        Findings, per-tensor rows, skip reasons, and coverage.

    Raises
    ------
    CheckConfigError
        On an invalid target (``check_target_invalid``), unknown ``within``
        names (``audit_within_unknown_name``), unknown or malformed
        ``bounds`` (``audit_bounds_unknown_name`` / ``check_bounds_invalid``),
        or out-of-range fraction knobs (``check_fraction_invalid``).
    """

    _validate_fraction("max_fraction", max_fraction)
    _validate_fraction("subnormal_fraction_threshold", subnormal_fraction_threshold)
    entries = named_entries_from_target(target, include_buffers=include_buffers)
    known = {name for name, _, _ in entries}
    if within is not None:
        requested = set(_validate_names("within", within, known, code="audit_within_unknown_name"))
        entries = [entry for entry in entries if entry[0] in requested]
    if bounds is not None:
        _validate_bounds(bounds)
        _validate_names("bounds", bounds.keys(), known, code="audit_bounds_unknown_name")

    rows = tuple(scan_named_tensors(entries, bounds=bounds))
    findings: list[CheckFinding] = []
    for row in rows:
        if not row.audited:
            continue
        findings.extend(
            _row_findings(
                row,
                bounds=bounds,
                max_fraction=max_fraction,
                subnormal_fraction_threshold=subnormal_fraction_threshold,
            )
        )
    skipped = tuple((row.name, row.skip_reason or "unscannable") for row in rows if not row.audited)
    checks_run = (
        "param_nonfinite",
        "param_bounds" if bounds else "param_bounds (no bounds declared)",
        "dtype_headroom",
        "param_subnormal",
        "param_all_same",
    )
    return ParamAudit(
        findings=severity_sorted(findings),
        rows=rows,
        checks_run=checks_run,
        skipped=skipped,
        coverage=_coverage(rows, len(skipped)),
    )


__all__ = ["ParamAudit", "audit_params"]
