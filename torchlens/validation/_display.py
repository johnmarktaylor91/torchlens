"""Validation printouts; independent of checks, tolerances and verdict production."""

from __future__ import annotations

from typing import Any

from .._utils._display_card import render_card
from .._utils._display_constructor import clipped, display_guard
from .._utils._display_numbers import format_count, format_exact_count


@display_guard
def status_repr(status: Any) -> str:
    """Keep the verdict and its source visible within one line."""

    lead = f"ValidationStatus: {status.state}, {status.backend}, {status.source}"
    if status.state in ("passed", "failed", "unverified"):
        lead += (
            f", {format_count(status.num_replayed_ops)} replayed, "
            f"{format_count(status.num_failed_ops)} failed"
        )
    return clipped(lead)


@display_guard
def status_card(status: Any) -> str:
    """Show exact counts and the unchanged availability explanation."""

    return render_card(
        status_repr(status),
        [
            ("available", str(status.is_available)),
            ("reason", status.reason or ""),
            ("message", status.message),
            ("replayed", format_exact_count(status.num_replayed_ops)),
            ("unverified", format_exact_count(status.num_unverified_ops)),
            ("failed", format_exact_count(status.num_failed_ops)),
        ],
        (".decisions", ".unverified_reason_counts", ".exempted_reason_counts"),
    )


@display_guard
def failure_repr(failure: Any) -> str:
    """Lead with the failing check and its location, with a bounded summary."""

    return clipped(f"ValidationFailure: {failure.summary()}")


@display_guard
def failure_card(failure: Any) -> str:
    """Expose recorded discrepancy details without changing their values."""

    return render_card(
        failure_repr(failure),
        [
            ("check", failure.check_name),
            ("op", failure.op_label or "not recorded"),
            ("function", failure.func_name or "not recorded"),
            ("message", failure.message),
            (
                "difference",
                f"max absolute {failure.max_abs_diff}, max relative {failure.max_rel_diff}",
            ),
        ],
        (".saved_shape", ".recomputed_shape", ".extra"),
    )
