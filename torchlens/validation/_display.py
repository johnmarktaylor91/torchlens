"""Validation printouts; independent of checks, tolerances and verdict production."""

from __future__ import annotations

from typing import Any

from .._utils._display_card import one_line, render_card
from .._utils._display_constructor import clipped, display_guard
from .._utils._display_keys import display_key
from .._utils._display_numbers import format_count, format_exact_count, format_sig
from .._utils.fail_open import fail_open


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
            ("available", "available" if status.is_available else "unavailable"),
            ("reason", status.reason or ""),
            ("message", status.message),
            (
                "replay",
                f"{format_exact_count(status.num_replayed_ops)} replayed, "
                f"{format_exact_count(status.num_unverified_ops)} unverified, "
                f"{format_exact_count(status.num_failed_ops)} failed",
            ),
        ],
        (".decisions", ".unverified_reason_counts", ".exempted_reason_counts"),
    )


@display_guard
def failure_repr(failure: Any) -> str:
    """Lead with the failing check and its location, with a bounded summary."""

    location = _failure_location(failure)
    if failure.func_name:
        location += f", function {failure.func_name}"
    difference = (
        f"max absolute {format_sig(failure.max_abs_diff)}"
        if failure.max_abs_diff is not None
        else "difference not computed"
    )
    return clipped(
        one_line(
            f"ValidationFailure {failure.check_name}",
            [location, difference],
            [failure.message] if failure.message else [],
        )
    )


@display_guard
def failure_card(failure: Any) -> str:
    """Expose recorded discrepancy details without changing their values."""

    return render_card(
        failure_repr(failure),
        [
            ("check", failure.check_name),
            ("location", _failure_location(failure)),
            ("function", failure.func_name or "not recorded"),
            ("message", failure.message),
            (
                "gap",
                f"max absolute {_difference_text(failure.max_abs_diff)}, "
                f"max relative {_difference_text(failure.max_rel_diff)}",
            ),
        ],
        (".op_label", ".saved_shape", ".recomputed_shape", ".extra"),
    )


def _difference_text(value: float | None) -> str:
    """Keep an unrecorded discrepancy distinct from a measured zero."""

    return "not computed" if value is None else format_sig(value)


def _failure_location(failure: Any) -> str:
    return getattr(failure, "_display_op_key", None) or failure.op_label or "not recorded"


def bind_failure_key(failure: Any, trace: Any) -> None:
    """Retain only the resolved display string, never a Trace or tensor reference."""

    def bind() -> None:
        record = getattr(trace, "_layer_dict_all_keys", {}).get(failure.op_label)
        if record is not None:
            failure._display_op_key = display_key(record)

    fail_open(bind, lambda _error: None)
