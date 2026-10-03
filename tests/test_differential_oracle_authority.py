"""The differential oracle's exception authority is the harness, not the subject.

b9-sol R75-1: ``align_streams`` excused mode events against the wrapper's own
live ``funcs_not_to_log`` export, so a capture regression that ADDS an op to
that table was excused by the oracle it should have tripped. These tests pin
the fix: excusal keys on the harness's ``PINNED_NOT_LOGGED`` and any set
drift between pin and live table is a loud ``NOT_LOGGED_AUTHORITY_DRIFT``
mismatch. Pure stream-alignment unit tests — no subprocess, no model build.
"""

import importlib.util
import sys
from pathlib import Path

_TOOL = Path(__file__).resolve().parent.parent / "tools" / "differential_capture.py"
_spec = importlib.util.spec_from_file_location("differential_capture", _TOOL)
dc = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("differential_capture", dc)
_spec.loader.exec_module(dc)


def _mode_events(names):
    return [{"name": n, "unmapped_repr": None} for n in names]


def _wrapper_result(ops, not_logged=None, expansions=None):
    return {
        "side": "wrapper",
        "model": "unit",
        "ops": list(ops),
        "funcs_not_to_log": sorted(dc.PINNED_NOT_LOGGED if not_logged is None else not_logged),
        "composite_expansions": expansions or {},
    }


def _mode_result(names):
    return {"side": "mode", "model": "unit", "events": _mode_events(names)}


def test_pinned_excusal_still_aligns():
    """Genuinely-unlogged protocol traffic is excused under the pin."""
    report = dc.align_streams(
        _mode_result(["mm", "size", "relu"]),
        _wrapper_result(["mm", "relu"]),
    )
    assert report["matched"], report["mismatches"]
    rules = [row["rule"] for row in report["ledger"]]
    assert rules == ["EXACT", "NOT_LOGGED", "EXACT"]


def test_subject_added_exemption_is_not_excused():
    """THE PLANT (b9-sol R75-1): a regression adds ``mm`` to the subject's
    ``funcs_not_to_log`` and stops logging it. The old oracle excused the
    orphaned mode event; the fixed oracle must fail BOTH ways — authority
    drift named, and the mm event not consumed by NOT_LOGGED."""
    report = dc.align_streams(
        _mode_result(["mm", "relu"]),
        _wrapper_result(["relu"], not_logged=sorted(dc.PINNED_NOT_LOGGED | {"mm"})),
    )
    assert not report["matched"]
    kinds = {row["kind"] for row in report["mismatches"]}
    assert "NOT_LOGGED_AUTHORITY_DRIFT" in kinds
    drift = next(row for row in report["mismatches"] if row["kind"] == "NOT_LOGGED_AUTHORITY_DRIFT")
    assert drift["added_by_subject"] == ["mm"]
    # The orphaned mode event surfaces as a real mismatch, never NOT_LOGGED.
    assert not any(row["rule"] == "NOT_LOGGED" and row["name"] == "mm" for row in report["ledger"])
    assert "NAME_MISMATCH" in kinds or "WRAPPER_STREAM_EXHAUSTED" in kinds


def test_subject_dropped_exemption_is_drift():
    """A pin entry the subject no longer declares is stale-pin drift: loud,
    reviewable, and the name is no longer excusable (intersection excusal
    keeps the streams honest while the pin is re-reviewed)."""
    live = sorted(dc.PINNED_NOT_LOGGED - {"size"})
    report = dc.align_streams(
        _mode_result(["size", "relu"]),
        _wrapper_result(["size", "relu"], not_logged=live),
    )
    assert not report["matched"]
    drift = next(row for row in report["mismatches"] if row["kind"] == "NOT_LOGGED_AUTHORITY_DRIFT")
    assert drift["missing_from_subject"] == ["size"]
    # The now-logged op still pairs EXACT through the intersection excusal.
    assert any(row["rule"] == "EXACT" and row["name"] == "size" for row in report["ledger"])


def test_shared_root_composite_omission_is_not_excused():
    """THE PLANT (b9-sol R75-1 round 4): a systematic wrapper omission.

    ``__add__`` is dropped from BOTH wrapper traces — the model stream AND
    the one-op probe that derives the composite expansion — because both run
    through the same wrappers. The pre-fix oracle trusted the subject-derived
    span as the expectation, so the identically-corrupted sides aligned
    ``matched=True``. Expansion now keys on the hand-reviewed
    ``PINNED_COMPOSITE_EXPANSIONS``: the corrupted stream mismatches the pin
    AND the corrupted probe derivation surfaces as authority drift.
    """

    report = dc.align_streams(
        _mode_result(["softsign"]),
        _wrapper_result(
            ["__abs__", "__truediv__"],
            expansions={"softsign": ["__abs__", "__truediv__"]},
        ),
    )
    assert not report["matched"], "the shared-root omission plant must FAIL alignment"
    kinds = {row["kind"] for row in report["mismatches"]}
    assert "COMPOSITE_AUTHORITY_DRIFT" in kinds
    drift = next(r for r in report["mismatches"] if r["kind"] == "COMPOSITE_AUTHORITY_DRIFT")
    assert drift["pinned_span"] == ["__abs__", "__add__", "__truediv__"]
    assert drift["derived_span"] == ["__abs__", "__truediv__"]
    assert "COMPOSITE_SPAN_MISMATCH" in kinds, "the stream must be compared against the PIN"


def test_unpinned_subject_composite_is_authority_drift():
    """A composite the subject declares but the pin does not know fails loudly.

    Without this arm a NEW composite family would silently fall back to the
    tautological subject-derived expectation.
    """

    report = dc.align_streams(
        _mode_result(["gelu_tanh"]),
        _wrapper_result(
            ["__mul__", "tanh"],
            expansions={"gelu_tanh": ["__mul__", "tanh"]},
        ),
    )
    assert not report["matched"]
    drift = next(r for r in report["mismatches"] if r["kind"] == "COMPOSITE_AUTHORITY_DRIFT")
    assert drift["name"] == "gelu_tanh"
    assert drift["pinned_span"] is None


def test_pinned_composite_span_still_aligns():
    """A healthy capture aligns: derived span equals the pin, stream matches."""

    report = dc.align_streams(
        _mode_result(["softsign", "relu"]),
        _wrapper_result(
            ["__abs__", "__add__", "__truediv__", "relu"],
            expansions={"softsign": ["__abs__", "__add__", "__truediv__"]},
        ),
    )
    assert report["matched"], report["mismatches"]
    assert report["ledger"][0]["rule"] == "COMPOSITE_EXPANSION"


def test_pin_matches_live_wrapper_table():
    """The pin tracks the real inventory through a REVIEWED harness edit:
    when the subject's table legitimately changes, this test is the reviewed
    place that changes with it."""
    from torchlens.backends.torch import wrappers

    assert set(wrappers.funcs_not_to_log) == set(dc.PINNED_NOT_LOGGED)
