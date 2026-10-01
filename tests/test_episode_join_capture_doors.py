"""The capture-door inventory lint (lane F40c, XS; foldA MEMO s5 item 8).

The one-door law (trace-verb verdict section 1): ``tl.trace`` is the ONLY
public live-capture door to a ``Trace``. This lint enumerates every root
callable whose return type mentions ``Trace`` (classes count by name) and
asserts the set equals the declared allowlist below, with a stated reason
per entry -- so a new capture door can never ship silently.

Growing this allowlist is a REVIEWED CONTRACT CHANGE: a new entry must be a
conversion, a loader, or product-bound replay, never a second live-capture
verb (new capture verbs are earned only by a new product).
"""

from __future__ import annotations

import inspect

import pytest

import torchlens as tl

pytestmark = pytest.mark.smoke

#: name -> the stated reason it may mention Trace without being a second
#: live-capture door.
CAPTURE_DOOR_ALLOWLIST: dict[str, str] = {
    "trace": "THE one public live-capture door (trace-verb verdict, the rule)",
    "load": "deserializes saved artifacts; loading, not live capture",
    "merge_ranks": (
        "stitches SAVED rank cores into a MergedTrace presenter; merging, "
        "not live capture (its name merely mentions Trace)"
    ),
    "push": "product-bound replay over an EXISTING capture; no new capture",
    "push_from": "product-bound replay over an EXISTING capture; no new capture",
    "run": "product-bound re-execution of an EXISTING capture's provider",
    "Trace": "the product class itself; constructing one records nothing",
    "ReentrantTraceError": "an exception class whose name mentions Trace",
}


def _root_callables_mentioning_trace() -> dict[str, str]:
    """Every ``tl.__all__`` callable whose return type mentions Trace."""

    found: dict[str, str] = {}
    for name in sorted(tl.__all__):
        obj = getattr(tl, name)
        if inspect.isclass(obj):
            if "Trace" in obj.__name__:
                found[name] = f"class {obj.__name__}"
            continue
        if not callable(obj):
            continue
        try:
            signature = inspect.signature(obj)
        except (TypeError, ValueError):
            continue
        annotation = signature.return_annotation
        if annotation is inspect.Signature.empty:
            continue
        if "Trace" in str(annotation):
            found[name] = str(annotation)
    return found


def test_capture_door_inventory_matches_the_declared_allowlist() -> None:
    found = _root_callables_mentioning_trace()
    unexpected = set(found) - set(CAPTURE_DOOR_ALLOWLIST)
    missing = set(CAPTURE_DOOR_ALLOWLIST) - set(found)
    assert not unexpected, (
        f"NEW root callable(s) whose return type mentions Trace: "
        f"{ {name: found[name] for name in sorted(unexpected)} }. "
        "tl.trace is the only live-capture door; if this entry is a "
        "conversion/loader/product-bound replay, add it to "
        "CAPTURE_DOOR_ALLOWLIST with its stated reason (a reviewed contract "
        "change), otherwise it must not ship."
    )
    assert not missing, (
        f"allowlisted capture-door entries no longer found at root: "
        f"{sorted(missing)}; prune the allowlist with the removal."
    )


def test_every_allowlist_entry_states_a_reason() -> None:
    for name, reason in CAPTURE_DOOR_ALLOWLIST.items():
        assert isinstance(reason, str) and len(reason) >= 20, (
            f"allowlist entry {name!r} needs a substantive stated reason"
        )


def test_record_is_not_a_trace_door() -> None:
    """``tl.record`` returns a Recording; ``Recording.to_trace()`` is a
    conversion on the product, not a second live-capture door."""

    annotation = str(inspect.signature(tl.record).return_annotation)
    assert "Trace" not in annotation
    assert "Recording" in annotation
