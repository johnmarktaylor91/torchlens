"""The count-only mask entry hides only the nested-call exit's two counts.

``tools/field_diff_lib/count_only.py`` lets head show fewer runs of one capture
phase and fewer code-context cache entries than base. These tests plant the
accepted difference and every nearby one it must not hide: a count change in
another phase, a higher count, a changed timing field, a changed cache value,
a cache entry base does not hold, and a longer cache. Each planted fault must
still reach the comparison.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "tools"))
from field_diff_lib.compare import diff  # noqa: E402
from field_diff_lib.count_only import accept_count_only  # noqa: E402
from field_diff_lib.mask import COUNT_ONLY  # noqa: E402

_BUCKET = COUNT_ONLY["_phase_timings"]


def _stats(count: int) -> dict[str, Any]:
    """Return one canonical phase bucket, its clock reading already masked."""

    return {"__dict__": "dict", "items": [["total_s", "<float>"], ["count", count]]}


def _dump(build: int, other: int, cache: list[list[Any]]) -> dict[str, Any]:
    """Return a canonical dump holding one trace's two count-only fields and a public field."""

    return {
        "traces": {
            "trace": {
                "trace": {
                    "_phase_timings": {
                        "__dict__": "dict",
                        "items": [["ctx_emit", _stats(other)], [_BUCKET, _stats(build)]],
                    },
                    "_code_context_cache": {"__dict__": "dict", "items": cache},
                    "num_operations": 9,
                }
            }
        }
    }


_CACHE = [
    [{"__tuple__": ["a.py", 3]}, ["ctx-a"]],
    [{"__tuple__": ["b.py", 5]}, ["ctx-b"]],
    [{"__tuple__": ["c.py", 7]}, ["ctx-c"]],
]


def _compared(head: dict[str, Any]) -> list[tuple[str, Any, Any]]:
    """Return the differences left after the count-only acceptance, as the compare step runs it."""

    base = _dump(13, 4, _CACHE)
    return diff(base, accept_count_only(base, head))


def test_lower_phase_count_and_fewer_equal_cache_entries_are_hidden() -> None:
    """The nested-call exit's own difference: one phase runs less, the cache keeps a subset."""

    assert _compared(_dump(11, 4, [_CACHE[0], _CACHE[2]])) == []
    assert _compared(_dump(13, 4, _CACHE)) == []


def test_count_change_in_another_phase_still_shows() -> None:
    """Only the named bucket's count may drop; another phase's count is compared."""

    found = _compared(_dump(11, 3, _CACHE))
    assert [path for path, _, _ in found] == [
        "/traces/trace/trace/_phase_timings/items[0]/ctx_emit/items[1]/count",
        f"/traces/trace/trace/_phase_timings/items[1]/{_BUCKET}/items[1]/count",
    ]


def test_higher_count_or_changed_timing_field_still_shows() -> None:
    """The named bucket may only lose runs, and its other fields are compared."""

    assert _compared(_dump(14, 4, _CACHE)) != []
    head = _dump(11, 4, _CACHE)
    timings = head["traces"]["trace"]["trace"]["_phase_timings"]
    timings["items"][1][1]["items"][0][1] = 0.5
    assert _compared(head) != []
    missing = _dump(13, 4, _CACHE)
    del missing["traces"]["trace"]["trace"]["_phase_timings"]["items"][0]
    assert _compared(missing) != []


def test_changed_cache_entry_value_still_shows() -> None:
    """A cache entry head holds must equal base's entry under the same key."""

    changed = copy.deepcopy([_CACHE[0], _CACHE[2]])
    changed[1][1] = ["ctx-changed"]
    assert _compared(_dump(13, 4, changed)) != []


def test_new_or_extra_cache_entries_still_show() -> None:
    """An entry base does not hold, a repeated entry, or a longer cache is compared."""

    foreign = [_CACHE[0], [{"__tuple__": ["d.py", 9]}, ["ctx-d"]]]
    assert _compared(_dump(13, 4, foreign)) != []
    assert _compared(_dump(13, 4, [_CACHE[0], _CACHE[0]])) != []
    assert _compared(_dump(13, 4, [*_CACHE, [{"__tuple__": ["e.py", 1]}, ["ctx-e"]]])) != []


def test_other_fields_beside_the_counts_still_show() -> None:
    """Accepting the counts leaves every other field of the trace compared."""

    head = _dump(11, 4, [_CACHE[1]])
    head["traces"]["trace"]["trace"]["num_operations"] = 8
    found = _compared(head)
    assert [path for path, _, _ in found] == ["/traces/trace/trace/num_operations"]
