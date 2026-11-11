"""The count-only mask entry: the nested-call exit's two private bookkeeping counts.

A higher-level call whose single tensor output a nested call already logged
skips its own emission, so on head one capture phase runs fewer times and the
code-context cache holds fewer entries. ``mask.COUNT_ONLY`` names exactly
those two counts. Unlike the other mask rules this one compares base with
head, because the narrow form needs both sides:

* ``_phase_timings``: every bucket is compared as before, with the same
  buckets in the same order; only the ``count`` of the bucket ``COUNT_ONLY``
  names may be lower on head than on base.
* ``_code_context_cache``: head may hold fewer entries than base, and every
  entry head holds must equal an entry base holds under the same key. A
  changed key or value, an entry base does not hold, or more entries on head
  still shows.

Any other difference on these fields, and every other field, is compared as
before.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

from field_diff_lib.mask import COUNT_ONLY


def _items(canonical: Any) -> list[Any] | None:
    """Return a canonical mapping's ``[key, value]`` pairs, or ``None`` for another shape."""

    if not isinstance(canonical, dict) or not isinstance(canonical.get("items"), list):
        return None
    pairs = canonical["items"]
    if not all(isinstance(pair, list) and len(pair) == 2 for pair in pairs):
        return None
    return pairs


def _same_shell(base: dict[str, Any], head: dict[str, Any]) -> bool:
    """Return whether two canonical mappings agree on everything but their items."""

    return {k: v for k, v in base.items() if k != "items"} == {
        k: v for k, v in head.items() if k != "items"
    }


def phase_count_only_lower(base: Any, head: Any, bucket: str) -> bool:
    """Return whether ``head`` differs from ``base`` only by a lower ``count`` in ``bucket``."""

    left, right = _items(base), _items(head)
    if left is None or right is None or not _same_shell(base, head) or len(left) != len(right):
        return False
    lowered = False
    for (base_name, base_stats), (head_name, head_stats) in zip(left, right, strict=True):
        if base_name != head_name:
            return False
        if base_stats == head_stats:
            continue
        if base_name != bucket or not _count_lower(base_stats, head_stats):
            return False
        lowered = True
    return lowered


def _count_lower(base: Any, head: Any) -> bool:
    """Return whether two canonical stats mappings differ only by a lower ``count`` on head."""

    left, right = _items(base), _items(head)
    if left is None or right is None or not _same_shell(base, head):
        return False
    base_fields, head_fields = dict(left), dict(right)
    if [pair[0] for pair in left] != [pair[0] for pair in right]:
        return False
    base_count, head_count = base_fields.get("count"), head_fields.get("count")
    if type(base_count) is not int or type(head_count) is not int or head_count >= base_count:
        return False
    return {k: v for k, v in base_fields.items() if k != "count"} == {
        k: v for k, v in head_fields.items() if k != "count"
    }


def cache_entries_kept(base: Any, head: Any) -> bool:
    """Return whether ``head`` holds fewer entries than ``base``, each equal to one of base's."""

    left, right = _items(base), _items(head)
    if left is None or right is None or not _same_shell(base, head) or len(right) >= len(left):
        return False
    base_entries = Counter(json.dumps(pair, sort_keys=True) for pair in left)
    head_entries = Counter(json.dumps(pair, sort_keys=True) for pair in right)
    return not head_entries - base_entries


def _accepted(name: str, base: Any, head: Any) -> bool:
    """Return whether the value under ``name`` differs only as ``COUNT_ONLY`` allows."""

    bucket = COUNT_ONLY.get(name, "")
    if name == "_phase_timings":
        return phase_count_only_lower(base, head, bucket)
    if name == "_code_context_cache":
        return cache_entries_kept(base, head)
    return False


def accept_count_only(base: Any, head: Any) -> Any:
    """Return ``head`` with every ``COUNT_ONLY`` value that differs only as allowed set to base's.

    Walks both canonical dumps together through mappings with the same keys
    and lists of the same length; anything else is returned unchanged, so the
    comparison that follows still reports it.
    """

    if isinstance(base, dict) and isinstance(head, dict):
        walked = {}
        for key, entry in head.items():
            if key not in base:
                walked[key] = entry
            elif key in COUNT_ONLY and entry != base[key] and _accepted(key, base[key], entry):
                walked[key] = base[key]
            else:
                walked[key] = accept_count_only(base[key], entry)
        return walked
    if isinstance(base, list) and isinstance(head, list) and len(base) == len(head):
        return [accept_count_only(left, right) for left, right in zip(base, head, strict=True)]
    return head
