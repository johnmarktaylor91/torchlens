"""Generated, closed Tier-R inventory (design-of-record section 2.4).

Walks the dataclass graph reachable from the record universe and fails on any
field the checked-in classification table does not cover — and on any table
entry whose field no longer exists. The inventory is therefore mechanically
closed in both directions; an unclassified new field is a build failure, not
a silent drift. ``OpRecord`` joins ``WALKER_ROOTS`` in P1.
"""

from __future__ import annotations

import dataclasses
import typing

from ._fields import ANNOTATION_NAMESPACE, TIER_CLASSIFICATION, WALKER_ROOTS


def _dataclass_types(annotation: object) -> list[type]:
    """Extract dataclass types from a (possibly nested) annotation."""

    result: list[type] = []
    origin = typing.get_origin(annotation)
    if origin is None:
        if isinstance(annotation, type) and dataclasses.is_dataclass(annotation):
            result.append(annotation)
        return result
    for argument in typing.get_args(annotation):
        result.extend(_dataclass_types(argument))
    return result


def _walk() -> dict[str, list[str]]:
    """Return {ClassName: [field, ...]} over the reachable record graph."""

    seen: dict[str, list[str]] = {}
    queue = list(WALKER_ROOTS)
    while queue:
        cls = queue.pop()
        name = cls.__name__
        if name in seen:
            continue
        try:
            hints = typing.get_type_hints(cls, localns=dict(ANNOTATION_NAMESPACE))
        except Exception:
            hints = {}
        seen[name] = [f.name for f in dataclasses.fields(cls)]
        for f in dataclasses.fields(cls):
            annotation = hints.get(f.name)
            if annotation is None:
                continue
            for nested in _dataclass_types(annotation):
                queue.append(nested)
    return seen


def test_tier_classification_is_closed() -> None:
    """Every reachable field classified; every classified field exists."""

    reachable = _walk()
    reachable_keys = {
        f"{cls_name}.{field_name}"
        for cls_name, field_names in reachable.items()
        for field_name in field_names
    }
    classified_keys = set(TIER_CLASSIFICATION)

    unclassified = sorted(reachable_keys - classified_keys)
    assert not unclassified, (
        "unclassified record fields (add them to TIER_CLASSIFICATION with an "
        f"explicit Tier-F/Tier-R disposition): {unclassified}"
    )
    stale = sorted(classified_keys - reachable_keys)
    assert not stale, f"classification rows for fields that no longer exist: {stale}"


def test_tier_values_are_legal() -> None:
    """Classification values come from the closed vocabulary."""

    legal = {"F", "R:journal-lifetime", "R:release-scrubbed", "R:compat-OpEvent-only"}
    bad = {key: value for key, value in TIER_CLASSIFICATION.items() if value not in legal}
    assert not bad, f"illegal tier values: {bad}"
