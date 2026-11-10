"""The fixed mask of the field differential.

A masked field is one whose value differs between two runs of the same
commit. Each entry names the record kind, the field and the reason. A field
is not masked only because it varies: where a stable property exists it is
compared instead, and the entry says which:

``type``
    the value is replaced by its canonical type (number, string, ``None``,
    tensor...), so a field that stops being measured, or starts raising,
    still shows.
``token``
    each distinct value is replaced by its order of first appearance in the
    trace, so uniqueness and equality between fields are still compared.

Entries that are not read at all are in ``NOT_READ`` with their reason; the
name is still recorded, so a field that disappears still shows.

This list is fixed by capture-core lane H3 and approved once. A later lane
may not add to it: a new difference between base and head is a finding.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class MaskRule:
    """One masked field.

    Parameters
    ----------
    kind:
        Record class name (``Trace``, ``Op``, ``Layer``...), ``to_pandas`` for
        a table column, or ``*`` for every kind.
    field:
        Field name; for a grouped member the leaf name (``cost.x`` is ``x``).
    rule:
        ``type`` or ``token`` (see the module docstring).
    reason:
        Why the value differs between two runs of one commit.
    """

    kind: str
    field: str
    rule: str
    reason: str


@dataclass(frozen=True)
class TextRule:
    """One substitution applied inside ``repr`` and ``str`` text.

    Parameters
    ----------
    pattern:
        Regular expression matched in the text.
    replacement:
        Text put in its place.
    reason:
        Why the matched text differs between two runs of one commit.
    """

    pattern: str
    replacement: str
    reason: str


#: Fields compared by a stable property instead of by value.
MASK: tuple[MaskRule, ...] = ()

#: Nested mapping keys masked wherever they appear inside a field's value.
NESTED_MASK: tuple[MaskRule, ...] = ()

#: Substitutions inside ``repr`` and ``str`` text.
TEXT_RULES: tuple[TextRule, ...] = ()

#: Fields that are never read, with the reason. Presence is still recorded.
NOT_READ: dict[str, str] = {
    "receptive_field": "lazy geometry solver that returns a view; cost unbounded on real models",
    "projective_field": "lazy geometry solver that returns a view; cost unbounded on real models",
}


class Mask:
    """Lookup over the fixed mask, with the token tables of one trace.

    Parameters
    ----------
    enabled:
        ``False`` turns every rule off (used to find what varies).
    """

    def __init__(self, enabled: bool = True) -> None:
        """Index the rules by kind and field."""

        self.enabled = enabled
        self._rules = {(rule.kind, rule.field): rule for rule in MASK}
        self._nested = {(rule.kind, rule.field): rule for rule in NESTED_MASK}
        self._text = [(re.compile(rule.pattern), rule.replacement) for rule in TEXT_RULES]
        self._tokens: dict[str, dict[str, int]] = {}

    def rule_for(self, kind: str, field: str) -> MaskRule | None:
        """Return the rule for ``kind.field``, or ``None``."""

        if not self.enabled:
            return None
        return self._rules.get((kind, field)) or self._rules.get(("*", field))

    def nested_rule_for(self, kind: str, key: str) -> MaskRule | None:
        """Return the rule for a nested mapping key, or ``None``."""

        if not self.enabled:
            return None
        return self._nested.get((kind, key)) or self._nested.get(("*", key))

    def text(self, value: str) -> str:
        """Apply the text substitutions to a ``repr`` or ``str``."""

        if self.enabled:
            for pattern, replacement in self._text:
                value = pattern.sub(replacement, value)
        return value

    def token(self, field: str, canonical: str) -> str:
        """Number a varying value by first appearance within the trace."""

        table = self._tokens.setdefault(field, {})
        return f"<{field}#{table.setdefault(canonical, len(table))}>"


def canonical_type(canonical: object) -> str:
    """Name the kind of a canonical value, for the ``type`` rule."""

    if canonical is None:
        return "<none>"
    if isinstance(canonical, bool):
        return "<bool>"
    if isinstance(canonical, int):
        return "<int>"
    if isinstance(canonical, str):
        return "<str>"
    if isinstance(canonical, list):
        return f"<list:{len(canonical)}>"
    if isinstance(canonical, dict):
        for tag in canonical:
            if tag.startswith("__"):
                if tag == "__raises__":
                    return f"<raises:{canonical[tag]}>"
                return f"<{tag.strip('_')}>"
        return "<dict>"
    return f"<{type(canonical).__name__}>"
