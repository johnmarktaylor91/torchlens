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
    the value is an object identity, or a container of them: each distinct
    identity is replaced by its order of first appearance in the trace, in
    one table shared by every identity field, so uniqueness, ordering and
    equality between fields are still compared.
``address``
    the value mixes identities with ordinary numbers: only integers of
    twelve digits or more (addresses, nanosecond file times) are numbered
    as identities; everything else is compared as it is.

Entries that are not read at all are in ``NOT_READ`` with their reason; the
name is still recorded, so a field that disappears still shows.

This list is fixed by capture-core lane H3 and approved once. A later lane
may not add to it: a new difference between base and head is a finding.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


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
        ``type``, ``token`` or ``address`` (see the module docstring).
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


_CLOCK = "clock reading: wall time of this run"
_IDENTITY = "object identity: a memory address of this process"
_THREAD = "thread identity of this process"


def _rules(kind: str, rule: str, reason: str, *fields: str) -> tuple[MaskRule, ...]:
    """Build one rule per field name."""

    return tuple(MaskRule(kind, field, rule, reason) for field in fields)


#: Fields compared by a stable property instead of by value.
MASK: tuple[MaskRule, ...] = (
    *_rules("Op", "type", _CLOCK, "func_duration", "duration"),
    *_rules("Layer", "type", _CLOCK, "duration", "total_func_duration"),
    *_rules(
        "Module",
        "type",
        _CLOCK,
        "forward_duration",
        "func_calls_duration",
        "total_forward_duration",
        "total_func_calls_duration",
    ),
    *_rules("ModuleCall", "type", _CLOCK, "forward_duration", "func_calls_duration"),
    *_rules("GradFn", "type", _CLOCK, "backward_duration", "total_backward_duration"),
    *_rules("BackwardPass", "type", _CLOCK, "duration"),
    *_rules(
        "Trace",
        "type",
        _CLOCK,
        "_cost_forward_duration",
        "_cost_func_calls_duration",
        "_cost_backward_durations",
        "_provenance_setup_duration",
        "_provenance_cleanup_duration",
        "_provenance_start_time",
        "_provenance_end_time",
        "forward_duration",
        "func_calls_duration",
        "backward_durations",
        "last_backward_duration",
        "total_backward_duration",
        "overhead_duration",
        "setup_duration",
        "cleanup_duration",
        "duration",
        "start_time",
        "end_time",
    ),
    *_rules("to_pandas", "type", _CLOCK, "func_duration"),
    *_rules("Op", "token", _IDENTITY, "grad_fn_object_id"),
    *_rules("Layer", "token", _IDENTITY, "grad_fn_object_id"),
    *_rules("GradFn", "token", _IDENTITY, "grad_fn_object_id", "next_grad_fn_ids"),
    *_rules("BackwardPass", "token", _IDENTITY, "root_grad_fn_ids"),
    *_rules(
        "Trace",
        "token",
        _IDENTITY,
        "_provenance_model_object_id",
        "_provenance_input_object_id",
        "model_object_id",
        "input_object_id",
        "_autograd_grad_fn_order",
        "_autograd_backward_root_grad_fn_object_ids",
        "_grad_fn_logs",
    ),
    *_rules("Trace", "token", _THREAD, "_provenance_owner_thread_id", "owner_thread_id"),
    MaskRule(
        "Trace",
        "_code_context_cache",
        "address",
        "private cache keyed by code-object addresses and file modification times",
    ),
)

#: Names masked wherever they appear as a mapping key or a dataclass field inside a value.
NESTED_MASK: tuple[MaskRule, ...] = (
    MaskRule("*", "total_s", "type", _CLOCK),
    MaskRule("*", "backend_handle_id", "token", _IDENTITY),
    MaskRule("*", "owner_thread_id", "token", _THREAD),
)

#: Substitutions inside ``repr`` and ``str`` text.
TEXT_RULES: tuple[TextRule, ...] = (
    TextRule(
        r"\b\d+(?:\.\d+)? (?:ns|us|µs|ms|s|min)\b",
        "<duration>",
        "clock reading printed in a summary line",
    ),
)

#: Fields that are never read, with the reason. Presence is still recorded.
NOT_READ: dict[str, str] = {
    "receptive_field": "lazy geometry solver that returns a view; cost unbounded on real models",
    "projective_field": "lazy geometry solver that returns a view; cost unbounded on real models",
}

_ADDRESS_DIGITS = 12


class Mask:
    """Lookup over the fixed mask, with the identity table of one trace.

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
        self._identities: dict[str, int] = {}

    def rule_for(self, kind: str, field: str) -> MaskRule | None:
        """Return the rule for ``kind.field``, or ``None``."""

        if not self.enabled:
            return None
        return self._rules.get((kind, field)) or self._rules.get(("*", field))

    def nested_rule_for(self, kind: str, key: object) -> MaskRule | None:
        """Return the rule for a nested mapping key or dataclass field, or ``None``."""

        if not self.enabled or not isinstance(key, str):
            return None
        return self._nested.get((kind, key)) or self._nested.get(("*", key))

    def text(self, value: str) -> str:
        """Apply the text substitutions to a ``repr`` or ``str``."""

        if self.enabled:
            for pattern, replacement in self._text:
                value = pattern.sub(replacement, value)
        return value

    def identity(self, value: object) -> str:
        """Number an identity by first appearance within the trace."""

        return f"<id#{self._identities.setdefault(repr(value), len(self._identities))}>"

    def apply(self, kind: str, leaf: str, canonical: Any) -> Any:
        """Mask one field's canonical value: its own rule, else the nested rules."""

        rule = self.rule_for(kind, leaf)
        if rule is None:
            return self._walk(kind, canonical) if self.enabled else canonical
        return self._by_rule(rule.rule, canonical)

    def _by_rule(self, rule: str, canonical: Any) -> Any:
        """Apply one rule to a canonical value."""

        if rule == "type":
            return canonical_type(canonical)
        return self._identities_in(canonical, 1 if rule == "token" else _ADDRESS_DIGITS)

    def _identities_in(self, canonical: Any, digits: int) -> Any:
        """Number every integer of at least ``digits`` digits inside a canonical value."""

        if isinstance(canonical, bool) or canonical is None:
            return canonical
        if isinstance(canonical, int):
            return self.identity(canonical) if len(str(abs(canonical))) >= digits else canonical
        if isinstance(canonical, str):
            numeric = canonical.isdigit() and len(canonical) >= digits
            return self.identity(int(canonical)) if numeric else canonical
        if isinstance(canonical, list):
            return [self._identities_in(member, digits) for member in canonical]
        if isinstance(canonical, dict):
            return {key: self._identities_in(entry, digits) for key, entry in canonical.items()}
        return canonical

    def _walk(self, kind: str, canonical: Any) -> Any:
        """Apply the nested rules, and the text rules to object reprs, inside a value."""

        if isinstance(canonical, list):
            return [self._walk(kind, member) for member in canonical]
        if not isinstance(canonical, dict):
            return canonical
        if "__object__" in canonical and isinstance(canonical.get("repr"), str):
            return {**canonical, "repr": self.text(canonical["repr"])}
        if "__dict__" in canonical and isinstance(canonical.get("items"), list):
            items = [[key, self._named(kind, key, entry)] for key, entry in canonical["items"]]
            return {**canonical, "items": items}
        if "__dataclass__" in canonical and isinstance(canonical.get("fields"), dict):
            fields = {
                key: self._named(kind, key, entry) for key, entry in canonical["fields"].items()
            }
            return {**canonical, "fields": fields}
        return {key: self._walk(kind, entry) for key, entry in canonical.items()}

    def _named(self, kind: str, key: object, entry: Any) -> Any:
        """Mask the value under one mapping key or dataclass field name."""

        rule = self.nested_rule_for(kind, key)
        return self._walk(kind, entry) if rule is None else self._by_rule(rule.rule, entry)


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
