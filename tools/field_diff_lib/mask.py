"""The fixed mask of the field differential.

A masked name is one whose value differs between two runs of the same
commit. Each entry gives the name, the rule and the reason. A name is
masked wherever it appears: as a field of a record, as a member of a concern
group, as a table column, or as a mapping key or dataclass field inside
another field's value. Nothing is dropped whole: each rule hides only the
part that varies and compares the rest.

``clock``
    every float inside the value is replaced by ``<float>``. Presence,
    ``None``, counts, keys and container lengths are still compared, so a
    reading that stops being taken still shows.
``identity``
    every integer (and every string of digits, and every float a table made
    of one) inside the value is a memory address or a thread identity and is
    replaced by ``<id>``. Presence and container shape are still compared. Equality between two addresses is
    not: an address is reused once its object is freed, so two unrelated
    short-lived objects share an address in one run and not in the next.
    Which fields hold the same live tensor, and which tensors share storage,
    is compared separately and exactly by the tensor form
    (``canonical.py``), which keeps every tensor alive while it numbers them.
``address``
    the value mixes addresses with ordinary numbers: only integers of twelve
    digits or more (addresses, nanosecond file times) are replaced by
    ``<id>``.
``barcode``
    every string inside the value is a random identifier, or a digest of
    one, and is replaced by its order of first appearance in the trace, so
    equality between the fields that carry the same identifier is still
    compared.
``id_keys``
    the value is a mapping keyed by memory address: only the integer keys
    are replaced by ``<id>``; every value under them is walked and compared
    like any other field.

The text rules apply to ``repr`` and ``str`` text and to the message of a
raised error, which can embed a record's ``repr``.

Names that are not read at all are in ``NOT_READ`` with their reason; the
name is still recorded, so a field that disappears still shows.

This list is fixed by capture-core lane H3 and approved once. A later lane
may not add to it: a new difference between base and head is a finding.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

RULES = ("clock", "identity", "address", "barcode", "id_keys")


@dataclass(frozen=True)
class MaskRule:
    """One masked name.

    Parameters
    ----------
    field:
        The name, wherever it appears; for a grouped member the leaf name
        (``cost.duration`` is ``duration``).
    rule:
        One of ``RULES`` (see the module docstring).
    reason:
        Why the value differs between two runs of one commit.
    """

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


def _rules(rule: str, reason: str, *fields: str) -> tuple[MaskRule, ...]:
    """Build one rule per name."""

    return tuple(MaskRule(field, rule, reason) for field in fields)


#: Names compared by a stable property instead of by value.
MASK: tuple[MaskRule, ...] = (
    *_rules(
        "clock",
        "wall time one function call took in this run",
        "func_duration",
        "duration",
        "total_func_duration",
        "func_calls_duration",
        "total_func_calls_duration",
    ),
    *_rules(
        "clock",
        "wall time one module call or the forward pass took in this run",
        "forward_duration",
        "total_forward_duration",
        "overhead_duration",
        "setup_duration",
        "cleanup_duration",
        "total_s",
    ),
    *_rules(
        "clock",
        "wall time a backward pass or one gradient function took in this run",
        "backward_duration",
        "total_backward_duration",
        "backward_durations",
        "last_backward_duration",
        "total_fire_duration",
        "grad_fn_fire_timings",
    ),
    *_rules(
        "clock",
        "clock reading taken when this run reached an event",
        "start_time",
        "end_time",
        "forward_start_time",
        "fire_started_monotonic",
        "fire_finished_monotonic",
        "timestamp",
        "started_at",
    ),
    *_rules(
        "clock",
        "stored form of a clock field listed above",
        "_cost_forward_duration",
        "_cost_func_calls_duration",
        "_cost_backward_durations",
        "_provenance_setup_duration",
        "_provenance_cleanup_duration",
        "_provenance_start_time",
        "_provenance_end_time",
    ),
    *_rules(
        "identity",
        "memory address of an autograd node in this process",
        "grad_fn_object_id",
        "next_grad_fn_ids",
        "root_grad_fn_ids",
        "backward_root_grad_fn_object_ids",
        "last_backward_root_grad_fn_object_id",
        "grad_fn_call_ref",
        "_autograd_backward_root_grad_fn_object_ids",
    ),
    *_rules(
        "identity",
        "a sequence of autograd node addresses and nothing else (discovery order, next-function "
        "ids): every integer is an object_id; length is compared, and the graph shape is compared "
        "through each GradFn record's own fields",
        "grad_fn_order",
        "topology",
        "_autograd_grad_fn_order",
    ),
    *_rules(
        "id_keys",
        "mapping keyed by autograd node address: only the integer keys are masked, and every "
        "GradFn record under them is compared field by field",
        "_grad_fn_logs",
    ),
    *_rules(
        "identity",
        "memory address of the model, the input, a tensor or a storage in this process",
        "model_object_id",
        "input_object_id",
        "_provenance_model_object_id",
        "_provenance_input_object_id",
        "backend_handle_id",
        "object_id",
        "source_id",
        "storage_ptr",
    ),
    *_rules(
        "identity",
        "identity of the capturing thread in this process",
        "owner_thread_id",
        "_provenance_owner_thread_id",
    ),
    MaskRule("storage_key", "address", "pair of a device and a storage address"),
    MaskRule(
        "_code_context_cache",
        "address",
        "private cache keyed by code-object addresses and file modification times",
    ),
    *_rules(
        "barcode",
        "random identifier drawn for a hook, an edit or a transaction",
        "handle",
        "handles",
        "_last_hook_handle_ids",
        "event_id",
        "transaction_id",
    ),
    *_rules(
        "barcode",
        "digest of a record that contains a random identifier",
        "resolve_digest",
        "event_digest",
    ),
)

#: Substitutions inside ``repr`` and ``str`` text, applied in this order.
TEXT_RULES: tuple[TextRule, ...] = (
    TextRule(
        r"\b\d+(?:\.\d+)? (?:ns|us|µs|ms|s|min)\b",
        "<duration>",
        "clock reading printed in a summary line",
    ),
    TextRule(
        r"<duration>(?:, almost all logging| \(<duration> of it logging\))",
        "<duration> and its logging share",
        "wording chosen from the ratio of two clock readings",
    ),
)

#: Fields that are never read, with the reason. Presence is still recorded.
NOT_READ: dict[str, str] = {
    "receptive_field": "lazy geometry solver that returns a view; cost unbounded on real models",
    "projective_field": "lazy geometry solver that returns a view; cost unbounded on real models",
}

#: Models small enough that the ``NOT_READ`` geometry fields are read and compared on them.
GEOMETRY_READ_MODELS: tuple[str, ...] = ("chain40", "varied8", "loopcell12")

_ADDRESS_DIGITS = 12


def _is_tag(key: str, entry: object) -> bool:
    """Return whether a canonical entry is a type tag (``"__dict__": "OrderedDict"``)."""

    return key.startswith("__") and isinstance(entry, str)


def _is_id_key(key: object) -> bool:
    """Return whether a canonical mapping key is an integer identity."""

    if isinstance(key, bool):
        return False
    return isinstance(key, int) or (isinstance(key, str) and key.isdigit())


class Mask:
    """Lookup over the fixed mask, with the barcode table of one trace.

    Parameters
    ----------
    enabled:
        ``False`` turns every rule off (used to find what varies).
    """

    def __init__(self, enabled: bool = True) -> None:
        """Index the rules by name."""

        self.enabled = enabled
        self._rules = {rule.field: rule.rule for rule in MASK}
        self._text = [(re.compile(rule.pattern), rule.replacement) for rule in TEXT_RULES]
        self._barcodes: dict[str, int] = {}

    def rule_for(self, name: object) -> str | None:
        """Return the rule of a name, or ``None``."""

        if not self.enabled or not isinstance(name, str):
            return None
        return self._rules.get(name)

    def text(self, value: str) -> str:
        """Apply the text substitutions to a ``repr`` or ``str``."""

        if self.enabled:
            for pattern, replacement in self._text:
                value = pattern.sub(replacement, value)
        return value

    def apply(self, name: str, canonical: Any) -> Any:
        """Mask the canonical value found under one name."""

        if not self.enabled:
            return canonical
        rule = self.rule_for(name)
        return self._walk(canonical) if rule is None else self._by_rule(rule, canonical)

    def _by_rule(self, rule: str, value: Any) -> Any:  # noqa: PLR0911
        """Apply one rule to every leaf of a canonical value."""

        if rule == "id_keys":
            return self._id_keys(value)
        if isinstance(value, list):
            return [self._by_rule(rule, member) for member in value]
        if isinstance(value, dict):
            if "__float__" in value and rule in ("clock", "identity"):
                if rule == "clock":
                    return "<float>"
                return value if value["__float__"] == "nan" else "<id>"
            return {
                key: entry if _is_tag(key, entry) else self._by_rule(rule, entry)
                for key, entry in value.items()
            }
        if isinstance(value, bool) or value is None:
            return value
        if isinstance(value, int) and rule in ("identity", "address"):
            digits = 1 if rule == "identity" else _ADDRESS_DIGITS
            return "<id>" if len(str(abs(value))) >= digits else value
        if isinstance(value, str):
            if rule == "identity" and value.isdigit():
                return "<id>"
            if rule == "barcode":
                return f"<barcode#{self._barcodes.setdefault(value, len(self._barcodes))}>"
        return value

    def _id_keys(self, value: Any) -> Any:
        """Mask the integer keys of a canonical mapping and walk its values."""

        if isinstance(value, dict) and isinstance(value.get("items"), list):
            items = [
                ["<id>" if _is_id_key(key) else key, self._walk(entry)]
                for key, entry in value["items"]
            ]
            return {**value, "items": items}
        return self._walk(value)

    def _walk(self, canonical: Any) -> Any:
        """Apply the rules to named entries, and the text rules to object reprs, inside a value."""

        if isinstance(canonical, list):
            return [self._walk(member) for member in canonical]
        if not isinstance(canonical, dict):
            return canonical
        if "__object__" in canonical and isinstance(canonical.get("repr"), str):
            return {**canonical, "repr": self.text(canonical["repr"])}
        if "__dict__" in canonical and isinstance(canonical.get("items"), list):
            items = [[key, self._named(key, entry)] for key, entry in canonical["items"]]
            return {**canonical, "items": items}
        if "__dataclass__" in canonical and isinstance(canonical.get("fields"), dict):
            fields = {key: self._named(key, entry) for key, entry in canonical["fields"].items()}
            return {**canonical, "fields": fields}
        return {key: self._walk(entry) for key, entry in canonical.items()}

    def _named(self, key: object, entry: Any) -> Any:
        """Mask the value under one mapping key or dataclass field name."""

        rule = self.rule_for(key)
        return self._walk(entry) if rule is None else self._by_rule(rule, entry)
