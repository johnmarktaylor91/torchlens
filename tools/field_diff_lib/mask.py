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
``reading``
    every number inside the value, integer or float, is a process
    measurement and is replaced by ``<reading>``. Presence, ``None``, keys
    and container lengths are still compared.

``TYPE_TAG_ONLY`` names private trace containers whose class was replaced
by a subclass that holds row keys and hands out the same records: on those
fields only, the subclass name is read as the base class name it stands
for (a list subclass as a plain list). Every item is still compared, and
any other class on those fields still shows.

``RELEASE_ORDER`` names the one sequence whose step outcomes are compared
by release order instead of by value: in ``sequence-8_release`` a watched
object may move only to an earlier release class (freed when the trace is
dropped, then when what was kept is dropped, then by the collector, then
never), with the same objects watched. A move toward the collector, a new
collector-only object, or a changed set of names still shows.

The text rules apply to ``repr`` and ``str`` text and to the message of a
raised error, which can embed a record's ``repr``.

Names that are not read at all are in ``NOT_READ`` with their reason; the
name is still recorded, so a field that disappears still shows.

This list is fixed by capture-core lane H3 and approved once. A later lane
may not add to it: a new difference between base and head is a finding.
Lead's capture-core stage 1 rulings (Mon 9 Nov 2026) added the backward
``peak_memory`` entry, the ``TYPE_TAG_ONLY`` entry and the ``RELEASE_ORDER`` entry.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

RULES = ("clock", "identity", "address", "barcode", "id_keys", "reading")


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
    *_rules(
        "reading",
        "process memory reading of a backward pass, which differs between two runs",
        "peak_memory",
    ),
)

#: Private trace containers read under their base class name: head class to base class.
#: ``"list"`` means the plain list form, which carries no type tag.
TYPE_TAG_ONLY: dict[str, dict[str, str]] = {
    "_layer_list": {"OpSequence": "list"},
    "_layer_dict_main_keys": {"OpLabelMap": "dict"},
    "_layer_dict_all_keys": {"OpOrderedLabelMap": "OrderedDict", "OpLabelMap": "dict"},
    "_layer_logs": {"LayerDictStore": "dict", "LayerStore": "OrderedDict"},
}
TYPE_TAG_ONLY_REASON = (
    "capture-core stage 1 replaced these containers by subclasses that hold row keys and"
    " hand out the same records"
)

#: The sequence dump compared by release order, and its classes from earliest to latest.
RELEASE_ORDER = "sequence-8_release"
RELEASE_CLASSES = (
    "freed_when_trace_dropped",
    "freed_when_kept_dropped",
    "freed_only_by_collector",
    "never_freed",
)
RELEASE_ORDER_REASON = (
    "each capture-core ownership lane frees records earlier by design (PLAN 3.3 item 8);"
    " only moves toward an earlier release are accepted"
)


def _release_classes(outcome: Any) -> dict[str, Any] | None:
    """Return a canonical release outcome as ``key -> value``, or ``None`` for another shape."""

    if not isinstance(outcome, dict) or not isinstance(outcome.get("items"), list):
        return None
    pairs = outcome["items"]
    if not all(isinstance(pair, list) and len(pair) == 2 for pair in pairs):
        return None
    table = {pair[0]: pair[1] for pair in pairs if isinstance(pair[0], str)}
    if not set(RELEASE_CLASSES) <= set(table):
        return None
    return table


def _release_rank(table: dict[str, Any]) -> dict[str, int] | None:
    """Return ``name -> class index``, or ``None`` when a name is listed twice or is not a string."""

    rank: dict[str, int] = {}
    for index, key in enumerate(RELEASE_CLASSES):
        names = table[key]
        if not isinstance(names, list):
            return None
        for name in names:
            if not isinstance(name, str) or name in rank:
                return None
            rank[name] = index
    return rank


def release_moved_earlier(base: Any, head: Any) -> bool:
    """Return whether ``head`` differs from ``base`` only by objects released earlier.

    Both are canonical outcomes of one release step. The same names must be
    watched (and watched the same way); every name's class in ``head`` must be
    the same as or earlier than in ``base``.
    """

    left, right = _release_classes(base), _release_classes(head)
    if left is None or right is None or set(left) != set(right):
        return False
    for key in set(left) - set(RELEASE_CLASSES):
        if left[key] != right[key]:
            return False
    base_rank, head_rank = _release_rank(left), _release_rank(right)
    if base_rank is None or head_rank is None or set(base_rank) != set(head_rank):
        return False
    return all(head_rank[name] <= rank for name, rank in base_rank.items())


def accept_release_moves(name: str, base: Any, head: Any) -> Any:
    """Return ``head`` with each release step that only moved objects earlier set to ``base``'s.

    Applies to the ``RELEASE_ORDER`` dump only (its ``{"steps": [...]}`` form or
    the bare step list); every other dump is returned unchanged, and so is any
    step that differs in another way.
    """

    if name != RELEASE_ORDER:
        return head
    if isinstance(base, dict) and isinstance(head, dict):
        if not isinstance(base.get("steps"), list) or not isinstance(head.get("steps"), list):
            return head
        return {**head, "steps": accept_release_moves(name, base["steps"], head["steps"])}
    if not isinstance(base, list) or not isinstance(head, list):
        return head
    base_steps = {
        step.get("step"): step for step in base if isinstance(step, dict) and "step" in step
    }
    accepted = []
    for step in head:
        before = base_steps.get(step.get("step")) if isinstance(step, dict) else None
        if (
            before is not None
            and before.get("outcome") != step.get("outcome")
            and release_moved_earlier(before.get("outcome"), step.get("outcome"))
        ):
            step = {**step, "outcome": before["outcome"]}
        accepted.append(step)
    return accepted


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
        self._tags = TYPE_TAG_ONLY
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
        canonical = self._base_tag(name, canonical)
        rule = self.rule_for(name)
        return self._walk(canonical) if rule is None else self._by_rule(rule, canonical)

    def _base_tag(self, name: object, canonical: Any) -> Any:
        """Read a ``TYPE_TAG_ONLY`` field's subclass tag as its base class tag."""

        if not self.enabled or not isinstance(name, str) or not isinstance(canonical, dict):
            return canonical
        names = self._tags.get(name)
        if not names:
            return canonical
        if "__seq__" in canonical and names.get(canonical["__seq__"]) == "list":
            return canonical.get("items")
        if "__dict__" in canonical and canonical["__dict__"] in names:
            return {**canonical, "__dict__": names[canonical["__dict__"]]}
        return canonical

    def _by_rule(self, rule: str, value: Any) -> Any:  # noqa: PLR0911
        """Apply one rule to every leaf of a canonical value."""

        if rule == "id_keys":
            return self._id_keys(value)
        if isinstance(value, list):
            return [self._by_rule(rule, member) for member in value]
        if isinstance(value, dict):
            if "__float__" in value and rule in ("clock", "identity", "reading"):
                if rule == "clock":
                    return "<float>"
                if rule == "reading":
                    return "<reading>"
                return value if value["__float__"] == "nan" else "<id>"
            return {
                key: entry if _is_tag(key, entry) else self._by_rule(rule, entry)
                for key, entry in value.items()
            }
        if isinstance(value, bool) or value is None:
            return value
        if isinstance(value, int) and rule == "reading":
            return "<reading>"
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

        entry = self._base_tag(key, entry)
        rule = self.rule_for(key)
        return self._walk(entry) if rule is None else self._by_rule(rule, entry)
