"""Apply code-owned family grammar data to decoded cells before record construction.

The nodes live in ``schema.FAMILIES``. This module interprets node shapes,
closed vocabularies and within-cell constraints without importing validators.
Cross-record relations remain with the reader's validation pass. The cell refusal
(``cell_error``) and the pending ``record`` object both modules raise and inspect are
defined here, below the decoder, so the decoder imports this module and not the reverse.
"""

from __future__ import annotations

import ast
import dataclasses
import math
import re
from collections.abc import Mapping, Sequence
from functools import cache
from typing import Any, Final

from ..format_errors import ArtifactError
from .contract import PHASE_TIMING_KEY_SOURCE
from .schema import FAMILIES

#: Code of a record cell that fails its typed schema (format.1; a key of ``REFUSAL_CODES``).
CELL_CODE: Final = "record_schema_invalid"

#: Field named in a family refusal, by code.
_FAMILY_FIELDS: Final[dict[str, str]] = {
    "artifact_distributed_scope_invalid": "Trace.provenance.distributed_scope",
    "artifact_grad_fn_timing_provenance_invalid": "Trace.grad_fn_timing_provenance",
    "artifact_checkpoint_witness_invalid": "Trace.checkpoint_invocation_witness",
    "artifact_intervention_audit_invalid": "Trace.intervention_audit",
    "artifact_kernel_telemetry_invalid": "Trace.annotations._kernel_telemetry",
    "artifact_sidecar_invalid": 'Trace.annotations["sidecar"]',
    "artifact_capture_advisories_invalid": 'Trace.annotations["capture_advisories"]',
    "artifact_source_snapshots_invalid": "Trace.source.snapshots",
    "artifact_structure_evidence_invalid": "Trace.graph.structure_evidence",
    "artifact_executed_flops_record_invalid": "BackwardPass._executed_flops_record",
    "artifact_episode_step_invalid": "Op.episode_step",
    "artifact_forward_peak_memory_invalid": "Trace.forward_peak_memory_reason",
    "artifact_backward_peak_memory_invalid": "Trace.backward_peak_memory_reason",
}


@dataclasses.dataclass
class PendingRecord:
    """A ``record`` object whose decoded fields still hold ``ref`` cells for the reader to settle."""

    cls: type
    fields: dict[str, Any]
    at: str


def cell_error(at: str, detail: str, *, code: str = CELL_CODE) -> ArtifactError:
    """The one record-cell refusal: the path, the detail, the re-save remedy."""

    remedy = "re-save the trace with trace.save(); the artifact was edited or corrupted"
    fields: dict[str, Any] = {"at": at, "detail": detail}
    field = _FAMILY_FIELDS.get(code)
    if code == "artifact_annotations_invalid":
        prefix = "Op" if at.startswith("op[") else "Trace"
        key = re.split(r"[.\[<]", at.partition(".annotations.")[2], maxsplit=1)[0]
        field = f'{prefix}.annotations["{key}"]' if key else f"{prefix}.annotations"
    if field is not None:
        fields.update(field=field, reason="not_a_string" if "peak_memory" in code else "schema")
    subject = f"{field} at record cell {at}" if field else f"Record cell {at}"
    message = f"{subject} does not satisfy the 3.0 saved-format contract: {detail}."
    if field is not None:
        message += f" Remedy: {remedy}."
    return ArtifactError(message, code=code, remedy=remedy, **fields)


_CELL_POSITION = re.compile(r"(\w+)\[\d+\]\.(\w+)")


class _FamilyContext:
    """Keep conditional grammar bindings independent of record-table decode order.

    A binding may require a singleton trace cell to be present. Its grammar
    waits for that cell to decode; an absent or None guard preserves the old
    validator's early return. No foreign-key or cross-record relation is checked.
    """

    def __init__(self) -> None:
        """Initialize the decoded guard facts and waiting cell grammars."""

        self._facts: dict[tuple[str, str], Any] = {}
        self._pending: list[tuple[dict[str, Any], Any, str]] = []

    def check(self, value: Any, at: str) -> None:
        """Apply bound cell grammars and settle any newly available guard.

        Parameters
        ----------
        value, at:
            Decoded value and its record, row and column path.
        """

        position = _CELL_POSITION.fullmatch(at)
        if position is None:
            return
        key = position.groups()
        if key in _guard_cells():
            self._facts[key] = value
            waiting, self._pending = self._pending, []
            for node, candidate, candidate_at in waiting:
                self._check(node, candidate, candidate_at)
        for node in _bound_families(*key):
            self._check(node, value, at)

    def _check(self, node: dict[str, Any], value: Any, at: str) -> None:
        """Apply one binding when its declared presence guard is settled.

        Parameters
        ----------
        node, value, at:
            Bound schema, decoded candidate and diagnostic path.
        """

        guard = node.get("requires")
        if guard is not None:
            key = (guard[0], guard[1])
            if key not in self._facts:
                self._pending.append((node, value, at))
                return
            if not _family_condition(self._facts[key], guard[2]):
                return
        for selected, selected_at in _family_targets(value, node.get("path", ()), at):
            _apply_family(node, selected, selected_at)


@cache
def _guard_cells() -> frozenset[tuple[str, str]]:
    """Return singleton cell identities used as conditional binding guards.

    Returns
    -------
    frozenset[tuple[str, str]]
        Code-owned record and column pairs, independent of artifact contents.
    """

    return frozenset(
        (node["requires"][0], node["requires"][1])
        for node in FAMILIES.values()
        if "requires" in node
    )


@cache
def _bound_families(record: str, column: str) -> tuple[dict[str, Any], ...]:
    """Return the grammar nodes bound to one record column.

    Parameters
    ----------
    record, column:
        Record kind and stored column name.
    """

    return tuple(
        node
        for node in FAMILIES.values()
        if node.get("record") == record and node.get("column") == column
    )


def _apply_family(node: Mapping[str, Any], value: Any, at: str) -> None:
    """Check one decoded value against a declarative grammar node.

    Parameters
    ----------
    node:
        Schema data with an inherited refusal code and optional family reference.
    value:
        Decoded value; record-tagged values can still be pending construction.
    at:
        Complete record, row, column and nested item path.
    """

    # Essential interpreter complexity: keep the closed node instructions together;
    # family-specific callbacks would duplicate these structural checks.

    if "family" in node:
        node = {**FAMILIES[node["family"]], **node}
    guard = node.get("when_attribute_present")
    if guard is not None and _family_attribute(value, guard) is None:
        return
    code = node.get("code", CELL_CODE)
    if value is None and node.get("nullable"):
        return
    if "vocab" in node and value not in _family_vocab(node["vocab"]):
        raise cell_error(at, f"value {value!r} is outside the family vocabulary", code=code)
    if "any_of" in node:
        for option in node["any_of"]:
            try:
                _apply_family({"code": code, **option}, value, at)
            except ArtifactError:
                continue
            return
        raise cell_error(at, "value matches none of the family schemas", code=code)
    kind = node.get("kind", "scalar")
    if kind == "object":
        _family_object(node, value, at, code)
    elif kind in ("list", "tuple"):
        _family_sequence(node, value, at, code)
    else:
        _family_scalar(node, value, at, code)


@cache
def _family_vocab(source: Any) -> tuple[Any, ...]:
    """Resolve a closed vocabulary from data or its code-owned source.

    Parameters
    ----------
    source:
        Literal vocabulary or the PIPELINE reference declared by the contract.
    """

    if source == "PIPELINE":
        from importlib import import_module

        module, _, attribute = PHASE_TIMING_KEY_SOURCE.rpartition(".")
        return tuple(step.name for step in getattr(import_module(module), attribute))
    return tuple(source)


def _family_scalar(node: Mapping[str, Any], value: Any, at: str, code: str) -> None:
    """Apply scalar kinds, bounds and literal syntax from a node.

    Parameters
    ----------
    node, value, at, code:
        Schema node, candidate scalar, diagnostic path and inherited refusal code.
    """

    # Essential interpreter complexity: keep the closed node instructions together;
    # family-specific callbacks would duplicate these structural checks.

    types: dict[str, Any] = {"str": str, "int": int, "bool": bool, "number": (int, float)}
    declared = node.get("type", ())
    declared = (declared,) if isinstance(declared, str) else declared
    typed = not declared or any(
        isinstance(value, types[name])
        and not (name in ("int", "number") and isinstance(value, bool))
        for name in declared
    )
    if not typed:
        raise cell_error(
            at, f"expected scalar kind {declared}, got {type(value).__name__}", code=code
        )
    for bound, compare in (
        ("minimum", lambda a, b: a >= b),
        ("min_length", lambda a, b: len(a) >= b),
        ("max_length", lambda a, b: len(a) <= b),
    ):
        if bound in node and not compare(value, node[bound]):
            raise cell_error(at, f"value violates {bound}={node[bound]}", code=code)
    if node.get("finite"):
        try:
            finite = math.isfinite(float(value))
        except OverflowError:
            finite = False
        if not finite:
            raise cell_error(at, "value must be finite", code=code)
    if (
        "pattern" in node
        and getattr(re.compile(node["pattern"]), node.get("pattern_method", "match"))(value) is None
    ):
        raise cell_error(at, "string does not match the family's syntax", code=code)
    if node.get("literal_tuple"):
        try:
            parsed = ast.literal_eval(value)
        except (MemoryError, RecursionError, SyntaxError, ValueError):
            raise cell_error(at, "string is not a literal tuple", code=code) from None
        if not isinstance(parsed, tuple):
            raise cell_error(at, "literal must be a tuple", code=code)


def _family_object(node: Mapping[str, Any], value: Any, at: str, code: str) -> None:
    """Check mapping fields, discriminated variants and within-cell rules.

    Parameters
    ----------
    node, value, at, code:
        Schema node, candidate object, diagnostic path and inherited refusal code.
    """

    value = _family_mapping(node, value, at, code)
    keys = node.get("keys", {})
    for key, item in value.items():
        if "key" in node:
            _apply_family({"code": code, **node["key"]}, key, f"{at}<key {key!r}>")
        child = keys.get(key, node.get("values"))
        if child is not None:
            _apply_family({"code": code, **child}, item, f"{at}.{key}")
    if "discriminator" in node:
        variant = value.get(node["discriminator"])
        selected = next((case for name, case in node["cases"].items() if variant == name), None)
        if selected is None:
            raise cell_error(at, f"unknown family variant {variant!r}", code=code)
        _apply_family({"code": code, **selected}, value, at)
    _family_constraints(node, value, at, code)


def _family_mapping(node: Mapping[str, Any], value: Any, at: str, code: str) -> Mapping[str, Any]:
    """Normalize an object representation and check its declared field set.

    Parameters
    ----------
    node, value, at, code:
        Schema node, candidate object, diagnostic path and inherited refusal code.
    """

    # Essential interpreter complexity: normalize both admitted object representations.
    keys = node.get("keys", {})
    if node.get("attributes") and not isinstance(value, Mapping):
        if isinstance(value, PendingRecord):
            value = {key: value.fields[key] for key in keys if key in value.fields}
        else:
            value = {key: getattr(value, key) for key in keys if hasattr(value, key)}
    if not isinstance(value, Mapping):
        raise cell_error(at, "family value must be a mapping", code=code)
    required = set(keys) - set(node.get("optional", ()))
    if not required <= set(value) or (
        "keys" in node and not node.get("open") and set(value) - set(keys)
    ):
        raise cell_error(at, f"family fields differ from {sorted(keys)}", code=code)
    return value


def _family_constraints(
    node: Mapping[str, Any], value: Mapping[str, Any], at: str, code: str
) -> None:
    """Apply the declared coherence rules inside one object cell.

    Parameters
    ----------
    node, value, at, code:
        Schema node, validated mapping, diagnostic path and inherited refusal code.
    """

    # Essential interpreter complexity: keep the closed node instructions together;
    # family-specific callbacks would duplicate these structural checks.

    for rule in node.get("rules", ()):
        if all(_family_condition(value.get(key), test) for key, test in rule["when"].items()):
            _apply_family(
                {"kind": "object", "open": True, "code": code, "keys": rule["keys"]}, value, at
            )
    if "ordinal_keys" in node:
        table, count = node["ordinal_keys"]
        # Cardinality plus bounded integer keys avoids allocating an attacker-sized range.
        members, total = value[table], value[count]
        if len(members) != total or set(members) != set(range(1, len(members) + 1)):
            raise cell_error(at, "mapping keys must be contiguous ordinals 1..count", code=code)
    if "sum" in node:
        total = value[node["sum"]["total"]]
        if total is not None:
            parts = [
                entry[index] or 0 for field, index in node["sum"]["terms"] for entry in value[field]
            ]
            if sum(parts) != total:
                raise cell_error(at, "item counts do not sum to the declared total", code=code)


def _family_condition(value: Any, condition: str) -> bool:
    """Evaluate a data-only presence or truth condition.

    Parameters
    ----------
    value:
        Field value in the current mapping.
    condition:
        Truthiness or presence predicate named by schema data.
    """

    if condition == "present":
        return value is not None
    return bool(value) if condition == "truthy" else not bool(value)


def _family_attribute(value: Any, name: str) -> Any:
    """Read an attribute of a pending or built value object.

    Parameters
    ----------
    value, name:
        Value object and code-owned attribute name.
    """

    return (
        value.fields.get(name) if isinstance(value, PendingRecord) else getattr(value, name, None)
    )


def _family_targets(value: Any, path: Sequence[str], at: str) -> tuple[tuple[Any, str], ...]:
    """Select mapping fields, object attributes or sequence members for a node.

    Parameters
    ----------
    value, path, at:
        Current value, remaining schema-owned path and diagnostic position.
    """

    if not path:
        return ((value, at),)
    key, *remaining = path
    if key == "*":
        if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
            return ()
        return tuple(
            target
            for index, item in enumerate(value)
            for target in _family_targets(item, remaining, f"{at}[{index}]")
        )
    if key.startswith("@"):
        name = key[1:]
        return _family_targets(_family_attribute(value, name), remaining, f"{at}.{name}")
    if isinstance(value, Mapping) and key in value:
        return _family_targets(value[key], remaining, f"{at}.{key}")
    return ()


def _family_sequence(node: Mapping[str, Any], value: Any, at: str, code: str) -> None:
    """Apply sequence items, positional fields and uniqueness constraints.

    Parameters
    ----------
    node, value, at, code:
        Schema node, candidate sequence, diagnostic path and inherited refusal code.
    """

    # Essential interpreter complexity: keep the closed node instructions together;
    # family-specific callbacks would duplicate these structural checks.

    allowed = isinstance(value, Sequence) and not isinstance(value, (str, bytes))
    if node.get("kind") == "tuple":
        allowed = isinstance(value, (list, tuple))
    if not allowed:
        raise cell_error(at, "family value must be a sequence", code=code)
    if "length" in node and len(value) != node["length"]:
        raise cell_error(at, f"sequence must have length {node['length']}", code=code)
    for index, item in enumerate(value):
        child = node.get("positions", {}).get(index, node.get("items"))
        if child is not None:
            _apply_family({"code": code, **child}, item, f"{at}[{index}]")
    if node.get("sorted_unique") and list(value) != sorted(set(value)):
        raise cell_error(at, "sequence must be sorted and unique", code=code)
    if "unique_by" in node:
        keys = [tuple(item[key] for key in node["unique_by"]) for item in value]
        if len(set(keys)) != len(keys):
            raise cell_error(at, "sequence repeats an identity", code=code)
