"""The checker's native output schema is derived from ``gate.v3`` and closes it.

Four rungs produced zero catalog records because the checker kept emitting a gate
object that ``gate.v3`` refused, and the refusal MOVED between nested blocks as
each was named in the prompt: ``terminal_disposition`` (``arm``,
``result_sha256``, ``recommendation_sha256``, ``reason``), then ``rung_check``
(``required`` -- a JSON Schema keyword), then ``integrity`` (``findings``). The
mechanism was that ``--output-schema`` pointed at a one-field ``result_json``
STRING transport, so the provider's structured-output constraint enforced nothing
about the gate and prose was the only channel carrying its shape.

These tests hold the structural fix: the schema handed to the model is DERIVED
from the repository's own gate item definition, every closed block is enumerated
from the schema rather than from the next failure, and each historically observed
invented key is refused ONE AT A TIME so no single rejection stands in for the
rest.

Rung 8 added the VALUE-level twin of the invented-key failure: the checker wrote
a real ``fidelity.verdict`` on a ``metadata_batch`` item -- a key the kind-neutral
schema admitted with the full verdict enum -- and ``gate.v3``'s per-kind
conditional refused the complete verdict after the fact, killing both batch
members (m10517 + m9666, ``metadata-c001cad863d95a67``). The schema is therefore
derived PER GATE KIND now: the machine-known correlations between ``gate_kind``
and ``fidelity``/``terminal_disposition`` are baked into what the model can
represent at all.
"""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Iterator

import pytest
from jsonschema import Draft202012Validator

from menagerie.crawler.constants import GATE_SCHEMA_VERSION_V3, GateKind
from menagerie.crawler.native_output_schema import (
    DEFERRED_APPLICATORS,
    NativeOutputSchemaError,
    _lower,
    closed_key_sets,
    decode_native_result,
    native_output_schema,
)
from menagerie.crawler.schema import load_schema, load_schema_resource
from menagerie.crawler.tests.conftest import make_gate

ALL_GATE_KINDS = (GateKind.METADATA_BATCH, GateKind.FIDELITY, GateKind.TERMINAL_DISPOSITION)

#: Full gate item key set, straight from ``gate.v3``'s item definition.
_ITEM_KEYS = (
    "campaign_root_work_id",
    "confidence",
    "family_representative_id",
    "fidelity",
    "fidelity_identity",
    "field_checks",
    "integrity",
    "required_repairs",
    "rung_check",
    "stable_id",
    "terminal_disposition",
    "unsupported_claims",
    "verdict",
    "verified_hashes",
    "vet_identity",
    "work_id",
)

#: Closed blocks shared by every gate kind's derived schema.
_COMMON_CLOSED_BLOCKS = {
    "/": ("items",),
    "/properties/items/items/properties/field_checks/items": (
        "checked_source_ids",
        "evidence_ids",
        "field",
        "reason",
        "required_repair",
        "verdict",
    ),
    "/properties/items/items/properties/integrity": (
        "excerpt_discrepancies",
        "hash_mismatches",
        "locator_failures",
        "verdict",
    ),
    "/properties/items/items/properties/rung_check": (
        "findings",
        "highest_applicable",
        "selected_rung",
        "verdict",
    ),
    "/properties/items/items/properties/verified_hashes/anyOf/0": (
        "code",
        "code_manifest",
        "evidence",
        "family_template",
        "proposal",
        "source_manifest",
        "source_to_code_map",
    ),
    "/properties/items/items/properties/verified_hashes/anyOf/1": (
        "code",
        "evidence",
        "family_template",
        "proposal",
        "source_manifest",
        "source_to_code_map",
    ),
}

_FIDELITY_CLOSED_BLOCKS = {
    "/properties/items/items/properties/fidelity": (
        "contradictions",
        "material_checks",
        "omissions",
        "permanent_scar",
        "required",
        "unsupported_choices",
        "verdict",
    ),
    "/properties/items/items/properties/fidelity/properties/material_checks/items": (
        "category",
        "code_locator",
        "code_path",
        "evidence_ids",
        "reason",
        "source_id",
        "source_locator",
        "verdict",
    ),
}

_TERMINAL_KEYS = (
    "author_result_id",
    "author_result_sha256",
    "evidence_identity",
    "evidence_ids",
    "findings",
    "handoff_proposal_id",
    "handoff_sha256",
    "kind",
    "license_identity",
    "predicate",
    "source_ids",
    "source_manifest_identity",
    "verdict",
)

#: Every nested block inside a gate item that is CLOSED, per gate kind.
#: Enumerated from the schema, not discovered one rung at a time.
#: ``verified_hashes`` appears as two closed presence arms because ``gate.v3``
#: makes ``code_manifest`` conditional. ``fidelity`` is ABSENT from the
#: ``metadata_batch`` item entirely -- it is machine-owned there and the wrapper
#: stamps it -- and ``terminal_disposition`` is a closed object only on the
#: terminal kind, where it is pinned to its object arm (elsewhere it is pinned
#: ``null`` and contributes no closed block).
EXPECTED_CLOSED_BLOCKS = {
    GateKind.METADATA_BATCH: {
        **_COMMON_CLOSED_BLOCKS,
        "/properties/items/items": tuple(key for key in _ITEM_KEYS if key != "fidelity"),
    },
    GateKind.FIDELITY: {
        **_COMMON_CLOSED_BLOCKS,
        **_FIDELITY_CLOSED_BLOCKS,
        "/properties/items/items": _ITEM_KEYS,
    },
    GateKind.TERMINAL_DISPOSITION: {
        **_COMMON_CLOSED_BLOCKS,
        **_FIDELITY_CLOSED_BLOCKS,
        "/properties/items/items": _ITEM_KEYS,
        "/properties/items/items/properties/terminal_disposition": _TERMINAL_KEYS,
    },
}

#: Every key the checker was OBSERVED inventing, with the block it invented it
#: into and the gate kind whose derived schema exposes that block. Each is
#: asserted separately: a batch assertion would let one rejection stand in for
#: four, which is how the prompt fix looked complete while three more blocks were
#: still exposed.
OBSERVED_INVENTED_KEYS = [
    (GateKind.METADATA_BATCH, "rung_check", "required", False),
    (GateKind.METADATA_BATCH, "integrity", "findings", []),
    (GateKind.TERMINAL_DISPOSITION, "terminal_disposition", "arm", "SKIP_RECOMMENDATION"),
    (GateKind.TERMINAL_DISPOSITION, "terminal_disposition", "reason", "restated prose"),
    (
        GateKind.TERMINAL_DISPOSITION,
        "terminal_disposition",
        "result_sha256",
        "sha256:" + "0" * 64,
    ),
    (
        GateKind.TERMINAL_DISPOSITION,
        "terminal_disposition",
        "recommendation_sha256",
        "sha256:" + "0" * 64,
    ),
]

_INVENTED_KEY_IDS = [
    f"{gate_kind.value}.{block}.{key}" for gate_kind, block, key, _value in OBSERVED_INVENTED_KEYS
]


def _validator(gate_kind: GateKind) -> Draft202012Validator:
    """Return a validator for the exact schema the model is handed.

    Parameters
    ----------
    gate_kind:
        Gate kind whose derived schema is validated against.

    Returns
    -------
    jsonschema.Draft202012Validator
        Validator over the derived native output schema.
    """

    return Draft202012Validator(native_output_schema(gate_kind))


def _terminal_disposition_block() -> dict[str, Any]:
    """Return one fully populated closed terminal recommendation block.

    Returns
    -------
    dict[str, Any]
        Schema-shaped ``terminal_disposition`` object.
    """

    return {
        "author_result_id": "result-1",
        "author_result_sha256": "sha256:" + "0" * 64,
        "handoff_proposal_id": None,
        "handoff_sha256": None,
        "kind": "SKIP_RECOMMENDATION",
        "predicate": "not-a-real-NN",
        "verdict": "accepted",
        "source_manifest_identity": "sha256:" + "0" * 64,
        "source_ids": ["src-1"],
        "evidence_identity": "sha256:" + "0" * 64,
        "evidence_ids": ["ev-1"],
        "license_identity": "sha256:" + "0" * 64,
        "findings": [],
    }


def _compliant_payload(gate_kind: GateKind) -> dict[str, Any]:
    """Return one native final message that the derived schema accepts.

    Parameters
    ----------
    gate_kind:
        Gate kind whose compliant shape is built. A ``metadata_batch`` item
        carries NO fidelity block (machine-owned, wrapper-stamped) and a null
        terminal recommendation; a ``fidelity`` item carries its decided
        fidelity block; a terminal item carries the closed recommendation
        object.

    Returns
    -------
    dict[str, Any]
        Native transport carrying exactly one fully populated item.
    """

    if gate_kind is GateKind.FIDELITY:
        item = deepcopy(make_gate(["m_native"], gate_kind="fidelity")["items"][0])
        return {"items": [item]}
    item = deepcopy(make_gate(["m_native"])["items"][0])
    if gate_kind is GateKind.METADATA_BATCH:
        del item["fidelity"]
        return {"items": [item]}
    item["terminal_disposition"] = _terminal_disposition_block()
    return {"items": [item]}


def _blames_a_closed_object(error: Any) -> bool:
    """Report whether one validation error traces to a closed-object refusal.

    ``anyOf`` reports the failure at the union and carries the real cause in
    ``context``, so the search has to descend rather than read the top-level
    message.

    Parameters
    ----------
    error:
        Validation error from the derived schema.

    Returns
    -------
    bool
        True when this error, or a nested cause, is an ``additionalProperties``
        refusal.
    """

    if error.validator == "additionalProperties":
        return True
    return any(_blames_a_closed_object(nested) for nested in error.context or ())


def _relaxed_schema(gate_kind: GateKind) -> dict[str, Any]:
    """Return the derived schema with every object reopened.

    Parameters
    ----------
    gate_kind:
        Gate kind whose derived schema is relaxed.

    Returns
    -------
    dict[str, Any]
        Same schema with ``additionalProperties`` flipped to ``True``.
    """

    def reopen(node: Any) -> Any:
        """Flip every closed object open."""

        if isinstance(node, dict):
            return {
                key: (True if key == "additionalProperties" else reopen(value))
                for key, value in node.items()
            }
        if isinstance(node, list):
            return [reopen(value) for value in node]
        return node

    relaxed = reopen(native_output_schema(gate_kind))
    assert isinstance(relaxed, dict)
    return relaxed


@pytest.mark.smoke
@pytest.mark.parametrize("gate_kind", ALL_GATE_KINDS)
def test_every_closed_block_is_enumerated_from_the_schema(gate_kind: GateKind) -> None:
    """The exposed blocks are read off the gate schema, not off the next failure.

    This is the drift oracle. If ``gate.v3`` grows or renames a nested block, this
    comparison fails immediately instead of a live rung discovering it.

    Parameters
    ----------
    gate_kind:
        Gate kind whose derived schema is enumerated.
    """

    assert closed_key_sets(gate_kind) == EXPECTED_CLOSED_BLOCKS[gate_kind]


@pytest.mark.smoke
def test_closed_blocks_match_the_gate_schema_definitions() -> None:
    """Each derived key set is exactly the corresponding ``gate.v3`` key set.

    Enumerating the blocks is only meaningful if the enumeration is the SCHEMA's
    and not a second hand-maintained copy of it, so each one is compared back to
    the definition it came from. The single deliberate delta is machine
    ownership: the ``metadata_batch`` item omits ``fidelity`` -- exactly that one
    key -- because the wrapper stamps its gate-kind constant.
    """

    common = load_schema_resource("gate-common.schema.json")["$defs"]
    gate_defs = load_schema(GATE_SCHEMA_VERSION_V3)["$defs"]
    terminal_derived = closed_key_sets(GateKind.TERMINAL_DISPOSITION)

    for pointer, definition in (
        ("/properties/items/items", gate_defs["item"]),
        (
            "/properties/items/items/properties/terminal_disposition",
            gate_defs["terminal_disposition"],
        ),
        ("/properties/items/items/properties/integrity", common["integrity"]),
        ("/properties/items/items/properties/rung_check", common["rung_check"]),
        ("/properties/items/items/properties/fidelity", common["fidelity"]),
        ("/properties/items/items/properties/field_checks/items", common["field_check"]),
        (
            "/properties/items/items/properties/fidelity/properties/material_checks/items",
            common["material_check"],
        ),
        (
            "/properties/items/items/properties/verified_hashes/anyOf/0",
            common["verified_hashes"],
        ),
    ):
        assert terminal_derived[pointer] == tuple(sorted(definition["properties"])), pointer

    metadata_derived = closed_key_sets(GateKind.METADATA_BATCH)
    item_keys = set(tuple(sorted(gate_defs["item"]["properties"])))
    assert set(metadata_derived["/properties/items/items"]) == item_keys - {"fidelity"}


@pytest.mark.smoke
@pytest.mark.parametrize("gate_kind", ALL_GATE_KINDS)
def test_a_compliant_native_payload_passes(gate_kind: GateKind) -> None:
    """The closure is not achieved by rejecting everything.

    Parameters
    ----------
    gate_kind:
        Gate kind whose compliant payload is validated.
    """

    _validator(gate_kind).validate(_compliant_payload(gate_kind))


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("gate_kind", "block", "key", "value"),
    OBSERVED_INVENTED_KEYS,
    ids=_INVENTED_KEY_IDS,
)
def test_each_observed_invented_key_is_refused_on_its_own(
    gate_kind: GateKind, block: str, key: str, value: Any
) -> None:
    """Every historically observed invented key is refused, one at a time.

    Parameters
    ----------
    gate_kind:
        Gate kind whose derived schema exposes the block.
    block:
        Nested block the checker was observed inventing into.
    key:
        The exact invented key.
    value:
        A plausible value for it.
    """

    payload = _compliant_payload(gate_kind)
    payload["items"][0][block][key] = value

    errors = list(_validator(gate_kind).iter_errors(payload))

    assert errors, f"{block}.{key} was accepted by the native output schema"
    # Causality, not message text. Under ``anyOf`` the failure message dumps the
    # whole instance, so a substring check for the invented key would pass no
    # matter WHY the payload was rejected. The refusal must trace to a closed
    # object, and deleting exactly this key must make the payload valid again.
    assert any(_blames_a_closed_object(error) for error in errors), [e.message for e in errors]
    del payload["items"][0][block][key]
    _validator(gate_kind).validate(payload)


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("gate_kind", "block", "key", "value"),
    OBSERVED_INVENTED_KEYS,
    ids=_INVENTED_KEY_IDS,
)
def test_reopening_the_objects_admits_every_invented_key(
    gate_kind: GateKind, block: str, key: str, value: Any
) -> None:
    """The failing direction: the closure, not something else, does the work.

    Flipping ``additionalProperties`` to ``True`` must make each refusal above
    disappear. Without this, a passing rejection could be coming from an
    unrelated clause and the guard under test would never have decided anything.

    Parameters
    ----------
    gate_kind:
        Gate kind whose derived schema exposes the block.
    block:
        Nested block the checker was observed inventing into.
    key:
        The exact invented key.
    value:
        A plausible value for it.
    """

    payload = _compliant_payload(gate_kind)
    payload["items"][0][block][key] = value

    Draft202012Validator(_relaxed_schema(gate_kind)).validate(payload)


@pytest.mark.smoke
def test_metadata_fidelity_verdict_is_unrepresentable() -> None:
    """The rung-8 batch killer cannot be expressed by a metadata checker at all.

    Batch ``metadata-c001cad863d95a67`` (2026-08-05): the checker wrote
    ``fidelity.verdict: "match"`` on a ``metadata_batch`` item, ``gate.v3``'s
    per-kind conditional refused the complete verdict, and BOTH batch members
    (m10517, m9666) terminalized ``failed:runner/protocol-violation`` in the
    same second. The block is machine-owned on a metadata gate -- a gate-kind
    constant with no judgment in it -- so the derived metadata schema omits it
    exactly like the scaffold, and no fidelity verdict, legal-looking or
    otherwise, can survive constrained decoding.
    """

    payload = _compliant_payload(GateKind.METADATA_BATCH)
    payload["items"][0]["fidelity"] = {
        "required": False,
        "verdict": "match",
        "material_checks": [],
        "unsupported_choices": [],
        "contradictions": [],
        "omissions": [],
        "permanent_scar": False,
    }

    errors = list(_validator(GateKind.METADATA_BATCH).iter_errors(payload))
    assert errors, "a metadata checker could still author a fidelity block"
    assert any(_blames_a_closed_object(error) for error in errors)

    # Even the machine's own constant is refused FROM THE CHECKER: ownership is
    # by field, not by value, so there is no accepted spelling to drift from.
    payload["items"][0]["fidelity"]["verdict"] = "not-applicable"
    assert list(_validator(GateKind.METADATA_BATCH).iter_errors(payload))

    del payload["items"][0]["fidelity"]
    _validator(GateKind.METADATA_BATCH).validate(payload)

    # The failing direction: reopening the objects admits the block again, so the
    # refusal above is the closure deciding, not an unrelated clause.
    payload["items"][0]["fidelity"] = {"required": False, "verdict": "match"}
    Draft202012Validator(_relaxed_schema(GateKind.METADATA_BATCH)).validate(payload)


@pytest.mark.smoke
def test_nonterminal_kinds_pin_terminal_disposition_to_null() -> None:
    """A terminal recommendation on a metadata or fidelity item is unrepresentable.

    ``gate.v3``'s second conditional demands ``null`` there for nonterminal gate
    kinds; the same one-slip-kills-the-batch mechanics as the fidelity verdict
    apply, so the machine-known correlation is baked into the derived schema.
    """

    for gate_kind in (GateKind.METADATA_BATCH, GateKind.FIDELITY):
        payload = _compliant_payload(gate_kind)
        payload["items"][0]["terminal_disposition"] = _terminal_disposition_block()
        assert list(_validator(gate_kind).iter_errors(payload)), gate_kind
        payload["items"][0]["terminal_disposition"] = None
        _validator(gate_kind).validate(payload)


@pytest.mark.smoke
def test_terminal_kind_requires_the_closed_recommendation_object() -> None:
    """A terminal checker cannot return ``null`` where its one judgment belongs."""

    payload = _compliant_payload(GateKind.TERMINAL_DISPOSITION)
    payload["items"][0]["terminal_disposition"] = None
    assert list(_validator(GateKind.TERMINAL_DISPOSITION).iter_errors(payload))


@pytest.mark.smoke
def test_fidelity_kind_pins_required_true() -> None:
    """A fidelity checker cannot declare its own lane optional.

    ``required`` is a machine fact of the lane -- the semantic validator refuses
    ``required is not True`` on a fidelity envelope -- so the derived schema pins
    it and the contradiction cannot be emitted in the first place.
    """

    payload = _compliant_payload(GateKind.FIDELITY)
    payload["items"][0]["fidelity"]["required"] = False
    assert list(_validator(GateKind.FIDELITY).iter_errors(payload))
    payload["items"][0]["fidelity"]["required"] = True
    _validator(GateKind.FIDELITY).validate(payload)


@pytest.mark.smoke
def test_machine_owned_scaffold_is_unreachable_from_the_native_schema() -> None:
    """A scaffold field is not merely discouraged; it cannot be expressed.

    The strict subset requires every declared property to be PRESENT, so listing
    the scaffold would have inverted "omitting is always correct" into "supplying
    a fabricated identity is mandatory". It is absent instead.
    """

    for gate_kind in ALL_GATE_KINDS:
        schema = native_output_schema(gate_kind)
        for field in (
            "schema_version",
            "gate_id",
            "ledger_seq",
            "payload_sha256",
            "gate_kind",
            "batch_size",
            "gate_round",
            "gate_identity",
            "checker",
            "result_envelope_sha256",
            "author_result_schema_identity",
            "dispatcher_identity",
        ):
            assert field not in schema["properties"], (gate_kind, field)

    errors = list(
        _validator(GateKind.METADATA_BATCH).iter_errors({"items": [], "gate_id": "gate-1"})
    )
    assert any("gate_id" in error.message for error in errors)


@pytest.mark.smoke
@pytest.mark.parametrize("gate_kind", ALL_GATE_KINDS)
def test_derived_schema_stays_inside_the_strict_structured_output_subset(
    gate_kind: GateKind,
) -> None:
    """Every construct the provider rejects is absent from what it is handed.

    Verified against the live API on 2026-07-30: ``oneOf``, ``allOf``, remote
    ``$ref``, a ``$ref`` with sibling keywords, and a ``required`` list missing a
    declared property each return HTTP 400 ``invalid_json_schema``.

    Parameters
    ----------
    gate_kind:
        Gate kind whose derived schema is walked.
    """

    schema = native_output_schema(gate_kind)

    def walk(node: Any, pointer: str) -> Iterator[tuple[str, dict[str, Any]]]:
        """Yield every subschema with its pointer."""

        if isinstance(node, dict):
            yield pointer, node
            for key, value in node.items():
                yield from walk(value, f"{pointer}/{key}")
        elif isinstance(node, list):
            for index, value in enumerate(node):
                yield from walk(value, f"{pointer}/{index}")

    forbidden = {"oneOf", "allOf", "if", "then", "else", "not", "$ref", "$defs"}
    for pointer, node in walk(schema, ""):
        assert not (forbidden & set(node)), f"{pointer}: {sorted(forbidden & set(node))}"
        if node.get("type") == "object":
            assert node.get("additionalProperties") is False, pointer
            assert sorted(node.get("required") or ()) == sorted(node.get("properties") or {}), (
                pointer
            )

    text = json.dumps(schema)
    assert len(text) < 120_000
    assert Draft202012Validator.check_schema(schema) is None


@pytest.mark.smoke
def test_the_lowering_refuses_an_unlisted_unsupported_construct() -> None:
    """A NEW conditional in the gate schema raises instead of vanishing quietly.

    Silently dropping an applicator the subset cannot express is how a derived
    schema drifts away from the schema it claims to derive from. Only the exact
    pointers in ``DEFERRED_APPLICATORS`` are allowed through.
    """

    with pytest.raises(NativeOutputSchemaError) as excinfo:
        _lower({"type": "object", "properties": {}, "if": {"const": 1}}, "/somewhere")
    assert "'if' at /somewhere/if" in str(excinfo.value)

    # And the deferral is exact: the same keyword at its allowlisted pointer is
    # accepted, so the guard is discriminating rather than uniformly permissive.
    assert set(DEFERRED_APPLICATORS) == {"/properties/verified_hashes/allOf"}


@pytest.mark.smoke
def test_deferred_constraints_are_still_enforced_by_the_gate_schema() -> None:
    """What the subset cannot express is not thereby unenforced.

    The native schema is an ADDITIONAL upstream constraint. The gate-kind
    conditionals and the ``code``/``code_manifest`` correlation still live in
    ``gate.v3`` exactly as before, and the tripwire is untouched.
    """

    gate = load_schema(GATE_SCHEMA_VERSION_V3)
    assert gate["allOf"], "gate.v3 lost its gate-kind conditionals"
    verified_hashes = load_schema_resource("gate-common.schema.json")["$defs"]["verified_hashes"]
    assert verified_hashes["allOf"], "gate.v3 lost the code_manifest correlation"
    assert gate["additionalProperties"] is False


@pytest.mark.smoke
@pytest.mark.parametrize(
    "payload",
    [
        {"result_json": "{}"},
        {"items": [], "gate_id": "gate-1"},
        {},
        {"items": {}},
        [],
        "items",
    ],
    ids=["legacy-string-transport", "extra-key", "empty", "items-not-array", "list", "string"],
)
def test_non_native_final_messages_are_refused(payload: Any) -> None:
    """The wrapper accepts exactly the native transport and nothing adjacent.

    The legacy ``result_json`` string transport is refused explicitly: leaving it
    accepted would leave the vacuous path reachable.

    Parameters
    ----------
    payload:
        Candidate final message.
    """

    with pytest.raises(NativeOutputSchemaError):
        decode_native_result(payload)


@pytest.mark.smoke
def test_native_decode_returns_only_the_authored_items() -> None:
    """Decoding yields the checker's authority and nothing that shadows scaffold."""

    payload = _compliant_payload(GateKind.METADATA_BATCH)
    decoded = decode_native_result(payload)
    assert set(decoded) == {"items"}
    assert decoded["items"] == payload["items"]
