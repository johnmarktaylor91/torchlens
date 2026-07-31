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
"""

from __future__ import annotations

import json
from copy import deepcopy
from typing import Any, Iterator

import pytest
from jsonschema import Draft202012Validator

from menagerie.crawler.constants import GATE_SCHEMA_VERSION_V3
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

#: Every nested block inside a gate item that is CLOSED, and therefore sits in
#: the position that produced the moving failure. Enumerated from the schema, not
#: discovered one rung at a time. ``verified_hashes`` appears as two closed
#: presence arms because ``gate.v3`` makes ``code_manifest`` conditional.
EXPECTED_CLOSED_BLOCKS = {
    "/": ("items",),
    "/properties/items/items": (
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
    ),
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
    "/properties/items/items/properties/terminal_disposition/anyOf/0": (
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

#: Every key the checker was OBSERVED inventing, with the block it invented it
#: into. Each is asserted separately: a batch assertion would let one rejection
#: stand in for four, which is how the prompt fix looked complete while three
#: more blocks were still exposed.
OBSERVED_INVENTED_KEYS = [
    ("rung_check", "required", False),
    ("integrity", "findings", []),
    ("terminal_disposition", "arm", "SKIP_RECOMMENDATION"),
    ("terminal_disposition", "reason", "restated prose"),
    ("terminal_disposition", "result_sha256", "sha256:" + "0" * 64),
    ("terminal_disposition", "recommendation_sha256", "sha256:" + "0" * 64),
]


def _validator() -> Draft202012Validator:
    """Return a validator for the exact schema the model is handed.

    Returns
    -------
    jsonschema.Draft202012Validator
        Validator over the derived native output schema.
    """

    return Draft202012Validator(native_output_schema())


def _compliant_payload() -> dict[str, Any]:
    """Return one native final message that the derived schema accepts.

    Returns
    -------
    dict[str, Any]
        Native transport carrying exactly one fully populated item.
    """

    item = deepcopy(make_gate(["m_native"])["items"][0])
    item["terminal_disposition"] = {
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


def _relaxed_schema() -> dict[str, Any]:
    """Return the derived schema with every object reopened.

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

    relaxed = reopen(native_output_schema())
    assert isinstance(relaxed, dict)
    return relaxed


@pytest.mark.smoke
def test_every_closed_block_is_enumerated_from_the_schema() -> None:
    """The exposed blocks are read off the gate schema, not off the next failure.

    This is the drift oracle. If ``gate.v3`` grows or renames a nested block, this
    comparison fails immediately instead of a live rung discovering it.
    """

    assert closed_key_sets() == EXPECTED_CLOSED_BLOCKS


@pytest.mark.smoke
def test_closed_blocks_match_the_gate_schema_definitions() -> None:
    """Each derived key set is exactly the corresponding ``gate.v3`` key set.

    Enumerating the blocks is only meaningful if the enumeration is the SCHEMA's
    and not a second hand-maintained copy of it, so each one is compared back to
    the definition it came from.
    """

    common = load_schema_resource("gate-common.schema.json")["$defs"]
    gate_defs = load_schema(GATE_SCHEMA_VERSION_V3)["$defs"]
    derived = closed_key_sets()

    for pointer, definition in (
        ("/properties/items/items", gate_defs["item"]),
        (
            "/properties/items/items/properties/terminal_disposition/anyOf/0",
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
        assert derived[pointer] == tuple(sorted(definition["properties"])), pointer


@pytest.mark.smoke
def test_a_compliant_native_payload_passes() -> None:
    """The closure is not achieved by rejecting everything."""

    _validator().validate(_compliant_payload())


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("block", "key", "value"),
    OBSERVED_INVENTED_KEYS,
    ids=[f"{block}.{key}" for block, key, _value in OBSERVED_INVENTED_KEYS],
)
def test_each_observed_invented_key_is_refused_on_its_own(
    block: str, key: str, value: Any
) -> None:
    """Every historically observed invented key is refused, one at a time.

    Parameters
    ----------
    block:
        Nested block the checker was observed inventing into.
    key:
        The exact invented key.
    value:
        A plausible value for it.
    """

    payload = _compliant_payload()
    payload["items"][0][block][key] = value

    errors = list(_validator().iter_errors(payload))

    assert errors, f"{block}.{key} was accepted by the native output schema"
    # Causality, not message text. Under ``anyOf`` the failure message dumps the
    # whole instance, so a substring check for the invented key would pass no
    # matter WHY the payload was rejected. The refusal must trace to a closed
    # object, and deleting exactly this key must make the payload valid again.
    assert any(_blames_a_closed_object(error) for error in errors), [e.message for e in errors]
    del payload["items"][0][block][key]
    _validator().validate(payload)


@pytest.mark.smoke
@pytest.mark.parametrize(
    ("block", "key", "value"),
    OBSERVED_INVENTED_KEYS,
    ids=[f"{block}.{key}" for block, key, _value in OBSERVED_INVENTED_KEYS],
)
def test_reopening_the_objects_admits_every_invented_key(
    block: str, key: str, value: Any
) -> None:
    """The failing direction: the closure, not something else, does the work.

    Flipping ``additionalProperties`` to ``True`` must make each refusal above
    disappear. Without this, a passing rejection could be coming from an
    unrelated clause and the guard under test would never have decided anything.

    Parameters
    ----------
    block:
        Nested block the checker was observed inventing into.
    key:
        The exact invented key.
    value:
        A plausible value for it.
    """

    payload = _compliant_payload()
    payload["items"][0][block][key] = value

    Draft202012Validator(_relaxed_schema()).validate(payload)


@pytest.mark.smoke
def test_machine_owned_scaffold_is_unreachable_from_the_native_schema() -> None:
    """A scaffold field is not merely discouraged; it cannot be expressed.

    The strict subset requires every declared property to be PRESENT, so listing
    the scaffold would have inverted "omitting is always correct" into "supplying
    a fabricated identity is mandatory". It is absent instead.
    """

    schema = native_output_schema()
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
        assert field not in schema["properties"], field

    errors = list(_validator().iter_errors({"items": [], "gate_id": "gate-1"}))
    assert any("gate_id" in error.message for error in errors)


@pytest.mark.smoke
def test_derived_schema_stays_inside_the_strict_structured_output_subset() -> None:
    """Every construct the provider rejects is absent from what it is handed.

    Verified against the live API on 2026-07-30: ``oneOf``, ``allOf``, remote
    ``$ref``, a ``$ref`` with sibling keywords, and a ``required`` list missing a
    declared property each return HTTP 400 ``invalid_json_schema``.
    """

    schema = native_output_schema()

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

    payload = _compliant_payload()
    decoded = decode_native_result(payload)
    assert set(decoded) == {"items"}
    assert decoded["items"] == payload["items"]
