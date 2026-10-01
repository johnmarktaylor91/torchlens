"""Writer-contract lockstep: the committed contract of record vs the live tree.

Three guarantees (ecosystem MEMO 3.1/3.3/3.4, megaplan P05):

1. LOCKSTEP -- regenerating the writer contract from the live tree reproduces
   ``torchlens/schemas/writer_contract_v<TLSPEC_VERSION>.json`` exactly. Any persistence
   contract change (a field added/removed/re-policied, a codec or component
   schema bumped) fails here until the golden is consciously regenerated in
   the same PR, making every contract diff reviewable. Regenerate with
   ``TORCHLENS_UPDATE_WRITER_CONTRACT=1 pytest tests/test_tlspec_envelope_contract.py``.

2. ALIAS-OR-FAIL (the per-PR cheap variant of the forward-load gate, G8) --
   a persisted field name present in the contract of record but absent from
   the live tree is a reader regression unless the live contract carries an
   alias rule for it. A deleted field without an alias must be LOUD.

3. DIGEST SEMANTICS -- the digest moves exactly when the canonical contract
   moves, and the persisted field-NAME sets are digest inputs (the
   v2.33.0-vs-v2.34.1 same-stamp drift is exactly what this catches).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from torchlens._io import TLSPEC_VERSION
from torchlens._io.field_registry import (
    RECORD_CONTRACT_CLASSES,
    FieldTier,
    field_tiers,
    persisted_field_names,
    record_field_policies,
)
from torchlens._io.writer_contract import writer_contract, writer_contract_digest

pytestmark = [pytest.mark.smoke]

GOLDEN_PATH = (
    Path(__file__).parent.parent
    / "torchlens"
    / "schemas"
    / f"writer_contract_v{TLSPEC_VERSION}.json"
)
UPDATE_FLAG = "TORCHLENS_UPDATE_WRITER_CONTRACT"


def _write_golden(payload: dict) -> None:
    with GOLDEN_PATH.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, sort_keys=True)
        fh.write("\n")


def test_contract_of_record_matches_live_tree() -> None:
    from _oracle_env import (
        flag_armed,
        guard_wrap_state_for_golden_update,
        require_update_reason,
    )

    contract = writer_contract()
    payload = {
        "writer_contract_digest": writer_contract_digest(contract),
        "contract": contract,
    }
    # Enums serialize as their .value through the canonical JSON path; the
    # golden comparison must see the identical plain-data projection.
    payload = json.loads(json.dumps(payload, sort_keys=True))
    if flag_armed(os.environ, UPDATE_FLAG):
        # In-process generation: refuse to freeze bytes on a torch earlier
        # tests already wrapped (SF-53 census rule; this golden is pure class
        # metadata, but the guard keeps the family uniformly conformant) and
        # demand the WHY before bytes are written. No provenance sidecar: the
        # golden is shipped package data (schemas/*.json), so provenance
        # lives in the regenerating commit message instead.
        guard_wrap_state_for_golden_update(UPDATE_FLAG)
        require_update_reason(UPDATE_FLAG)
        _write_golden(payload)
    committed = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    assert committed == payload, (
        "The live persistence contract differs from the committed contract of "
        f"record ({GOLDEN_PATH.name}). If this change is intentional, it is a "
        "schema-territory change (P05/C07 fence): regenerate the golden in the "
        f"same PR with {UPDATE_FLAG}=1 and, for any REMOVED persisted field, "
        "add an alias rule (PORTABLE_STATE_ALIASES on the owning class) so "
        "governed artifacts keep loading."
    )


def test_removed_persisted_fields_require_aliases() -> None:
    """Alias-or-fail: a dropped persisted name must be covered by an alias."""

    committed = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    live = json.loads(json.dumps(writer_contract(), sort_keys=True))
    failures: list[str] = []
    for record_key, committed_record in committed["contract"]["record_contracts"].items():
        live_record = live["record_contracts"].get(record_key)
        if live_record is None:
            failures.append(f"record contract {record_key!r} disappeared entirely")
            continue
        removed = set(committed_record["persisted_field_names"]) - set(
            live_record["persisted_field_names"]
        )
        uncovered = removed - set(live_record["aliases"])
        if uncovered:
            failures.append(
                f"{record_key}: persisted field(s) {sorted(uncovered)} were removed "
                "without an alias rule"
            )
    assert not failures, (
        "Reader regression (MEMO 3.3 alias-or-fail): a field a governed writer "
        "persisted no longer loads and carries no alias. Add "
        "PORTABLE_STATE_ALIASES coverage on the owning class or restore the "
        f"field. Details: {failures}"
    )


def test_digest_tracks_persisted_field_name_sets() -> None:
    """The same-stamp drift catcher: field-name sets are digest inputs."""

    contract = writer_contract()
    baseline = writer_contract_digest(contract)
    mutated = json.loads(json.dumps(contract, sort_keys=True))
    mutated["record_contracts"]["op"]["persisted_field_names"].append(
        "a_field_a_future_release_added"
    )
    assert writer_contract_digest(mutated) != baseline


def test_tier_partition_is_closed_and_total() -> None:
    """Every persisted field of every record contract has exactly one tier."""

    for record_key in RECORD_CONTRACT_CLASSES:
        tiers = field_tiers(record_key)
        assert set(tiers) == set(persisted_field_names(record_key))
        assert all(isinstance(tier, FieldTier) for tier in tiers.values())


def test_tier_partition_moves_no_fields() -> None:
    """P05 is the envelope only: tiers annotate, policies are untouched.

    Every field the tier table classifies must still be persisted under its
    declared FieldPolicy -- the partition can never itself change what a
    writer emits (field moves land at C07 via the 8->9 bump).
    """

    for record_key in RECORD_CONTRACT_CLASSES:
        policies = record_field_policies(record_key)
        for name in field_tiers(record_key):
            assert name in policies
            assert policies[name].value != "drop"


def test_registry_mirrors_the_lockstep_catalogs() -> None:
    """RECORD_CONTRACT_CLASSES covers exactly the FIELD_ORDER catalog owners."""

    from tests.test_schema_lockstep import CATALOGS

    lockstep_owners = {catalog.owner.__name__ for catalog in CATALOGS if catalog.owner is not None}
    registry_owners = {class_name for _, class_name in RECORD_CONTRACT_CLASSES.values()}
    assert registry_owners == lockstep_owners


def test_component_spec_module_list_is_complete() -> None:
    """Every file declaring a PORTABLE_STATE_SPEC is known to the registry."""

    from torchlens._io.field_registry import COMPONENT_SPEC_MODULES

    package_root = Path(__file__).parent.parent / "torchlens"
    declaring: set[str] = set()
    for path in package_root.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if "PORTABLE_STATE_SPEC: ClassVar" in text or "PORTABLE_STATE_SPEC: dict" in text:
            rel = path.relative_to(package_root.parent).with_suffix("")
            declaring.add(".".join(rel.parts))
    known = set(COMPONENT_SPEC_MODULES) | {
        module_name for module_name, _ in RECORD_CONTRACT_CLASSES.values()
    }
    missing = declaring - known
    assert not missing, (
        "New PORTABLE_STATE_SPEC-declaring module(s) unknown to the field "
        f"registry: {sorted(missing)}. Add them to COMPONENT_SPEC_MODULES in "
        "torchlens/_io/field_registry.py (P05/C07 schema fence) and regenerate "
        "the writer contract golden."
    )
