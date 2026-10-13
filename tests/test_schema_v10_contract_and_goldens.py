"""tlspec v10 coordinated schema write: the contract pins and the v10 field registry.

v10 is ONE bump for every new saved field of its window (R2-05 with R2-38).
This module is the field registry of record for that window:
``V10_FIELD_ADMISSIONS`` lists every persisted field a lane may add over the
frozen v9 contract, with its registered ``FieldPolicy``. A persisted-name
addition missing from the registry fails here, so no field reaches the v10
contract without a registration; an admitted field whose writer has not landed
yet is simply absent from the live contract.

The module also pins the rest of the v10 write: the v9 contract of record stays
byte-identical, the v10 contract states the true floor release (2.31.0, the
first tlspec-6 writer) where v9 froze "2.33", the generation step-score names
are reserved and credited to inseq (U22.3), the facet snapshot stays a DROP
field (R2-36), and the floor, ceiling and older-artifact load doors hold.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import warnings
from pathlib import Path

import pytest

import torchlens as tl
from torchlens._io import (
    MIN_TLSPEC_VERSION,
    MIN_TORCHLENS_VERSION_TEXT,
    TLSPEC_VERSION,
    ArtifactSchemaAgeWarning,
    ArtifactVersionAboveRuntimeError,
    ArtifactVersionBelowFloorError,
    read_tlspec_version,
)
from torchlens._io.compat_ledger import floor_metadata, pair_is_governed
from torchlens._io.format_contract import (
    RESERVED_STEP_SCORE_CREDIT,
    RESERVED_STEP_SCORE_NAMES,
)
from torchlens._io.writer_contract import writer_contract

_HERE = Path(__file__).parent
SCHEMAS = _HERE.parent / "torchlens" / "schemas"
V9_CONTRACT = SCHEMAS / "writer_contract_v9.json"
V10_CONTRACT = SCHEMAS / "writer_contract_v10.json"

#: SHA-256 of the frozen v9 contract of record. The v10 write must never touch
#: its bytes: the v9 contract is a ledger digest input for every v9 artifact.
V9_CONTRACT_SHA256 = (
    "a2a8c3b1bcd0589ecff53d379b9519b017f36a9fe063f648fe986565e7747fc8"  # pragma: allowlist secret
)

#: The v10 FIELD REGISTRY: record contract key -> {field name -> FieldPolicy
#: value} for every persisted field admitted over the v9 contract. A lane adds
#: a field only after its SCHEMA-REQUESTS row is answered by the commit that
#: adds the admission here. Removals and retypes of v9 fields are registered
#: in ``V10_FIELD_RETYPES`` (removals stay alias-or-fail, see
#: ``tests/test_tlspec_envelope_contract.py``).
V10_FIELD_ADMISSIONS: dict[str, dict[str, str]] = {
    # SCHEMA-REQUESTS row 2 (RUN-B, R2-37): the byte disclosure for the
    # autograd graph the attached saved activations keep alive. DROP: a loaded
    # or unpickled trace pins nothing, so a persisted capture-time number
    # would read as current; it loads as None ("not measured").
    # SCHEMA-REQUESTS row 9 (REAL-2B2, AGT-2362): the uncalled-module record,
    # taken while the source model is live. DROP: a restored trace reads
    # num_uncalled_modules as None (unknown), never a stale 0.
    "trace": {"pinned_autograd_memory": "drop", "_uncalled_module_addresses": "drop"},
    # SCHEMA-REQUESTS row 6 (OBSERVER, AGT-2360): the record of the plain
    # attributes a forward wrote (arbitrary user objects). DROP, both the public
    # property and its slot: a loaded, unpickled or deep-copied trace reads None.
    "module": {"attribute_writes": "drop", "_attribute_writes": "drop"},
}

#: Registered retypes of existing v9 persisted fields: record key -> {field
#: name -> one-line description of the old and new type}. A retype keeps the
#: field name, so the name diff cannot see it; the registration is the record.
V10_FIELD_RETYPES: dict[str, dict[str, str]] = {}

_INSEQ_STEP_SCORES_MAP_KEYS = (
    "logit",
    "probability",
    "entropy",
    "crossentropy",
    "perplexity",
    "contrast_logits",
    "contrast_prob",
    "contrast_logits_diff",
    "contrast_prob_diff",
    "pcxmi",
    "kl_divergence",
    "in_context_pvi",
    "mc_dropout_prob_avg",
    "top_p_size",
)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _record_contracts(path: Path) -> dict:
    return _load(path)["contract"]["record_contracts"]


@pytest.mark.smoke
def test_v9_to_v10_additions_are_all_registered() -> None:
    """Every field added over v9 (persisted or DROP) is admitted, under its registered policy.

    DROP fields never enter ``persisted_field_names`` but do enter the policy
    table, so the gate reads both.
    """

    old = _record_contracts(V9_CONTRACT)
    new = _record_contracts(V10_CONTRACT)
    assert set(old) == set(new), "no record contract may appear or disappear at v10"
    assert set(V10_FIELD_ADMISSIONS) <= set(new), "admissions name unknown record contracts"
    for record_key in sorted(new):
        new_names = set(new[record_key]["persisted_field_names"]) | set(
            new[record_key]["field_policies"]
        )
        old_names = set(old[record_key]["persisted_field_names"]) | set(
            old[record_key]["field_policies"]
        )
        admitted = V10_FIELD_ADMISSIONS.get(record_key, {})
        unregistered = new_names - old_names - set(admitted)
        assert unregistered == set(), (
            f"{record_key}: field(s) {sorted(unregistered)} reach the v10 "
            "contract without a registration; request them in SCHEMA-REQUESTS and "
            "add the admission to V10_FIELD_ADMISSIONS in the same change"
        )
        policies = new[record_key]["field_policies"]
        for name in sorted(set(admitted) & new_names):
            assert policies[name] == admitted[name], (
                f"{record_key}.{name}: live policy {policies[name]!r} differs from the "
                f"registered {admitted[name]!r}"
            )


def test_admissions_are_new_names_and_retypes_are_existing_names() -> None:
    old = _record_contracts(V9_CONTRACT)
    policy_values = {"keep", "blob", "blob_recursive", "drop", "stringify", "weakref_strip"}
    for record_key, admitted in V10_FIELD_ADMISSIONS.items():
        v9_names = set(old[record_key]["persisted_field_names"]) | set(
            old[record_key]["field_policies"]
        )
        clash = set(admitted) & v9_names
        assert clash == set(), f"{record_key}: {sorted(clash)} are already v9 fields"
        assert set(admitted.values()) <= policy_values
    for record_key, retyped in V10_FIELD_RETYPES.items():
        missing = set(retyped) - set(old[record_key]["persisted_field_names"])
        assert missing == set(), f"{record_key}: retypes name non-v9 fields {sorted(missing)}"


def test_v9_contract_of_record_is_byte_identical() -> None:
    digest = hashlib.sha256(V9_CONTRACT.read_bytes()).hexdigest()
    assert digest == V9_CONTRACT_SHA256, (
        "writer_contract_v9.json changed; the v9 contract of record is frozen since "
        "the v10 bump (regenerate only writer_contract_v10.json)"
    )


def test_contract_versions_floor_text_and_digests() -> None:
    v9 = _load(V9_CONTRACT)
    v10 = _load(V10_CONTRACT)
    assert v9["contract"]["tlspec_version"] == 9
    assert v10["contract"]["tlspec_version"] == TLSPEC_VERSION == 10
    assert v9["contract"]["min_torchlens_version"] == "2.33", "v9 stays frozen"
    assert v10["contract"]["min_torchlens_version"] == "2.31.0"
    assert floor_metadata()["producer_floor_writer"] == MIN_TORCHLENS_VERSION_TEXT
    assert v10["contract"]["min_tlspec_version"] == MIN_TLSPEC_VERSION == 6
    assert v9["writer_contract_digest"] != v10["writer_contract_digest"]


def test_step_score_names_are_reserved_and_credited() -> None:
    """U22.3: inseq's step-score names are reserved in the v10 contract of record."""

    assert RESERVED_STEP_SCORE_NAMES == _INSEQ_STEP_SCORES_MAP_KEYS
    reserved = _load(V10_CONTRACT)["contract"]["reserved_vocabularies"]["step_scores"]
    assert reserved["names"] == list(_INSEQ_STEP_SCORES_MAP_KEYS)
    assert reserved["status"] == "reserved"
    assert "inseq" in reserved["credit"] and reserved["credit"] == RESERVED_STEP_SCORE_CREDIT
    assert writer_contract()["reserved_vocabularies"]["step_scores"] == reserved


def test_facet_snapshot_stays_a_drop_field() -> None:
    """R2-36: the facet snapshot never persists (pickle round-trip: test_fw2b_*)."""

    trace_contract = _record_contracts(V10_CONTRACT)["trace"]
    assert trace_contract["field_policies"]["facet_registry_snapshot"] == "drop"


def test_below_floor_refusal_names_the_corrected_floor() -> None:
    state = {"tlspec_version": MIN_TLSPEC_VERSION - 1}
    with pytest.raises(ArtifactVersionBelowFloorError) as excinfo:
        read_tlspec_version(state, cls_name="Trace")
    fields = excinfo.value.fields
    assert fields["code"] == "artifact_version_below_floor"
    assert fields["floor_tlspec_version"] == 6
    assert fields["floor_torchlens_version"] == "2.31.0"
    assert "(torchlens 2.31.0)" in str(excinfo.value)
    assert "2.33" not in str(excinfo.value)


def test_modellog_216_refusal_keeps_the_verified_reader_remedy() -> None:
    from torchlens._io.compat_ledger import raise_if_modellog_portable

    with pytest.raises(ArtifactVersionBelowFloorError) as excinfo:
        raise_if_modellog_portable("v2.16_modellog_portable", "Artifact", "x.tlspec")
    assert "torchlens==2.17.0" in excinfo.value.fields["remedy"]


def test_future_version_refuses_on_state_and_producer_windows() -> None:
    with pytest.raises(ArtifactVersionAboveRuntimeError) as excinfo:
        read_tlspec_version({"tlspec_version": TLSPEC_VERSION + 1}, cls_name="Trace")
    assert excinfo.value.fields["ceiling_tlspec_version"] == 10
    assert pair_is_governed("2.36.0", 10)
    assert not pair_is_governed("2.35.0", 10), "v10 was never written before 2.36.0"
    assert pair_is_governed("2.34.1", 9)


def test_v9_artifact_loads_at_its_recorded_schema(tmp_path: Path) -> None:
    """A committed v9 artifact (writer 2.34.1) loads under v10 with the age advisory."""

    source = _HERE / "agent_surface_goldens" / "clean.tlspec"
    manifest = _load(source / "manifest.json")
    assert manifest["tlspec_version"] == 9, "fixture must stay a v9 artifact"
    work = tmp_path / "clean_v9.tlspec"
    shutil.copytree(source, work)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        trace = tl.load(work)
    assert any(issubclass(w.category, ArtifactSchemaAgeWarning) for w in caught)
    assert trace.tlspec_version == 9, "a loaded artifact keeps its recorded schema"
    assert len(trace.ops) > 0
