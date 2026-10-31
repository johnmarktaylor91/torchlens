"""tlspec v10 coordinated schema write: the contract pins and the v10 field registry.

v10 is ONE bump for every new saved field of its window (R2-05 with R2-38).
This module is the field registry of record for that window:
``V10_FIELD_ADMISSIONS`` lists every persisted field a lane added over the
v9 contract, with its registered ``FieldPolicy``. A persisted-name
addition missing from the registry fails here, so no field reaches the v10
contract without a registration; an admitted field whose writer has not landed
yet is simply absent from the live contract.

The module also pins the rest of the v10 write: the v10 contract states the true
floor release (2.31.0, the first tlspec-6 writer), the generation step-score names
are reserved and credited to inseq (U22.3), the facet snapshot stays a DROP
field (R2-36), and the floor, ceiling and 2.x refusal doors hold. The v9 and v8
contracts left with the 2.x readers, so the diff of admitted fields over v9 is gone;
the next window diffs against the v10 contract of record.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from torchlens._io import (
    MIN_TLSPEC_VERSION,
    MIN_TORCHLENS_VERSION_TEXT,
    TLSPEC_VERSION,
    ArtifactTooNewError,
    ArtifactTooOldError,
    read_tlspec_version,
)
from torchlens._io._pre3_module_paths import modellog_216_error
from torchlens._io.format_contract import (
    RESERVED_STEP_SCORE_CREDIT,
    RESERVED_STEP_SCORE_NAMES,
)
from torchlens._io.writer_contract import writer_contract

_HERE = Path(__file__).parent
SCHEMAS = _HERE.parent / "torchlens" / "schemas"
V10_CONTRACT = SCHEMAS / "writer_contract_v10.json"

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
    # SCHEMA-REQUESTS row 8 (REPORT, forks.13): why a CPU forward peak is None.
    # KEEP: the reason travels with the None it explains; v9 artifacts read None.
    # SCHEMA-REQUESTS row 9 (REAL-2B2, AGT-2362): the uncalled-module record,
    # taken while the source model is live. DROP: a restored trace reads
    # num_uncalled_modules as None (unknown), never a stale 0.
    # SCHEMA-REQUESTS rows 16 and 17 (measure.2, JMT 28 Oct 2026): the CPU peaks
    # are opt-in. KEEP: why a CPU backward peak is None travels with the None;
    # DROP: the measure_host_peak_memory session knob (load restores False).
    "trace": {
        "pinned_autograd_memory": "drop",
        "forward_peak_memory_reason": "keep",
        "_uncalled_module_addresses": "drop",
        "backward_peak_memory_reason": "keep",
        "measure_host_peak_memory": "drop",
    },
    # SCHEMA-REQUESTS row 6 (OBSERVER, AGT-2360): the record of the plain
    # attributes a forward wrote (arbitrary user objects). DROP, both the public
    # property and its slot: a loaded, unpickled or deep-copied trace reads None.
    "module": {"attribute_writes": "drop", "_attribute_writes": "drop"},
    # SCHEMA-REQUESTS row 15 (FLOPS-2, U05.4): the counted executed backward
    # and its attribution, a closed dict of ints, None and short strings
    # written at save. KEEP: a loaded trace reads the counted answer; v9 and
    # earlier v10 artifacts default-fill None and read unknown.
    "backward_pass": {"executed_flops_record": "keep"},
}

#: Registered retypes of existing v9 persisted fields: record key -> {field
#: name -> one-line description of the old and new type}. A retype keeps the
#: field name, so the name diff cannot see it; the registration is the record.
V10_FIELD_RETYPES: dict[str, dict[str, str]] = {
    # SCHEMA-REQUESTS row 7 (REPORT, forks.13): a CPU forward delta of 0 is
    # recorded as None with forward_peak_memory_reason; KEEP policy unchanged.
    # SCHEMA-REQUESTS row 16 (measure.2): a CPU backward peak without the
    # measure_host_peak_memory opt-in is None with backward_peak_memory_reason.
    "trace": {
        "forward_peak_memory": "Bytes -> Bytes | None (CPU zero delta reads None)",
        "backward_peak_memory": "Bytes -> Bytes | None (CPU without the opt-in reads None)",
    },
}

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


def test_v10_admissions_match_the_v10_contract() -> None:
    """Every admitted field that has landed carries its registered policy in the contract."""

    contracts = _record_contracts(V10_CONTRACT)
    assert set(V10_FIELD_ADMISSIONS) <= set(contracts), "admissions name unknown record contracts"
    policy_values = {"keep", "blob", "blob_recursive", "drop", "stringify", "weakref_strip"}
    for record_key, admitted in V10_FIELD_ADMISSIONS.items():
        policies = contracts[record_key]["field_policies"]
        assert set(admitted.values()) <= policy_values
        for name, policy in admitted.items():
            if name in policies:
                assert policies[name] == policy, (
                    f"{record_key}.{name}: contract policy {policies[name]!r} differs from "
                    f"the registered {policy!r}"
                )
    for record_key, retyped in V10_FIELD_RETYPES.items():
        missing = set(retyped) - set(contracts[record_key]["persisted_field_names"])
        assert missing == set(), (
            f"{record_key}: retypes name non-persisted fields {sorted(missing)}"
        )


def test_v10_contract_floor_text_and_version() -> None:
    contract = _load(V10_CONTRACT)["contract"]
    assert contract["tlspec_version"] == TLSPEC_VERSION == 10
    assert contract["min_torchlens_version"] == MIN_TORCHLENS_VERSION_TEXT == "2.31.0"
    assert contract["min_tlspec_version"] == MIN_TLSPEC_VERSION == 6


def test_below_floor_state_refuses_as_2x() -> None:
    state = {"tlspec_version": MIN_TLSPEC_VERSION - 1}
    with pytest.raises(ArtifactTooOldError) as excinfo:
        read_tlspec_version(state, cls_name="Trace")
    fields = excinfo.value.fields
    assert fields["code"] == "artifact_version_below_floor"
    assert fields["floor_tlspec_version"] == 6
    assert "saved by torchlens 2.x" in str(excinfo.value)
    assert "2.33" not in str(excinfo.value)


def test_modellog_216_refusal_names_the_one_release_that_reads_it() -> None:
    error = modellog_216_error("x.tlspec")
    assert "torchlens==2.17.0" in error.fields["remedy"]
    assert "save it again with torchlens 2.36.x" in error.fields["remedy"]
    assert error.fields["code"] == "artifact_version_below_floor"


def test_future_version_refuses_on_state_window() -> None:
    with pytest.raises(ArtifactTooNewError) as excinfo:
        read_tlspec_version({"tlspec_version": TLSPEC_VERSION + 1}, cls_name="Trace")
    assert excinfo.value.fields["ceiling_tlspec_version"] == 10
