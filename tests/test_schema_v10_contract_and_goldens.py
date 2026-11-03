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

from torchlens._io.format_contract import (
    RESERVED_STEP_SCORE_CREDIT,
    RESERVED_STEP_SCORE_NAMES,
)
from torchlens._io.writer_contract import writer_contract

_HERE = Path(__file__).parent
SCHEMAS = _HERE.parent / "torchlens" / "schemas"
V10_CONTRACT = SCHEMAS / "writer_contract_v10.json"

#: SHA-256 of the frozen v10 contract of record (frozen since the v11 bump).
V10_CONTRACT_SHA256 = (
    "f49ae6370c46c171074c335a4ac2e6ce3734935bb7194668840672c2ea570fea"  # pragma: allowlist secret
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


def test_step_score_names_are_reserved_and_credited() -> None:
    """U22.3: inseq's step-score names are reserved in the v10 contract of record."""

    assert RESERVED_STEP_SCORE_NAMES == _INSEQ_STEP_SCORES_MAP_KEYS
    reserved = _load(V10_CONTRACT)["contract"]["reserved_vocabularies"]["step_scores"]
    assert reserved["names"] == list(_INSEQ_STEP_SCORES_MAP_KEYS)
    assert reserved["status"] == "reserved"
    assert "inseq" in reserved["credit"] and reserved["credit"] == RESERVED_STEP_SCORE_CREDIT
    assert writer_contract()["reserved_vocabularies"]["step_scores"] == reserved
