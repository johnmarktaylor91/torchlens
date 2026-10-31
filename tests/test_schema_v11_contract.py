"""Field-admission gate for the v11 writer contract.

Every field the v11 contract of record carries over v10 (persisted or DROP)
must be registered here under its policy. An unregistered new persisted field
fails this test, so a field never reaches the stored shape by accident.
"""

from __future__ import annotations

import json
from pathlib import Path

_HERE = Path(__file__).resolve().parent
SCHEMAS = _HERE.parent / "torchlens" / "schemas"
V10_CONTRACT = SCHEMAS / "writer_contract_v10.json"
V11_CONTRACT = SCHEMAS / "writer_contract_v11.json"

#: The v11 FIELD REGISTRY: record contract key -> {field name -> policy value}
#: for every field added over v10. The v11 renames enter under their new
#: spellings (the old spellings leave with no alias).
V11_FIELD_ADMISSIONS: dict[str, dict[str, str]] = {
    "grad_fn_call": {"fire_record": "keep"},
    "layer": {"is_edited": "keep"},
    "op": {"is_edited": "keep"},
    "trace": {
        "_replay_frontier": "drop",
        "facet_functions": "drop",
        "is_replayable": "keep",
        "model_name": "keep",
        "name": "keep",
        "stop_reason": "keep",
    },
}


def _record_contracts(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))["contract"]["record_contracts"]


def _field_names(record: dict) -> set[str]:
    return set(record["persisted_field_names"]) | set(record["field_policies"])


def test_v10_to_v11_additions_are_all_registered() -> None:
    """Every field added over v10 (persisted or DROP) is admitted, under its registered policy.

    DROP fields never enter ``persisted_field_names`` but do enter the policy
    table, so the gate reads both.
    """

    old = _record_contracts(V10_CONTRACT)
    new = _record_contracts(V11_CONTRACT)
    assert set(old) == set(new), "no record contract may appear or disappear at v11"
    assert set(V11_FIELD_ADMISSIONS) <= set(new), "admissions name unknown record contracts"
    for record_key in sorted(new):
        added = _field_names(new[record_key]) - _field_names(old[record_key])
        admitted = V11_FIELD_ADMISSIONS.get(record_key, {})
        unregistered = added - set(admitted)
        assert unregistered == set(), (
            f"{record_key}: field(s) {sorted(unregistered)} reach the v11 contract "
            "without a registration; add the admission to V11_FIELD_ADMISSIONS in "
            "the same change"
        )
        policies = new[record_key]["field_policies"]
        for name in sorted(set(admitted) & added):
            assert policies[name] == admitted[name], (
                f"{record_key}.{name}: live policy {policies[name]!r} differs from the "
                f"registered {admitted[name]!r}"
            )


def test_v11_admissions_name_only_real_additions() -> None:
    """A registration names a field that is new in v11, never a stale or v10 field."""

    old = _record_contracts(V10_CONTRACT)
    new = _record_contracts(V11_CONTRACT)
    for record_key, admitted in sorted(V11_FIELD_ADMISSIONS.items()):
        added = _field_names(new[record_key]) - _field_names(old[record_key])
        stale = set(admitted) - added
        assert stale == set(), (
            f"{record_key}: admission(s) {sorted(stale)} are not fields added at v11"
        )
