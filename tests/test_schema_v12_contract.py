"""Field-admission gate for the v12 writer contract.

Every field the v12 contract of record carries over the frozen v11 one
(persisted or DROP) must be registered here under its policy. An unregistered
new field fails this test, so a field never reaches the stored shape by
accident.
"""

from __future__ import annotations

import json
from pathlib import Path

_HERE = Path(__file__).resolve().parent
SCHEMAS = _HERE.parent / "torchlens" / "schemas"
V11_CONTRACT = SCHEMAS / "writer_contract_v11.json"
V12_CONTRACT = SCHEMAS / "writer_contract_v12.json"

#: The v12 FIELD REGISTRY: record contract key -> {field name -> policy value}
#: for every field added over v11.
V12_FIELD_ADMISSIONS: dict[str, dict[str, str]] = {
    "op": {"arg_expressions": "keep", "is_constant": "keep"},
}


def _document(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _record_contracts(path: Path) -> dict:
    return _document(path)["contract"]["record_contracts"]


def _field_names(record: dict) -> set[str]:
    return set(record["persisted_field_names"]) | set(record["field_policies"])


def test_v11_to_v12_additions_are_exactly_the_registered_ones() -> None:
    """The fields added over v11 are the registry, under the registered policies.

    DROP fields never enter ``persisted_field_names`` but do enter the policy
    table, so the gate reads both.
    """

    old = _record_contracts(V11_CONTRACT)
    new = _record_contracts(V12_CONTRACT)
    assert set(old) == set(new), "no record contract may appear or disappear at v12"
    assert set(V12_FIELD_ADMISSIONS) <= set(new), "admissions name unknown record contracts"
    for record_key in sorted(new):
        added = _field_names(new[record_key]) - _field_names(old[record_key])
        admitted = V12_FIELD_ADMISSIONS.get(record_key, {})
        assert added == set(admitted), (
            f"{record_key}: v12 adds {sorted(added)} but the registry admits "
            f"{sorted(admitted)}; update V12_FIELD_ADMISSIONS in the same change"
        )
        policies = new[record_key]["field_policies"]
        for name in sorted(added):
            assert policies[name] == admitted[name], (
                f"{record_key}.{name}: live policy {policies[name]!r} differs from the "
                f"registered {admitted[name]!r}"
            )


def test_v12_admissions_are_persisted() -> None:
    """Both v12 additions are KEEP fields that every save writes."""

    new = _record_contracts(V12_CONTRACT)
    for record_key, admitted in sorted(V12_FIELD_ADMISSIONS.items()):
        persisted = set(new[record_key]["persisted_field_names"])
        assert set(admitted) <= persisted, f"{record_key}: {sorted(set(admitted) - persisted)}"


def test_v12_removes_no_field() -> None:
    """Nothing persisted at v11 leaves the stored shape at v12."""

    old = _record_contracts(V11_CONTRACT)
    new = _record_contracts(V12_CONTRACT)
    for record_key in sorted(old):
        removed = _field_names(old[record_key]) - _field_names(new[record_key])
        assert removed == set(), f"{record_key}: v12 removes {sorted(removed)}"


def test_v12_reader_floor_moves_with_the_contract() -> None:
    """The v12 contract reads only stamp 12; the frozen v11 contract keeps floor 11."""

    from torchlens._io import MIN_TLSPEC_VERSION, TLSPEC_VERSION

    assert MIN_TLSPEC_VERSION == TLSPEC_VERSION == 12
    v12 = _document(V12_CONTRACT)["contract"]
    v11 = _document(V11_CONTRACT)["contract"]
    assert v12["tlspec_version"] == 12
    assert v12["min_tlspec_version"] == 12
    assert v11["tlspec_version"] == 11, "v11 stays frozen"
    assert v11["min_tlspec_version"] == 11, "v11 stays frozen"
