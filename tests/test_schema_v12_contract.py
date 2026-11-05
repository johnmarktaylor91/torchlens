"""Field-admission gate for the v12 writer contract.

Every key the v12 contract of record persists beyond the frozen v11 one (a new
field, or a v11 DROP field flipped to KEEP) must be registered here under its
policy. An unregistered key fails this test, so a field never reaches the
stored shape by accident.
"""

from __future__ import annotations

import json
from pathlib import Path

_HERE = Path(__file__).resolve().parent
SCHEMAS = _HERE.parent / "torchlens" / "schemas"
V11_CONTRACT = SCHEMAS / "writer_contract_v11.json"
V12_CONTRACT = SCHEMAS / "writer_contract_v12.json"

#: The v12 FIELD REGISTRY: record contract key -> {field name -> policy value}
#: for every field newly PERSISTED over v11: a new field, or a v11 DROP field
#: whose policy flipped to KEEP (``arg_expressions`` was a v11 DROP field).
V12_FIELD_ADMISSIONS: dict[str, dict[str, str]] = {
    "op": {"arg_expressions": "keep", "is_constant": "keep"},
}


def _document(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _record_contracts(path: Path) -> dict:
    return _document(path)["contract"]["record_contracts"]


def _field_names(record: dict) -> set[str]:
    return set(record["persisted_field_names"]) | set(record["field_policies"])


def test_v11_to_v12_persisted_additions_are_exactly_the_registered_ones() -> None:
    """The keys newly persisted over v11 are the registry, under the registered policies."""

    old = _record_contracts(V11_CONTRACT)
    new = _record_contracts(V12_CONTRACT)
    assert set(old) == set(new), "no record contract may appear or disappear at v12"
    assert set(V12_FIELD_ADMISSIONS) <= set(new), "admissions name unknown record contracts"
    for record_key in sorted(new):
        added = set(new[record_key]["persisted_field_names"]) - set(
            old[record_key]["persisted_field_names"]
        )
        admitted = V12_FIELD_ADMISSIONS.get(record_key, {})
        assert added == set(admitted), (
            f"{record_key}: v12 persists {sorted(added)} newly but the registry admits "
            f"{sorted(admitted)}; update V12_FIELD_ADMISSIONS in the same change"
        )
        policies = new[record_key]["field_policies"]
        for name in sorted(added):
            assert policies[name] == admitted[name], (
                f"{record_key}.{name}: live policy {policies[name]!r} differs from the "
                f"registered {admitted[name]!r}"
            )


def test_v12_policy_changes_are_only_the_arg_expressions_flip() -> None:
    """The only v11 field whose policy moved is ``op.arg_expressions``, DROP to KEEP."""

    old = _record_contracts(V11_CONTRACT)
    new = _record_contracts(V12_CONTRACT)
    moved = {
        (record_key, name): (policy, new[record_key]["field_policies"].get(name))
        for record_key in sorted(old)
        for name, policy in old[record_key]["field_policies"].items()
        if new[record_key]["field_policies"].get(name) != policy
    }
    assert moved == {("op", "arg_expressions"): ("drop", "keep")}


def test_v12_removes_no_field() -> None:
    """No v11 field leaves the stored shape or the policy table at v12."""

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


def test_component_spec_keys_keep_the_persisted_intervention_module_path() -> None:
    """Edit and FireRecord stay keyed by the module path the contract was digested under.

    ``torchlens.intervention.types`` became private (``intervention._types``); the
    writer contract keeps the persisted path, so the move cannot change the digest
    and artifacts saved before it keep loading at the same tlspec version.
    """

    from torchlens._io.field_registry import component_spec_tables
    from torchlens._io.writer_contract import writer_contract_digest
    from torchlens.intervention._types import Edit, FireRecord

    tables = component_spec_tables()
    committed = _document(V12_CONTRACT)["contract"]["component_specs"]
    for cls in (Edit, FireRecord):
        assert cls.__module__ == "torchlens.intervention._types"
        persisted = f"torchlens.intervention.types.{cls.__qualname__}"
        assert persisted in tables
        assert f"{cls.__module__}.{cls.__qualname__}" not in tables
        assert tables[persisted] == committed[persisted]
    assert writer_contract_digest() == _document(V12_CONTRACT)["writer_contract_digest"]


def test_format3_class_paths_keep_the_persisted_intervention_module_path() -> None:
    """Saved files name Edit and FireRecord by the persisted path, which resolves back."""

    from torchlens._io.format3.values import class_path, record_type_table
    from torchlens.intervention._types import Edit, FireRecord

    table = record_type_table()
    for cls in (Edit, FireRecord):
        path = f"torchlens.intervention.types.{cls.__qualname__}"
        assert class_path(cls) == path
        assert table[path] is cls


def test_format3_recipe_edit_path_is_the_persisted_class_path() -> None:
    """The recipe encoder writes a nested Edit under the same persisted path the reader admits."""

    from torchlens._io.format3 import recipe
    from torchlens._io.format3.values import class_path, record_type_table
    from torchlens.intervention._types import Edit

    assert class_path(Edit) == recipe._EDIT_CLASS_PATH
    assert record_type_table()[recipe._EDIT_CLASS_PATH] is Edit
