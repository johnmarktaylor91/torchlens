"""r20j hardening: IR container spec correctness + registry honesty.

Covers ledger findings N4 (false-VERIFIED tensor-key dict witness), N5 (rebuild
must reject malformed specs), N6 (most-derived registration dispatch), and N7
(snapshot dedup must not drop the later observation's event index).
"""

from __future__ import annotations

import pytest
import torch

from torchlens._ir.container import (
    _CONTAINER_REGISTRY,
    ContainerReconstructionError,
    ContainerSpec,
    TupleIndex,
    get_registered_container,
    rebuild_container_from_spec,
    register_container,
)
from torchlens._ir.container_registry import (
    ContainerRegistry,
    FuncSite,
    Phase,
    Role,
    walk_container,
)

# ---------------------------------------------------------------------------
# N4 -- tensor-key dict must NOT be blessed reconstructable (false-VERIFIED class)
# ---------------------------------------------------------------------------


def test_n4_tensor_key_dict_is_not_reconstructable() -> None:
    """A mixed tensor-key dict is reported unreconstructable AND cannot silently lie.

    The witness previously said ``reconstructable=True`` (kind ``dict``, length counts the
    tensor key) while ``rebuild_container_from_spec`` raised "Not enough leaves". The honest
    fix degrades the spec to ``opaque`` so the flag matches reality.
    """

    visible = torch.tensor([1.0])
    payload = {"visible": visible, torch.tensor([9.0]): torch.tensor([2.0])}
    result = walk_container(payload, role=Role.MODEL_OUTPUT, capability="full_spec")
    assert result is not None
    # The witness must be honest: it can NOT be rebuilt.
    assert result.reconstructable is False
    assert result.spec.kind == "opaque"
    # And the honesty holds under actual reconstruction: rebuilding an opaque spec is a
    # clean typed refusal, never a misleading "Not enough leaves".
    with pytest.raises(ValueError, match="Opaque ContainerSpec cannot be reconstructed"):
        rebuild_container_from_spec(result.spec, [visible])


def test_n4_clean_nested_containers_stay_reconstructable() -> None:
    """The honest recursion must NOT regress the clean nested list/dict/tuple+literal path."""

    payload = {
        "pair": (torch.tensor([1.0]), torch.tensor([2.0])),
        "nested": {"inner": [torch.tensor([3.0]), 7]},
    }
    result = walk_container(payload, role=Role.MODEL_OUTPUT, capability="full_spec")
    assert result is not None
    assert result.reconstructable is True
    assert result.spec.kind == "dict"
    leaves = [torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([3.0])]
    rebuilt = rebuild_container_from_spec(result.spec, leaves)
    assert set(rebuilt) == {"pair", "nested"}
    assert rebuilt["nested"]["inner"][1] == 7


def test_n4_nested_opaque_child_makes_parent_unreconstructable() -> None:
    """Class-wide: an opaque node ANYWHERE makes the reconstructable witness False.

    The prior shallow ``spec.kind != "opaque"`` blessed a container whose nested child
    (here a generator) is opaque -- a lying witness. The recursive honesty check catches it.
    """

    def gen() -> object:
        yield torch.tensor([1.0])

    payload = [torch.tensor([5.0]), gen()]
    result = walk_container(payload, role=Role.MODEL_OUTPUT, capability="full_spec")
    assert result is not None
    # top-level kind is a list (not opaque), but a nested child is opaque -> not rebuildable.
    assert result.spec.kind == "list"
    assert any(child.kind == "opaque" for _c, child in result.spec.child_specs)
    assert result.reconstructable is False


# ---------------------------------------------------------------------------
# N5 -- rebuild must REJECT malformed specs, never normalize garbage
# ---------------------------------------------------------------------------


def test_n5_negative_length_is_rejected() -> None:
    """A negative ``length`` was coerced to an empty container; now it is refused."""

    spec = ContainerSpec(kind="tuple", length=-4)
    with pytest.raises(ContainerReconstructionError, match="non-negative"):
        rebuild_container_from_spec(spec, [])


def test_n5_length_keys_disagreement_is_rejected() -> None:
    """A ``dict`` whose ``length`` disagrees with ``keys`` was silently normalized to ``{}``."""

    spec = ContainerSpec(kind="dict", length=9, keys=())
    with pytest.raises(ContainerReconstructionError, match="disagrees"):
        rebuild_container_from_spec(spec, [])


def test_n5_out_of_range_child_component_is_rejected() -> None:
    """An out-of-range ``TupleIndex`` child was dropped without error; now it is refused."""

    spec = ContainerSpec(
        kind="tuple",
        length=2,
        child_specs=((TupleIndex(4), ContainerSpec(kind="literal", literal_value="X")),),
    )
    with pytest.raises(ContainerReconstructionError, match="out of "):
        rebuild_container_from_spec(spec, [10, 20])


def test_n5_duplicate_child_component_is_rejected() -> None:
    """Duplicate child path components silently collapsed via ``dict()``; now refused."""

    spec = ContainerSpec(
        kind="tuple",
        length=3,
        child_specs=(
            (TupleIndex(0), ContainerSpec(kind="literal", literal_value=1)),
            (TupleIndex(0), ContainerSpec(kind="literal", literal_value=2)),
        ),
    )
    with pytest.raises(ContainerReconstructionError, match="duplicate"):
        rebuild_container_from_spec(spec, [])


def test_n5_valid_specs_still_rebuild() -> None:
    """Tighten-only: real nested/tuple/dict/namedtuple specs must NOT be rejected."""

    import collections

    point_t = collections.namedtuple("Point", ["x", "y"])
    payloads = [
        (torch.tensor([1.0]), [torch.tensor([2.0]), 3], {"k": torch.tensor([4.0])}),
        point_t(torch.tensor([5.0]), 6),
        {"nested": {"deep": (torch.tensor([7.0]),)}},
    ]
    for payload in payloads:
        result = walk_container(payload, role=Role.MODEL_OUTPUT, capability="full_spec")
        assert result is not None and result.reconstructable
        leaves = [torch.zeros(1) for _ in result.leaf_occurrences]
        # Must not raise -- a faithful capture spec always validates.
        rebuild_container_from_spec(result.spec, leaves)


# ---------------------------------------------------------------------------
# N6 -- get_registered_container must return the MOST-DERIVED match
# ---------------------------------------------------------------------------


@pytest.fixture
def clean_container_registry() -> object:
    """Snapshot and restore the module-global container registry around a test."""

    saved = dict(_CONTAINER_REGISTRY)
    _CONTAINER_REGISTRY.clear()
    try:
        yield _CONTAINER_REGISTRY
    finally:
        _CONTAINER_REGISTRY.clear()
        _CONTAINER_REGISTRY.update(saved)


def _reg(marker: str):
    return (lambda v: (list(v), marker), lambda aux, children: list(children))


def test_n6_subclass_wins_when_base_registered_first(clean_container_registry: object) -> None:
    """Base registered BEFORE subclass: the subclass must resolve to its OWN registration."""

    class Base(list):
        pass

    class Sub(Base):
        pass

    register_container(Base, *_reg("BASE"), state_complete=True)
    register_container(Sub, *_reg("SUB"), state_complete=True)
    reg = get_registered_container(Sub)
    assert reg is not None
    assert reg.flatten(Sub([1]))[1] == "SUB"


def test_n6_subclass_wins_when_base_registered_last(clean_container_registry: object) -> None:
    """Order-independence: subclass registered FIRST, base LAST -- subclass still wins."""

    class Base(list):
        pass

    class Sub(Base):
        pass

    register_container(Sub, *_reg("SUB"), state_complete=True)
    register_container(Base, *_reg("BASE"), state_complete=True)
    assert get_registered_container(Sub).flatten(Sub([1]))[1] == "SUB"
    assert get_registered_container(Base).flatten(Base([1]))[1] == "BASE"


def test_n6_multiple_inheritance_and_exact_and_missing(clean_container_registry: object) -> None:
    """Exact match, most-derived via MRO, and no-match all resolve correctly."""

    class A(list):
        pass

    class B(A):
        pass

    class C(B):
        pass

    register_container(A, *_reg("A"), state_complete=True)
    register_container(C, *_reg("C"), state_complete=True)
    # C is an exact match -> C.
    assert get_registered_container(C).flatten(C([1]))[1] == "C"
    # B is between A and C, only A registered on its ancestry -> A (nearest registered ancestor).
    assert get_registered_container(B).flatten(B([1]))[1] == "A"

    class Unrelated(tuple):
        pass

    assert get_registered_container(Unrelated) is None


# ---------------------------------------------------------------------------
# N7 -- snapshot dedup must not discard the later observation's event index
# ---------------------------------------------------------------------------


def _register_same_body_at_two_sites() -> tuple[object, object, object]:
    """Register one identical container body at two sites (events 3 and 99)."""

    registry = ContainerRegistry()
    obj = {"a": torch.tensor([1.0])}
    spec = ContainerSpec(kind="dict", length=1, keys=("a",))
    site_early = FuncSite(func_call_id=1, position="out")
    site_late = FuncSite(func_call_id=2, position="out")
    common = {
        "role": Role.CALL_OUTPUT,
        "phase": Phase.POST_CALL,
        "spec": spec,
        "leaf_occurrences": (),
        "reconstructable": True,
    }
    registry.register_snapshot(obj, site=site_early, observed_at_event_index=3, **common)
    record = registry.register_snapshot(obj, site=site_late, observed_at_event_index=99, **common)
    return record, site_early, site_late


def test_n7_dedup_preserves_each_sites_event_index() -> None:
    """Deduped snapshot keeps ONE body plus aliases, and each site keeps its OWN index."""

    record, site_early, site_late = _register_same_body_at_two_sites()
    assert len(record.snapshots) == 1
    snapshot = record.snapshots[0]
    assert len(snapshot.site_aliases) == 1
    # Primary site keeps its index; the later observation's index is NO LONGER discarded.
    assert snapshot.observed_index_for_site(site_early) == 3
    assert snapshot.observed_index_for_site(site_late) == 99
    # snapshot_at(alias) resolves to the single deduped body (aliases are matches).
    assert record.snapshot_at(site=site_late) is snapshot


def test_n7_alias_indices_survive_pickle_roundtrip() -> None:
    """The per-site indices are portable: they survive a serialize/deserialize round-trip."""

    import pickle

    record, site_early, site_late = _register_same_body_at_two_sites()
    restored = pickle.loads(pickle.dumps(record.snapshots[0]))
    assert restored.observed_index_for_site(site_early) == 3
    assert restored.observed_index_for_site(site_late) == 99
