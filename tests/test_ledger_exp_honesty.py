"""F03 ledger memo items 0 + 0b: bundle honesty batch and fork lineage.

Covers the XS honesty batch (typed non-Trace membership refusal, apply()
member-name idiom, relate as a real method, most_changed doc semantics) and
the fork-lineage carriage (relation-table carry, bundle_id minting,
member-construction anchors, hash-chained BundleOperation rows).
"""

from __future__ import annotations

import inspect

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.bundle._lineage import (
    BundleOperation,
    MemberEffectRow,
    MemberEffectTable,
    validate_member_construction,
    validate_operation_chain,
)
from torchlens.errors.episode import BundleRelationError
from torchlens.intervention.errors import BundleMemberError

pytestmark = pytest.mark.smoke


class _Tiny(nn.Module):
    """Two-op model for cheap bundle captures."""

    def __init__(self, offset: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 3)
        self.offset = offset

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.linear(x)) + self.offset


def _capture(offset: float = 0.0, seed: int = 7) -> tl.Trace:
    torch.manual_seed(seed)
    model = _Tiny(offset=offset)
    x = torch.randn(2, 3)
    return tl.trace(model, x, capture=tl.options.CaptureOptions(intervention_ready=True))


def _pair() -> tl.Bundle:
    return tl.Bundle(
        {"baseline": _capture(), "changed": _capture(offset=0.5)},
        baseline="baseline",
    )


# ---------------------------------------------------------------------------
# item 0: membership type refusal
# ---------------------------------------------------------------------------


def test_constructor_refuses_non_trace_member_typed() -> None:
    """A dict can no longer silently become a member (the measured defect)."""

    with pytest.raises(BundleMemberError) as excinfo:
        tl.Bundle({"a": _capture(), "b": {"not": "a trace"}})  # type: ignore[dict-item]
    assert excinfo.value.fields["code"] == "bundle_member_type_invalid"
    assert excinfo.value.fields["received_type"] == "dict"


def test_add_refuses_non_trace_member_typed() -> None:
    bundle = _pair()
    with pytest.raises(BundleMemberError) as excinfo:
        bundle.add({"not": "a trace"})  # type: ignore[arg-type]
    assert excinfo.value.fields["code"] == "bundle_member_type_invalid"
    # The refusal happened before any mutation.
    assert bundle.names == ["baseline", "changed"]


def test_add_refuses_string_member_typed() -> None:
    bundle = _pair()
    with pytest.raises(BundleMemberError) as excinfo:
        bundle.add("baseline")  # type: ignore[arg-type]
    assert excinfo.value.fields["code"] == "bundle_member_type_invalid"


# ---------------------------------------------------------------------------
# item 0: apply() member-name idiom
# ---------------------------------------------------------------------------


def test_apply_passes_member_name_to_two_arg_callable() -> None:
    bundle = _pair()
    result = bundle.apply(lambda log, name: name)
    assert result == {"baseline": "baseline", "changed": "changed"}


def test_apply_single_arg_contract_unchanged() -> None:
    bundle = _pair()
    result = bundle.apply(lambda log: type(log).__name__)
    assert set(result.values()) == {"Trace"}


def test_apply_var_positional_receives_name() -> None:
    bundle = _pair()

    def collect(*args: object) -> int:
        return len(args)

    assert set(bundle.apply(collect).values()) == {2}


def test_apply_uninspectable_callable_falls_back_single_arg() -> None:
    bundle = _pair()
    # repr is (obj, /): one positional parameter -> single-arg call.
    result = bundle.apply(repr)
    assert all(isinstance(value, str) for value in result.values())


# ---------------------------------------------------------------------------
# item 0: relate / derive_episode_status are real methods
# ---------------------------------------------------------------------------


def test_relate_is_a_real_method() -> None:
    assert hasattr(tl.Bundle, "relate")
    assert hasattr(tl.Bundle, "derive_episode_status")
    assert inspect.isfunction(tl.Bundle.relate)
    bundle = _pair()
    bundle.relate({"kind": "alternative_of", "from": "changed", "to": "baseline", "params": {}})
    assert bundle.member_relations[0].kind == "alternative_of"


# ---------------------------------------------------------------------------
# item 0b: bundle_id + fork lineage carriage
# ---------------------------------------------------------------------------


def test_bundle_id_minted_random_and_distinct() -> None:
    first = _pair()
    second = _pair()
    assert first.bundle_id != second.bundle_id
    assert len(first.bundle_id) == 16
    int(first.bundle_id, 16)  # hex


def test_fork_carries_relations_baseline_and_anchors() -> None:
    bundle = _pair()
    bundle.relate({"kind": "alternative_of", "from": "changed", "to": "baseline", "params": {}})
    child = bundle.fork()

    # Relation table carried (the historical drop is the 0b defect).
    assert [row.kind for row in child.member_relations] == ["alternative_of"]
    assert child.baseline_name == "baseline"

    # New identity, anchored source.
    assert child.bundle_id != bundle.bundle_id
    assert child._forked_from_bundle_id == bundle.bundle_id

    # Per-member origin anchors.
    anchor = child._member_construction["changed"]
    assert anchor["origin"] == "forked"
    assert anchor["source_bundle_id"] == bundle.bundle_id
    assert anchor["source_member"] == "changed"

    # Chronology rows on BOTH containers, chained from seq 1.
    assert bundle._operations[-1].kind == "fork"
    assert bundle._operations[-1].params["child_bundle_id"] == child.bundle_id
    assert child._operations[0].kind == "fork"
    assert child._operations[0].params["source_bundle_id"] == bundle.bundle_id
    validate_operation_chain(bundle._operations)
    validate_operation_chain(child._operations)


def test_member_construction_tracks_add_and_remove() -> None:
    bundle = _pair()
    assert bundle._member_construction["baseline"] == {"origin": "constructed"}
    extra = _capture(offset=1.0)
    bundle.add(extra, names="extra")
    assert bundle._member_construction["extra"] == {"origin": "added"}
    bundle.remove("extra")
    assert "extra" not in bundle._member_construction


# ---------------------------------------------------------------------------
# item 0b/3 shapes: payload codecs are fail-closed
# ---------------------------------------------------------------------------


def test_bundle_operation_round_trip_and_tamper_refusal() -> None:
    row = BundleOperation(
        operation_id="abc123", seq=1, kind="fork", member_names=("a",), params={"x": 1}
    )
    payload = row.to_payload()
    rebuilt = BundleOperation.from_payload(payload)
    assert rebuilt == row
    tampered = dict(payload)
    tampered["kind"] = "vary"
    with pytest.raises(ValueError, match="digest mismatch"):
        BundleOperation.from_payload(tampered)


def test_bundle_operation_refuses_unknown_kind_and_bad_seq() -> None:
    with pytest.raises(ValueError, match="closed"):
        BundleOperation(operation_id="a", seq=1, kind="mystery")
    with pytest.raises(ValueError, match="positive int"):
        BundleOperation(operation_id="a", seq=0, kind="fork")


def test_operation_chain_break_refuses_typed() -> None:
    first = BundleOperation(operation_id="a", seq=1, kind="fork")
    # A second row that does not chain from the first.
    orphan = BundleOperation(operation_id="b", seq=2, kind="do", prev_operation_digest="0" * 64)
    with pytest.raises(BundleRelationError) as excinfo:
        validate_operation_chain([first, orphan])
    assert excinfo.value.fields["code"] == "bundle_lineage_invalid"


def test_member_construction_validator_fail_closed() -> None:
    good = {"m": {"origin": "forked", "source_bundle_id": "abc", "source_member": "m"}}
    validated = validate_member_construction(good, member_names=["m"])
    assert validated["m"]["origin"] == "forked"
    with pytest.raises(BundleRelationError) as excinfo:
        validate_member_construction({"ghost": {"origin": "added"}}, member_names=["m"])
    assert excinfo.value.fields["code"] == "bundle_lineage_invalid"
    with pytest.raises(BundleRelationError):
        validate_member_construction({"m": {"origin": "teleported"}}, member_names=["m"])
    with pytest.raises(BundleRelationError):
        validate_member_construction({"m": {"origin": "added", "extra_key": 1}}, member_names=["m"])


def test_effect_table_round_trip_and_refusals() -> None:
    table = MemberEffectTable(
        operation_id="op1",
        rows=(
            MemberEffectRow(candidate_id="c0", status="completed", member_name="m0", value=1.5),
            MemberEffectRow(candidate_id="c1", status="released", value=0.25),
        ),
        baseline_member="baseline",
        metric_repr="logit_diff",
        edit_repr="zero_ablate",
        lane="live_hook",
        retain_policy="all",
    )
    rebuilt = MemberEffectTable.from_payload(table.to_payload())
    assert rebuilt == table
    with pytest.raises(ValueError, match="closed"):
        MemberEffectRow(candidate_id="c", status="vanished")
    with pytest.raises(ValueError, match="duplicate"):
        MemberEffectTable(
            operation_id="op",
            rows=(
                MemberEffectRow(candidate_id="c", status="completed"),
                MemberEffectRow(candidate_id="c", status="failed"),
            ),
        )
