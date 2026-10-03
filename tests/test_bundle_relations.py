"""S6 bundle member-relation table + episode fold tests (L2 build).

Covers the S6 rules R1-R7, the R5 mutator guards, the section-2.3 episode
fold arms through both the Bundle surface and the ledger-module fold, and
the S3-gated ``member_relations`` persistence paths.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._io import PreReleaseArtifactError
from torchlens._io.prerelease import PRERELEASE_MARKER
from torchlens.bundle._relations import MemberRelationRow, MemberRelationTable
from torchlens.capture._episode_ledger import (
    EpisodeLedger,
    EpisodeLedgerHeader,
    EpisodeLedgerRow,
    derive_episode_status,
)
from torchlens.errors import BundleRelationError


class _TinyRelu(nn.Module):
    """Small model for cheap member captures."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.linear(x))


class _BombModule(nn.Module):
    """Model whose forward fails mid-pass for the failed-partial case."""

    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        raise RuntimeError("bomb: mid-forward failure")


@pytest.fixture(scope="module")
def tiny_trace() -> Any:
    """One captured tiny trace shared (by object) across member slots."""

    torch.manual_seed(0)
    model = _TinyRelu()
    x = torch.randn(2, 3)
    trace = tl.trace(model, x)
    try:
        yield trace
    finally:
        trace.cleanup()


def _pair_row(kind: str = "alternative_of", src: str = "a", dst: str = "b", **params: Any) -> dict:
    return {"kind": kind, "from": src, "to": dst, "params": dict(params)}


def _episode_row(member: str, at_step: int, role: str, episode_id: str = "ep") -> dict:
    return {
        "kind": "episode_member",
        "member": member,
        "params": {"episode_id": episode_id, "at_step": at_step, "role": role},
    }


def _episode_ledger(n_declared: int, n_rows: int) -> EpisodeLedger:
    header = EpisodeLedgerHeader(
        episode_id="ep",
        stepped_module="block",
        entry_seed=0,
        n_steps_declared=n_declared,
    )
    rows = [
        EpisodeLedgerRow(
            episode_step=step,
            role="prefill" if step == 0 else "decode",
            status="complete",
            coord={"member": f"m{step}"},
            step_output=(step,),
        )
        for step in range(n_rows)
    ]
    return EpisodeLedger(header, rows)


# ---------------------------------------------------------------------------
# R1: no dangling edges, ever
# ---------------------------------------------------------------------------


def test_ctor_refuses_row_naming_absent_member(tiny_trace: Any) -> None:
    """Constructing a Bundle with a relation naming a non-member refuses typed."""

    with pytest.raises(BundleRelationError) as excinfo:
        tl.bundle(
            {"a": tiny_trace, "b": tiny_trace},
            member_relations=[_pair_row(dst="ghost")],
        )
    assert excinfo.value.fields["code"] == "bundle_relation_member_missing"
    assert excinfo.value.fields["missing_member"] == "ghost"


def test_relate_refuses_absent_member_and_keeps_table(tiny_trace: Any) -> None:
    """relate() with a dangling row refuses typed and leaves the table unchanged."""

    bundle = tl.bundle({"a": tiny_trace, "b": tiny_trace})
    with pytest.raises(BundleRelationError) as excinfo:
        bundle.relate(_pair_row(kind="successor_of", dst="ghost"))
    assert excinfo.value.fields["code"] == "bundle_relation_member_missing"
    assert bundle.member_relations == ()


# ---------------------------------------------------------------------------
# R2: closed schema — unknown kind / wrong shape / param vocabulary
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "payload",
    [
        # Unknown kind.
        {"kind": "buddies_with", "from": "a", "to": "b", "params": {}},
        # PAIR kind given the MEMBER row shape.
        {"kind": "alternative_of", "member": "a", "params": {}},
        # Undeclared param key.
        {"kind": "forked_from", "from": "a", "to": "b", "params": {"at_step": 0, "bogus": 1}},
        # Missing required at_step param.
        {"kind": "escalates", "from": "a", "to": "b", "params": {}},
        # Ill-typed at_step.
        {"kind": "forked_from", "from": "a", "to": "b", "params": {"at_step": "zero"}},
        # Role outside the closed vocabulary.
        {
            "kind": "episode_member",
            "member": "a",
            "params": {"episode_id": "ep", "at_step": 0, "role": "verify"},
        },
    ],
)
def test_off_schema_rows_refuse(tiny_trace: Any, payload: dict) -> None:
    """Off-schema rows raise ValueError at parse level, typed via public wiring."""

    with pytest.raises(ValueError):
        MemberRelationRow.from_payload(payload)
    with pytest.raises(BundleRelationError) as excinfo:
        tl.bundle({"a": tiny_trace, "b": tiny_trace}, member_relations=[payload])
    assert excinfo.value.fields["code"] == "bundle_relation_schema_invalid"


def test_episode_member_invariants(tiny_trace: Any) -> None:
    """Duplicate at_step / missing prefill-at-0 refuse as schema-invalid (R3)."""

    with pytest.raises(BundleRelationError) as excinfo:
        tl.bundle(
            {"a": tiny_trace, "b": tiny_trace},
            member_relations=[
                _episode_row("a", 0, "prefill"),
                _episode_row("b", 0, "decode"),
            ],
        )
    assert excinfo.value.fields["code"] == "bundle_relation_schema_invalid"
    with pytest.raises(BundleRelationError) as excinfo:
        tl.bundle(
            {"a": tiny_trace, "b": tiny_trace},
            member_relations=[_episode_row("a", 1, "decode")],
        )
    assert excinfo.value.fields["code"] == "bundle_relation_schema_invalid"


# ---------------------------------------------------------------------------
# R4: identity-stable immutable view; relate() installs a NEW version
# ---------------------------------------------------------------------------


def test_view_identity_stability_and_new_version(tiny_trace: Any) -> None:
    """member_relations is the SAME tuple across reads; relate() swaps it."""

    bundle = tl.bundle(
        {"a": tiny_trace, "b": tiny_trace},
        member_relations=[_pair_row()],
    )
    first_view = bundle.member_relations
    assert isinstance(first_view, tuple)
    assert bundle.member_relations is first_view
    bundle.relate(_pair_row(kind="successor_of", src="b", dst="a"))
    second_view = bundle.member_relations
    assert second_view is not first_view
    assert len(second_view) == 2
    assert bundle.member_relations is second_view


# ---------------------------------------------------------------------------
# R5: mutator guards — refuse typed or cascade explicitly, never orphan
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_remove_refuses_then_cascades_exactly(tiny_trace: Any) -> None:
    """remove() refuses typed BEFORE removal; cascade drops only rows naming it."""

    bundle = tl.bundle(
        {"a": tiny_trace, "b": tiny_trace, "c": tiny_trace},
        member_relations=[_pair_row(), _episode_row("c", 0, "prefill")],
    )
    with pytest.raises(BundleRelationError) as excinfo:
        bundle.remove("b")
    assert excinfo.value.fields["code"] == "bundle_member_has_relations"
    assert "b" in bundle.names
    assert len(bundle.member_relations) == 2

    bundle.remove("b", cascade_relations=True)
    assert "b" not in bundle.names
    remaining = bundle.member_relations
    assert len(remaining) == 1
    assert remaining[0].kind == "episode_member"
    assert remaining[0].member == "c"


@pytest.mark.smoke
def test_clear_and_remove_except_guards(tiny_trace: Any) -> None:
    """clear()/remove_except() refuse typed, then cascade rows of removed members."""

    def build() -> Any:
        return tl.bundle(
            {"a": tiny_trace, "b": tiny_trace, "c": tiny_trace},
            baseline="a",
            member_relations=[_pair_row(src="b", dst="c")],
        )

    bundle = build()
    with pytest.raises(BundleRelationError) as excinfo:
        bundle.clear()
    assert excinfo.value.fields["code"] == "bundle_member_has_relations"
    assert bundle.names == ["a", "b", "c"]
    bundle.clear(cascade_relations=True)
    assert bundle.names == ["a"]
    assert bundle.member_relations == ()

    bundle = build()
    with pytest.raises(BundleRelationError) as excinfo:
        bundle.remove_except(["a", "b"])
    assert excinfo.value.fields["code"] == "bundle_member_has_relations"
    assert bundle.names == ["a", "b", "c"]
    bundle.remove_except(["a", "b"], cascade_relations=True)
    assert bundle.names == ["a", "b"]
    assert bundle.member_relations == ()


@pytest.mark.smoke
def test_capacity_eviction_refuses_typed(tiny_trace: Any) -> None:
    """LRU eviction of a related member refuses typed, never silently orphans."""

    bundle = tl.bundle(
        {"a": tiny_trace, "b": tiny_trace, "c": tiny_trace},
        baseline="a",
        member_relations=[_pair_row(src="b", dst="c")],
    )
    with pytest.raises(BundleRelationError) as excinfo:
        bundle.set_capacity(2)
    assert excinfo.value.fields["code"] == "bundle_member_has_relations"
    assert bundle.names == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# Episode fold: section 2.3 arms (fold input constructed directly where a
# non-COMPLETE settled status is needed — never forged onto real traces)
# ---------------------------------------------------------------------------


@pytest.mark.smoke
def test_fold_arm2_episode_complete_via_bundle_ledger(tiny_trace: Any) -> None:
    """All-COMPLETE members matching the ledger-declared N fold complete."""

    bundle = tl.bundle(
        {"m0": tiny_trace, "m1": tiny_trace},
        member_relations=[
            _episode_row("m0", 0, "prefill"),
            _episode_row("m1", 1, "decode"),
        ],
    )
    result = bundle.derive_episode_status("ep", ledger=_episode_ledger(2, 2))
    assert result.status == "episode_complete"
    # Without the ledger, arm 2 cannot fire: fail-closed to unknown.
    assert bundle.derive_episode_status("ep").status == "episode_unknown"


def test_fold_arm3_halted_member() -> None:
    """A trailing HALTED member folds to episode_halted_at_step k."""

    result = derive_episode_status(
        [("complete", None), ("halted", None)], n_declared=None, ledger=None
    )
    assert result.status == "episode_halted_at_step"
    assert result.at_step == 1
    assert result.member_status == "HALTED"


def test_fold_arm5_aborted_nonfinite_verbatim() -> None:
    """ABORTED_NONFINITE is disclosed verbatim on the fold result."""

    result = derive_episode_status(
        [("complete", None), ("aborted_nonfinite", None)], n_declared=None, ledger=None
    )
    assert result.status == "episode_aborted_at_step"
    assert result.at_step == 1
    assert result.member_status == "ABORTED_NONFINITE"


def test_fold_arm6_failed_including_phaseless() -> None:
    """FAILED members fold with verbatim phase, or 'unattributed' when absent."""

    result = derive_episode_status(
        [("complete", None), ("failed", "forward")], n_declared=None, ledger=None
    )
    assert result.status == "episode_failed_at_step"
    assert result.at_step == 1
    assert result.member_status == "FAILED"
    assert result.member_phase == "forward"

    phaseless = derive_episode_status([("failed", None)], n_declared=None, ledger=None)
    assert phaseless.status == "episode_failed_at_step"
    assert phaseless.at_step == 0
    assert phaseless.member_phase == "unattributed"


def test_fold_arm7_default_post_forward_failure_then_more_members() -> None:
    """A post-forward FAILED member followed by more members folds unknown."""

    result = derive_episode_status(
        [("complete", None), ("failed", "postprocess"), ("complete", None)],
        n_declared=3,
        ledger=None,
    )
    assert result.status == "episode_unknown"


def test_fold_arm1_unattested_member() -> None:
    """Any UNATTESTED member routes the whole fold to episode_unknown."""

    result = derive_episode_status(
        [("unattested", None), ("complete", None)], n_declared=2, ledger=None
    )
    assert result.status == "episode_unknown"


def test_fold_escalates_exclusion_via_bundle(tiny_trace: Any) -> None:
    """E-B5: escalation members leave the fold domain before evaluation."""

    bundle = tl.bundle(
        {"m0": tiny_trace, "m1": tiny_trace, "esc": tiny_trace},
        member_relations=[
            _episode_row("m0", 0, "prefill"),
            _episode_row("m1", 1, "decode"),
            _episode_row("esc", 2, "decode"),
            {"kind": "escalates", "from": "esc", "to": "m1", "params": {"at_step": 1}},
        ],
    )
    # With esc excluded the domain is 2 COMPLETE members == n_declared=2;
    # without the exclusion the count would be 3 and the fold unknown.
    result = bundle.derive_episode_status("ep", ledger=_episode_ledger(2, 2))
    assert result.status == "episode_complete"


def test_fold_min_provenance_tier() -> None:
    """Mixed exact + ledger_only member tiers fold to the MINIMUM tier."""

    result = derive_episode_status(
        [("complete", None), ("complete", None)],
        n_declared=2,
        ledger=None,
        member_tiers=["exact", "ledger_only"],
    )
    assert result.status == "episode_complete"
    assert result.provenance_tier == "ledger_only"


@pytest.mark.smoke
def test_fold_integration_real_failed_partial_member(tiny_trace: Any) -> None:
    """A real mid-forward failure folds via the failed member's real outcome."""

    torch.manual_seed(0)
    bomb = _BombModule()
    x = torch.randn(2, 3)
    with pytest.raises(Exception) as excinfo:
        tl.trace(bomb, x)
    partial = tl.partial.from_failed_capture(excinfo.value)
    outcome = partial.outcome
    assert outcome is not None and outcome.status.value == "failed"

    bundle = tl.bundle(
        {"m0": tiny_trace, "m1": partial},
        member_relations=[
            _episode_row("m0", 0, "prefill"),
            _episode_row("m1", 1, "decode"),
        ],
    )
    result = bundle.derive_episode_status("ep")
    assert result.status == "episode_failed_at_step"
    assert result.at_step == 1
    assert result.member_status == "FAILED"


# ---------------------------------------------------------------------------
# Gated persistence (S3): disclosed drop when inactive, round-trip when
# active, fail-closed on tamper and dangling artifacts
# ---------------------------------------------------------------------------


def _relation_bundle(tiny_trace: Any) -> Any:
    return tl.bundle(
        {"a": tiny_trace, "b": tiny_trace},
        baseline="a",
        member_relations=[
            _pair_row(),
            _episode_row("a", 0, "prefill"),
        ],
    )


@pytest.mark.smoke
def test_persistence_empty_table_omits_the_key(tiny_trace: Any, tmp_path: Path) -> None:
    """A relation-free bundle writes no key and reloads plain (S6 R7)."""

    from torchlens.bundle import Bundle

    bundle = Bundle({"a": tiny_trace})
    target = tmp_path / "plain.tlspec"
    bundle.save(target)
    metadata = json.loads((target / "bundle.json").read_text(encoding="utf-8"))
    assert "member_relations" not in metadata
    assert "_tlspec_prerelease" not in metadata
    loaded = tl.load(target)
    assert loaded.member_relations == ()


def test_persistence_plain_round_trip_byte_faithful(tiny_trace: Any, tmp_path: Path) -> None:
    """tlspec v8: relations round-trip byte-faithfully on a PLAIN save (R7),
    with no pre-release marker riding the artifact."""

    bundle = _relation_bundle(tiny_trace)
    target = tmp_path / "active.tlspec"
    bundle.save(target)
    metadata = json.loads((target / "bundle.json").read_text(encoding="utf-8"))
    assert "_tlspec_prerelease" not in metadata
    expected_payload = MemberRelationTable(bundle.member_relations).to_payload()
    assert metadata["member_relations"] == expected_payload
    loaded = tl.load(target)
    assert [row.to_payload() for row in loaded.member_relations] == expected_payload


def test_persistence_stale_marker_refuses_typed(tiny_trace: Any, tmp_path: Path) -> None:
    """A switched-era artifact (marker present, switch off) still refuses."""

    bundle = _relation_bundle(tiny_trace)
    target = tmp_path / "tampered.tlspec"
    bundle.save(target)
    metadata_path = target / "bundle.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["_tlspec_prerelease"] = {
        "marker": PRERELEASE_MARKER,
        "fields": ["Bundle.member_relations"],
    }
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(PreReleaseArtifactError):
        tl.load(target)


def test_persistence_dangling_member_refuses_typed(tiny_trace: Any, tmp_path: Path) -> None:
    """A plain v8 artifact naming a ghost member refuses R1-typed."""

    bundle = _relation_bundle(tiny_trace)
    target = tmp_path / "dangling.tlspec"
    bundle.save(target)
    metadata_path = target / "bundle.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["member_relations"] = [_pair_row(dst="ghost")]
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(BundleRelationError) as excinfo:
        tl.load(target)
    assert excinfo.value.fields["code"] == "bundle_relation_member_missing"
