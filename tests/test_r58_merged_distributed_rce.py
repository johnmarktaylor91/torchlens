"""r58 A3e/A3g: hostile-input parity for the merged + distributed load surfaces.

Two gaps this closes.

A3e -- the distributed group-lifecycle ledger was NOT fail-closed at its parse
boundary. ``GroupLifecycleEvent.from_payload`` assigned its closed-vocabulary fields
(``kind`` / ``ordinal_source`` / ``install_epoch``) straight from the payload, and
``lineage_vectors()`` then SILENTLY DROPPED an event whose kind it did not recognize --
so a forged sidecar could erase a generation from the lineage evidence the pre-join
membership audit reads, or promote a ``seeded`` rank to the complete witness it never
was. ``GroupLifecycleLedger.from_payload`` also bypassed ``append()``, so duplicate or
decreasing event indices loaded happily.

A3g -- the RCE corpus (13 ``test_r*_rce.py`` files at the time) had ZERO coverage of the
merged-directory / ``load_merged`` / ledger surface the sprint had just added. The
artifact-boundary invariant is that a hostile artifact refuses TYPED and never crashes,
never executes, and never degrades a tamper into a presence gap.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import torch

from torchlens.distributed._ledger import (
    GroupLifecycleEvent,
    GroupLifecycleLedger,
    membership_digest_for_ranks,
)
from torchlens.merged._artifact import load_merged
from torchlens.merged._enums import (
    MERGED_BUNDLE_FORMAT,
    MERGED_DESCRIPTOR_KIND,
    MERGED_DESCRIPTOR_SCHEMA_VERSION,
    MERGED_TLSPEC_VERSION,
    MergedErrorCode,
)
from torchlens.merged._errors import MergedArtifactError, MergeInputError

_DIGEST = membership_digest_for_ranks([0, 1])


def _event_payload(**overrides: Any) -> dict[str, Any]:
    """Return a valid group-lifecycle event payload with optional overrides."""

    payload = GroupLifecycleEvent(
        event_index=0,
        kind="create",
        membership_digest=_DIGEST,
        ordinal=0,
        ordinal_source="wrapped",
        install_epoch="armed_before_any_group",
        local_creation_index=0,
        group_name="group:0",
        name_scheme="pg.group_name",
    ).to_payload()
    payload.update(overrides)
    return payload


# --------------------------------------------------------------------------- #
# A3e: the ledger parse boundary is fail-closed                                #
# --------------------------------------------------------------------------- #


@pytest.mark.smoke
def test_valid_ledger_payload_round_trips() -> None:
    """The tightened parser still accepts what ``to_payload`` emits."""

    ledger = GroupLifecycleLedger()
    ledger.append(GroupLifecycleEvent.from_payload(_event_payload()))
    ledger.append(GroupLifecycleEvent.from_payload(_event_payload(event_index=1, kind="destroy")))
    rebuilt = GroupLifecycleLedger.from_payload(ledger.to_payload())
    assert rebuilt.events == ledger.events
    assert rebuilt.lineage_vectors()[_DIGEST].entries[0].destroyed is True


@pytest.mark.parametrize(
    "overrides",
    [
        pytest.param({"kind": "frobnicate"}, id="kind-outside-vocabulary"),
        pytest.param({"install_epoch": "armed_before_anything"}, id="epoch-outside-vocabulary"),
        pytest.param({"ordinal_source": "conjured"}, id="ordinal-source-outside-vocabulary"),
        pytest.param({"membership_digest": "not-a-digest"}, id="digest-not-sha256"),
        pytest.param({"membership_digest": 7}, id="digest-not-a-string"),
        pytest.param({"event_index": -1}, id="negative-event-index"),
        pytest.param({"event_index": "0"}, id="event-index-not-an-int"),
        pytest.param({"ordinal": True}, id="ordinal-is-a-bool"),
        pytest.param({"group_name": 3}, id="diagnostic-not-a-string"),
        pytest.param({"smuggled": "field"}, id="unknown-key"),
    ],
)
def test_forged_event_field_refuses(overrides: dict[str, Any]) -> None:
    """Every closed-vocabulary / typed field refuses at the parse boundary.

    Fail-before: an unknown ``kind`` parsed fine and was then silently omitted from
    ``lineage_vectors()``, i.e. the forged sidecar ERASED evidence rather than being
    rejected.
    """

    with pytest.raises((ValueError, TypeError)):
        GroupLifecycleEvent.from_payload(_event_payload(**overrides))


def test_missing_required_field_refuses() -> None:
    """A truncated payload refuses rather than defaulting a verdict-steering field."""

    payload = _event_payload()
    del payload["install_epoch"]
    with pytest.raises(ValueError, match="missing keys"):
        GroupLifecycleEvent.from_payload(payload)


def test_non_monotone_ledger_payload_refuses() -> None:
    """A loaded ledger obeys the SAME monotone contract as a live one.

    Fail-before: ``from_payload`` handed the list straight to ``__init__``, bypassing
    ``append()``, so duplicate or decreasing indices could reorder or mask evidence.
    """

    with pytest.raises(ValueError, match="strictly increasing"):
        GroupLifecycleLedger.from_payload([_event_payload(), _event_payload()])
    with pytest.raises(ValueError, match="strictly increasing"):
        GroupLifecycleLedger.from_payload([_event_payload(event_index=5), _event_payload()])


def test_non_list_ledger_payload_refuses() -> None:
    """A payload of the wrong shape refuses typed instead of iterating a mapping."""

    with pytest.raises(TypeError, match="must be a list"):
        GroupLifecycleLedger.from_payload({"event_index": 0})  # type: ignore[arg-type]


def test_lineage_derivation_never_silently_drops_a_kind() -> None:
    """The derivation raises on an unrecognized kind rather than omitting it.

    Constructed directly (bypassing the parse boundary) to prove the belt is inside the
    derivation too, so a future kind cannot be added and silently ignored.
    """

    rogue = GroupLifecycleEvent(
        event_index=0,
        kind="frobnicate",  # type: ignore[arg-type]
        membership_digest=_DIGEST,
        ordinal=0,
        ordinal_source="wrapped",
        install_epoch="seeded",
    )
    ledger = GroupLifecycleLedger([rogue])
    with pytest.raises(ValueError, match="outside the closed"):
        ledger.lineage_vectors()


def test_ledger_epoch_disagreement_refuses_at_evidence_extraction() -> None:
    """A ledger whose events disagree with the record epoch refuses typed.

    ``lineage_vectors()`` stamps each vector's epoch from the EVENTS while the audit
    also reads the record-level epoch, so a disagreement is a sidecar trying to promote
    a ``seeded`` rank to a complete witness.
    """

    from torchlens.merged._evidence import extract_rank_evidence

    boundary = {
        "schema": "collective_boundary_v1",
        "kind": "all_reduce",
        "correlation": {
            "membership_digest": _DIGEST,
            "lifetime_ordinal": 0,
            "channel": "default",
            "seq": 0,
        },
        "group": {"global_ranks": [0, 1], "my_global_rank": 0},
        "events": {"completion_binding": "issue_sync"},
        "witness": {"policy_resolved": "digest"},
        "op_labels_raw": ["all_reduce_1_1"],
    }

    class _FakeTrace:
        annotations = {
            "distributed": {
                "boundaries": [boundary],
                "install_epoch": "seeded",
                "group_lifecycle_ledger": [_event_payload(install_epoch="armed_before_any_group")],
            }
        }

    with pytest.raises(MergeInputError) as caught:
        extract_rank_evidence(_FakeTrace(), "forged")
    assert caught.value.fields["code"] == MergedErrorCode.MERGED_SCHEMA_INVALID.value


# --------------------------------------------------------------------------- #
# A3g: the merged-directory load surface refuses hostile artifacts typed        #
# --------------------------------------------------------------------------- #


def _merged_artifact(root: Path, descriptor: dict[str, Any], **manifest: Any) -> Path:
    """Write a syntactically well-formed merged-directory artifact."""

    root.mkdir(parents=True, exist_ok=True)
    (root / "merge").mkdir(exist_ok=True)
    descriptor_bytes = json.dumps(descriptor).encode("utf-8")
    (root / "merge" / "descriptor.json").write_bytes(descriptor_bytes)
    payload = {
        "bundle_format": MERGED_BUNDLE_FORMAT,
        "tlspec_version": MERGED_TLSPEC_VERSION,
        "descriptor_sha256": hashlib.sha256(descriptor_bytes).hexdigest(),
        "members": {"0": "a" * 64},
    }
    payload.update(manifest)
    (root / "manifest.json").write_text(json.dumps(payload), encoding="utf-8")
    return root


def _descriptor(**overrides: Any) -> dict[str, Any]:
    """Return a well-formed merged descriptor with optional overrides."""

    payload: dict[str, Any] = {
        "descriptor_kind": MERGED_DESCRIPTOR_KIND,
        "schema_version": MERGED_DESCRIPTOR_SCHEMA_VERSION,
        "members": [{"rank": 0, "path": "rank0", "tree_sha256": "a" * 64}],
    }
    payload.update(overrides)
    return payload


def test_deeply_nested_merged_manifest_refuses_typed(tmp_path: Path) -> None:
    """A depth-5000 root manifest is a typed schema refusal, never a RecursionError."""

    root = tmp_path / "deep.merged"
    root.mkdir()
    (root / "manifest.json").write_text('{"d": ' + "[" * 5000 + "0" + "]" * 5000 + "}")
    with pytest.raises(MergedArtifactError) as caught:
        load_merged(root)
    assert caught.value.fields["code"] == MergedErrorCode.MERGED_SCHEMA_INVALID.value


def test_deeply_nested_merged_descriptor_refuses_typed(tmp_path: Path) -> None:
    """An over-nested descriptor refuses typed AFTER its checksum verifies."""

    root = tmp_path / "deepdesc.merged"
    root.mkdir()
    (root / "merge").mkdir()
    descriptor_bytes = ("[" * 5000 + "0" + "]" * 5000).encode("utf-8")
    (root / "merge" / "descriptor.json").write_bytes(descriptor_bytes)
    (root / "manifest.json").write_text(
        json.dumps(
            {
                "bundle_format": MERGED_BUNDLE_FORMAT,
                "tlspec_version": MERGED_TLSPEC_VERSION,
                "descriptor_sha256": hashlib.sha256(descriptor_bytes).hexdigest(),
                "members": {"0": "a" * 64},
            }
        )
    )
    with pytest.raises(MergedArtifactError) as caught:
        load_merged(root)
    assert caught.value.fields["code"] == MergedErrorCode.MERGED_SCHEMA_INVALID.value


def test_descriptor_checksum_tamper_is_a_typed_tamper_not_a_gap(tmp_path: Path) -> None:
    """Rewritten descriptor bytes refuse as TAMPER; tamper is never a presence gap."""

    root = _merged_artifact(tmp_path / "tampered.merged", _descriptor())
    (root / "merge" / "descriptor.json").write_bytes(b'{"descriptor_kind": "swapped"}')
    with pytest.raises(MergedArtifactError) as caught:
        load_merged(root)
    assert caught.value.fields["code"] == MergedErrorCode.MERGED_DESCRIPTOR_TAMPER.value


@pytest.mark.parametrize(
    "member_path",
    [
        pytest.param("../escape", id="parent-traversal"),
        pytest.param("/etc", id="absolute-path"),
        pytest.param(".", id="self-referential"),
    ],
)
def test_member_path_escape_refuses_typed(tmp_path: Path, member_path: str) -> None:
    """A descriptor member path outside the artifact root refuses typed."""

    root = _merged_artifact(
        tmp_path / f"escape_{abs(hash(member_path))}.merged",
        _descriptor(members=[{"rank": 0, "path": member_path, "tree_sha256": "a" * 64}]),
    )
    with pytest.raises(MergedArtifactError) as caught:
        load_merged(root)
    assert caught.value.fields["code"] == MergedErrorCode.MERGED_SCHEMA_INVALID.value


@pytest.mark.parametrize(
    "manifest_override",
    [
        pytest.param({"bundle_format": "trace-bundle"}, id="wrong-bundle-format"),
        pytest.param({"tlspec_version": 999}, id="unsupported-tlspec-version"),
        pytest.param({"members": []}, id="members-table-not-a-mapping"),
    ],
)
def test_merged_manifest_vocabulary_violations_refuse_typed(
    tmp_path: Path, manifest_override: dict[str, Any]
) -> None:
    """Closed-vocabulary violations in the root manifest refuse typed."""

    root = _merged_artifact(
        tmp_path / f"vocab_{next(iter(manifest_override))}.merged",
        _descriptor(),
        **manifest_override,
    )
    with pytest.raises(MergedArtifactError) as caught:
        load_merged(root)
    assert caught.value.fields["code"] == MergedErrorCode.MERGED_SCHEMA_INVALID.value


@pytest.mark.smoke
def test_missing_merged_members_refuse_typed(tmp_path: Path) -> None:
    """A descriptor naming a rank core that is absent refuses as tamper."""

    root = _merged_artifact(tmp_path / "absent.merged", _descriptor())
    with pytest.raises(MergedArtifactError) as caught:
        load_merged(root)
    assert caught.value.fields["code"] == MergedErrorCode.MERGED_DESCRIPTOR_TAMPER.value


@pytest.mark.parametrize(
    "entry",
    [
        pytest.param({}, id="empty-object"),
        pytest.param("rank0", id="not-a-mapping"),
        pytest.param({"rank": 0, "path": "rank0"}, id="missing-tree-sha256"),
        pytest.param(
            {"rank": 0, "path": "rank0", "tree_sha256": "a" * 64, "smuggled": 1},
            id="unknown-key",
        ),
        pytest.param(
            {"rank": True, "path": "rank0", "tree_sha256": "a" * 64},
            id="rank-is-a-bool",
        ),
        pytest.param(
            {"rank": "0", "path": "rank0", "tree_sha256": "a" * 64},
            id="rank-not-an-int",
        ),
        pytest.param(
            {"rank": -1, "path": "rank0", "tree_sha256": "a" * 64},
            id="rank-negative",
        ),
        pytest.param({"rank": 0, "path": 5, "tree_sha256": "a" * 64}, id="path-not-a-string"),
        pytest.param({"rank": 0, "path": "rank0", "tree_sha256": 7}, id="digest-not-a-string"),
        pytest.param(
            {"rank": 0, "path": "rank0", "tree_sha256": "Z" * 64},
            id="digest-not-lowercase-hex",
        ),
        pytest.param(
            {"rank": 0, "path": "rank0", "tree_sha256": "a" * 63},
            id="digest-wrong-length",
        ),
    ],
)
def test_malformed_member_entry_refuses_typed(tmp_path: Path, entry: Any) -> None:
    """Every member entry is validated against closed keys/types before use.

    Fail-before (p2 R58 sol-R58-1): a validly hashed descriptor with
    ``"members": [{}]`` escaped the documented ``MergedArtifactError`` as a raw
    ``KeyError('rank')``; non-mapping entries escaped via ``TypeError`` and
    booleans were accepted by ``int()``.
    """

    root = _merged_artifact(tmp_path / "malformed.merged", _descriptor(members=[entry]))
    with pytest.raises(MergedArtifactError) as caught:
        load_merged(root)
    assert caught.value.fields["code"] == MergedErrorCode.MERGED_SCHEMA_INVALID.value


def test_duplicate_member_rank_refuses_typed(tmp_path: Path) -> None:
    """Two member entries claiming the same rank refuse at the schema pass."""

    root = _merged_artifact(
        tmp_path / "dup.merged",
        _descriptor(
            members=[
                {"rank": 0, "path": "rank0", "tree_sha256": "a" * 64},
                {"rank": 0, "path": "rank0b", "tree_sha256": "b" * 64},
            ]
        ),
    )
    with pytest.raises(MergedArtifactError) as caught:
        load_merged(root)
    assert caught.value.fields["code"] == MergedErrorCode.MERGED_SCHEMA_INVALID.value


# --------------------------------------------------------------------------- #
# A-R58-2: refuse-not-degrade at the member bundle-load entry point            #
#                                                                             #
# Every tamper test above stops at the descriptor/manifest/tree-hash layer;   #
# NONE reached ``load_bundle(member_path)`` with a real rank core, so the     #
# guarded-unpickler-refusal-laundered-as-degradation class (A-R58-1) had zero #
# gadget coverage. These build a real two-rank artifact and corrupt one       #
# member's ``metadata.pkl`` so ``load_bundle`` raises a guarded-unpickler /   #
# corrupt-stream error, proving the failure refuses as TAMPER, never a merge  #
# that silently succeeds at partial off the honest member.                    #
# --------------------------------------------------------------------------- #


def _real_two_rank_artifact(tmp_path: Path) -> Path:
    """Build a real, honest two-rank merged artifact via the public API."""

    from torch import nn

    import torchlens as tl

    def boundary(rank: int) -> dict[str, Any]:
        return {
            "schema": "collective_boundary_v1",
            "kind": "all_reduce",
            "func": "torch.distributed.all_reduce",
            "correlation": {
                "membership_digest": _DIGEST,
                "lifetime_ordinal": 0,
                "channel": "coll",
                "seq": 0,
            },
            "group": {
                "global_ranks": [0, 1],
                "size": 2,
                "backend": "gloo",
                "my_global_rank": rank,
                "my_group_rank": None,
                "coord_provenance": "test",
            },
            "reduce_op": "RedOpType.SUM",
            "peer": None,
            "events": {"async_op": False, "completion_binding": "issue_sync"},
            "roles": [
                {
                    "role": "contribution_destination",
                    "index": 0,
                    "shape": [2, 4],
                    "logical_shape": None,
                    "placements": None,
                }
            ],
            "witness": {
                "policy_resolved": "digest",
                "contribution_digests": [hashlib.sha256(b"cc").hexdigest()],
                "destination_digests": [hashlib.sha256(b"aa").hexdigest()],
                "not_present_reason": None,
            },
            "lifetime_evidence": {
                "ordinal_source": "seeded",
                "install_epoch": "seeded",
                "arming_source": "explicit",
            },
            "c10d_group_seq": None,
            "disclosures": [],
            "op_labels_raw": ["allreduce_1_raw"],
            "op_node": True,
        }

    ledger = GroupLifecycleLedger()
    ledger.append(GroupLifecycleEvent(0, "seed", _DIGEST, 0, "seeded", "seeded", 0))
    ledger_payload = ledger.to_payload()

    def rank_trace(rank: int):
        torch.manual_seed(0)
        log = tl.trace(nn.Linear(4, 4), torch.randn(2, 4))
        log.annotations["distributed"] = {
            "boundaries": [boundary(rank)],
            "group_lifecycle_ledger": ledger_payload,
            "install_epoch": "seeded",
        }
        return log

    merged = tl.merge_ranks([rank_trace(0), rank_trace(1)])
    art = tmp_path / "merged.tlspec"
    merged.save(art)
    return art


def _rehash_artifact(art: Path) -> None:
    """Recompute member tree hashes + descriptor checksum after a byte edit."""

    from torchlens.merged._artifact import canonical_json_bytes, tree_hash

    desc_path = art / "merge" / "descriptor.json"
    descriptor = json.loads(desc_path.read_text())
    for entry in descriptor["members"]:
        entry["tree_sha256"] = tree_hash(art / entry["path"])
    desc_bytes = canonical_json_bytes(descriptor)
    desc_path.write_bytes(desc_bytes)

    man_path = art / "manifest.json"
    manifest = json.loads(man_path.read_text())
    manifest["members"] = {
        str(entry["rank"]): entry["tree_sha256"] for entry in descriptor["members"]
    }
    manifest["descriptor_sha256"] = hashlib.sha256(desc_bytes).hexdigest()
    man_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def test_denylisted_member_pickle_is_tamper_not_degradation(tmp_path: Path) -> None:
    """A member ``metadata.pkl`` denylist RCE gadget refuses as tamper (A-R58-1)."""

    import os
    import pickle

    class _Evil:
        def __reduce__(self):  # pragma: no cover - denied at load, never executed
            return (os.system, ("echo pwned",))

    art = _real_two_rank_artifact(tmp_path)
    (art / "members" / "rank_0001.tlspec" / "metadata.pkl").write_bytes(pickle.dumps(_Evil()))
    _rehash_artifact(art)
    with pytest.raises(MergedArtifactError) as caught:
        load_merged(art)
    assert caught.value.fields["code"] == MergedErrorCode.MERGED_DESCRIPTOR_TAMPER.value
    assert "bundle-integrity" in str(caught.value)


def test_corrupt_member_pickle_is_tamper_not_degradation(tmp_path: Path) -> None:
    """A corrupt member ``metadata.pkl`` stream refuses as tamper (A-R58-1)."""

    art = _real_two_rank_artifact(tmp_path)
    (art / "members" / "rank_0001.tlspec" / "metadata.pkl").write_bytes(
        b"\x80\x05not-a-valid-pickle-stream\xff\xff"
    )
    _rehash_artifact(art)
    with pytest.raises(MergedArtifactError) as caught:
        load_merged(art)
    assert caught.value.fields["code"] == MergedErrorCode.MERGED_DESCRIPTOR_TAMPER.value
    assert "bundle-integrity" in str(caught.value)
