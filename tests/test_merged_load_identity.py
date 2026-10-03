"""Load-side integrity of the ``merged-directory`` artifact (p4 3.4 R59).

These pins do NOT need a live process group: they build rank cores from plain
captures with injected ``collective_boundary_v1`` distributed annotations (the
same shape ``torchlens.distributed`` emits), merge them, save the artifact,
then attack the on-disk bytes and require a TYPED tamper/schema refusal --
never a silent accept, never a laundered runtime degradation.

Covered:

* Rank-IDENTITY binding: one honest core duplicated across N member slots
  must not read back as N distinct attesting ranks (ATTESTED laundering).
* ``expected_ranks`` forged to a non-list / non-integer refuses typed rather
  than escaping as a raw ``TypeError``/``ValueError``.
* A member that loads as a bundle but is not a valid rank core refuses as
  tamper, not laundered into the benign runtime-parse degradation channel.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.distributed._ledger import (
    GroupLifecycleEvent,
    GroupLifecycleLedger,
    membership_digest_for_ranks,
)
from torchlens.merged._artifact import canonical_json_bytes, load_merged, tree_hash
from torchlens.merged._enums import MergedErrorCode
from torchlens.merged._errors import MergedArtifactError

WORLD = membership_digest_for_ranks([0, 1])
_CC = hashlib.sha256(b"cc").hexdigest()
_AA = hashlib.sha256(b"aa").hexdigest()


def _ledger_payload() -> list[dict[str, Any]]:
    ledger = GroupLifecycleLedger()
    ledger.append(GroupLifecycleEvent(0, "seed", WORLD, 0, "seeded", "seeded", 0))
    return ledger.to_payload()


def _boundary(rank: int) -> dict[str, Any]:
    return {
        "schema": "collective_boundary_v1",
        "kind": "all_reduce",
        "func": "torch.distributed.all_reduce",
        "correlation": {
            "membership_digest": WORLD,
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
            "contribution_digests": [_CC],
            "destination_digests": [_AA],
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


def _rank_trace(rank: int):
    torch.manual_seed(0)
    log = tl.trace(nn.Linear(4, 4), torch.randn(2, 4))
    log.annotations["distributed"] = {
        "boundaries": [_boundary(rank)],
        "group_lifecycle_ledger": _ledger_payload(),
        "install_epoch": "seeded",
    }
    return log


def _saved_two_rank(tmp_path: Path) -> Path:
    merged = tl.merge_ranks([_rank_trace(0), _rank_trace(1)])
    assert merged.value_status.value == "attested_complete"
    art = tmp_path / "merged.tlspec"
    merged.save(art)
    return art


def _rehash(art: Path) -> None:
    """Recompute member tree hashes + descriptor checksum after a byte edit."""

    desc_path = art / "merge" / "descriptor.json"
    descriptor = json.loads(desc_path.read_text())
    for entry in descriptor["members"]:
        member = art / entry["path"]
        entry["tree_sha256"] = tree_hash(member)
    desc_bytes = canonical_json_bytes(descriptor)
    desc_path.write_bytes(desc_bytes)

    man_path = art / "manifest.json"
    manifest = json.loads(man_path.read_text())
    manifest["members"] = {
        str(entry["rank"]): entry["tree_sha256"] for entry in descriptor["members"]
    }
    manifest["descriptor_sha256"] = hashlib.sha256(desc_bytes).hexdigest()
    man_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def test_duplicated_core_across_slots_refuses_typed(tmp_path: Path) -> None:
    art = _saved_two_rank(tmp_path)
    # Duplicate rank 0's core (my_global_rank=0) over the rank-1 slot.
    shutil.rmtree(art / "members" / "rank_0001.tlspec")
    shutil.copytree(art / "members" / "rank_0000.tlspec", art / "members" / "rank_0001.tlspec")
    _rehash(art)
    with pytest.raises(MergedArtifactError) as excinfo:
        load_merged(art)
    assert excinfo.value.fields["code"] == MergedErrorCode.MERGED_DESCRIPTOR_TAMPER.value
    assert "prove it is rank 0" in str(excinfo.value)


def test_forged_expected_ranks_refuses_typed(tmp_path: Path) -> None:
    art = _saved_two_rank(tmp_path)
    desc_path = art / "merge" / "descriptor.json"
    descriptor = json.loads(desc_path.read_text())
    descriptor["derivation"]["expected_ranks"] = "not-a-list"
    desc_bytes = canonical_json_bytes(descriptor)
    desc_path.write_bytes(desc_bytes)
    man_path = art / "manifest.json"
    manifest = json.loads(man_path.read_text())
    manifest["descriptor_sha256"] = hashlib.sha256(desc_bytes).hexdigest()
    man_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    with pytest.raises(MergedArtifactError) as excinfo:
        load_merged(art)
    assert excinfo.value.fields["code"] == MergedErrorCode.MERGED_SCHEMA_INVALID.value
    assert "expected_ranks" in str(excinfo.value)


def test_non_rank_core_member_refuses_not_laundered(tmp_path: Path) -> None:
    art = _saved_two_rank(tmp_path)
    # Replace rank 1's core with a plain (non-distributed) bundle that loads
    # fine but carries no rank-core evidence. Pre-fix this laundered into a
    # runtime-parse degradation and loaded at partial; it must refuse as tamper.
    torch.manual_seed(0)
    plain = tl.trace(nn.Linear(4, 4), torch.randn(2, 4))
    shutil.rmtree(art / "members" / "rank_0001.tlspec")
    tl.save(plain, art / "members" / "rank_0001.tlspec")
    _rehash(art)
    with pytest.raises(MergedArtifactError) as excinfo:
        load_merged(art)
    assert excinfo.value.fields["code"] == MergedErrorCode.MERGED_DESCRIPTOR_TAMPER.value
    assert "not a valid rank capture" in str(excinfo.value) or "no coherent" in str(excinfo.value)


def test_denylisted_member_pickle_refuses_not_laundered(tmp_path: Path) -> None:
    """A-R58-1: a guarded-unpickler denylist refusal on a member core is TAMPER.

    Plant a classic RCE gadget pickle (``__reduce__`` -> ``os.system``) as one
    member's ``metadata.pkl`` and re-hash so the descriptor's tree hash matches
    (the byte-integrity layer would otherwise refuse first). The guarded
    unpickler denies the gadget with ``pickle.UnpicklingError``. Pre-fix,
    ``load_merged``'s bare ``except Exception`` around ``load_bundle`` laundered
    that denylist refusal into the "no longer parses on this runtime"
    degradation channel and SUCCEEDED the merge off the honest member -- a tamper
    signal silently downgraded. It must refuse typed as tamper instead.
    """

    import pickle as _pickle

    class _Evil:
        def __reduce__(self):  # pragma: no cover - never executed (denied at load)
            return (os.system, ("echo pwned",))

    art = _saved_two_rank(tmp_path)
    metadata = art / "members" / "rank_0001.tlspec" / "metadata.pkl"
    assert metadata.exists()
    metadata.write_bytes(_pickle.dumps(_Evil()))
    _rehash(art)
    with pytest.raises(MergedArtifactError) as excinfo:
        load_merged(art)
    assert excinfo.value.fields["code"] == MergedErrorCode.MERGED_DESCRIPTOR_TAMPER.value
    assert "bundle-integrity" in str(excinfo.value)


def test_corrupt_member_pickle_stream_refuses_not_laundered(tmp_path: Path) -> None:
    """A-R58-1 sibling: a truncated/garbage member pickle stream is TAMPER too.

    Not every integrity failure is a denylist hit -- a corrupt or truncated
    ``metadata.pkl`` also surfaces as ``pickle.UnpicklingError`` and was
    laundered identically. Garbage bytes must refuse as tamper, never degrade.
    """

    art = _saved_two_rank(tmp_path)
    metadata = art / "members" / "rank_0001.tlspec" / "metadata.pkl"
    metadata.write_bytes(b"\x80\x05not-a-valid-pickle-stream\xff\xff")
    _rehash(art)
    with pytest.raises(MergedArtifactError) as excinfo:
        load_merged(art)
    assert excinfo.value.fields["code"] == MergedErrorCode.MERGED_DESCRIPTOR_TAMPER.value
    assert "bundle-integrity" in str(excinfo.value)


def test_shared_member_path_refuses_typed(tmp_path: Path) -> None:
    art = _saved_two_rank(tmp_path)
    desc_path = art / "merge" / "descriptor.json"
    descriptor = json.loads(desc_path.read_text())
    # Point both member entries at rank 0's core directory.
    rank0_path = next(e["path"] for e in descriptor["members"] if e["rank"] == 0)
    for entry in descriptor["members"]:
        entry["path"] = rank0_path
    desc_bytes = canonical_json_bytes(descriptor)
    desc_path.write_bytes(desc_bytes)
    man_path = art / "manifest.json"
    manifest = json.loads(man_path.read_text())
    manifest["descriptor_sha256"] = hashlib.sha256(desc_bytes).hexdigest()
    man_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    with pytest.raises(MergedArtifactError) as excinfo:
        load_merged(art)
    assert excinfo.value.fields["code"] == MergedErrorCode.MERGED_SCHEMA_INVALID.value
    assert "shared by more than one rank" in str(excinfo.value)


def test_tamper_refusal_carries_a_remedy(tmp_path: Path) -> None:
    """Every merged tamper refusal now carries fields['remedy'] (R65)."""

    art = _saved_two_rank(tmp_path)
    # Corrupt the descriptor checksum: a checksum-mismatch tamper refusal.
    man_path = art / "manifest.json"
    manifest = json.loads(man_path.read_text())
    manifest["descriptor_sha256"] = "0" * 64
    man_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    with pytest.raises(MergedArtifactError) as excinfo:
        load_merged(art)
    assert excinfo.value.fields["code"] == MergedErrorCode.MERGED_DESCRIPTOR_TAMPER.value
    assert excinfo.value.fields.get("remedy"), "tamper refusal must carry a remedy"


def test_honest_two_rank_round_trip_still_loads(tmp_path: Path) -> None:
    art = _saved_two_rank(tmp_path)
    loaded = load_merged(art)
    assert loaded.alignment.value == "aligned"
    assert loaded.value_status.value == "attested_complete"
    assert loaded.rank_ids == (0, 1)
    assert loaded.load_degradations == ()


@pytest.mark.parametrize("resource_exc", [MemoryError, OSError])
def test_member_load_resource_failure_surfaces_not_laundered(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, resource_exc: type[Exception]
) -> None:
    """B2R4-17 (the R58 resource half): ENOMEM/EIO during member load surfaces.

    The member's bytes were just read successfully by the tree-hash integrity
    layer, so the bare handler's "no longer parses on this runtime" was an
    actively misleading description of an I/O failure -- and it silently capped
    the merge at ``partial`` off the remaining members. A ``MemoryError`` or
    ``OSError`` escaping the member bundle load must propagate raw as the
    environment error it is, never enter ``load_degradations``.
    """

    art = _saved_two_rank(tmp_path)
    import torchlens._io.bundle as bundle_module

    real_load = bundle_module.load
    calls = {"count": 0}

    def _failing_load(path, *args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 2:
            raise resource_exc("simulated resource failure during member core load")
        return real_load(path, *args, **kwargs)

    monkeypatch.setattr(bundle_module, "load", _failing_load)
    with pytest.raises(resource_exc, match="simulated resource failure"):
        load_merged(art)
    assert calls["count"] == 2, "the resource failure must abort the member loop mid-flight"
