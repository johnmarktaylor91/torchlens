"""F03 ledger memo item 3: Bundle lineage persistence in the .tlspec artifact.

The C07X split-off join, landed: ``bundle_id`` / ``forked_from_bundle_id`` /
``member_construction`` anchors / hash-chained ``operations`` /
``member_effect_tables`` persist as writer-owned ``bundle.json`` sections and
load back fail-closed. The effect-table registry survives save/load/released
members (D3c); pre-F03 artifacts (sections absent) load with honest
``origin="loaded"`` anchors.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.bundle._lineage import MemberEffectRow, MemberEffectTable
from torchlens.errors.episode import BundleRelationError

pytestmark = pytest.mark.smoke


class _Tiny(nn.Module):
    def __init__(self, offset: float = 0.0) -> None:
        super().__init__()
        self.linear = nn.Linear(3, 3)
        self.offset = offset

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.linear(x)) + self.offset


def _pair() -> tl.Bundle:
    torch.manual_seed(11)
    x = torch.randn(2, 3)
    baseline = tl.trace(_Tiny(), x, capture=tl.options.CaptureOptions(intervention_ready=True))
    changed = tl.trace(
        _Tiny(offset=0.5), x, capture=tl.options.CaptureOptions(intervention_ready=True)
    )
    return tl.Bundle({"baseline": baseline, "changed": changed}, baseline="baseline")


def _rich_bundle() -> tl.Bundle:
    bundle = _pair()
    bundle.relate({"kind": "alternative_of", "from": "changed", "to": "baseline", "params": {}})
    child = bundle.fork()
    operation = child.operations[0]
    child._effect_tables[operation.operation_id] = MemberEffectTable(
        operation_id=operation.operation_id,
        rows=(
            MemberEffectRow(
                candidate_id="c0",
                status="completed",
                member_name="changed",
                value=1.25,
                resolved_site_count=1,
                retained=True,
            ),
            MemberEffectRow(candidate_id="c1", status="released", value=0.5),
        ),
        baseline_member="baseline",
        metric_repr="logit_diff",
        edit_repr="zero_ablate()",
        lane="live_hook",
        retain_policy="top_k(1)",
    )
    return child


def test_lineage_round_trips_through_tlspec(tmp_path: Path) -> None:
    child = _rich_bundle()
    path = tmp_path / "f03_lineage_bundle"
    child.save(str(path))
    loaded = tl.load(str(path))

    assert loaded.bundle_id == child.bundle_id
    assert loaded._forked_from_bundle_id == child._forked_from_bundle_id
    assert loaded.member_construction == child.member_construction
    assert [row.to_payload() for row in loaded.operations] == [
        row.to_payload() for row in child.operations
    ]
    table = loaded._effect_tables[child.operations[0].operation_id]
    assert table == child._effect_tables[child.operations[0].operation_id]
    # Released candidates stay table rows on the loaded artifact (D3c).
    assert table.rows[1].status == "released"
    assert table.rows[1].member_name is None


def test_pre_f03_artifact_loads_with_loaded_anchors(tmp_path: Path) -> None:
    bundle = _pair()
    path = tmp_path / "f03_pre_bundle"
    bundle.save(str(path))
    metadata_path = path / "bundle.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    for key in (
        "bundle_id",
        "forked_from_bundle_id",
        "member_construction",
        "operations",
        "member_effect_tables",
    ):
        metadata.pop(key, None)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")

    loaded = tl.load(str(path))
    # A fresh identity is minted; anchors disclose no construction evidence.
    assert loaded.bundle_id
    assert loaded.member_construction == {
        "baseline": {"origin": "loaded"},
        "changed": {"origin": "loaded"},
    }
    assert loaded.operations == ()


def _tamper(path: Path, mutate) -> None:
    metadata_path = path / "bundle.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    mutate(metadata)
    metadata_path.write_text(json.dumps(metadata), encoding="utf-8")


def test_tampered_operation_chain_refuses_typed(tmp_path: Path) -> None:
    child = _rich_bundle()
    path = tmp_path / "f03_tampered_ops"
    child.save(str(path))

    def _flip_kind(metadata: dict) -> None:
        metadata["operations"][0]["kind"] = "vary"

    _tamper(path, _flip_kind)
    with pytest.raises(BundleRelationError) as excinfo:
        tl.load(str(path))
    assert excinfo.value.fields["code"] == "bundle_lineage_invalid"


def test_tampered_anchor_origin_refuses_typed(tmp_path: Path) -> None:
    child = _rich_bundle()
    path = tmp_path / "f03_tampered_anchor"
    child.save(str(path))

    def _bad_origin(metadata: dict) -> None:
        metadata["member_construction"]["changed"]["origin"] = "teleported"

    _tamper(path, _bad_origin)
    with pytest.raises(BundleRelationError) as excinfo:
        tl.load(str(path))
    assert excinfo.value.fields["code"] == "bundle_lineage_invalid"


def test_effect_table_key_mismatch_refuses_typed(tmp_path: Path) -> None:
    child = _rich_bundle()
    path = tmp_path / "f03_tampered_table"
    child.save(str(path))

    def _rekey(metadata: dict) -> None:
        tables = metadata["member_effect_tables"]
        (only_key,) = list(tables)
        tables["not_that_operation"] = tables.pop(only_key)

    _tamper(path, _rekey)
    with pytest.raises(BundleRelationError) as excinfo:
        tl.load(str(path))
    assert excinfo.value.fields["code"] == "bundle_lineage_invalid"
