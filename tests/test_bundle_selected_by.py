"""The ``selected_by`` control relation kind (harness bundles, tlspec v10).

"writer selected_by router" records that the router's decision selected the
writer call, with no claim that the router's text reached the writer's
prompt. Covers the closed params (``chosen``, ``candidates``, ``evidence``),
the direction pin, the walks and the comparison gate, the v10 round trip, the
version admission, and the unchanged refusal of unknown kinds.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from _format3_envelope import redigest

import torchlens as tl
from torchlens._io import ArtifactTooOldError
from torchlens.bundle import Bundle
from torchlens.bundle._relations import MemberRelationRow, MemberRelationTable
from torchlens.errors.episode import BundleRelationError


class _TinyLM(torch.nn.Module):
    """Tokenizer-free stand-in: ids in, logits out."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(16, 4)
        self.head = torch.nn.Linear(4, 16)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.head(torch.relu(self.embed(ids)))


@pytest.fixture(scope="module")
def model() -> _TinyLM:
    torch.manual_seed(0)
    return _TinyLM()


def _routed(model: _TinyLM, router_ids: list[int] | None = None) -> Bundle:
    ids = router_ids or [1, 2, 3]
    return Bundle(
        {
            "router": tl.trace(model, torch.tensor([ids])),
            "writer": tl.trace(model, torch.tensor([[1, 2, 3]])),
        },
        baseline="router",
    )


def _code(excinfo: pytest.ExceptionInfo[BaseException]) -> str:
    return str(excinfo.value.fields["code"])  # type: ignore[attr-defined]


def _evidence() -> dict[str, object]:
    return {
        "schema": "route_scores_v1",
        "items": [{"grade": "verified", "basis": "router_logits", "scores": [0.2, 0.8]}],
        "facts_digest": "sha256:00",
    }


def test_round_trip_keeps_the_row(tmp_path: Path, model: _TinyLM) -> None:
    bundle = _routed(model).relate(
        "selected_by", "writer", "router", chosen="lookup", candidates=["lookup", "none"]
    )
    bundle.save(tmp_path / "routed.tlspec")
    manifest = json.loads((tmp_path / "routed.tlspec" / "manifest.json").read_text())
    assert manifest["tlspec_version"] == 12
    loaded = tl.load(tmp_path / "routed.tlspec")
    assert loaded.relations == bundle.relations
    assert loaded.upstream("writer") == ("router",)


# shrink-r4: rows kept carry a sole library arc or a campaign kill; the others were proven redundant (slice 1 evidence)
@pytest.mark.parametrize(
    "params",
    [
        {"candidates": []},
        {"candidates": [1, 1]},
        {"candidates": [{"route": 1}]},
        {"chosen": True, "candidates": [0, 1]},
        {"evidence": {"schema": "s", "items": []}},
    ],
)
def test_malformed_params_refuse_typed(model: _TinyLM, params: dict[str, object]) -> None:
    bundle = _routed(model)
    with pytest.raises(BundleRelationError) as excinfo:
        bundle.relate("selected_by", "writer", "router", **params)
    assert _code(excinfo) == "bundle_relation_schema_invalid"
    assert bundle.relations == (), "a refused relate leaves the table unchanged"


def test_tampered_artifact_param_refuses_at_load(tmp_path: Path, model: _TinyLM) -> None:
    bundle = _routed(model).relate("selected_by", "writer", "router", chosen=1, candidates=[0, 1])
    path = tmp_path / "routed.tlspec"
    bundle.save(path)
    metadata_path = path / "relations.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["member_relations"][0]["params"]["chosen"] = 7
    metadata_path.write_text(json.dumps(metadata))
    redigest(metadata_path.parent, "relations.json")
    with pytest.raises(BundleRelationError) as excinfo:
        tl.load(path)
    assert _code(excinfo) == "bundle_relation_schema_invalid"


def test_v9_stamped_artifact_with_the_kind_refuses(tmp_path: Path, model: _TinyLM) -> None:
    path = tmp_path / "routed.tlspec"
    _routed(model).relate("selected_by", "writer", "router").save(path)
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["tlspec_version"] = 9
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ArtifactTooOldError) as excinfo:
        tl.load(path)
    assert _code(excinfo) == "artifact_version_below_floor"


@pytest.mark.parametrize("kind", ["chosen_by", "routed_by", "Selected_by", "selected-by"])
def test_unknown_bare_kinds_still_refuse(tmp_path: Path, model: _TinyLM, kind: str) -> None:
    """ARMED NEGATIVE: one new kind admits exactly one spelling; neighbours refuse as before."""

    bundle = _routed(model)
    with pytest.raises(BundleRelationError) as excinfo:
        bundle.relate(kind, "writer", "router")
    assert _code(excinfo) == "bundle_relation_schema_invalid"
    payload = [{"kind": kind, "from": "writer", "to": "router", "params": {}}]
    for policy in ("opaque", "refuse"):
        with pytest.raises(ValueError, match="outside the closed vocabulary"):
            MemberRelationTable.from_payload(payload, unknown_kinds=policy)  # type: ignore[arg-type]
    path = tmp_path / f"{kind}.tlspec"
    bundle.relate("selected_by", "writer", "router").save(path)
    metadata_path = path / "relations.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["member_relations"][0]["kind"] = kind
    metadata_path.write_text(json.dumps(metadata))
    redigest(metadata_path.parent, "relations.json")
    with pytest.raises(BundleRelationError) as excinfo:
        tl.load(path)
    assert _code(excinfo) == "bundle_relation_schema_invalid"


def test_graph_edge_names_the_decision() -> None:
    from torchlens._visualization._bundle_graph import _relation_edge_label

    row = MemberRelationRow(
        kind="selected_by", from_member="writer", to_member="router", params={"chosen": 1}
    )
    label, checked = _relation_edge_label(row)
    assert label.splitlines() == ["selected_by", "chose 1"] and not checked
