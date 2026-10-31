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

import torchlens as tl
from torchlens.bundle import Bundle
from torchlens.bundle._relations import MemberRelationRow, MemberRelationTable
from torchlens.errors.episode import BundleRelationError
from torchlens.intervention.errors import BundleComparisonError


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


def test_direction_pin_and_params(model: _TinyLM) -> None:
    bundle = _routed(model).relate(
        "selected_by", "writer", "router", chosen=1, candidates=[0, 1], evidence=_evidence()
    )
    row = bundle.member_relations[0]
    assert (row.kind, row.from_member, row.to_member) == ("selected_by", "writer", "router")
    payload = row.to_payload()
    assert (payload["from"], payload["to"]) == ("writer", "router")
    assert payload["params"]["chosen"] == 1 and payload["params"]["candidates"] == [0, 1]
    item = payload["params"]["evidence"]["items"][0]
    assert item["grade"] == "disclosed" and item["declared_grade"] == "verified"
    assert bundle.upstream("writer") == ("router",)
    assert bundle.downstream("router") == ("writer",)


def test_selected_and_decider_are_ordered_for_the_comparison_gate(model: _TinyLM) -> None:
    bundle = _routed(model)
    assert bundle.delta_map("relative_l2"), "same model, same input: comparable"
    bundle.relate("selected_by", "writer", "router")
    with pytest.raises(BundleComparisonError) as excinfo:
        bundle.delta_map("relative_l2")
    assert _code(excinfo) == "bundle_gate_ordering_topology"


def test_round_trip_keeps_the_row(tmp_path: Path, model: _TinyLM) -> None:
    bundle = _routed(model).relate(
        "selected_by", "writer", "router", chosen="lookup", candidates=["lookup", "none"]
    )
    bundle.save(tmp_path / "routed.tlspec")
    manifest = json.loads((tmp_path / "routed.tlspec" / "manifest.json").read_text())
    assert manifest["tlspec_version"] == 10
    loaded = tl.load(tmp_path / "routed.tlspec")
    assert loaded.member_relations == bundle.member_relations
    assert loaded.upstream("writer") == ("router",)


@pytest.mark.parametrize(
    "params",
    [
        {"chosen": [1]},
        {"chosen": float("nan")},
        {"chosen": ""},
        {"candidates": []},
        {"candidates": "ab"},
        {"candidates": [1, 1]},
        {"candidates": [{"route": 1}]},
        {"chosen": 2, "candidates": [0, 1]},
        {"chosen": True, "candidates": [0, 1]},
        {"evidence": {"schema": "s", "items": []}},
        {"label": "router"},
        {"token_spans": [[0, 1, 0, 1]]},
    ],
)
def test_malformed_params_refuse_typed(model: _TinyLM, params: dict[str, object]) -> None:
    bundle = _routed(model)
    with pytest.raises(BundleRelationError) as excinfo:
        bundle.relate("selected_by", "writer", "router", **params)
    assert _code(excinfo) == "bundle_relation_schema_invalid"
    assert bundle.member_relations == (), "a refused relate leaves the table unchanged"


def test_tampered_artifact_param_refuses_at_load(tmp_path: Path, model: _TinyLM) -> None:
    bundle = _routed(model).relate("selected_by", "writer", "router", chosen=1, candidates=[0, 1])
    path = tmp_path / "routed.tlspec"
    bundle.save(path)
    metadata_path = path / "bundle.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["member_relations"][0]["params"]["chosen"] = 7
    metadata_path.write_text(json.dumps(metadata))
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
    with pytest.raises(BundleRelationError) as excinfo:
        tl.load(path)
    assert _code(excinfo) == "bundle_relation_schema_invalid"
    assert excinfo.value.fields["row_kind"] == "selected_by"  # type: ignore[attr-defined]


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
    metadata_path = path / "bundle.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["member_relations"][0]["kind"] = kind
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(BundleRelationError) as excinfo:
        tl.load(path)
    assert _code(excinfo) == "bundle_relation_schema_invalid"


def test_graph_edge_names_the_decision() -> None:
    from torchlens.visualization._bundle_graph import _relation_edge_label

    row = MemberRelationRow(
        kind="selected_by", from_member="writer", to_member="router", params={"chosen": 1}
    )
    label, checked = _relation_edge_label(row)
    assert label.splitlines() == ["selected_by", "chose 1"] and not checked


def test_writer_contract_records_the_kind() -> None:
    from torchlens._io.writer_contract import writer_contract

    kinds = writer_contract()["component_schema_versions"]["bundle_relation_kinds"]
    assert "selected_by" in kinds
