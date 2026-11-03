"""Harness bundles, phase 1: the ``fed_by`` relation kind and its reads.

Covers the grammar and direction of ``fed_by`` (the later member's prompt
contains the earlier member's output), the positional ``relate`` spelling,
the token-stream binding that decides whether a row is ``verified`` or only
``declared``, the upstream/downstream walks, the member-scoped ``node``, the
comparison gate on the delta family, empty bundles, and the current tlspec
round trip with spans. Every protective claim tries the forbidden thing.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import torchlens as tl
from torchlens._errors import InvalidArgumentError
from torchlens._io.sidecar import SidecarError
from torchlens.bundle import Bundle
from torchlens.bundle._fed_by import (
    TOKEN_BOUNDARY_FAMILY_ID,
    ids_digest,
)
from torchlens.bundle._relations import MemberRelationRow
from torchlens.errors.episode import BundleRelationError
from torchlens.intervention.errors import BundleComparisonError
from torchlens.io import attach_sidecar

_VOCAB = 32


class _TinyLM(torch.nn.Module):
    """Tokenizer-free causal-LM stand-in: ids in, logits out."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(_VOCAB, 8)
        self.block = torch.nn.Linear(8, 8)
        self.head = torch.nn.Linear(8, _VOCAB)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.head(torch.relu(self.block(self.embed(ids))))


class _Other(torch.nn.Module):
    """A foreign architecture (the small router model of a harness)."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(_VOCAB, 4)
        self.proj = torch.nn.Linear(4, 2)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.proj(self.embed(ids))


# draft = prompt [1, 2, 3] + completion [7, 8, 9, 10]; the critique prompt
# quotes the completion at positions 3..7 between template tokens.
_DRAFT_IDS = [1, 2, 3, 7, 8, 9, 10]
_CRITIQUE_IDS = [4, 5, 6, 7, 8, 9, 10, 11]
_SPAN = [3, 7, 3, 7]


@pytest.fixture(scope="module")
def model() -> _TinyLM:
    torch.manual_seed(0)
    return _TinyLM()


def _trace(model: torch.nn.Module, ids: list[int]) -> tl.Trace:
    return tl.trace(model, torch.tensor([ids]))


def _harness(model: _TinyLM) -> Bundle:
    return Bundle({"draft": _trace(model, _DRAFT_IDS), "critique": _trace(model, _CRITIQUE_IDS)})


def _bound_harness(model: _TinyLM) -> Bundle:
    """The harness with the draft's completion range bound (its output is [3, 7))."""

    bundle = _harness(model)
    completion = {"stream": "input", "batch_row": 0, "completion": [3, 7]}
    attach_sidecar(bundle["draft"], TOKEN_BOUNDARY_FAMILY_ID, completion)
    return bundle


def _code(excinfo: pytest.ExceptionInfo[BaseException]) -> str:
    return str(excinfo.value.fields["code"])  # type: ignore[attr-defined]


def _fed_by(**params: object) -> MemberRelationRow:
    return MemberRelationRow(
        kind="fed_by", from_member="critique", to_member="draft", params=params
    )


# ---------------------------------------------------------------------------
# Grammar and direction.
# ---------------------------------------------------------------------------


def test_fed_by_row_admits_exactly_its_four_params() -> None:
    row = _fed_by(token_spans=[(0, 2, 5, 7)], basis="declared", coverage=0.5, label="critique")
    assert row.params["token_spans"] == [[0, 2, 5, 7]], "rows normalize to lists of four ints"
    assert row.to_payload()["from"] == "critique" and row.to_payload()["to"] == "draft"
    with pytest.raises(ValueError, match="undeclared param"):
        _fed_by(role="judge")


# shrink-r4: rows kept carry a sole library arc or a campaign kill; the others were proven redundant (slice 1 evidence)
@pytest.mark.parametrize(
    "spans",
    [
        [[0, 2, 5, -1]],
        [[0, 2, 5, 5]],
        "auto",
    ],
)
def test_malformed_token_spans_refuse(spans: object) -> None:
    with pytest.raises(ValueError):
        _fed_by(token_spans=spans)


# shrink-r4: rows kept carry a sole library arc or a campaign kill; the others were proven redundant (slice 1 evidence)
@pytest.mark.parametrize(
    "params",
    [
        {"basis": "measured"},
        {"coverage": 1.5},
        {"coverage": float("nan")},
        {"label": ""},
        {"label": "x" * 257},
        {"basis": "matched", "token_spans": []},
    ],
)
def test_fed_by_param_values_are_closed(params: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        _fed_by(**params)


def test_positional_relate_refusals(model: _TinyLM) -> None:
    bundle = _harness(model)
    with pytest.raises(BundleRelationError) as excinfo:
        bundle.relate("fed_by", "critique", "draft", token_spans="auto")
    assert _code(excinfo) == "bundle_relation_auto_spans_unavailable"
    assert "no completion range" in str(excinfo.value)  # the draft is not bound here
    with pytest.raises(BundleRelationError) as excinfo:
        bundle.relate("fed_by", "critique")
    assert _code(excinfo) == "bundle_relation_schema_invalid"
    with pytest.raises(BundleRelationError) as excinfo:
        bundle.relate(_fed_by(), label="x")
    assert _code(excinfo) == "bundle_relation_schema_invalid"
    with pytest.raises(BundleRelationError) as excinfo:
        bundle.relate("fed_by", "critique", "nobody")
    assert _code(excinfo) == "bundle_relation_member_missing"
    assert bundle.relations == (), "every refusal left the table unchanged"


def test_positional_relate_names_one_member_for_member_kinds(model: _TinyLM) -> None:
    bundle = _harness(model)
    bundle.relate("episode_member", "draft", episode_id="ep", at_step=0, role="prefill")
    (row,) = bundle.relations
    assert row.member == "draft"


# ---------------------------------------------------------------------------
# Binding and basis.
# ---------------------------------------------------------------------------


def test_mismatched_spans_stay_declared(model: _TinyLM) -> None:
    bundle = _harness(model)
    bundle.relate("fed_by", "critique", "draft", token_spans=[[0, 3, 0, 3]], coverage=0.9)
    (row,) = bundle.relations
    assert row.params["basis"] == "declared", "retokenized or paraphrased text is a declaration"
    assert row.params["coverage"] == 0.9


def test_claimed_verified_row_rechecks_at_every_door(model: _TinyLM) -> None:
    bundle = _bound_harness(model)
    bundle.relate("fed_by", "critique", "draft", token_spans=[_SPAN])  # binds both
    fresh = Bundle(dict(bundle.members), relations=[_fed_by(token_spans=[_SPAN], basis="verified")])
    assert fresh.relations[0].params["basis"] == "verified"
    matched = Bundle(
        dict(bundle.members), relations=[_fed_by(token_spans=[_SPAN], basis="matched")]
    )
    # The matcher, re-run on the members, re-derives exactly these spans.
    assert matched.relations[0].params["basis"] == "matched"
    with pytest.raises(BundleRelationError) as excinfo:
        Bundle(
            dict(bundle.members),
            relations=[_fed_by(token_spans=[[0, 3, 0, 3]], basis="verified")],
        )
    assert _code(excinfo) == "bundle_relation_span_unverified"


def test_unverifiable_claim_clamps_to_declared(model: _TinyLM) -> None:
    bundle = _harness(model)  # no bindings attached
    bundle.relate(_fed_by(token_spans=[_SPAN], basis="verified"))
    assert bundle.relations[0].params["basis"] == "declared"


def test_forged_binding_digest_contradicts(model: _TinyLM) -> None:
    bundle = _harness(model)
    forged = {
        "stream": "input",
        "batch_row": 0,
        "length": 7,
        "ids_digest": ids_digest([0] * 7),
        "completion": [3, 7],
    }
    attach_sidecar(bundle["draft"], TOKEN_BOUNDARY_FAMILY_ID, forged)
    bundle.relate("fed_by", "critique", "draft", token_spans=[_SPAN])
    assert bundle.relations[0].params["basis"] == "declared"
    with pytest.raises(BundleRelationError) as excinfo:
        bundle.relate(_fed_by(token_spans=[_SPAN], basis="verified"))
    assert _code(excinfo) == "bundle_relation_span_unverified"


def test_spans_past_the_stream_refuse(model: _TinyLM) -> None:
    bundle = _harness(model)
    with pytest.raises(BundleRelationError) as excinfo:
        bundle.relate("fed_by", "critique", "draft", token_spans=[[3, 9, 3, 9]])
    assert _code(excinfo) == "bundle_relation_span_out_of_bounds"
    assert excinfo.value.fields["side"] == "source"


def test_malformed_attached_binding_refuses_typed(model: _TinyLM) -> None:
    bundle = _harness(model)
    with pytest.raises(SidecarError) as excinfo:
        attach_sidecar(
            bundle["draft"], TOKEN_BOUNDARY_FAMILY_ID, {"stream": "logits", "batch_row": 0}
        )
    assert _code(excinfo) == "sidecar_payload_invalid"


def test_empty_bundle_reads_refuse_typed_or_return_empty() -> None:
    from torchlens.intervention.errors import BaselineUndeterminedError

    empty = tl.Bundle()
    assert empty.ops.shared == [] and empty.ops.divergent == []
    with pytest.raises(BaselineUndeterminedError):
        empty._delta_map("relative_l2")


def test_draw_diff_gates_only_the_rendered_pair(tmp_path: Path, model: _TinyLM) -> None:
    clean = _trace(model, [1, 2, 3])
    bundle = Bundle(
        {"clean": clean, "patched": clean.fork("patched"), "router": _trace(_Other(), [4, 5])},
        baseline="clean",
    )
    dot = bundle.draw(graph="diff", members=("clean", "patched"), path=str(tmp_path / "d")).dot
    assert "clean vs patched" in dot
    with pytest.raises(BundleComparisonError):
        bundle._delta_map("relative_l2")


def test_binding_descriptor_is_closed() -> None:
    from torchlens.bundle._fed_by import validate_descriptor

    good = {"stream": "input", "batch_row": 0, "length": 3, "ids_digest": ids_digest([1, 2, 3])}
    validate_descriptor(good)
    for bad in (
        {**good, "stream": "logits"},
        {**good, "extra": 1},
        {**good, "ids_digest": "md5:00"},
        {**good, "completion": [2, 9]},
        {**good, "excluded_ids": ["pad"]},
        {key: value for key, value in good.items() if key != "stream"},
    ):
        with pytest.raises(ValueError):
            validate_descriptor(bad)


# ---------------------------------------------------------------------------
# Walks and scoped node.
# ---------------------------------------------------------------------------


def test_walks_are_transitive_and_follow_ordering_kinds(model: _TinyLM) -> None:
    bundle = Bundle({name: _trace(model, [i + 1, i + 2]) for i, name in enumerate("abcd")})
    bundle.relate("fed_by", "b", "a").relate("successor_of", "c", "b").relate("fed_by", "d", "b")
    bundle.relate("alternative_of", "a", "d")  # peers: not an ordering kind
    assert bundle.upstream("c") == ("b", "a")
    assert bundle.downstream("a") == ("b", "c", "d")
    assert bundle.upstream("a") == ()
    assert bundle.upstream(bundle["d"]) == ("b", "a"), "a member Trace resolves to its name"
    with pytest.raises(KeyError):
        bundle.upstream("zzz")


def test_at_on_a_member_subset_skips_a_foreign_member(model: _TinyLM) -> None:
    torch.manual_seed(1)
    ids = [1, 2, 3]
    bundle = Bundle(
        {"a": _trace(model, ids), "b": _trace(model, ids), "router": _trace(_Other(), ids)}
    )
    site = tl.func("relu")
    with pytest.raises(BundleComparisonError) as excinfo:
        bundle.at(site)
    assert _code(excinfo) == "bundle_gate_model_axis_unmet"
    view = Bundle({name: bundle[name] for name in ("a", "b")}).at(site)
    assert set(view.members) == {"a", "b"}


# ---------------------------------------------------------------------------
# The delta family goes through the comparison gate.
# ---------------------------------------------------------------------------


_DELTA_READS = [
    lambda b: b._delta_map("relative_l2"),
    lambda b: b._norm_delta(),
    lambda b: b._output_delta("draft"),
    lambda b: b.compare("draft", "critique", metric="relative_l2"),
    lambda b: b.store_comparison("relative_l2"),
]


@pytest.mark.parametrize("read", _DELTA_READS)
def test_delta_family_compares_prompts_whose_graphs_line_up(model: _TinyLM, read) -> None:
    """Two calls on different prompts of one length line up and compare."""

    bundle = Bundle(
        {"draft": _trace(model, [1, 2, 3, 4]), "critique": _trace(model, [5, 6, 7, 8])},
        baseline="draft",
    )
    assert read(bundle) is not None


@pytest.mark.parametrize("read", _DELTA_READS)
def test_delta_family_refuses_prompts_of_different_lengths(model: _TinyLM, read) -> None:
    """Prompts of different token counts refuse typed at the first mismatched site."""

    bundle = Bundle(
        {"draft": _trace(model, [1, 2, 3, 4]), "critique": _trace(model, [5, 6, 7])},
        baseline="draft",
    )
    with pytest.raises(InvalidArgumentError) as excinfo:
        read(bundle)
    assert excinfo.value.fields["code"] == "metric_shape_mismatch"
    assert excinfo.value.fields["site"]


def test_chrome_trace_diff_keeps_the_timeline_and_refuses_only_the_deltas(
    tmp_path: Path, model: _TinyLM
) -> None:
    unrelated = Bundle({"draft": _trace(model, [1, 2, 3]), "critique": _trace(model, [4, 5])})
    payload = json.loads(tl.export.chrome_trace_diff(unrelated, tmp_path / "a.json").read_text())
    status = payload["metadata"]["deltas"]
    assert status["status"] == "refused"
    assert status["code"] == "metric_shape_mismatch"
    spans = [event for event in payload["traceEvents"] if event["ph"] == "X"]
    assert spans and all(event["args"]["delta"] is None for event in spans)
    same = Bundle({"a": _trace(model, [1, 2, 3]), "b": _trace(model, [1, 2, 3])}, baseline="a")
    payload = json.loads(tl.export.chrome_trace_diff(same, tmp_path / "b.json").read_text())
    assert payload["metadata"]["deltas"]["status"] == "computed"
    assert any(event["args"].get("delta") is not None for event in payload["traceEvents"])


# ---------------------------------------------------------------------------
# Empty bundles and tlspec v10.
# ---------------------------------------------------------------------------


def test_empty_bundle_constructs_and_round_trips(tmp_path: Path, model: _TinyLM) -> None:
    empty = tl.Bundle()
    assert len(empty) == 0 and empty.names == []
    empty.save(tmp_path / "empty.tlspec")
    loaded = tl.load(tmp_path / "empty.tlspec")
    assert isinstance(loaded, Bundle) and len(loaded) == 0
    filled = _harness(model)
    filled.clear()
    filled.save(tmp_path / "cleared.tlspec")
    assert len(tl.load(tmp_path / "cleared.tlspec")) == 0
    empty.add(_trace(model, [1, 2]), names="first")
    assert empty.names == ["first"]
