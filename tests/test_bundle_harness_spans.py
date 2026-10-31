"""Harness bundles, phase 2: automatic span finding, token pairs, render, load.

Covers the integer token-id matcher behind ``token_spans="auto"`` (exact
runs, trimming, repeat and tie refusal, missing evidence), how a ``matched``
row is attested at every door, the strict ``Bundle.token_pairs`` read (only
re-verified one-to-one positions are paired; the two DISSENT names.3
counterexamples are reported as excluded), the members-level picture,
``tl.load(..., lazy=True)`` reaching members with their token ids intact,
and the v9 refusal of a ``fed_by`` row. Every protective claim tries the
forbidden thing.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import pytest
import torch

import torchlens as tl
from torchlens.bundle import Bundle
from torchlens.bundle._fed_by import TOKEN_BOUNDARY_FAMILY_ID, bind_member
from torchlens.bundle._token_spans import find_token_spans
from torchlens.errors.episode import BundleRelationError
from torchlens.io import attach_sidecar

_VOCAB = 32
_EOS = 31


class _TinyLM(torch.nn.Module):
    """Tokenizer-free causal-LM stand-in: ids in, logits out."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(_VOCAB, 8)
        self.block = torch.nn.Linear(8, 8)
        self.head = torch.nn.Linear(8, _VOCAB)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        return self.head(torch.relu(self.block(self.embed(ids))))


# draft = prompt [1, 2, 3] + completion [7, 8, 9, 10]; the critique prompt
# quotes the completion at positions 3..7 between template tokens.
_DRAFT_IDS = [1, 2, 3, 7, 8, 9, 10]
_CRITIQUE_IDS = [4, 5, 6, 7, 8, 9, 10, 11]


@pytest.fixture(scope="module")
def model() -> _TinyLM:
    torch.manual_seed(0)
    return _TinyLM()


def _code(excinfo: pytest.ExceptionInfo[BaseException]) -> str:
    return str(excinfo.value.fields["code"])  # type: ignore[attr-defined]


def _member(model: torch.nn.Module, ids: list[int], **binding: Any) -> tl.Trace:
    trace = tl.trace(model, torch.tensor([ids]))
    if binding:
        attach_sidecar(
            trace, TOKEN_BOUNDARY_FAMILY_ID, {"stream": "input", "batch_row": 0, **binding}
        )
    return trace


def _harness(
    model: torch.nn.Module,
    reader_ids: list[int] = _CRITIQUE_IDS,
    *,
    draft_ids: list[int] = _DRAFT_IDS,
    completion: tuple[int, int] = (3, 7),
    **draft_binding: Any,
) -> Bundle:
    draft = _member(model, draft_ids, completion=list(completion), **draft_binding)
    return Bundle({"draft": draft, "critique": _member(model, reader_ids)})


# ---------------------------------------------------------------------------
# The matcher (pure integer runs).
# ---------------------------------------------------------------------------


def test_matcher_finds_an_exact_run() -> None:
    assert find_token_spans(_DRAFT_IDS, [3, 7], _CRITIQUE_IDS) == [[3, 7, 3, 7]]


def test_matcher_trims_retokenized_boundary_tokens() -> None:
    # The quote's first token retokenized (7 -> 30) when embedded in the prompt.
    reader = [4, 5, 30, 8, 9, 10, 11]
    assert find_token_spans(_DRAFT_IDS, [3, 7], reader) == [[4, 7, 3, 6]]


def test_matcher_refuses_a_repeated_quote() -> None:
    reader = [7, 8, 9, 10, 0, 7, 8, 9, 10]
    with pytest.raises(BundleRelationError) as excinfo:
        find_token_spans(_DRAFT_IDS, [3, 7], reader)
    assert _code(excinfo) == "bundle_relation_auto_spans_ambiguous"


def test_matcher_refuses_a_tie_between_two_trims() -> None:
    # Neither [7, 8, 9, 10] nor a unique one-token trim: [8, 9, 10] and
    # [7, 8, 9] each occur once, at the same trim level.
    reader = [8, 9, 10, 0, 7, 8, 9, 0]
    with pytest.raises(BundleRelationError) as excinfo:
        find_token_spans(_DRAFT_IDS, [3, 7], reader)
    assert _code(excinfo) == "bundle_relation_auto_spans_ambiguous"


def test_matcher_refuses_when_nothing_matches() -> None:
    with pytest.raises(BundleRelationError) as excinfo:
        find_token_spans(_DRAFT_IDS, [3, 7], [20, 21, 22, 23])
    assert _code(excinfo) == "bundle_relation_auto_spans_unmatched"


def test_matcher_never_trims_a_run_below_two_tokens() -> None:
    # A two-token completion cannot trim: one shared token is not a match.
    with pytest.raises(BundleRelationError) as excinfo:
        find_token_spans([1, 7, 8], [1, 3], [4, 8, 5])
    assert _code(excinfo) == "bundle_relation_auto_spans_unmatched"


def test_matcher_splits_runs_at_excluded_ids_and_searches_only_the_prompt() -> None:
    source = [1, 2, 7, 8, _EOS, 9, 10]
    reader = [4, 7, 8, 5, 9, 10, 6, 7, 8]
    spans = find_token_spans(source, [2, 7], reader, excluded_ids=[_EOS], search_end=7)
    assert spans == [[2, 4, 1, 3], [5, 7, 4, 6]]
    with pytest.raises(BundleRelationError) as excinfo:  # the reader's own output repeats it
        find_token_spans(source, [2, 7], reader, excluded_ids=[_EOS])
    assert _code(excinfo) == "bundle_relation_auto_spans_ambiguous"


def test_matcher_leaves_a_single_token_run_unmatched() -> None:
    # Run [12] occurs once in the reader by chance (a template token): not a match.
    source = [1, 2, 3, 7, 8, 9, _EOS, 12]
    reader = [4, 7, 8, 9, 5, 12]
    assert find_token_spans(source, [3, 8], reader, excluded_ids=[_EOS]) == [[3, 6, 1, 4]]


def test_matcher_refuses_two_runs_on_the_same_reader_tokens() -> None:
    source = [1, 7, 8, _EOS, 7, 8]
    with pytest.raises(BundleRelationError) as excinfo:
        find_token_spans(source, [1, 6], [4, 7, 8, 5], excluded_ids=[_EOS])
    assert _code(excinfo) == "bundle_relation_auto_spans_ambiguous"


# ---------------------------------------------------------------------------
# token_spans="auto" through relate, and the matched attestation.
# ---------------------------------------------------------------------------


def test_auto_spans_write_a_matched_row(model: _TinyLM) -> None:
    bundle = _harness(model)
    bundle.relate("fed_by", "critique", "draft", token_spans="auto", label="critique")
    params = bundle.relations[0].params
    assert params["token_spans"] == [[3, 7, 3, 7]]
    assert params["basis"] == "matched"
    assert params["coverage"] == 1.0
    assert params["label"] == "critique"


def test_auto_spans_trimmed_coverage_is_partial(model: _TinyLM) -> None:
    bundle = _harness(model, [4, 5, 30, 8, 9, 10, 11])
    bundle.relate("fed_by", "critique", "draft", token_spans="auto")
    params = bundle.relations[0].params
    assert (params["basis"], params["coverage"]) == ("matched", 0.75)


def test_auto_spans_refuse_without_the_evidence(model: _TinyLM) -> None:
    no_completion = Bundle(
        {"draft": _member(model, _DRAFT_IDS), "critique": _member(model, _CRITIQUE_IDS)}
    )
    with pytest.raises(BundleRelationError) as excinfo:
        no_completion.relate("fed_by", "critique", "draft", token_spans="auto")
    assert _code(excinfo) == "bundle_relation_auto_spans_unavailable"
    assert no_completion.relations == ()


def test_auto_spans_refuse_when_ids_are_not_retained(model: _TinyLM) -> None:
    options = tl.options.CaptureOptions(layers_to_save=[])
    draft = tl.trace(model, torch.tensor([_DRAFT_IDS]), capture=options)
    attach_sidecar(
        draft, TOKEN_BOUNDARY_FAMILY_ID, {"stream": "input", "batch_row": 0, "completion": [3, 7]}
    )
    bundle = Bundle({"draft": draft, "critique": _member(model, _CRITIQUE_IDS)})
    with pytest.raises(BundleRelationError) as excinfo:
        bundle.relate("fed_by", "critique", "draft", token_spans="auto")
    assert _code(excinfo) == "bundle_relation_auto_spans_unavailable"


def test_auto_spans_work_with_a_selection_that_names_the_input(model: _TinyLM) -> None:
    options = tl.options.CaptureOptions(layers_to_save=["input_1", "block"])
    draft = tl.trace(model, torch.tensor([_DRAFT_IDS]), capture=options)
    attach_sidecar(
        draft, TOKEN_BOUNDARY_FAMILY_ID, {"stream": "input", "batch_row": 0, "completion": [3, 7]}
    )
    critique = tl.trace(model, torch.tensor([_CRITIQUE_IDS]), capture=options)
    bundle = Bundle({"draft": draft, "critique": critique})
    bundle.relate("fed_by", "critique", "draft", token_spans="auto")
    assert bundle.relations[0].params["basis"] == "matched"


@pytest.mark.parametrize(
    "args,params",
    [
        (("successor_of", "critique", "draft"), {}),
        (("fed_by", "critique", "draft"), {"basis": "matched"}),
        (("fed_by", "critique", "draft"), {"coverage": 1.0}),
    ],
)
def test_auto_spans_refuse_off_their_one_spelling(
    model: _TinyLM, args: tuple[str, ...], params: dict[str, Any]
) -> None:
    with pytest.raises(BundleRelationError) as excinfo:
        _harness(model).relate(*args, token_spans="auto", **params)
    assert _code(excinfo) == "bundle_relation_schema_invalid"


def test_auto_spans_refuse_a_non_member(model: _TinyLM) -> None:
    with pytest.raises(BundleRelationError) as excinfo:
        _harness(model).relate("fed_by", "critique", "ghost", token_spans="auto")
    assert _code(excinfo) == "bundle_relation_member_missing"


def test_auto_spans_refuse_an_ambiguous_prompt(model: _TinyLM) -> None:
    bundle = _harness(model, [7, 8, 9, 10, 0, 7, 8, 9, 10])
    with pytest.raises(BundleRelationError) as excinfo:
        bundle.relate("fed_by", "critique", "draft", token_spans="auto")
    assert _code(excinfo) == "bundle_relation_auto_spans_ambiguous"


def test_a_matched_claim_the_matcher_does_not_reproduce_settles_verified(
    model: _TinyLM,
) -> None:
    # [4, 6, 4, 6] re-checks (ids agree) but is not what the matcher finds.
    bundle = _harness(model)
    for name in ("draft", "critique"):
        bind_member(bundle[name])
    bundle.relate(
        {
            "kind": "fed_by",
            "from": "critique",
            "to": "draft",
            "params": {"token_spans": [[4, 6, 4, 6]], "basis": "matched"},
        }
    )
    assert bundle.relations[0].params["basis"] == "verified"


def _saved_matched(tmp_path: Path, model: _TinyLM) -> Path:
    bundle = _harness(model)
    bundle.relate("fed_by", "critique", "draft", token_spans="auto")
    path = tmp_path / "matched.tlspec"
    bundle.save(path)
    return path


def _rewrite_spans(source: Path, target: Path, spans: list[list[int]]) -> None:
    shutil.copytree(source, target)
    metadata_path = target / "bundle.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["member_relations"][0]["params"]["token_spans"] = spans
    metadata_path.write_text(json.dumps(metadata))


def test_forged_matched_spans_refuse_at_load(tmp_path: Path, model: _TinyLM) -> None:
    source = _saved_matched(tmp_path, model)
    _rewrite_spans(source, tmp_path / "forged.tlspec", [[3, 7, 0, 4]])
    with pytest.raises(BundleRelationError) as excinfo:
        tl.load(tmp_path / "forged.tlspec")
    assert _code(excinfo) == "bundle_relation_span_unverified"


def test_edited_matched_spans_that_still_check_lose_matched(tmp_path: Path, model: _TinyLM) -> None:
    source = _saved_matched(tmp_path, model)
    _rewrite_spans(source, tmp_path / "edited.tlspec", [[4, 6, 4, 6]])
    loaded = tl.load(tmp_path / "edited.tlspec")
    assert loaded.relations[0].params["basis"] == "verified"


@pytest.mark.parametrize("lazy", [False, True])
def test_retained_ids_survive_save_and_load(tmp_path: Path, model: _TinyLM, lazy: bool) -> None:
    """The ids-retention check: a matched row re-attests after a round trip."""

    loaded = tl.load(_saved_matched(tmp_path, model), lazy=lazy)
    params = loaded.relations[0].params
    assert (params["basis"], params["token_spans"]) == ("matched", [[3, 7, 3, 7]])
    pairs = loaded.token_pairs("critique", "draft")
    assert pairs.pairs == ((3, 3), (4, 4), (5, 5), (6, 6))


# ---------------------------------------------------------------------------
# Bundle.token_pairs: strict one-to-one positions.
# ---------------------------------------------------------------------------


def test_token_pairs_pairs_a_verified_row(model: _TinyLM) -> None:
    bundle = _harness(model)
    bundle.relate("fed_by", "critique", "draft", token_spans="auto")
    pairs = bundle.token_pairs("critique", "draft")
    assert (pairs.reader, pairs.source) == ("critique", "draft")
    assert pairs.pairs == ((3, 3), (4, 4), (5, 5), (6, 6))
    assert pairs.source_positions == [3, 4, 5, 6]
    assert pairs.reader_positions == [3, 4, 5, 6]
    assert pairs.coverage == 1.0
    assert pairs.excluded == ()


def test_token_pairs_reports_trimmed_tokens_as_uncovered(model: _TinyLM) -> None:
    bundle = _harness(model, [4, 5, 30, 8, 9, 10, 11])
    bundle.relate("fed_by", "critique", "draft", token_spans="auto")
    pairs = bundle.token_pairs("critique", "draft")
    assert pairs.pairs == ((4, 3), (5, 4), (6, 5))
    assert pairs.coverage == 0.75
    (uncovered,) = pairs.excluded
    assert (uncovered["side"], uncovered["start"], uncovered["end"]) == ("source", 3, 4)
    assert "no verified span" in uncovered["reason"]


def test_two_source_tokens_into_one_reader_token_are_not_paired(model: _TinyLM) -> None:
    """DISSENT names.3, case 1: a many-to-one retokenization stays a correspondence."""

    # Source tokens [7, 8] became one reader token (29) after a two-token prefix.
    bundle = _harness(model, [4, 5, 29, 9, 10, 11])
    bundle.relate("fed_by", "critique", "draft", token_spans=[[3, 5, 2, 3]], basis="declared")
    pairs = bundle.token_pairs("critique", "draft")
    assert pairs.pairs == ()
    reasons = [entry["reason"] for entry in pairs.excluded if entry["row_index"] == 0]
    assert reasons and all("2 source tokens against 1 reader" in reason for reason in reasons)
    assert pairs.coverage == 0.0


def test_a_reader_token_spanning_a_delimiter_and_copied_text_is_not_paired(
    model: _TinyLM,
) -> None:
    """DISSENT names.3, case 2: equal lengths with a mixed-origin token are not a match."""

    # Reader token 30 is "<delimiter>7" fused; the rest of the quote copies through.
    bundle = _harness(model, [4, 5, 30, 8, 9, 10, 11])
    bundle.relate("fed_by", "critique", "draft", token_spans=[[3, 7, 2, 6]])
    assert bundle.relations[0].params["basis"] == "declared"  # ids differ at 2
    pairs = bundle.token_pairs("critique", "draft")
    assert pairs.pairs == ()  # positions 3..5 agree, yet nothing inside the span pairs
    sides = {(entry["side"], entry["start"], entry["end"]) for entry in pairs.excluded}
    assert {("source", 3, 7), ("reader", 2, 6)} <= sides


def test_a_source_token_copied_twice_is_not_paired(model: _TinyLM) -> None:
    bundle = _harness(model, [7, 8, 9, 10, 0, 7, 8, 9, 10])
    bundle.relate("fed_by", "critique", "draft", token_spans=[[3, 7, 0, 4], [3, 7, 5, 9]])
    assert bundle.relations[0].params["basis"] == "verified"
    pairs = bundle.token_pairs("critique", "draft")
    assert pairs.pairs == ()
    assert all("several reader tokens" in entry["reason"] for entry in pairs.excluded)


def test_a_declared_span_vetoes_verified_pairs_on_the_tokens_it_touches(model: _TinyLM) -> None:
    bundle = _harness(model)
    bundle.relate("fed_by", "critique", "draft", token_spans=[[3, 7, 3, 7]])
    bundle.relate("fed_by", "critique", "draft", token_spans=[[3, 5, 3, 4]], basis="declared")
    pairs = bundle.token_pairs("critique", "draft")
    assert pairs.pairs == ((5, 5), (6, 6))
    vetoed = [e for e in pairs.excluded if e["reason"].startswith("also claimed")]
    assert {(e["side"], e["start"]) for e in vetoed} == {
        ("source", 3),
        ("source", 4),
        ("reader", 3),
        ("reader", 4),
    }


def test_a_span_into_the_readers_own_output_never_verifies(model: _TinyLM) -> None:
    reader_ids = [4, 5, 6, 7, 8, 9, 10, 11, 7, 8, 9, 10]  # the reader's output repeats it
    bundle = _harness(model, reader_ids)
    attach_sidecar(
        bundle["critique"],
        TOKEN_BOUNDARY_FAMILY_ID,
        {"stream": "input", "batch_row": 0, "completion": [8, 12]},
    )
    bundle.relate("fed_by", "critique", "draft", token_spans=[[3, 7, 8, 12]])
    assert bundle.relations[0].params["basis"] == "declared"
    with pytest.raises(BundleRelationError) as excinfo:
        Bundle(
            dict(bundle.members),
            relations=[
                {
                    "kind": "fed_by",
                    "from": "critique",
                    "to": "draft",
                    "params": {"token_spans": [[3, 7, 8, 12]], "basis": "verified"},
                }
            ],
        )
    assert _code(excinfo) == "bundle_relation_span_unverified"


def test_token_pairs_with_an_out_of_range_completion_reports_no_coverage(model: _TinyLM) -> None:
    bundle = _harness(model, completion=(0, 100))  # unmeasured descriptor, never bound
    bundle.relate("fed_by", "critique", "draft", token_spans=[[3, 7, 3, 7]], basis="declared")
    pairs = bundle.token_pairs("critique", "draft")
    assert (pairs.pairs, pairs.coverage) == ((), None)


def test_token_pairs_skip_excluded_ids(model: _TinyLM) -> None:
    draft_ids = [1, 2, 3, 7, 8, 9, _EOS]
    bundle = _harness(model, [4, 7, 8, 9, _EOS, 5], draft_ids=draft_ids, excluded_ids=[_EOS])
    bundle.relate("fed_by", "critique", "draft", token_spans=[[3, 7, 1, 5]])
    pairs = bundle.token_pairs("critique", "draft")
    assert pairs.pairs == ((3, 1), (4, 2), (5, 3))
    assert pairs.coverage == 1.0
    assert [(e["side"], e["start"]) for e in pairs.excluded] == [("source", 6), ("reader", 4)]


def test_token_pairs_refuse_streams_from_different_tokenizers(model: _TinyLM) -> None:
    bundle = _harness(model, tokenizer="tok-a")
    attach_sidecar(
        bundle["critique"],
        TOKEN_BOUNDARY_FAMILY_ID,
        {"stream": "input", "batch_row": 0, "tokenizer": "tok-b"},
    )
    bundle.relate("fed_by", "critique", "draft", token_spans=[[3, 7, 3, 7]])
    pairs = bundle.token_pairs("critique", "draft")
    assert pairs.pairs == ()
    assert any("different tokenizers" in entry["reason"] for entry in pairs.excluded)


def test_token_pairs_without_a_row_pair_nothing(model: _TinyLM) -> None:
    pairs = _harness(model).token_pairs("critique", "draft")
    assert (pairs.pairs, pairs.coverage) == ((), 0.0)
    with pytest.raises(KeyError):
        _harness(model).token_pairs("critique", "ghost")


def test_token_pairs_resolves_on_an_instance(model: _TinyLM) -> None:
    assert "token_pairs" in dir(_harness(model))


# ---------------------------------------------------------------------------
# The members-level picture.
# ---------------------------------------------------------------------------


def test_members_render_draws_rows_as_labelled_edges(tmp_path: Path, model: _TinyLM) -> None:
    pytest.importorskip("graphviz")
    bundle = _harness(model)
    bundle.add(_member(model, [4, 5]), names="revise")
    bundle.relate("fed_by", "critique", "draft", token_spans="auto", label="critique")
    bundle.relate("successor_of", "revise", "critique")
    source = tl.visualization.show_bundle_graph(
        bundle,
        nodes="members",
        path=str(tmp_path / "members"),
        vis_save_only=True,
        file_format="svg",
    )
    assert source is not None
    for index, name in enumerate(("draft", "critique", "revise")):
        assert name in _statement(source, f"member_{index} [")
    fed_by_edge = _statement(source, "member_0 -> member_1")  # draft -> critique
    assert "matched, coverage 1.00" in fed_by_edge and "style=solid" in fed_by_edge
    successor = _statement(source, "member_1 -> member_2")  # critique -> revise
    assert "successor_of" in successor and "style=dashed" in successor
    assert "fwd_" not in source  # no op nodes in the members picture


def _statement(source: str, head: str) -> str:
    """Return one DOT statement from ``head`` to its closing bracket."""

    start = source.index(head)
    return source[start : source.index("]", start) + 1]


def test_members_render_refuses_a_backward_direction(model: _TinyLM) -> None:
    with pytest.raises(ValueError, match="direction"):
        tl.visualization.show_bundle_graph(_harness(model), nodes="members", direction="backward")
    with pytest.raises(ValueError, match="nodes"):
        tl.visualization.show_bundle_graph(_harness(model), nodes="layers")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Loading: lazy members and the version gate.
# ---------------------------------------------------------------------------


def test_lazy_load_reaches_bundle_members(tmp_path: Path, model: _TinyLM) -> None:
    path = _saved_matched(tmp_path, model)
    eager, lazy = tl.load(path), tl.load(path, lazy=True)
    for name in ("draft", "critique"):
        assert lazy[name].payload_load_status == "loaded_lazy"
        assert eager[name].payload_load_status != "loaded_lazy"
        pairs = zip(lazy[name]._layer_list, eager[name]._layer_list, strict=True)
        lazy_op, eager_op = next((a, b) for a, b in pairs if a.has_saved_activation)
        assert lazy_op.out_ref is not None  # the payload stayed on disk until read
        assert torch.equal(lazy_op.out, eager_op.out)


def test_v9_stamped_artifact_with_a_fed_by_row_refuses(tmp_path: Path, model: _TinyLM) -> None:
    path = _saved_matched(tmp_path, model)
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["tlspec_version"] = 9
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(BundleRelationError) as excinfo:
        tl.load(path)
    assert _code(excinfo) == "bundle_relation_schema_invalid"
    assert excinfo.value.fields["row_kind"] == "fed_by"  # type: ignore[attr-defined]
