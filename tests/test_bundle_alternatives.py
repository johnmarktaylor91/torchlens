"""Steer one call and re-run the harness: ``token_divergence`` and ``add_alternative``.

The user's harness (draft, critique, revise on a tokenizer-free stand-in) runs
twice, as is and with one call steered; ``add_alternative`` pairs the calls by
name, checks the pairing, merges the second run as ``cf_<name>`` and reports
per call whether it was steered and whether its prompt and completion
changed. Every refusal is fired, and the tripwire (an unsteered call with the
same prompt that wrote different text) is pinned by a test that fails without
it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import torch

import torchlens as tl
from torchlens.bundle import AlternativeReport, Bundle, TokenDivergence
from torchlens.errors.episode import BundleRelationError
from torchlens.intervention import at_positions


class _TinyCausalLM(torch.nn.Module):
    """Each position mixes in a running mean of the earlier ones (causal)."""

    def __init__(self) -> None:
        super().__init__()
        self.embed = torch.nn.Embedding(64, 16)
        self.block = torch.nn.Linear(16, 16)
        self.head = torch.nn.Linear(16, 64)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        x = self.embed(ids)
        steps = torch.arange(1, ids.shape[1] + 1, dtype=x.dtype).view(1, -1, 1)
        x = x + x.cumsum(dim=1) / steps
        return self.head(torch.relu(self.block(x)))


torch.manual_seed(0)
_LM = _TinyCausalLM().eval()
_Q = torch.tensor([[20, 21, 22]])
_STEER = tl.when(tl.module("block"), tl.steer(torch.randn(16), magnitude=6.0, feature_axis=-1))


def _generate(model: Any, ids: torch.Tensor, seed: int) -> torch.Tensor:
    rng = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for _ in range(6):
            probs = torch.softmax(model(ids)[:, -1], dim=-1)
            ids = torch.cat([ids, torch.multinomial(probs, 1, generator=rng)], dim=1)
    return ids


def _chat(*parts: torch.Tensor) -> torch.Tensor:
    return torch.cat([torch.tensor([[1, 2]]), *parts, torch.tensor([[3]])], dim=1)


def _critique_loop(intervene: dict[str, Any], seed: int = 0) -> Bundle:
    """The user's harness: each call generates natively, then one capture under the same spec."""

    bundle = Bundle()

    def call(name: str, prompt: torch.Tensor, offset: int) -> torch.Tensor:
        spec = intervene.get(name)
        out = _generate(spec.bind(_LM) if spec is not None else _LM, prompt, seed + offset)
        trace = tl.trace(_LM, out, intervene=spec) if spec is not None else tl.trace(_LM, out)
        binding = {"stream": "input", "batch_row": 0, "completion": [prompt.shape[1], out.shape[1]]}
        tl.io.attach_sidecar(trace, "torchlens.boundary_facts", binding)
        bundle.add(trace, names=name)
        return out[:, prompt.shape[1] :]

    draft = call("draft", _chat(_Q), 1)
    critique = call("critique", _chat(_Q, torch.tensor([[4]]), draft), 2)
    call("revise", _chat(_Q, torch.tensor([[4]]), draft, torch.tensor([[5]]), critique), 3)
    bundle.relate("fed_by", "critique", "draft", token_spans="auto")
    bundle.relate("fed_by", "revise", "draft", token_spans="auto")
    bundle.relate("fed_by", "revise", "critique", token_spans="auto")
    return bundle


def _code(excinfo: pytest.ExceptionInfo[BaseException]) -> str:
    return str(excinfo.value.fields["code"])  # type: ignore[attr-defined]


def _state(bundle: Bundle) -> tuple[Any, ...]:
    return (tuple(bundle.names), bundle.member_relations, bundle.operations)


def test_steering_the_draft_reaches_critique_and_revise() -> None:
    b = _critique_loop({})
    report = b.add_alternative(_critique_loop({"draft": _STEER}), steered=("draft",))
    assert isinstance(report, AlternativeReport) and report.steered == ("draft",)
    by_call = {row.call: row for row in report.calls}
    draft = by_call["draft"]
    assert (draft.status, draft.member, draft.prompt_divergence) == ("steered", "cf_draft", None)
    assert draft.completion_divergence is not None, "the steer changed the draft's text"
    quoted_from = 6  # [1, 2] + q + [4], then the draft
    for name in ("critique", "revise"):
        row = by_call[name]
        assert row.status in ("absorbed", "changed")
        assert row.prompt_divergence == quoted_from + draft.completion_divergence
    assert b.downstream("draft") == ("critique", "revise"), "alternative_of copies stay out"
    assert b.downstream("cf_draft") == ("cf_critique", "cf_revise"), "the copy's own flow"
    pairs = [(r.kind, r.from_member, r.to_member) for r in b.member_relations]
    assert ("alternative_of", "cf_draft", "draft") in pairs
    assert ("fed_by", "cf_critique", "cf_draft") in pairs
    assert b.operations[-1].kind == "add"
    statuses = {c["call"]: c["status"] for c in b.operations[-1].params["alternative"]["calls"]}
    assert statuses == {row.call: row.status for row in report.calls}
    assert "draft" in report.table() and "steered" in report.table()


def test_token_divergence_gives_first_differing_positions() -> None:
    b = _critique_loop({})
    report = b.add_alternative(_critique_loop({"draft": _STEER}), steered="draft")
    split = {row.call: row for row in report.calls}["draft"].completion_divergence
    d = b.token_divergence("critique", "cf_critique")
    assert isinstance(d, TokenDivergence)
    assert d.prompt_divergence == 6 + split and not d.prompt_same
    assert d.intervened == (False, False)
    (prompt_len, _), _ = d.lengths
    assert d.shared[0] == (0, d.prompt_divergence, 0, d.prompt_divergence)
    tail = d.shared[1]
    assert tail[1] == prompt_len, "the template token after the quote is shared"
    steered = b.token_divergence("draft", "cf_draft")
    assert steered.prompt_same and steered.completion_divergence == split
    assert steered.intervened == (False, True)


def test_steering_the_last_call_leaves_earlier_calls_unchanged() -> None:
    b = _critique_loop({})
    report = b.add_alternative(_critique_loop({"revise": _STEER}), steered=("revise",))
    assert [row.status for row in report.calls] == ["unchanged", "unchanged", "steered"]


def test_unpaired_calls_are_added_or_dropped() -> None:
    b = _critique_loop({})
    second = _critique_loop({"draft": _STEER})
    second.remove("revise", cascade_relations=True)
    second.add(tl.trace(_LM, torch.tensor([[1, 2, 3]])), names="lookup")
    report = b.add_alternative(second, steered=("draft",))
    by_call = {row.call: row for row in report.calls}
    assert (by_call["revise"].status, by_call["revise"].member) == ("dropped", None)
    assert (by_call["lookup"].status, by_call["lookup"].member) == ("added", "cf_lookup")
    assert "cf_lookup" in b.names
    peers = {(r.from_member, r.to_member) for r in b.member_relations if r.kind == "alternative_of"}
    assert peers == {("cf_draft", "draft"), ("cf_critique", "critique")}


def test_unchanged_call_that_diverged_refuses_and_adds_nothing() -> None:
    """ARMED NEGATIVE: an unsteered call with the same prompt and different text refuses."""

    b = _critique_loop({})
    before = _state(b)
    reseeded = _critique_loop({"revise": _STEER}, seed=100)
    with pytest.raises(BundleRelationError) as excinfo:
        b.add_alternative(reseeded, steered=("revise",))
    assert _code(excinfo) == "alternative_unchanged_call_diverged"
    assert excinfo.value.fields["call"] == "draft"  # type: ignore[attr-defined]
    assert _state(b) == before


def _position_steer() -> Any:
    edit = at_positions(
        tl.steer(torch.randn(16), magnitude=6.0, feature_axis=-1), "last", assume_no_padding=True
    )
    return tl.when(tl.module("block"), edit)


@pytest.mark.parametrize(
    ("second", "steered", "code"),
    [
        (lambda: _critique_loop({"draft": _STEER}), ("judge",), "alternative_target_missing"),
        (lambda: _critique_loop({}), ("draft",), "alternative_target_unsteered"),
        (
            lambda: _critique_loop({"draft": _position_steer()}),
            ("draft",),
            "alternative_capture_position_dependent",
        ),
        (
            lambda: _critique_loop({"draft": _STEER, "revise": _STEER}),
            ("draft",),
            "alternative_steer_undeclared",
        ),
    ],
)
def test_pairing_refusals_leave_the_bundle_unchanged(second: Any, steered: Any, code: str) -> None:
    b = _critique_loop({})
    before = _state(b)
    with pytest.raises(BundleRelationError) as excinfo:
        b.add_alternative(second(), steered=steered)
    assert _code(excinfo) == code
    assert _state(b) == before


def test_reused_prefix_refuses() -> None:
    b = _critique_loop({})
    second = _critique_loop({"draft": _STEER})
    b.add_alternative(second, steered=("draft",))
    before = _state(b)
    with pytest.raises(BundleRelationError) as excinfo:
        b.add_alternative(second, steered=("draft",))
    assert _code(excinfo) == "alternative_member_name_taken"
    assert _state(b) == before
    b.add_alternative(second, steered=("draft",), prefix="cf2_")
    assert "cf2_draft" in b.names


def test_divergence_without_a_completion_range_refuses() -> None:
    b = Bundle({"a": tl.trace(_LM, torch.tensor([[1, 2, 3]])), "b": tl.trace(_LM, _Q)})
    with pytest.raises(BundleRelationError) as excinfo:
        b.token_divergence("a", "b")
    assert _code(excinfo) == "bundle_token_divergence_unavailable"
    tl.io.attach_sidecar(b["a"], "torchlens.boundary_facts", {"stream": "input", "batch_row": 0})
    with pytest.raises(BundleRelationError) as excinfo:
        b.token_divergence("a", "b")
    assert _code(excinfo) == "bundle_token_divergence_unavailable"
    assert excinfo.value.fields["member"] == "a"  # type: ignore[attr-defined]
    with pytest.raises(KeyError):
        b.token_divergence("a", "missing")


def test_unbound_paired_call_refuses_the_merge() -> None:
    b = _critique_loop({})
    second = _critique_loop({"draft": _STEER})
    second.add(tl.trace(_LM, _Q), names="judge")
    b.add(tl.trace(_LM, _Q), names="judge")
    before = _state(b)
    with pytest.raises(BundleRelationError) as excinfo:
        b.add_alternative(second, steered=("draft",))
    assert _code(excinfo) == "bundle_token_divergence_unavailable"
    assert _state(b) == before


def test_argument_errors() -> None:
    b = _critique_loop({})
    with pytest.raises(TypeError):
        b.add_alternative(b, steered=("draft",))
    with pytest.raises(TypeError):
        b.add_alternative(_critique_loop({}), steered=())
    with pytest.raises(TypeError):
        b.add_alternative(_critique_loop({}), steered=("draft",), prefix="")


def test_merged_bundle_round_trips(tmp_path: Path) -> None:
    b = _critique_loop({})
    b.add_alternative(_critique_loop({"draft": _STEER}), steered=("draft",))
    b.save(tmp_path / "harness.tlspec")
    loaded = tl.load(tmp_path / "harness.tlspec")
    assert loaded.member_relations == b.member_relations
    assert loaded.operations[-1].params == b.operations[-1].params
    assert loaded.token_divergence("critique", "cf_critique") == b.token_divergence(
        "critique", "cf_critique"
    )
