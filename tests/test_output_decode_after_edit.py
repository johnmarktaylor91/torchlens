"""Decoded output follows the trace's own current output after an edit.

The automatic decode runs once at the end of a capture and stores rows on the
trace. An edit that rewrites the outputs (a ``do()`` replay on a fork, an
in-place rerun) must re-derive those rows: ``decode_output()`` and
``output_table()`` describe the edited forward, never the parent's or the
pre-edit one, and the parent keeps its own rows.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl

_LABELS = {0: "cat", 1: "dog", 2: "fox"}


class _Config:
    def __init__(self, labels: dict[int, str]) -> None:
        self.id2label = labels
        self.label2id = {label: index for index, label in labels.items()}
        self.num_labels = len(labels)


class _TinyClassifier(nn.Module):
    """Two linear layers; zero-ablating ``hidden`` leaves the head bias, which ranks ``fox`` first."""

    def __init__(self) -> None:
        super().__init__()
        self.hidden = nn.Linear(2, 3, bias=False)
        self.head = nn.Linear(3, 3)
        with torch.no_grad():
            self.hidden.weight.copy_(torch.tensor([[2.0, 0.0], [0.0, 1.0], [0.0, 0.0]]))
            self.head.weight.copy_(torch.eye(3) * 2.0)
            self.head.bias.copy_(torch.tensor([0.0, 0.5, 1.0]))
        self.config = _Config(_LABELS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self.head(torch.relu(self.hidden(x)))


class _Tokenizer:
    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:  # noqa: D102
        return " ".join(f"tok{int(index)}" for index in token_ids)


class _TinyLM(nn.Module):
    """Per-position vocabulary logits; zero-ablating ``proj`` makes every position pick token 3."""

    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(2, 4, bias=False)
        self.head = nn.Linear(4, 4)
        with torch.no_grad():
            self.proj.weight.copy_(torch.tensor([[2.0, 0.0], [0.0, 2.0], [0.0, 0.0], [0.0, 0.0]]))
            self.head.weight.copy_(torch.eye(4) * 3.0)
            self.head.bias.copy_(torch.tensor([0.0, 0.0, 0.0, 1.0]))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:  # noqa: D102
        return {"logits": self.head(torch.relu(self.proj(x)))}


def _expected_rows(logits: torch.Tensor, k: int) -> list[tuple[str, float]]:
    """Top-``k`` (label, prob) pairs of the first batch item, computed by hand."""

    probs = logits.detach().softmax(-1)[0]
    values, indices = probs.topk(k)
    return [(_LABELS[int(i)], float(v)) for v, i in zip(values, indices, strict=True)]


def _table_rows(trace: Any, k: int) -> list[tuple[str, float]]:
    table = trace.output_table(k=k)
    return [(str(row.label), float(row.prob)) for row in table.itertuples()]


def _assert_rows_match(actual: list[tuple[str, float]], expected: list[tuple[str, float]]) -> None:
    assert [label for label, _ in actual] == [label for label, _ in expected]
    for (_, got), (_, want) in zip(actual, expected, strict=True):
        assert got == pytest.approx(want, abs=1e-6)


def _capture_classifier() -> tuple[nn.Module, torch.Tensor, Any]:
    model = _TinyClassifier().eval()
    x = torch.ones(1, 2)
    return model, x, tl.trace(model, x)


@pytest.mark.parametrize("engine", ["replay", "rerun"])
def test_edited_fork_decodes_its_own_output(engine: str) -> None:
    """``output_table``/``decode_output`` on an edited fork read the fork's edited logits."""

    model, x, trace = _capture_classifier()
    out = trace.output_layers[0]
    _assert_rows_match(_table_rows(trace, 3), _expected_rows(trace[out].out, 3))
    assert _table_rows(trace, 1)[0][0] == "cat"

    options = tl.options.InterventionOptions(engine=engine)
    edited = trace.fork().do(
        tl.module("hidden"), tl.zero_ablate(), model=model, inputs=x, intervention=options
    )

    assert not torch.allclose(edited[out].out, trace[out].out)
    expected = _expected_rows(edited[out].out, 3)
    assert expected[0][0] == "fox"
    _assert_rows_match(_table_rows(edited, 3), expected)
    decoded_rows = edited.decode_output(k=3)["rows"]
    _assert_rows_match([(row["label"], row["prob"]) for row in decoded_rows], expected)
    # The parent keeps describing its own, unedited forward.
    _assert_rows_match(_table_rows(trace, 3), _expected_rows(trace[out].out, 3))
    assert _table_rows(trace, 1)[0][0] == "cat"


@pytest.mark.usefixtures("root_edits_in_place")  # D.12: no per-call mutation keyword
def test_in_place_replay_edit_refreshes_the_decode() -> None:
    """A confirmed in-place edit on the root trace re-derives its rows the same way."""

    _model, _x, trace = _capture_classifier()
    out = trace.output_layers[0]
    trace.do(tl.module("hidden"), tl.zero_ablate())
    _assert_rows_match(_table_rows(trace, 3), _expected_rows(trace[out].out, 3))
    assert _table_rows(trace, 1)[0][0] == "fox"


def _capture_lm() -> Any:
    model = _TinyLM().eval()
    model._torchlens_output_tokenizer = _Tokenizer()
    return tl.trace(model, torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]))


def test_edited_fork_text_decode_reports_the_edited_token_ids() -> None:
    """Changed argmax ids replace the captured ones; the stale captured text is not kept."""

    trace = _capture_lm()
    assert trace.decode_output() == [
        {"batch_item": 0, "rank": 1, "text": "tok0 tok1", "token_ids": [0, 1]}
    ]

    edited = trace.fork().do(tl.module("proj"), tl.zero_ablate())

    # The session tokenizer is gone after capture, so the changed ids carry no text;
    # passing a tokenizer at read time renders it.
    assert edited.decode_output() == [
        {"batch_item": 0, "rank": 1, "text": None, "token_ids": [3, 3]}
    ]
    assert edited.output_table().iloc[0]["token_ids"] == [3, 3]
    assert edited.decode_output(style="hf_text", tokenizer=_Tokenizer())[0]["text"] == "tok3 tok3"
    assert trace.decode_output()[0]["text"] == "tok0 tok1"


def test_text_decode_keeps_captured_text_when_the_ids_do_not_change() -> None:
    """An edit that leaves every argmax id in place keeps the captured text."""

    trace = _capture_lm()
    edited = trace.fork().do(tl.module("head"), tl.scale(0.5))

    out = trace.output_layers[0]
    assert not torch.allclose(edited[out].out, trace[out].out)
    assert edited.decode_output() == [
        {"batch_item": 0, "rank": 1, "text": "tok0 tok1", "token_ids": [0, 1]}
    ]
