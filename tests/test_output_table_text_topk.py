"""``output_table(k)`` on a language model: the top ``k`` next tokens at the last position.

A text decode reads each batch item's logits at its last position, takes the
softmax over the vocabulary and returns ``k`` rows per batch item: ``rank``,
``token_id``, the decoded token ``text`` and its ``prob``, the same top-k the
classifier rows give.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl

pytest.importorskip("pandas")

_VOCAB = 8


class _Tokenizer:
    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:  # noqa: D102
        return " ".join(f"tok{int(index)}" for index in token_ids)


class _BiasLM(nn.Module):
    """Every position's logits are the head bias plus a per-position input term."""

    def __init__(self) -> None:
        super().__init__()
        self.head = nn.Linear(1, _VOCAB)
        with torch.no_grad():
            self.head.weight.copy_(torch.linspace(-1.0, 1.0, _VOCAB).unsqueeze(1))
            self.head.bias.copy_(torch.arange(_VOCAB, dtype=torch.float32) * 0.5)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:  # noqa: D102
        return {"logits": self.head(x)}


def _capture(x: torch.Tensor) -> Any:
    model = _BiasLM().eval()
    model._torchlens_output_tokenizer = _Tokenizer()
    return tl.trace(model, x)


def _expected(logits: torch.Tensor, k: int) -> list[tuple[int, int, int, str, float]]:
    """``(batch_item, rank, token_id, text, prob)`` computed by hand from the last position."""

    probs = torch.softmax(logits[:, -1, :].float(), dim=-1)
    values, indices = torch.topk(probs, k=k, dim=-1)
    return [
        (b, rank + 1, int(indices[b, rank]), f"tok{int(indices[b, rank])}", float(values[b, rank]))
        for b in range(probs.shape[0])
        for rank in range(k)
    ]


def _rows(table: Any) -> list[tuple[int, int, int, str, float]]:
    return [
        (int(r.batch_item), int(r.rank), int(r.token_id), str(r.text), float(r.prob))
        for r in table.itertuples()
    ]


def test_output_table_gives_k_next_tokens_per_batch_item() -> None:
    """Two batch items of three positions: three rows each, from the last position only."""

    x = torch.tensor([[[0.0], [3.0], [-2.0]], [[1.0], [-3.0], [2.5]]])
    trace = _capture(x)
    table = trace.output_table(k=3)

    assert list(table.columns) == ["batch_item", "rank", "token_id", "text", "prob"]
    logits = trace[trace.output_layers[0]].out
    got, want = _rows(table), _expected(logits, 3)
    assert [row[:4] for row in got] == [row[:4] for row in want]
    assert [row[4] for row in got] == pytest.approx([row[4] for row in want])
    # The two items' last positions differ, so their top tokens differ.
    assert got[0][2] != got[3][2]


def test_text_decode_rows_carry_the_probability_of_each_token() -> None:
    """``decode_output`` returns the same rows; probabilities fall with rank and sum below 1."""

    trace = _capture(torch.tensor([[[0.5], [1.5]]]))
    rows = trace.decode_output(k=5)
    assert [row["rank"] for row in rows] == [1, 2, 3, 4, 5]
    probs = [row["prob"] for row in rows]
    assert probs == sorted(probs, reverse=True)
    assert 0.0 < sum(probs) < 1.0
    assert all(row["text"] == f"tok{row['token_id']}" for row in rows)


def test_batch_items_selects_items_and_k_bounds_rows() -> None:
    x = torch.tensor([[[0.0], [3.0]], [[1.0], [-3.0]]])
    trace = _capture(x)
    table = trace.output_table(k=2, batch_items=[1])
    assert table["batch_item"].tolist() == [1, 1]
    assert table["rank"].tolist() == [1, 2]


def test_read_time_text_decode_takes_k_above_the_captured_bound() -> None:
    """A decode on demand with a tokenizer returns ``k`` rows even above the captured five."""

    trace = _capture(torch.tensor([[[0.5], [1.5]]]))
    rows = trace.decode_output(style="hf_text", tokenizer=_Tokenizer(), k=7)
    assert [row["rank"] for row in rows] == list(range(1, 8))
    table = trace.output_table(k=7, style="hf_text")
    assert len(table) == 7
    assert table["text"].isna().all(), "no tokenizer at read time: ids only"


def test_unbatched_logits_decode_as_one_batch_item() -> None:
    class _Unbatched(_BiasLM):
        def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:  # noqa: D102
            return {"logits": self.head(x)[0]}

    model = _Unbatched().eval()
    model._torchlens_output_tokenizer = _Tokenizer()
    trace = tl.trace(model, torch.tensor([[[0.0], [2.0]]]))
    table = trace.output_table(k=2)
    assert table["batch_item"].tolist() == [0, 0]
    logits = trace[trace.output_layers[0]].out.unsqueeze(0)
    assert [row[:4] for row in _rows(table)] == [row[:4] for row in _expected(logits, 2)]


def test_an_edit_keeps_text_only_for_tokens_the_capture_decoded() -> None:
    """After an edit, a next token the captured rows never held has ``text=None``."""

    trace = _capture(torch.zeros(1, 2, 1))
    captured = {row["token_id"]: row["text"] for row in trace.decode_output()}
    assert set(captured) == {7, 6, 5, 4, 3}

    edited = trace.fork().do(tl.module("head"), tl.scale(-1.0))
    rows = edited.decode_output()
    assert [row["token_id"] for row in rows] == [0, 1, 2, 3, 4]
    assert [row["text"] for row in rows] == [None, None, None, "tok3", "tok4"]
    # The parent keeps its own rows.
    assert [row["token_id"] for row in trace.decode_output()] == [7, 6, 5, 4, 3]
