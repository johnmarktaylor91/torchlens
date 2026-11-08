"""Automatic text decode on language models carrying placeholder labels."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._errors import InvalidArgumentError


class _Config:
    def __init__(self, labels: dict[int, str]) -> None:
        self.id2label = labels
        self.label2id = {label: index for index, label in labels.items()}
        self.num_labels = len(labels)
        self.model_type = "tiny"
        self.architectures = ["TinyLM"]


class _Tokenizer:
    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:  # noqa: D102
        return " ".join(f"tok{int(index)}" for index in token_ids)


class _PlaceholderLM(nn.Module):
    """Vocabulary-width per-token logits beside the default two placeholder labels."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.config = _Config({0: "LABEL_0", 1: "LABEL_1"})

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:  # noqa: D102
        logits = torch.tensor([[[0.0, 1.0, 5.0, 2.0], [6.0, 0.0, 1.0, 2.0]]]) * self.weight
        return {"logits": logits + x.sum() * 0}


def _next_token_rows(last: list[float], *, text: bool, k: int = 5) -> list[dict[str, Any]]:
    """Expected text-decode rows: the top ``k`` next tokens of the last position's logits."""

    probs = torch.softmax(torch.tensor(last), dim=-1)
    values, indices = torch.topk(probs, k=min(k, len(last)))
    return [
        {
            "batch_item": 0,
            "rank": rank + 1,
            "token_id": int(index),
            "text": f"tok{int(index)}" if text else None,
            "prob": pytest.approx(float(value)),
        }
        for rank, (value, index) in enumerate(zip(values, indices, strict=True))
    ]


def test_placeholder_labels_do_not_claim_vocabulary_logits() -> None:
    """With a tokenizer attached, the text detector wins over the placeholder labels."""

    model = _PlaceholderLM().eval()
    model._torchlens_output_tokenizer = _Tokenizer()
    trace = tl.trace(model, torch.ones(1, 2))

    assert trace.provenance.output_postprocessor is not None
    assert trace.provenance.output_postprocessor.style == "hf_text"
    assert trace.decode_output() == _next_token_rows([6.0, 0.0, 1.0, 2.0], text=True)
    table = trace.output_table()
    assert list(table.columns) == ["batch_item", "rank", "token_id", "text", "prob"]
    assert table["text"].tolist() == ["tok0", "tok3", "tok2", "tok1"]


def test_placeholder_labels_without_a_tokenizer_decode_nothing() -> None:
    """Without a tokenizer no detector resolves, and the read-time door refuses typed."""

    trace = tl.trace(_PlaceholderLM().eval(), torch.ones(1, 2))

    assert trace.provenance.output_postprocessor is None
    with pytest.raises(InvalidArgumentError) as excinfo:
        trace.decode_output()
    assert excinfo.value.fields["code"] == "decode_style_unknown"


def test_two_wide_labels_still_classify() -> None:
    """Labels that cover the logits keep winning: the placeholder guard is width-based."""

    class _Classifier(_PlaceholderLM):
        def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:  # noqa: D102
            return {"logits": torch.tensor([[0.0, 4.0]]) * self.weight + x.sum() * 0}

    model = _Classifier().eval()
    model._torchlens_output_tokenizer = _Tokenizer()
    trace = tl.trace(model, torch.ones(1, 2))
    assert trace.provenance.output_postprocessor.style == "classification"
    assert trace.decode_output()["rows"][0]["label"] == "LABEL_1"
