"""The read-time output decode door: ``Trace.decode_output(style=, head=, tokenizer=)``.

Output decoding is not a capture option. The capture runs the automatic
detector pass with no setting; a user who wants another style, a named head
of a multi-output model, or a text decode with their own tokenizer decodes
again from the retained output tensors at read time. The door returns rows
and never rewrites ``decoded_output``; an unknown style, an unknown head and
an ambiguous head refuse typed.
"""

from __future__ import annotations

import inspect
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._autoroute import _decode_spellings as spellings
from torchlens._data_classes.trace import ResolvedPostprocessing
from torchlens._errors import InvalidArgumentError, PayloadUnavailableError


class _Config:
    def __init__(self, labels: dict[int, str]) -> None:
        self.id2label = labels
        self.label2id = {label: index for index, label in labels.items()}
        self.num_labels = len(labels)


class _Fixed(nn.Module):
    """Return fixed logits connected to the input."""

    def __init__(self, logits: torch.Tensor, labels: dict[int, str] | None = None) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.register_buffer("_logits", logits)
        if labels is not None:
            self.config = _Config(labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return self._logits.to(x.device) + (x.sum() + self.weight) * 0


class _TwoHeads(_Fixed):
    """Two same-width heads: ambiguous for the automatic pass."""

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:  # noqa: D102
        logits = super().forward(x)
        return {"aux": logits.flip(-1), "main": logits}


class _Tokenizer:
    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:  # noqa: D102
        return " ".join(f"tok{int(index)}" for index in token_ids)


@pytest.mark.smoke
def test_capture_options_have_no_decode_fields() -> None:
    """The two removed fields are unknown to ``CaptureOptions`` (no alias)."""

    for field in ("output_style", "output_head"):
        with pytest.raises(TypeError):
            tl.options.CaptureOptions(**{field: "classification"})
        assert not hasattr(tl.options.CaptureOptions(), field)


@pytest.mark.smoke
def test_spellings_are_pinned_to_the_signatures() -> None:
    """The public keywords live in one module; the signatures must match it."""

    decode_params = inspect.signature(tl.Trace.decode_output).parameters
    table_params = inspect.signature(tl.Trace.output_table).parameters
    for keyword in (spellings.STYLE_KW, spellings.HEAD_KW, spellings.TOKENIZER_KW):
        assert keyword in decode_params, keyword
        assert decode_params[keyword].kind is inspect.Parameter.KEYWORD_ONLY, keyword
    for keyword in (spellings.STYLE_KW, spellings.HEAD_KW):
        assert keyword in table_params, keyword
        assert table_params[keyword].kind is inspect.Parameter.KEYWORD_ONLY, keyword


@pytest.mark.smoke
def test_forced_imagenet_style_decodes_an_unverified_1000_way_model() -> None:
    """The automatic pass declines a bare 1000-way tensor; ``style="imagenet"`` decodes it."""

    logits = torch.zeros(1, 1000)
    logits[0, 1] = 9.0  # ImageNet-1k class 1 is "goldfish"
    trace = tl.trace(_Fixed(logits).eval(), torch.ones(1, 2))

    assert trace.decoded_output is None
    decoded = trace.decode_output(style="imagenet", k=1)
    assert decoded["kind"] == "batch_topk"
    assert decoded["rows"][0]["class_index"] == 1
    assert decoded["rows"][0]["label"] == "goldfish"
    assert trace.decoded_output is None, "the read-time decode must not rewrite the record"
    assert trace.output_table(k=1, style="imagenet1k")["label"].tolist() == ["goldfish"]


def test_named_head_resolves_an_ambiguous_multi_output_model() -> None:
    labels = {0: "cold", 1: "hot"}
    trace = tl.trace(_TwoHeads(torch.tensor([[1.0, 5.0]]), labels).eval(), torch.ones(1, 2))

    assert trace.decoded_output is None
    assert trace.output_postprocessor is not None and trace.output_postprocessor.ambiguous
    assert trace.decode_output(head="main")["rows"][0]["label"] == "hot"
    assert trace.decode_output(head="aux")["rows"][0]["label"] == "cold"


def test_unknown_head_refuses_with_the_available_heads() -> None:
    labels = {0: "cold", 1: "hot"}
    trace = tl.trace(_TwoHeads(torch.tensor([[1.0, 5.0]]), labels).eval(), torch.ones(1, 2))

    with pytest.raises(InvalidArgumentError) as excinfo:
        trace.decode_output(head="logits")
    assert excinfo.value.fields["code"] == spellings.HEAD_UNKNOWN_CODE
    assert sorted(excinfo.value.fields["available_heads"]) == ["aux", "main"]


def test_ambiguous_heads_refuse_when_none_is_named() -> None:
    labels = {0: "cold", 1: "hot"}
    trace = tl.trace(_TwoHeads(torch.tensor([[1.0, 5.0]]), labels).eval(), torch.ones(1, 2))

    with pytest.raises(InvalidArgumentError) as excinfo:
        trace.decode_output(style="classification")
    assert excinfo.value.fields["code"] == spellings.HEAD_AMBIGUOUS_CODE
    assert sorted(excinfo.value.fields["available_heads"]) == ["aux", "main"]


@pytest.mark.smoke
def test_unknown_style_refuses_with_did_you_mean() -> None:
    trace = tl.trace(_Fixed(torch.zeros(1, 3)).eval(), torch.ones(1, 2))

    with pytest.raises(tl.errors.DecodeStyleUnknownError) as excinfo:
        trace.decode_output(style="clasification")
    assert excinfo.value.fields["code"] == spellings.STYLE_UNKNOWN_CODE
    assert "did you mean 'classification'?" in str(excinfo.value)


def test_hf_text_style_decodes_with_the_passed_tokenizer() -> None:
    logits = torch.tensor([[[0.0, 5.0], [6.0, 1.0]]])
    trace = tl.trace(_Fixed(logits).eval(), torch.ones(1, 2))

    assert trace.decoded_output is None  # no tokenizer attached: no automatic text decode
    decoded = trace.decode_output(style="hf_text", tokenizer=_Tokenizer())
    assert decoded == [{"batch_item": 0, "rank": 1, "text": "tok1 tok0", "token_ids": [1, 0]}]
    with pytest.raises(InvalidArgumentError) as excinfo:
        trace.decode_output(style="hf_text")
    assert excinfo.value.fields["code"] == spellings.STYLE_UNKNOWN_CODE


def test_attached_tokenizer_decodes_text_automatically_only_from_a_logits_head() -> None:
    """The bridge's attached tokenizer switches text decode on; hidden states never decode."""

    class _Encoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(()))

        def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:  # noqa: D102
            hidden = torch.ones(1, 2, 4) * self.weight
            return {"last_hidden_state": hidden}

    encoder = _Encoder().eval()
    encoder._torchlens_output_tokenizer = _Tokenizer()
    trace = tl.trace(encoder, torch.ones(1, 2))
    assert trace.decoded_output is None
    assert trace.output_postprocessor is None

    class _LM(_Encoder):
        def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:  # noqa: D102
            return {"logits": torch.tensor([[[0.0, 5.0], [6.0, 1.0]]]) * self.weight}

    language_model = _LM().eval()
    language_model._torchlens_output_tokenizer = _Tokenizer()
    trace = tl.trace(language_model, torch.ones(1, 2))
    assert trace.output_postprocessor is not None
    assert trace.output_postprocessor.style == "hf_text"
    assert trace.output_postprocessor.selected_output_head == "logits"
    assert trace.decoded_output == [
        {"batch_item": 0, "rank": 1, "text": "tok1 tok0", "token_ids": [1, 0]}
    ]


def test_attached_tokenizer_never_overrides_a_labelled_classifier() -> None:
    """A text classifier traced through the bridge gets its labels, not argmax text."""

    model = _Fixed(torch.tensor([[0.0, 4.0]]), {0: "neg", 1: "pos"}).eval()
    model._torchlens_output_tokenizer = _Tokenizer()
    trace = tl.trace(model, torch.ones(1, 2))

    assert trace.output_postprocessor is not None
    assert trace.output_postprocessor.style == "classification"
    assert trace.decoded_output["rows"][0]["label"] == "pos"


def test_custom_detector_serves_the_read_time_style() -> None:
    @tl._autoroute.output.register(name="walk_labels", priority=5)
    def walk_labels(outputs: Any, meta: dict[str, Any]) -> ResolvedPostprocessing | None:
        if meta.get(spellings.STYLE_KW) != "walk_labels":
            return None
        return ResolvedPostprocessing(
            source="project",
            identifier="walk_labels",
            verified=True,
            config={"id2label": {0: "a", 1: "b", 2: "c"}},
            description="test label bank",
            style="classification",
            selected_output_head=meta.get(spellings.HEAD_KW),
            label_source="test",
            label_source_version="v1",
            confidence=1.0,
        )

    try:
        trace = tl.trace(_Fixed(torch.tensor([[0.0, 0.0, 3.0]])).eval(), torch.ones(1, 2))
        assert trace.decoded_output is None  # the custom detector declines the automatic pass
        assert trace.decode_output(style="walk_labels")["rows"][0]["label"] == "c"
    finally:
        tl._autoroute.output.unregister("walk_labels")


def test_structure_only_trace_refuses_a_read_time_decode_as_unavailable() -> None:
    """No retained output values: the decode door refuses typed instead of guessing."""

    trace = tl.trace(
        _Fixed(torch.zeros(1, 3)).eval(),
        torch.ones(1, 2),
        capture=tl.options.CaptureOptions(structure_only=True),
    )
    with pytest.raises(PayloadUnavailableError) as excinfo:
        trace.decode_output(style="classification")
    assert excinfo.value.fields["code"] == "decoded_output_unavailable"
