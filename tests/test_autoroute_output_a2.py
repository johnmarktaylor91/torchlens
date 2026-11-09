"""Sprint A2 semantic output decode tests."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens import _autoroute as autoroute
from torchlens._autoroute._builtin_output import load_imagenet1k_labels, semantic_output_cache_key
from torchlens._data_classes.trace import ResolvedPostprocessing


class _Config:
    """Small config object exposing classifier label metadata."""

    def __init__(self, labels: dict[int, str]) -> None:
        """Initialize label metadata."""

        self.id2label = labels
        self.label2id = {label: index for index, label in labels.items()}
        self.num_labels = len(labels)
        self.model_type = "tiny"
        self.architectures = ["TinyForSequenceClassification"]


class _Classifier(nn.Module):
    """Classifier returning fixed logits plus a trainable zero term."""

    def __init__(self, logits: torch.Tensor, labels: dict[int, str] | None = None) -> None:
        """Initialize the classifier."""

        super().__init__()
        self.weight = nn.Parameter(torch.ones(()))
        self.register_buffer("_logits", logits)
        if labels is not None:
            self.config = _Config(labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return fixed logits connected to ``x`` and ``weight``."""

        return self._logits.to(x.device) + (x.sum() + self.weight) * 0


class _TupleClassifier(_Classifier):
    """Classifier returning tuple outputs."""

    def __init__(
        self,
        logits: torch.Tensor,
        labels: dict[int, str],
        *,
        ambiguous: bool = False,
    ) -> None:
        """Initialize tuple output behavior."""

        super().__init__(logits, labels)
        self.ambiguous = ambiguous

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return loss-like and logits-like outputs."""

        logits = super().forward(x)
        if self.ambiguous:
            return logits + 1, logits
        return logits.sum(), logits


class TinyModelOutput:
    """Duck-typed Hugging Face ModelOutput with a ``logits`` head."""

    def __init__(self, logits: torch.Tensor) -> None:
        """Initialize with logits."""

        self.logits = logits

    def keys(self) -> list[str]:
        """Return available keys."""

        return ["logits"]

    def __getitem__(self, key: str) -> torch.Tensor:
        """Return a named output item."""

        if key != "logits":
            raise KeyError(key)
        return self.logits


class _HFLikeClassifier(_Classifier):
    """Classifier returning an HF-like model output."""

    def forward(self, x: torch.Tensor) -> TinyModelOutput:
        """Return a model output object with logits."""

        return TinyModelOutput(super().forward(x))


class _DictHeadClassifier(_Classifier):
    """Classifier returning multiple named heads."""

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return named output heads."""

        logits = super().forward(x)
        return {"aux": logits - 10, "main": logits}


class _Tokenizer:
    """Minimal tokenizer with a decode method."""

    def decode(self, token_ids: list[int]) -> str:
        """Decode token ids into a display string."""

        return " ".join(f"tok{token_id}" for token_id in token_ids)


def test_read_time_head_decodes_the_named_retained_head() -> None:
    """Two same-width heads are ambiguous at capture; ``head=`` decodes one at read time."""

    labels = {0: "cold", 1: "hot"}
    model = _DictHeadClassifier(torch.tensor([[1.0, 5.0]]), labels).eval()

    trace = tl.trace(model, torch.ones(1, 2))

    assert trace._decoded_output is None
    assert trace.provenance.output_postprocessor is not None
    assert trace.provenance.output_postprocessor.ambiguous is True

    decoded = trace.decode_output(head="main")
    assert decoded["kind"] == "batch_topk"
    assert decoded["rows"][0]["label"] == "hot"
    aux = trace.decode_output(style="classification", head="aux")
    assert aux["rows"][0]["label"] == "hot"
    # The read-time decode returns rows and never rewrites the captured record.
    assert trace._decoded_output is None


def test_custom_registered_output_detector_decodes() -> None:
    """A custom output detector registered via ``tl._autoroute.output.register`` decodes."""

    labels = {0: "left", 1: "right"}
    model = _Classifier(torch.tensor([[3.0, 0.0]])).eval()

    with autoroute.output.snapshot():

        @autoroute.output.register(name="custom_labels", priority=-1)
        def custom_labels(outputs: Any, meta: dict[str, Any]) -> ResolvedPostprocessing | None:
            """Return custom labels for the read-time custom style."""

            if meta.get("style") != "custom_labels":
                return None
            return ResolvedPostprocessing(
                source="custom",
                identifier="custom_labels",
                verified=True,
                config={"id2label": labels, "num_labels": 2},
                description="custom labels",
                style="classification",
                label_source="test",
                label_source_version="test",
                confidence=1.0,
            )

        trace = tl.trace(model, torch.ones(1, 2))
        assert trace._decoded_output is None
        decoded = trace.decode_output(style="custom_labels")

    assert decoded["rows"][0]["label"] == "left"


def test_multi_output_selects_logits_and_fails_closed_on_ambiguity() -> None:
    """Tuple/HF logits are selected conservatively and ambiguous tuples do not decode."""

    labels = {0: "no", 1: "yes"}
    x = torch.ones(1, 2)

    tuple_trace = tl.trace(_TupleClassifier(torch.tensor([[0.0, 3.0]]), labels).eval(), x)
    assert tuple_trace._decoded_output is not None
    assert tuple_trace.provenance.output_postprocessor is not None
    assert tuple_trace.provenance.output_postprocessor.selected_output_head == "1"

    hf_trace = tl.trace(_HFLikeClassifier(torch.tensor([[4.0, 1.0]]), labels).eval(), x)
    assert hf_trace._decoded_output is not None
    assert hf_trace.provenance.output_postprocessor is not None
    assert hf_trace.provenance.output_postprocessor.selected_output_head == "logits"

    ambiguous = tl.trace(
        _TupleClassifier(torch.tensor([[0.0, 3.0]]), labels, ambiguous=True).eval(),
        x,
    )
    assert ambiguous._decoded_output is None
    assert ambiguous.provenance.output_postprocessor is not None
    assert (
        "call trace.decode_output(style=..., head=...)"
        in ambiguous.provenance.output_postprocessor.description
    )


def test_hf_text_decode_uses_attached_tokenizer() -> None:
    """HF text style decodes the top next tokens with the attached tokenizer."""

    model = _Classifier(torch.tensor([[[0.0, 5.0], [6.0, 1.0]]])).eval()
    model._torchlens_output_tokenizer = _Tokenizer()

    trace = tl.trace(model, torch.ones(1, 2))

    assert trace.provenance.output_postprocessor is not None
    assert trace.provenance.output_postprocessor.style == "hf_text"
    probs = torch.softmax(torch.tensor([6.0, 1.0]), dim=-1)
    assert trace._decoded_output == [
        {
            "batch_item": 0,
            "rank": 1,
            "token_id": 0,
            "text": "tok0",
            "prob": pytest.approx(float(probs[0])),
        },
        {
            "batch_item": 0,
            "rank": 2,
            "token_id": 1,
            "text": "tok1",
            "prob": pytest.approx(float(probs[1])),
        },
    ]


def test_decoded_output_roundtrips_tlspec_and_decode_output_is_bounded(tmp_path: Path) -> None:
    """Decoded output rows survive TLSPEC save/load and bounded reads work."""

    labels = {0: "cat", 1: "dog", 2: "eel"}
    trace = tl.trace(_Classifier(torch.tensor([[1.0, 2.0, 0.0]]), labels).eval(), torch.ones(1, 2))
    bundle_path = tmp_path / "decoded.tlspec"

    trace.save(bundle_path)
    restored = tl.load(bundle_path)

    assert restored._decoded_output == trace._decoded_output
    assert restored.decode_output(k=1) == {
        "kind": "batch_topk",
        "rows": [trace._decoded_output["rows"][0]],
    }
    assert len(restored.decode_output(k=10)["rows"]) == 3


def test_cache_key_changes_when_only_id2label_changes(tmp_path: Path) -> None:
    """Semantic cache keys include config labels to avoid stale decoded labels."""

    model = _Classifier(torch.tensor([[0.0, 3.0]]), {0: "old-no", 1: "old-yes"}).eval()
    input_tensor = torch.ones(1, 2)

    first = tl.trace(
        model, input_tensor, capture=tl.options.CaptureOptions(cache=True, cache_dir=tmp_path)
    )
    model.config.id2label = {0: "new-no", 1: "new-yes"}
    second = tl.trace(
        model, input_tensor, capture=tl.options.CaptureOptions(cache=True, cache_dir=tmp_path)
    )

    assert first.provenance.cache_key != second.provenance.cache_key
    assert first._decoded_output is not None
    assert second._decoded_output is not None
    assert first._decoded_output["rows"][0]["label"] == "old-yes"
    assert second._decoded_output["rows"][0]["label"] == "new-yes"
    assert semantic_output_cache_key(model)["config"]["id2label"] == {0: "new-no", 1: "new-yes"}


def test_imagenet_label_bank_package_data_loads() -> None:
    """The shipped ImageNet-1k label bank loads from package data."""

    labels = load_imagenet1k_labels()

    assert len(labels) == 1000
    assert labels[:3] == ["tench", "goldfish", "great white shark"]


class _NestedTensorOutput(nn.Module):
    """Model whose forward returns a strided nested tensor (shape reads raise)."""

    def __init__(self) -> None:
        super().__init__()
        self.lin = nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.lin(x)
        return torch.nested.nested_tensor([y[0], y[1]])


def test_nested_tensor_output_never_aborts_capture_with_raw_torch_error() -> None:
    """A nested-tensor model output surfaces typed, never a raw torch error.

    Fail-before (R65): the opportunistic output sniffer read ``.shape`` on the
    nested output and ``tl.trace`` died with torch's internal ``RuntimeError:
    NestedTensorImpl doesn't support sizes`` -- a raw abort blaming a TorchLens
    internal on the user. The detector must treat unreadable-layout tensors as
    "not logits", and any refusal on the output path must stay typed.
    """

    from torchlens.errors import TorchLensError

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            trace = tl.trace(_NestedTensorOutput().eval(), torch.randn(2, 3))
        except Exception as exc:
            assert isinstance(exc, TorchLensError), (
                "nested-tensor output aborted capture with a raw non-TorchLens "
                f"error: {type(exc).__name__}: {exc}"
            )
        else:
            assert trace is not None


def test_detector_exception_degrades_to_disclosed_warning_not_abort() -> None:
    """A raising registered detector is disclosed and skipped, never fatal."""

    from torchlens._autoroute import output as autoroute_output
    from torchlens.errors import TorchLensWarning

    @autoroute_output.register(name="raising_probe_detector", priority=5)
    def _raising_detector(outputs: Any, meta: dict[str, Any]) -> None:
        raise RuntimeError("detector blew up")

    try:
        with pytest.warns(TorchLensWarning, match="Semantic output decode skipped"):
            trace = tl.trace(nn.Linear(3, 3).eval(), torch.randn(2, 3))
        assert trace._decoded_output is None
    finally:
        autoroute_output.unregister("raising_probe_detector")
