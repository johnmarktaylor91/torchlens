"""Sprint A2 semantic output decode tests."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens import autoroute
from torchlens.autoroute._builtin_output import load_imagenet1k_labels
from torchlens.data_classes.trace import ResolvedPostprocessing
from torchlens.user_funcs import semantic_output_cache_key


def _decoded_rows(trace: tl.Trace) -> list[dict[str, Any]]:
    """Return classification decoded rows from the typed A3 representation."""

    assert trace.decoded_output is not None
    assert trace.decoded_output["kind"] == "batch_topk"
    return trace.decoded_output["rows"]


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


def test_config_id2label_decodes_topk_labels_and_probs() -> None:
    """Classifier config labels decode into bounded top-k JSON rows."""

    labels = {0: "negative", 1: "positive", 2: "neutral"}
    model = _Classifier(torch.tensor([[0.0, 4.0, 1.0]]), labels).eval()

    trace = tl.trace(model, torch.ones(1, 2))

    assert trace.output_postprocessor is not None
    assert trace.output_postprocessor.source == "hf_config"
    rows = _decoded_rows(trace)
    assert rows[0]["label"] == "positive"
    assert rows[0]["class_index"] == 1
    assert pytest.approx(rows[0]["prob"], rel=1e-5) == float(
        torch.softmax(torch.tensor([0.0, 4.0, 1.0]), dim=0)[1]
    )


def test_output_style_override_and_output_head_select_named_head() -> None:
    """Explicit style and output head decode the selected live output tensor."""

    labels = {0: "cold", 1: "hot"}
    model = _DictHeadClassifier(torch.tensor([[1.0, 5.0]]), labels).eval()

    trace = tl.trace(
        model,
        torch.ones(1, 2),
        output_style="classification",
        output_head="main",
    )

    assert trace.output_postprocessor is not None
    assert trace.output_postprocessor.selected_output_head == "main"
    assert _decoded_rows(trace)[0]["label"] == "hot"


def test_custom_registered_output_detector_decodes() -> None:
    """A custom output detector registered via ``tl.autoroute.output.register`` decodes."""

    labels = {0: "left", 1: "right"}
    model = _Classifier(torch.tensor([[3.0, 0.0]])).eval()

    with autoroute.output.snapshot():

        @autoroute.output.register(name="custom_labels", priority=-1)
        def custom_labels(outputs: Any, meta: dict[str, Any]) -> ResolvedPostprocessing | None:
            """Return custom labels for an explicit custom style."""

            if meta.get("output_style") != "custom_labels":
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

        trace = tl.trace(model, torch.ones(1, 2), output_style="custom_labels")

    assert _decoded_rows(trace)[0]["label"] == "left"


def test_backward_ready_decode_keeps_graph_connected_output() -> None:
    """Decode uses a copy and leaves retained outputs connected to autograd."""

    labels = {0: "low", 1: "high"}
    model = _Classifier(torch.tensor([[0.0, 2.0]]), labels).train()
    x = torch.ones(1, 2, requires_grad=True)

    trace = tl.trace(model, x, backward_ready=True)
    output_op = trace.output_ops[0]
    output_tensor = output_op.out
    assert isinstance(output_tensor, torch.Tensor)
    assert output_tensor.requires_grad

    output_tensor.sum().backward()

    assert model.weight.grad is not None
    assert x.grad is not None


def test_multi_output_selects_logits_and_fails_closed_on_ambiguity() -> None:
    """Tuple/HF logits are selected conservatively and ambiguous tuples do not decode."""

    labels = {0: "no", 1: "yes"}
    x = torch.ones(1, 2)

    tuple_trace = tl.trace(_TupleClassifier(torch.tensor([[0.0, 3.0]]), labels).eval(), x)
    assert tuple_trace.decoded_output is not None
    assert tuple_trace.output_postprocessor is not None
    assert tuple_trace.output_postprocessor.selected_output_head == "1"

    hf_trace = tl.trace(_HFLikeClassifier(torch.tensor([[4.0, 1.0]]), labels).eval(), x)
    assert hf_trace.decoded_output is not None
    assert hf_trace.output_postprocessor is not None
    assert hf_trace.output_postprocessor.selected_output_head == "logits"

    ambiguous = tl.trace(
        _TupleClassifier(torch.tensor([[0.0, 3.0]]), labels, ambiguous=True).eval(),
        x,
    )
    assert ambiguous.decoded_output is None
    assert ambiguous.output_postprocessor is not None
    assert "pass output_style=/output_head=" in ambiguous.output_postprocessor.description


def test_generic_1000_width_tensor_without_verified_metadata_does_not_decode() -> None:
    """A bare ``[B,1000]`` tensor is not silently treated as ImageNet."""

    model = _Classifier(torch.zeros(1, 1000)).eval()
    trace = tl.trace(model, torch.ones(1, 2))

    assert trace.decoded_output is None
    assert trace.output_postprocessor is None


def test_hf_text_decode_uses_attached_tokenizer() -> None:
    """HF text style decodes argmax token ids with the attached tokenizer."""

    model = _Classifier(torch.tensor([[[0.0, 5.0], [6.0, 1.0]]])).eval()
    model._torchlens_output_tokenizer = _Tokenizer()

    trace = tl.trace(model, torch.ones(1, 2), output_style="hf_text")

    assert trace.output_postprocessor is not None
    assert trace.output_postprocessor.style == "hf_text"
    assert trace.decoded_output == [
        {"batch_item": 0, "rank": 1, "text": "tok1 tok0", "token_ids": [1, 0]}
    ]


def test_decoded_output_roundtrips_tlspec_and_decode_output_is_bounded(tmp_path: Path) -> None:
    """Decoded output rows survive TLSPEC save/load and bounded reads work."""

    labels = {0: "cat", 1: "dog", 2: "eel"}
    trace = tl.trace(_Classifier(torch.tensor([[1.0, 2.0, 0.0]]), labels).eval(), torch.ones(1, 2))
    bundle_path = tmp_path / "decoded.tlspec"

    tl.save(trace, bundle_path)
    restored = tl.load(bundle_path)

    assert restored.decoded_output == trace.decoded_output
    assert restored.decode_output(top_n=1) == {
        "kind": "batch_topk",
        "rows": [trace.decoded_output["rows"][0]],
    }
    assert len(restored.decode_output(top_n=10)["rows"]) == 3


def test_cache_key_changes_when_only_id2label_changes(tmp_path: Path) -> None:
    """Semantic cache keys include config labels to avoid stale decoded labels."""

    model = _Classifier(torch.tensor([[0.0, 3.0]]), {0: "old-no", 1: "old-yes"}).eval()
    input_tensor = torch.ones(1, 2)

    first = tl.trace(model, input_tensor, cache=True, cache_dir=tmp_path)
    model.config.id2label = {0: "new-no", 1: "new-yes"}
    second = tl.trace(model, input_tensor, cache=True, cache_dir=tmp_path)

    assert first.capture_cache_key != second.capture_cache_key
    assert first.decoded_output is not None
    assert second.decoded_output is not None
    assert first.decoded_output["rows"][0]["label"] == "old-yes"
    assert second.decoded_output["rows"][0]["label"] == "new-yes"
    assert semantic_output_cache_key(model, output_style=None, output_head=None)["config"][
        "id2label"
    ] == {0: "new-no", 1: "new-yes"}


def test_imagenet_label_bank_package_data_loads() -> None:
    """The shipped ImageNet-1k label bank loads from package data."""

    labels = load_imagenet1k_labels()

    assert len(labels) == 1000
    assert labels[:3] == ["tench", "goldfish", "great white shark"]


def test_import_torchlens_does_not_import_optional_output_detector_deps() -> None:
    """Importing TorchLens does not import the optional output-detector deps.

    transformers/timm imports live INSIDE the detector functions and must stay
    lazy. torchvision is intentionally excluded here: TorchLens imports it at
    import time to register and wrap torchvision custom ops (nms/roi_align/...),
    which is pre-existing, load-bearing behavior.
    """

    code = """
import sys
for name in ['transformers', 'timm']:
    sys.modules.pop(name, None)
import torchlens
print({name: name in sys.modules for name in ['transformers', 'timm']})
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
        text=True,
        capture_output=True,
    )

    assert result.stdout.strip() == "{'transformers': False, 'timm': False}"
