"""Regression tests for r18b bridge/hf.py hardening.

Covers four A3 findings on ``torchlens/bridge/transformers.py``:

* A3-06 -- autoroute predicates must decline implausible modality/chat values.
* A3-07 -- list image preprocessing must not run per-item transforms N+1 times.
* A3-13 -- tokenizer provenance must report honest ``verified``/``padding``.
* A3-05 -- the multimodal detection heuristic must be offline-safe (cache-only).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

import torchlens as tl
import torchlens.bridge.transformers as hf

# ---------------------------------------------------------------------------
# A3-06 -- predicate misclassification
# ---------------------------------------------------------------------------


def test_multimodal_predicate_declines_none_and_bare_object() -> None:
    """Implausible audio/video values must not trigger the multimodal route."""

    assert hf._is_hf_multimodal_input({"audio": None}) is False
    assert hf._is_hf_multimodal_input({"videos": object()}) is False
    assert hf._is_hf_multimodal_input({"audio": []}) is False
    assert hf._is_hf_multimodal_input({"audio": 5}) is False


def test_multimodal_predicate_accepts_plausible_media() -> None:
    """Real waveform/frame payloads must still route as multimodal."""

    assert hf._is_hf_multimodal_input({"audio": torch.zeros(16000)}) is True
    assert hf._is_hf_multimodal_input({"audio": [0.1, 0.2, 0.3]}) is True
    assert hf._is_hf_multimodal_input({"videos": (torch.zeros(3, 2, 2),)}) is True
    assert hf._is_hf_multimodal_input({"audio": b"\x00\x01"}) is True


def test_text_predicate_declines_malformed_chat() -> None:
    """Chat records with non-string role or non-str/list content must decline."""

    assert hf._is_hf_text_input([{"role": None, "content": object()}]) is False
    assert hf._is_hf_text_input([{"role": "user"}]) is False
    assert hf._is_hf_text_input([{"content": "hi"}]) is False
    assert hf._is_hf_text_input([{"role": 1, "content": "hi"}]) is False


def test_text_predicate_accepts_valid_chat_and_strings() -> None:
    """Valid chat messages and plain strings still route as text."""

    assert hf._is_hf_text_input("hello") is True
    assert hf._is_hf_text_input(["a", "b"]) is True
    assert hf._is_hf_text_input([{"role": "user", "content": "hi"}]) is True
    assert (
        hf._is_hf_text_input([{"role": "user", "content": [{"type": "text", "text": "hi"}]}])
        is True
    )


# ---------------------------------------------------------------------------
# A3-07 -- list image preprocessing N+1 calls
# ---------------------------------------------------------------------------


def test_image_transform_batch_native_called_once() -> None:
    """A batch-native (tagged) processor must be invoked exactly once on a list."""

    calls: list[Any] = []

    def batch_processor(value: Any) -> dict[str, Any]:
        calls.append(value)
        return {"call": len(calls), "value": value}

    batch_processor._tl_batch_input = True  # type: ignore[attr-defined]
    wrapped = hf._make_image_transform(batch_processor)
    result = wrapped(["a", "b", "c"])

    assert len(calls) == 1
    assert calls == [["a", "b", "c"]]
    assert result == {"call": 1, "value": ["a", "b", "c"]}


def test_image_transform_untagged_mapping_no_wasted_peritem_calls() -> None:
    """An untagged mapping (stateful) transform must not run N discarded per-item passes."""

    calls: list[Any] = []

    def stateful_mapping_transform(value: Any) -> dict[str, Any]:
        calls.append(value)
        return {"call_number": len(calls), "value": value}

    wrapped = hf._make_image_transform(stateful_mapping_transform)
    wrapped(["a", "b", "c", "d"])

    # Old behavior: N per-item probes (discarded) + 1 whole-list = N+1 = 5.
    # New behavior: a single per-item probe + 1 whole-list = 2.
    assert len(calls) <= 2
    assert calls[-1] == ["a", "b", "c", "d"]


def test_image_transform_per_item_stacks_without_waste() -> None:
    """A per-item tensor transform must be called once per item and stacked."""

    calls: list[Any] = []

    def per_item(value: Any) -> torch.Tensor:
        calls.append(value)
        return torch.zeros(3, 2, 2)

    wrapped = hf._make_image_transform(per_item)
    result = wrapped(["a", "b", "c"])

    assert len(calls) == 3
    assert calls == ["a", "b", "c"]
    assert isinstance(result, torch.Tensor)
    assert result.shape == (3, 3, 2, 2)


def test_image_transform_single_image_unsqueezes() -> None:
    """A single-image tensor transform result gains a batch dimension."""

    wrapped = hf._make_image_transform(lambda value: torch.zeros(3, 2, 2))
    result = wrapped("one-image")

    assert isinstance(result, torch.Tensor)
    assert result.shape == (1, 3, 2, 2)


# ---------------------------------------------------------------------------
# A3-13 -- tokenizer provenance honesty (false-VERIFIED class)
# ---------------------------------------------------------------------------


def test_record_explicit_tokenizer_is_not_verified() -> None:
    """An explicit user tokenizer must not be recorded as model-verified."""

    tokenizer = SimpleNamespace(name_or_path="my/custom", model_max_length=128)
    record = hf._tokenizer_preprocessing_record(tokenizer, nn.Linear(1, 1), explicit=True)

    assert record.verified is False


def test_record_unknown_identifier_is_not_verified() -> None:
    """When no identifier can be resolved, verified must be False (not True)."""

    tokenizer = SimpleNamespace(model_max_length=128)  # no name_or_path
    record = hf._tokenizer_preprocessing_record(tokenizer, nn.Linear(1, 1), explicit=False)

    assert record.identifier == "unknown"
    assert record.verified is False


def test_record_auto_resolved_identifier_stays_verified() -> None:
    """Auto-resolved tokenizers with a real identifier remain verified."""

    tokenizer = SimpleNamespace(name_or_path="bert-base-uncased", model_max_length=512)
    record = hf._tokenizer_preprocessing_record(tokenizer, nn.Linear(1, 1), explicit=False)

    assert record.identifier == "bert-base-uncased"
    assert record.verified is True


def test_record_reports_actual_padding_fallback() -> None:
    """A no-pad-token fallback must be recorded as padding=False, not True."""

    class NoPadTokenizer:
        name_or_path = "gpt2"
        model_max_length = 1024

        def __call__(self, text: Any, return_tensors: Any = None, padding: Any = None) -> Any:
            if padding:
                raise ValueError("Asking to pad but the tokenizer does not have a padding token.")
            return {"input_ids": [[1, 2, 3]]}

    tok = NoPadTokenizer()
    transform, state = hf._make_text_transform(tok)
    transform("hello")  # single string -> triggers the padding=False fallback

    assert state["padding"] is False
    record = hf._tokenizer_preprocessing_record(
        tok, nn.Linear(1, 1), explicit=False, padding=state["padding"]
    )
    assert record.config["padding"] is False


def test_trace_text_explicit_tokenizer_end_to_end(monkeypatch: Any) -> None:
    """Full trace_text with an explicit custom tokenizer reports honest provenance."""

    class Model(nn.Module):
        def forward(self, x: Any) -> Any:
            return x

    class ExplicitCustomTokenizer:
        def __call__(self, text: Any, **kwargs: Any) -> Any:
            return {"input_ids": text}

    monkeypatch.setattr(tl, "trace", lambda *a, **k: SimpleNamespace(input_preprocessor=None))
    log = hf.trace_text(Model(), "hello", tokenizer=ExplicitCustomTokenizer())
    record = log.input_preprocessor

    assert record.verified is False
    assert record.identifier == "unknown"


# ---------------------------------------------------------------------------
# A3-05 -- detection heuristic must be offline-safe (cache-only)
# ---------------------------------------------------------------------------


def test_processor_gate_probes_cache_only(monkeypatch: Any) -> None:
    """The multimodal gate must resolve AutoProcessor with local_files_only=True.

    The fake AutoProcessor simulates an offline box: any non-cache-only call
    raises (as a real Hub retry loop eventually would). If the gate is
    cache-only it resolves; the recorded kwargs prove local_files_only was set.
    """

    import sys

    recorded: list[dict[str, Any]] = []

    class FakeAutoProcessor:
        @classmethod
        def from_pretrained(cls, name_or_path: str, **kwargs: Any) -> Any:
            recorded.append(kwargs)
            if not kwargs.get("local_files_only"):
                raise RuntimeError("network access attempted in a detection heuristic")
            return object()

    monkeypatch.setitem(
        sys.modules, "transformers", SimpleNamespace(AutoProcessor=FakeAutoProcessor)
    )

    model = SimpleNamespace(name_or_path="fixture/model")
    assert hf._can_resolve_hf_processor(model) is True
    assert recorded == [{"local_files_only": True}]


def test_processor_gate_declines_when_uncached(monkeypatch: Any) -> None:
    """An uncached model declines the route (no network) instead of downloading."""

    import sys

    class FakeAutoProcessor:
        @classmethod
        def from_pretrained(cls, name_or_path: str, **kwargs: Any) -> Any:
            # Simulate a cache miss under local_files_only=True.
            raise OSError("not found in local cache")

    monkeypatch.setitem(
        sys.modules, "transformers", SimpleNamespace(AutoProcessor=FakeAutoProcessor)
    )

    model = SimpleNamespace(name_or_path="fixture/uncached")
    assert hf._can_resolve_hf_processor(model) is False
