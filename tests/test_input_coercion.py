"""Tests for ergonomic TorchLens input coercion."""

from __future__ import annotations

from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._input_coerce import _coerce_input


class TokenOutput:
    """Simple tokenizer return object with HuggingFace-style input ids."""

    def __init__(self, input_ids: torch.Tensor) -> None:
        """Store token ids.

        Parameters
        ----------
        input_ids:
            Token ids exposed through the HuggingFace-style attribute.
        """

        self.input_ids = input_ids


class PixelOutput:
    """Simple image processor return object with pixel values."""

    def __init__(self, pixel_values: torch.Tensor) -> None:
        """Store pixel values.

        Parameters
        ----------
        pixel_values:
            Processed image batch.
        """

        self.pixel_values = pixel_values


class AudioOutput:
    """Simple audio processor return object with input values."""

    def __init__(self, input_values: torch.Tensor) -> None:
        """Store input values.

        Parameters
        ----------
        input_values:
            Processed audio batch.
        """

        self.input_values = input_values


class TLTextModel:
    """Duck-typed TransformerLens text model for coercion tests."""

    def __init__(self) -> None:
        """Initialize call bookkeeping."""

        self.calls: list[Any] = []

    def to_tokens(self, x: str | list[str]) -> torch.Tensor:
        """Convert text to token ids.

        Parameters
        ----------
        x:
            Text input.

        Returns
        -------
        torch.Tensor
            Dummy token ids.
        """

        self.calls.append(x)
        return torch.tensor([[1, 2, 3]])


class HFTextModel:
    """Duck-typed HuggingFace text model for coercion tests."""

    def __init__(self) -> None:
        """Initialize call bookkeeping."""

        self.calls: list[tuple[Any, str]] = []

    def tokenizer(self, x: str | list[str], *, return_tensors: str) -> TokenOutput:
        """Tokenize text using a HuggingFace-style callable.

        Parameters
        ----------
        x:
            Text input.
        return_tensors:
            Requested tensor backend.

        Returns
        -------
        TokenOutput
            Dummy token ids.
        """

        self.calls.append((x, return_tensors))
        return TokenOutput(torch.tensor([[4, 5, 6]]))


class ImageModel:
    """Duck-typed image model for coercion tests."""

    def __init__(self) -> None:
        """Initialize call bookkeeping."""

        self.calls: list[tuple[Any, str]] = []

    def image_processor(self, x: Any, *, return_tensors: str) -> PixelOutput:
        """Process image input using a HuggingFace-style callable.

        Parameters
        ----------
        x:
            PIL image input.
        return_tensors:
            Requested tensor backend.

        Returns
        -------
        PixelOutput
            Dummy pixel tensor.
        """

        self.calls.append((x, return_tensors))
        return PixelOutput(torch.ones(1, 3, 8, 8))


class AudioModel:
    """Duck-typed audio model for coercion tests."""

    def __init__(self) -> None:
        """Initialize processor attributes and call bookkeeping."""

        self.sampling_rate = 16_000
        self.calls: list[tuple[Any, int | None, str]] = []

    def feature_extractor(
        self,
        x: Any,
        *,
        return_tensors: str,
        sampling_rate: int | None = None,
    ) -> AudioOutput:
        """Process audio input using a HuggingFace-style callable.

        Parameters
        ----------
        x:
            Raw audio input.
        return_tensors:
            Requested tensor backend.
        sampling_rate:
            Optional sampling rate.

        Returns
        -------
        AudioOutput
            Dummy audio tensor.
        """

        self.calls.append((x, sampling_rate, return_tensors))
        return AudioOutput(torch.ones(1, 3))


class TokenEmbeddingModel(nn.Module):
    """Tiny token model used for end-to-end string capture."""

    def __init__(self) -> None:
        """Initialize embedding and attached tokenizer."""

        super().__init__()
        self.embedding = nn.Embedding(16, 4)
        self.tokenizer_calls: list[tuple[Any, str]] = []

    def tokenizer(self, x: str | list[str], *, return_tensors: str) -> TokenOutput:
        """Tokenize text for the tiny embedding model.

        Parameters
        ----------
        x:
            Text input.
        return_tensors:
            Requested tensor backend.

        Returns
        -------
        TokenOutput
            Token ids for the embedding.
        """

        self.tokenizer_calls.append((x, return_tensors))
        return TokenOutput(torch.tensor([[1, 2, 3]], dtype=torch.long))

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run a tiny token embedding forward pass.

        Parameters
        ----------
        input_ids:
            Token ids.

        Returns
        -------
        torch.Tensor
            Mean embedding.
        """

        return self.embedding(input_ids).mean(dim=1)


def test_tensor_passthrough() -> None:
    """Tensor inputs are returned unchanged."""

    tensor = torch.tensor([1, 2, 3])
    assert _coerce_input(object(), tensor) is tensor


def test_numpy_array_to_tensor() -> None:
    """NumPy arrays are converted with dtype and values preserved."""

    np = pytest.importorskip("numpy")
    array = np.array([[1, 2], [3, 4]], dtype=np.int64)
    result = _coerce_input(object(), array)
    assert isinstance(result, torch.Tensor)
    assert result.dtype is torch.int64
    assert torch.equal(result, torch.tensor([[1, 2], [3, 4]], dtype=torch.int64))


def test_transformerlens_text_to_tokens() -> None:
    """String input uses a duck-typed TransformerLens to_tokens method."""

    model = TLTextModel()
    result = _coerce_input(model, "hello world")
    assert model.calls == ["hello world"]
    assert torch.equal(result, torch.tensor([[1, 2, 3]]))


def test_hf_text_tokenizer_input_ids() -> None:
    """String input uses an attached HuggingFace tokenizer."""

    model = HFTextModel()
    result = _coerce_input(model, "hello world")
    assert model.calls == [("hello world", "pt")]
    assert torch.equal(result, torch.tensor([[4, 5, 6]]))


def test_hf_text_batch_tokenizer_input_ids() -> None:
    """Batched string input uses the attached HuggingFace tokenizer once."""

    model = HFTextModel()
    result = _coerce_input(model, ["a", "b", "c"])
    assert model.calls == [(["a", "b", "c"], "pt")]
    assert torch.equal(result, torch.tensor([[4, 5, 6]]))


def test_pil_image_processor() -> None:
    """PIL image input uses an attached image processor."""

    pil_image = pytest.importorskip("PIL.Image")
    image = pil_image.new("RGB", (8, 8))
    model = ImageModel()
    result = _coerce_input(model, image)
    assert model.calls == [(image, "pt")]
    assert torch.equal(result, torch.ones(1, 3, 8, 8))


def test_pil_image_batch_processor() -> None:
    """Batched PIL image input uses an attached image processor once."""

    pil_image = pytest.importorskip("PIL.Image")
    images = [pil_image.new("RGB", (8, 8)), pil_image.new("RGB", (8, 8))]
    model = ImageModel()
    result = _coerce_input(model, images)
    assert model.calls == [(images, "pt")]
    assert torch.equal(result, torch.ones(1, 3, 8, 8))


def test_audio_feature_extractor_with_sampling_rate() -> None:
    """Raw audio input uses an explicit feature extractor and sampling rate."""

    np = pytest.importorskip("numpy")
    waveform = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    model = AudioModel()
    result = _coerce_input(model, waveform)
    assert len(model.calls) == 1
    assert model.calls[0][0] is waveform
    assert model.calls[0][1:] == (16_000, "pt")
    assert torch.equal(result, torch.ones(1, 3))


def test_unsupported_string_error_message() -> None:
    """Plain modules receive a helpful string-dispatch error."""

    model = nn.Linear(3, 2)
    with pytest.raises(TypeError, match="String input requires either model.to_tokens"):
        _coerce_input(model, "hello")


def test_trace_with_attached_tokenizer_string() -> None:
    """End-to-end capture accepts raw text when a tokenizer is attached."""

    model = TokenEmbeddingModel()
    trace = tl.trace(model, "hello world")
    try:
        assert len(trace.layer_list) > 0
        assert model.tokenizer_calls == [("hello world", "pt")]
    finally:
        trace.cleanup()


def test_fastlog_record_with_attached_tokenizer_string() -> None:
    """Fastlog one-shot recording accepts raw text when a tokenizer is attached."""

    model = TokenEmbeddingModel()
    recording = tl.fastlog.record(model, "hello world", default_op=True)
    assert recording.records
    assert model.tokenizer_calls == [("hello world", "pt")]


def test_trace_rerun_with_attached_tokenizer_string() -> None:
    """Trace rerun accepts new raw text input when a tokenizer is attached."""

    model = TokenEmbeddingModel()
    trace = tl.trace(model, "hello world", intervention_ready=True)
    try:
        trace.rerun("goodbye")
        assert model.tokenizer_calls == [("hello world", "pt"), ("goodbye", "pt")]
    finally:
        trace.cleanup()


@pytest.mark.slow
def test_trace_with_transformerlens_gpt2_string_if_available() -> None:
    """End-to-end TransformerLens GPT-2 capture accepts raw text when locally available."""

    hooked_transformer = pytest.importorskip("transformer_lens").HookedTransformer
    try:
        model = hooked_transformer.from_pretrained(
            "gpt2",
            device="cpu",
            n_ctx=8,
            first_n_layers=1,
            local_files_only=True,
        )
    except Exception as exc:
        pytest.skip(f"TransformerLens GPT-2 weights/tokenizer are not available locally: {exc}")

    trace = tl.trace(model, "hello world", layers_to_save=None)
    try:
        assert len(trace.layer_list) > 0
    finally:
        trace.cleanup()


def test_trace_with_pil_image_and_attached_processor() -> None:
    """End-to-end capture accepts a PIL image when an image processor is attached."""

    pil_image = pytest.importorskip("PIL.Image")

    class TinyVisionModel(nn.Module):
        """Tiny image model with an attached image processor."""

        def __init__(self) -> None:
            """Initialize convolution and processor bookkeeping."""

            super().__init__()
            self.conv = nn.Conv2d(3, 2, kernel_size=1)
            self.processor_calls = 0

        def image_processor(self, x: Any, *, return_tensors: str) -> PixelOutput:
            """Process PIL image input into a tensor.

            Parameters
            ----------
            x:
                PIL image input.
            return_tensors:
                Requested tensor backend.

            Returns
            -------
            PixelOutput
                Pixel tensor for the convolution.
            """

            del x, return_tensors
            self.processor_calls += 1
            return PixelOutput(torch.ones(1, 3, 8, 8))

        def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
            """Run the tiny image model.

            Parameters
            ----------
            pixel_values:
                Image tensor.

            Returns
            -------
            torch.Tensor
                Convolution output.
            """

            return self.conv(pixel_values)

    model = TinyVisionModel()
    trace = tl.trace(model, pil_image.new("RGB", (8, 8)))
    try:
        assert len(trace.layer_list) > 0
        assert model.processor_calls == 1
    finally:
        trace.cleanup()
