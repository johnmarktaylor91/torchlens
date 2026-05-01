"""Tests for the Phase 13 ``torchlens.compat.report`` API."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.compat import CompatReport, report
from torchlens.options import CaptureOptions
from torchlens.utils.rng import log_current_rng_states, set_rng_from_saved_states
from torchlens.utils.tensor_utils import tensor_nanequal


class SmallCnn(nn.Module):
    """Tiny convolutional reference model."""

    def __init__(self) -> None:
        """Initialize layers."""

        super().__init__()
        self.conv = nn.Conv2d(1, 2, kernel_size=3)
        self.head = nn.Linear(8, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run one forward pass.

        Parameters
        ----------
        x:
            Input image batch.

        Returns
        -------
        torch.Tensor
            Logits.
        """

        hidden = torch.relu(self.conv(x))
        return self.head(hidden.flatten(1))


class MockPreTrainedModel(nn.Module):
    """Hugging Face-like model without importing transformers."""

    __module__ = "transformers.modeling_utils"

    def __init__(self) -> None:
        """Initialize the mock model."""

        super().__init__()
        self.config = {"model_type": "mock"}
        self.proj = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run one forward pass.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Projected tensor.
        """

        return self.proj(x)


class QuantizedInputModel(nn.Module):
    """Reference model used with a quantized input tensor."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a dequantized tensor for ordinary execution.

        Parameters
        ----------
        x:
            Quantized or floating tensor.

        Returns
        -------
        torch.Tensor
            Floating tensor.
        """

        if x.is_quantized:
            return x.dequantize()
        return x


class MultiGpuEmulationModel(nn.Module):
    """CPU-only model used while monkeypatching CUDA device count."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run one forward pass.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Shifted tensor.
        """

        return x + 1


class FullyShardedDataParallel(nn.Module):
    """FSDP-like placeholder that does not require distributed initialization."""

    __module__ = "torch.distributed.fsdp.fully_sharded_data_parallel"

    def __init__(self) -> None:
        """Initialize the placeholder wrapper."""

        super().__init__()
        self.module = nn.Linear(4, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Delegate to the wrapped module.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Wrapped module output.
        """

        return self.module(x)


class TiedEmbeddingModel(nn.Module):
    """Model with shared parameter objects."""

    def __init__(self) -> None:
        """Initialize tied embeddings."""

        super().__init__()
        self.input_embedding = nn.Embedding(8, 4)
        self.output_embedding = self.input_embedding

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run one forward pass.

        Parameters
        ----------
        x:
            Token ids.

        Returns
        -------
        torch.Tensor
            Embedding output.
        """

        return self.output_embedding(x)


class DeviceContextFactoryModel(nn.Module):
    """Model that creates a factory tensor under ``torch.device``."""

    def __init__(self) -> None:
        """Initialize observed device storage."""

        super().__init__()
        self.factory_device_type = "unset"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Create a tensor inside a DeviceContext during active logging.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Shifted input tensor.
        """

        with torch.device("meta"):
            probe = torch.empty((1,))
        self.factory_device_type = probe.device.type
        return x + 1


def _reference_models() -> Iterator[tuple[nn.Module, torch.Tensor]]:
    """Yield the required five Phase 13 reference model/input pairs.

    Yields
    ------
    tuple[nn.Module, torch.Tensor]
        Model/input pair for ``tl.compat.report``.
    """

    yield SmallCnn(), torch.randn(2, 1, 4, 4)
    yield MockPreTrainedModel(), torch.randn(2, 4)
    yield (
        QuantizedInputModel(),
        torch.quantize_per_tensor(
            torch.tensor([1.0, 2.0]), scale=0.1, zero_point=10, dtype=torch.quint8
        ),
    )
    yield MultiGpuEmulationModel(), torch.randn(2, 4)
    yield FullyShardedDataParallel(), torch.randn(2, 4)


def test_report_runs_on_five_reference_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """``tl.compat.report`` runs without executing or crashing on required references."""

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    reports = [report(model, x) for model, x in _reference_models()]

    assert len(reports) == 5
    assert all(isinstance(item, CompatReport) for item in reports)
    assert all(len(item.rows) == 17 for item in reports)
    assert reports[1].row("hf_transformers").detected is True
    assert reports[2].row("quantized_tensor").status == "known_broken"
    assert reports[3].row("multi_gpu_rng").detected is True
    assert reports[4].row("fsdp").status == "scope"


def test_report_renderers_include_truth_table_rows() -> None:
    """``show`` and ``to_markdown`` expose stable truth-table information."""

    compat_report = report(SmallCnn(), torch.randn(2, 1, 4, 4))

    text_table = compat_report.show()
    markdown_table = compat_report.to_markdown()

    assert "HF Transformers wrapper" in text_table
    assert "Single-thread design" in text_table
    assert "| Row | Status | Severity | Detected | Details | Suggestion |" in markdown_table
    assert "`pass`" in markdown_table


def test_report_detects_known_scope_and_broken_rows() -> None:
    """Wrappers with known semantics produce the expected row statuses."""

    data_parallel_report = report(nn.DataParallel(SmallCnn()), torch.randn(2, 1, 4, 4))
    fsdp_report = report(FullyShardedDataParallel(), torch.randn(2, 4))
    tied_report = report(TiedEmbeddingModel(), torch.tensor([1, 2, 3]))

    assert data_parallel_report.row("data_parallel").status == "known_broken"
    assert fsdp_report.row("fsdp").status == "scope"
    assert tied_report.row("tied_parameters").detected is True
    assert tied_report.row("tied_parameters").status == "pass"


def test_quantized_tensor_nanequal_no_longer_crashes() -> None:
    """Quantized tensors are compared without calling unsupported floating ops."""

    left = torch.quantize_per_tensor(
        torch.tensor([1.0, 2.0]), scale=0.1, zero_point=10, dtype=torch.quint8
    )
    right = torch.quantize_per_tensor(
        torch.tensor([1.0, 2.0]), scale=0.1, zero_point=10, dtype=torch.quint8
    )
    mismatch = torch.quantize_per_tensor(
        torch.tensor([1.0, 3.0]), scale=0.1, zero_point=10, dtype=torch.quint8
    )

    assert tensor_nanequal(left, right)
    assert not tensor_nanequal(left, mismatch)


def test_rng_snapshot_uses_all_cuda_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    """RNG helpers use all-device CUDA state APIs when CUDA is available."""

    calls: list[str] = []
    fake_states = [torch.tensor([1], dtype=torch.uint8), torch.tensor([2], dtype=torch.uint8)]

    monkeypatch.setattr("torchlens.utils.tensor_utils._cuda_available", True)
    monkeypatch.setattr("torchlens.utils.rng._is_cuda_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_rng_state_all", lambda: fake_states)

    def fake_set_rng_state_all(states: list[torch.Tensor]) -> None:
        """Record all-device restore calls.

        Parameters
        ----------
        states:
            CUDA RNG states.
        """

        assert states == fake_states
        calls.append("all")

    monkeypatch.setattr(torch.cuda, "set_rng_state_all", fake_set_rng_state_all)

    states = log_current_rng_states(torch_only=True)
    set_rng_from_saved_states(states)

    assert states["torch_cuda_all"] == fake_states
    assert calls == ["all"]


def test_device_context_factory_injection_during_active_logging() -> None:
    """Factory functions honor ``torch.device`` contexts while TorchLens is logging."""

    model = DeviceContextFactoryModel()
    tl.log_forward_pass(model, torch.randn(1), capture=CaptureOptions(layers_to_save="none"))

    assert model.factory_device_type == "meta"
