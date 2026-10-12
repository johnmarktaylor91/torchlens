"""Tests for the Phase 13 ``torchlens.compat.report`` API."""

from __future__ import annotations

import warnings
from collections.abc import Iterator

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.backends.tf._tf_compat import get_tf_capability_snapshot
from torchlens.compat import CompatReport, _report as compat_report_module, report
from torchlens.options import CaptureOptions
from torchlens.utils._torch_compat import get_torch_capability_snapshot
from torchlens.utils.rng import log_current_rng_states, set_rng_from_saved_states
from torchlens.utils.tensor_utils import tensor_nanequal

EXPECTED_COMPAT_ROW_KEYS = {
    "accelerate_cpu_disk_offload",
    "accelerate_device_map_auto",
    "bitsandbytes_8bit_4bit",
    "peft_lora_adapters",
    "data_parallel",
    "deepspeed",
    "device_context_factory",
    "device_mesh",
    "fp8_dtype",
    "distributed_data_parallel",
    "dtensor",
    "fsdp",
    "fx_graph_module",
    "hf_transformers",
    "lightning_training_step",
    "mechanical_belt",
    "multi_gpu_rng",
    "quantized_tensor",
    "pipeline_parallel",
    "single_thread_design",
    "tensor_parallel",
    "tied_parameters",
    "torch_capabilities",
    "torch_compile",
    "vmap_functorch",
}


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


class FalsePositiveOptimizedModule(nn.Module):
    """Model whose class name should not imply ``torch.compile``."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return the input unchanged.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Input tensor.
        """

        return x


def _reference_quantized_input() -> torch.Tensor:
    """Build the quantized compatibility-report fixture under a warning filter.

    Returns
    -------
    torch.Tensor
        Quantized reference input.
    """

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"torch\.quantize_per_tensor, torch\.quantize_per_channel.*deprecated",
            category=UserWarning,
        )
        return torch.quantize_per_tensor(
            torch.tensor([1.0, 2.0]), scale=0.1, zero_point=10, dtype=torch.quint8
        )


def _reference_models() -> Iterator[tuple[nn.Module, torch.Tensor]]:
    """Yield the required five Phase 13 reference model/input pairs.

    Yields
    ------
    tuple[nn.Module, torch.Tensor]
        Model/input pair for ``tl.compat.report``.
    """

    yield SmallCnn(), torch.randn(2, 1, 4, 4)
    yield MockPreTrainedModel(), torch.randn(2, 4)
    yield (QuantizedInputModel(), _reference_quantized_input())
    yield MultiGpuEmulationModel(), torch.randn(2, 4)
    yield FullyShardedDataParallel(), torch.randn(2, 4)


def test_report_runs_on_five_reference_models(monkeypatch: pytest.MonkeyPatch) -> None:
    """``tl.compat.report`` runs without executing or crashing on required references."""

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    reports = [report(model, x) for model, x in _reference_models()]

    assert len(reports) == 5
    assert all(isinstance(item, CompatReport) for item in reports)
    assert all({row.key for row in item.rows} == EXPECTED_COMPAT_ROW_KEYS for item in reports)
    assert reports[1].row("hf_transformers").detected is True
    assert reports[1].row("torch_capabilities").status in {"pass", "not_tested"}
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


def test_torch_compile_row_uses_feature_detection_not_class_name(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A user class named like Dynamo's wrapper must not trip the compile row."""

    monkeypatch.setattr(
        compat_report_module,
        "get_dynamo_optimized_module_type",
        lambda: None,
    )

    row = report(FalsePositiveOptimizedModule(), torch.randn(1)).row("torch_compile")

    assert row.detected is False
    assert row.status == "pass"


def test_report_surfaces_every_runtime_capability() -> None:
    """Compatibility report capability row stays lockstep with defined flags."""

    compat_report = report(SmallCnn(), torch.randn(2, 1, 4, 4))
    row = compat_report.row("torch_capabilities")
    expected = set(get_torch_capability_snapshot()) | set(get_tf_capability_snapshot())
    # The row cell carries the grouped absences-first summary; the FULL dump
    # moved to the detail accessor (sumfam wave-0 item 2), which is where the
    # lockstep-with-defined-flags tripwire now bites.
    surfaced = set(compat_report.capability_snapshot())
    assert surfaced == expected
    assert row.details.startswith("Runtime capabilities: ")
    assert "capabilities present" in row.details
    assert "capability_snapshot()" in row.details
    # Every ABSENT flag stays named in the row cell itself (absences-first).
    snapshot = compat_report.capability_snapshot()
    for name, available in snapshot.items():
        if not available:
            assert name in row.details


def test_report_detects_known_scope_and_broken_rows() -> None:
    """Wrappers with known semantics produce the expected row statuses."""

    data_parallel_report = report(nn.DataParallel(SmallCnn()), torch.randn(2, 1, 4, 4))
    fsdp_report = report(FullyShardedDataParallel(), torch.randn(2, 4))
    tied_report = report(TiedEmbeddingModel(), torch.tensor([1, 2, 3]))

    assert data_parallel_report.row("data_parallel").status == "pass"
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


def test_report_quantized_row_uses_shared_cycle_safe_tensor_walker() -> None:
    """Quantized input detection survives shared tensors and cyclic containers."""

    tensor = torch.quantize_per_tensor(
        torch.tensor([1.0, 2.0]), scale=0.1, zero_point=10, dtype=torch.quint8
    )
    payload: list[object] = [tensor, {"again": tensor}]
    payload.append(payload)

    compat_report = report(QuantizedInputModel(), payload)

    assert compat_report.row("quantized_tensor").status == "known_broken"


def test_rng_snapshot_uses_all_cuda_devices(monkeypatch: pytest.MonkeyPatch) -> None:
    """RNG helpers use all-device CUDA state APIs when CUDA RNG state is live.

    ``is_initialized`` is part of the fiction: the snapshot deliberately skips
    CUDA entirely until this process has actually initialized it, so that a
    CPU-only capture never force-initializes a visible device (see
    ``torchlens/utils/rng.py::_snapshot_cuda_rng_states``).
    """

    calls: list[str] = []
    fake_states = [torch.tensor([1], dtype=torch.uint8), torch.tensor([2], dtype=torch.uint8)]

    monkeypatch.setattr("torchlens.utils.tensor_utils._cuda_available", True)
    monkeypatch.setattr("torchlens.utils.rng._is_cuda_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: True)
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
    tl.trace(model, torch.randn(1), capture=CaptureOptions(layers_to_save="none"))

    assert model.factory_device_type == "meta"


def test_torch_compile_row_discloses_mid_forward_creation_exception() -> None:
    """The clean-preflight compile row must not overclaim stance coverage.

    The stance engages only when Dynamo is already imported at capture entry,
    so a compiled callable CREATED inside the forward (the process's first
    torch._dynamo import happening mid-capture) is bypassed-and-disclosed, not
    logged. The row's claim must carry that exception instead of stating that
    every compiled callable "runs eager and is logged" (b6-opus R16, measured:
    inline torch.compile interior absent while the row read pass/logged).
    """

    from torchlens.utils import _torch_compat

    if not _torch_compat.HAS_SET_STANCE:
        pytest.skip("set_stance unavailable on this torch")

    row = report(SmallCnn(), torch.randn(2, 1, 4, 4)).row("torch_compile")

    assert row.detected is False
    assert "created inside the forward" in row.details.lower()
    assert "dynamo_region_not_logged" in row.details
