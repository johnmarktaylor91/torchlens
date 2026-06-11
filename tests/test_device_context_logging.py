"""DeviceContext injection must work under ACTIVE logging, not just the fast path.

Bug DEVICE-CONTEXT-LOGGING: when a ``TorchFunctionMode`` device context (e.g.
``with torch.device("meta"):``, used internally by HuggingFace
``from_pretrained``) is active, factory functions like ``torch.zeros`` must
produce tensors on the context device even while a TorchLens trace is being
captured. The wrapper bypasses PyTorch's C-level mode dispatch, so TorchLens
replicates the device-kwarg injection itself (``_maybe_inject_device_kwarg``
in ``backends/torch/wrappers.py``, applied on BOTH the fast path and the
active-logging path) — and the capture pipeline must then tolerate the
resulting data-less meta tensors instead of crashing on content operations
(hashing, equality) that meta tensors cannot support.

These tests assert traced behavior matches untraced behavior exactly.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl


class _MetaFactoryConsumerModel(nn.Module):
    """Forward enters a meta-device context, calls a factory, and consumes the result.

    Mirrors the HuggingFace ``from_pretrained`` pattern: model code runs
    factory functions inside ``with torch.device("meta")`` and keeps operating
    on the resulting meta tensors during a forward pass that TorchLens is
    actively logging. The meta tensors are NOT part of the model output.
    """

    def __init__(self) -> None:
        super().__init__()
        self.seen_devices: list[str] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.device("meta"):
            z = torch.zeros(3, 4)
        z2 = z + 1  # consume the meta tensor mid-forward (stays meta)
        self.seen_devices.append(z.device.type)
        self.seen_devices.append(z2.device.type)
        return x * 2


class _MetaFactoryOutputModel(nn.Module):
    """Forward returns the meta factory tensor as part of the model output."""

    def __init__(self) -> None:
        super().__init__()
        self.seen_devices: list[str] = []

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        with torch.device("meta"):
            z = torch.zeros(3, 4)
        self.seen_devices.append(z.device.type)
        return x * 2, z


class _CpuFactoryModel(nn.Module):
    """Forward calls a factory function with no device context active."""

    def __init__(self) -> None:
        super().__init__()
        self.seen_devices: list[str] = []

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = torch.zeros(3, 4)
        self.seen_devices.append(z.device.type)
        return x * 2, z


def _layer_by_func(log, func_substring: str):
    """Return the first layer whose label contains ``func_substring``."""
    labels = [name for name in log.layer_labels if func_substring in name]
    assert labels, f"no layer matching {func_substring!r} in {list(log.layer_labels)}"
    return log[labels[0]]


@pytest.mark.smoke
def test_meta_device_context_untraced_baseline() -> None:
    """Sanity: without TorchLens, the context produces meta tensors."""
    model = _MetaFactoryConsumerModel()
    model(torch.randn(2, 4))
    assert model.seen_devices == ["meta", "meta"]


@pytest.mark.smoke
def test_meta_device_context_factory_under_active_logging() -> None:
    """Factory + consumption inside ``torch.device('meta')`` works under trace.

    Pre-fix failure modes (both during ACTIVE logging):
      - saving the zeros activation crashed in content hashing
        (``Cannot copy out of meta tensor; no data!``)
      - the ``z + 1`` op crashed in mutation detection
        (``aten::equal ... Meta tensors``)
    """
    model = _MetaFactoryConsumerModel()
    x = torch.randn(2, 4)

    log = tl.trace(model, x)

    # Runtime behavior matches untraced: factory output and downstream op
    # both landed on the meta device.
    assert model.seen_devices == ["meta", "meta"]

    # The non-meta branch of the graph is captured intact.
    mul_layer = _layer_by_func(log, "mul")
    assert mul_layer.out.device.type == "cpu"
    assert torch.allclose(mul_layer.out, x * 2)


@pytest.mark.smoke
def test_meta_device_context_with_save_arg_values() -> None:
    """The arg-snapshot/variation-tracking path also tolerates meta tensors."""
    from torchlens.options import CaptureOptions

    model = _MetaFactoryConsumerModel()
    tl.trace(model, torch.randn(2, 4), capture=CaptureOptions(save_arg_values=True))
    assert model.seen_devices == ["meta", "meta"]


def test_meta_factory_output_metadata_with_selective_save() -> None:
    """A meta factory tensor in the model output records shape/dtype metadata.

    With the meta op excluded from activation saving, the trace completes and
    the captured metadata reflects the post-injection (meta-device) tensor.
    """
    model = _MetaFactoryOutputModel()
    log = tl.trace(model, torch.randn(2, 4), save=tl.func("mul"))

    assert model.seen_devices == ["meta"]
    zeros_layer = _layer_by_func(log, "zeros")
    # Shape/dtype come from the post-injection meta tensor: torch.zeros(3, 4)
    # under the meta context (a CPU tensor would have identical shape only if
    # injection worked — device parity is asserted via seen_devices above).
    assert zeros_layer.shape == (3, 4)
    assert zeros_layer.dtype == torch.float32
    assert not zeros_layer.has_saved_activation


def test_meta_factory_as_model_output_full_save() -> None:
    """Strictest form of the repro: meta factory tensor returned as output, full save.

    Requires meta-aware ``tensor_nanequal`` (meta/meta tensors compare equal once
    shape and dtype match — there is no data to differ), exercised here through
    postprocess ``_add_output_layers`` saved-activation comparison.
    """
    model = _MetaFactoryOutputModel()
    log = tl.trace(model, torch.randn(2, 4))

    assert model.seen_devices == ["meta"]
    zeros_layer = _layer_by_func(log, "zeros")
    # Captured metadata records the post-injection device.
    assert zeros_layer.out.device.type == "meta"
    assert zeros_layer.shape == (3, 4)


@pytest.mark.smoke
def test_cpu_default_unaffected_under_active_logging() -> None:
    """Without a device context, factory functions stay on CPU under trace."""
    model = _CpuFactoryModel()
    x = torch.randn(2, 4)

    log = tl.trace(model, x)

    assert model.seen_devices == ["cpu"]
    zeros_layer = _layer_by_func(log, "zeros")
    assert zeros_layer.out.device.type == "cpu"
    assert torch.equal(zeros_layer.out, torch.zeros(3, 4))


def test_meta_device_context_fast_path_with_torch_wrapped() -> None:
    """Fast path (wrapped but not logging) keeps injecting the device kwarg."""
    # Ensure torch is wrapped (lazy wrapping happens on first capture).
    tl.trace(_CpuFactoryModel(), torch.randn(2, 4))

    with torch.device("meta"):
        z = torch.zeros(2, 2)
    assert z.device.type == "meta"

    # Explicit device wins over the context, traced or not.
    with torch.device("meta"):
        c = torch.zeros(2, 2, device="cpu")
    assert c.device.type == "cpu"
