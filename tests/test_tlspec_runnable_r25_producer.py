"""Round-25 runnable producer capability regressions."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any
from typing import Callable

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._io.runnable import build_sparse_run_descriptor, preflight_sparse_run_descriptor
from torchlens.intervention.types import FunctionRegistryKey
from torchlens.options import CaptureOptions
from torchlens.runnable import PathFaithfulness, ReadinessStatus, RunnableErrorCode

_CAPTURE = CaptureOptions(
    intervention_ready=True,
    capture_container_structure=True,
    cache=False,
)


class BareBatchNorm1d(nn.Module):
    """Bare affine ``BatchNorm1d`` module."""

    def __init__(self) -> None:
        """Initialize affine BatchNorm."""

        super().__init__()
        self.bn = nn.BatchNorm1d(4, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply BatchNorm."""

        return self.bn(x)


class BareBatchNorm2d(nn.Module):
    """Bare affine ``BatchNorm2d`` module."""

    def __init__(self) -> None:
        """Initialize affine BatchNorm."""

        super().__init__()
        self.bn = nn.BatchNorm2d(3, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply BatchNorm."""

        return self.bn(x)


class ResidualBatchNorm1d(nn.Module):
    """Small residual block containing affine ``BatchNorm1d``."""

    def __init__(self) -> None:
        """Initialize a linear residual block."""

        super().__init__()
        self.proj = nn.Linear(4, 4, bias=False)
        self.bn = nn.BatchNorm1d(4, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a residual block."""

        return torch.relu(self.bn(self.proj(x)) + x)


class ResidualBatchNorm2d(nn.Module):
    """Small ResNet-style block containing affine ``BatchNorm2d``."""

    def __init__(self) -> None:
        """Initialize a convolutional residual block."""

        super().__init__()
        self.conv = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(3, affine=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a residual block."""

        return torch.relu(self.bn(self.conv(x)) + x)


class TensorPropertyModel(nn.Module):
    """Return one tensor property plus a scalar."""

    def __init__(self, property_name: str) -> None:
        """Store the property name to read."""

        super().__init__()
        self.property_name = property_name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Read a safe tensor property."""

        return getattr(x, self.property_name) + 1


class DataCopyLabeledRhs(nn.Module):
    """Write a labelled RHS through an unlabelled ``.data`` copy receiver."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a tensor mutated through ``.data.copy_``."""

        h = torch.zeros_like(x)
        h.data.copy_(x * 3)
        return h


class DataIaddLabeledRhs(nn.Module):
    """Write a labelled RHS through an unlabelled ``.data`` augmented assignment."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a tensor mutated through ``.data +=``."""

        h = torch.zeros_like(x)
        h.data += x * 3
        return h


def _clone_state(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return a detached clone of ``model`` state."""

    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def _expected_output(
    model_factory: Callable[[], nn.Module],
    state: dict[str, torch.Tensor],
    training: bool,
    x: torch.Tensor,
) -> torch.Tensor:
    """Return output from a fresh identically initialized model."""

    model = model_factory()
    model.load_state_dict(state)
    model.train(training)
    with torch.no_grad():
        return model(x.clone())


def _save_runnable(model: nn.Module, x: torch.Tensor, path: Path) -> tuple[Any, Path]:
    """Trace, save, and return one runnable artifact with embedded state."""

    trace = tl.trace(model, x.clone(), capture=_CAPTURE)
    trace.save(path, level="runnable", include_weights=True)
    return trace, path


def _assert_batchnorm_param_bindings(trace: Any, weight_name: str, bias_name: str) -> None:
    """Assert affine BatchNorm arguments bind weight and bias to the correct slots."""

    descriptor = build_sparse_run_descriptor(trace)
    batchnorm_calls = [
        call
        for call in descriptor.calls
        if any("batchnorm" in label.lower() for label in call.op_labels)
    ]
    assert len(batchnorm_calls) == 1
    by_path = {
        argument.argument_path: argument.slot_id for argument in batchnorm_calls[0].tensor_arguments
    }
    assert by_path[("args", 1)] == f"state:{weight_name}"
    assert by_path[("args", 2)] == f"state:{bias_name}"


@pytest.mark.parametrize(
    ("model_factory", "shape", "weight_name", "bias_name"),
    (
        (BareBatchNorm1d, (6, 4), "bn.weight", "bn.bias"),
        (BareBatchNorm2d, (4, 3, 5, 5), "bn.weight", "bn.bias"),
        (ResidualBatchNorm1d, (6, 4), "bn.weight", "bn.bias"),
        (ResidualBatchNorm2d, (4, 3, 5, 5), "bn.weight", "bn.bias"),
    ),
)
@pytest.mark.parametrize("training", (True, False))
def test_affine_batchnorm_saves_runs_verified_and_value_correct(
    tmp_path: Path,
    model_factory: Callable[[], nn.Module],
    shape: tuple[int, ...],
    weight_name: str,
    bias_name: str,
    training: bool,
) -> None:
    """Affine BatchNorm weight/bias bind by identity and replay correctly."""

    torch.manual_seed(2501)
    model = model_factory()
    model.train(training)
    state = _clone_state(model)
    capture_x = torch.randn(*shape)
    changed_x = torch.randn(*shape)
    trace, path = _save_runnable(model, capture_x, tmp_path / "bn.tlspec")

    _assert_batchnorm_param_bindings(trace, weight_name, bias_name)
    loaded = tl.load(path)
    for runtime_x in (capture_x, changed_x):
        result = loaded.run(runtime_x.clone(), seed=0, on_divergence="return_diverged")
        assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
        expected = _expected_output(model_factory, state, training, runtime_x)
        torch.testing.assert_close(result.output, expected)


_COMPLEX_2X3 = torch.tensor(
    [[1 + 1j, 2 + 2j, 3 + 3j], [4 + 4j, 5 + 5j, 6 + 6j]], dtype=torch.complex64
)


@pytest.mark.parametrize(
    ("property_name", "x"),
    (
        ("T", torch.arange(6.0).reshape(2, 3)),
        ("mT", torch.arange(6.0).reshape(2, 3)),
        ("real", torch.tensor([1 + 2j, 3 + 4j])),
        # r45: ``.H`` / ``.mH`` are the pure-view siblings of ``.T`` / ``.mT`` (the r44
        # corr1_1 / secF_1 over-deny). Cover both real and complex sources; ``.imag`` is
        # complex-only. Non-square inputs sidestep the square/real capture quirk noted in
        # the finding.
        ("H", torch.arange(6.0).reshape(2, 3)),
        ("mH", torch.arange(6.0).reshape(2, 3)),
        ("H", _COMPLEX_2X3),
        ("mH", _COMPLEX_2X3),
        ("real", _COMPLEX_2X3),
        ("imag", _COMPLEX_2X3),
        ("imag", torch.tensor([1 + 2j, 3 + 4j])),
    ),
)
def test_safe_tensor_properties_resolve_and_run_verified(
    tmp_path: Path,
    property_name: str,
    x: torch.Tensor,
) -> None:
    """Safe tensor properties replay without custom import trust."""

    trace = tl.trace(TensorPropertyModel(property_name), x.clone(), capture=_CAPTURE)
    path = tmp_path / f"{property_name}.tlspec"
    trace.save(path, level="runnable")
    loaded = tl.load(path)
    result = loaded.run(x.clone(), seed=0, on_divergence="return_diverged")

    assert loaded.readiness.status is ReadinessStatus.READY
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    torch.testing.assert_close(result.output, getattr(x, property_name) + 1)


def test_foreign_getset_descriptor_still_fails_closed(tmp_path: Path) -> None:
    """A custom getset descriptor key without a safe tensor-property name is denied."""

    trace = tl.trace(TensorPropertyModel("T"), torch.ones(2, 3), capture=_CAPTURE)
    descriptor = build_sparse_run_descriptor(trace)
    entry = descriptor.callable_registry[0]
    foreign = replace(
        descriptor,
        callable_registry=(
            replace(
                entry,
                key=FunctionRegistryKey(
                    "custom",
                    "getset_descriptor.__get__",
                    "dunder",
                    import_path=None,
                ),
            ),
            *descriptor.callable_registry[1:],
        ),
    )

    report, attachments = preflight_sparse_run_descriptor(foreign)

    assert report.status is ReadinessStatus.UNAVAILABLE
    assert attachments is None
    assert RunnableErrorCode.UNTRUSTED_CUSTOM_IMPORT in {
        diagnostic.code for diagnostic in report.diagnostics
    }


@pytest.mark.parametrize("model_factory", (DataCopyLabeledRhs, DataIaddLabeledRhs))
def test_labeled_rhs_data_write_saves_and_runs_unverifiable(
    tmp_path: Path,
    model_factory: Callable[[], nn.Module],
) -> None:
    """A labelled-RHS ``.data`` write saves and fails closed at run faithfulness."""

    x = torch.ones(2, 3)
    with pytest.warns(UserWarning, match="no graph/source provenance"):
        trace = tl.trace(model_factory(), x, capture=_CAPTURE)
    path = tmp_path / "data_write.tlspec"
    trace.save(path, level="runnable")
    result = tl.load(path).run(torch.full((2, 3), 2.0), seed=0, on_divergence="return_diverged")

    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    torch.testing.assert_close(result.output, torch.full((2, 3), 6.0))
