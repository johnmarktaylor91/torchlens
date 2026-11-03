"""Round-25 runnable producer capability regressions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens._io.runnable import build_sparse_run_descriptor, preflight_sparse_run_descriptor
from torchlens._runnable import PathFaithfulness, ReadinessStatus, RunnableErrorCode
from torchlens.intervention.types import FunctionRegistryKey
from torchlens.options import CaptureOptions

_CAPTURE = CaptureOptions(
    replayable=True,
    capture_container_structure=True,
    cache=False,
)


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


_COMPLEX_2X3 = torch.tensor(
    [[1 + 1j, 2 + 2j, 3 + 3j], [4 + 4j, 5 + 5j, 6 + 6j]], dtype=torch.complex64
)


# shrink-r4: rows kept carry a sole library arc or a campaign kill; the others were proven redundant (slice 1 evidence)
@pytest.mark.parametrize(
    ("property_name", "x"),
    (
        # r45: ``.H`` / ``.mH`` are the pure-view siblings of ``.T`` / ``.mT`` (the r44
        # corr1_1 / secF_1 over-deny). Cover both real and complex sources; ``.imag`` is
        # complex-only. Non-square inputs sidestep the square/real capture quirk noted in
        # the finding.
        ("imag", _COMPLEX_2X3),
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
    result = loaded.run(x.clone(), seed=0, on_divergence="return")

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
    result = tl.load(path).run(torch.full((2, 3), 2.0), seed=0, on_divergence="return")

    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    torch.testing.assert_close(result.output, torch.full((2, 3), 6.0))
