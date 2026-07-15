"""Round-20 state TOCTOU sampling and buffer-slot attestation regressions."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.options import CaptureOptions
from torchlens.runnable import NumericAttestationStatus, PathFaithfulness

_CAPTURE = CaptureOptions(
    intervention_ready=True,
    capture_container_structure=True,
    cache=False,
)


def _save(model: nn.Module, capture_input: torch.Tensor, path: Path, **save_kwargs: object) -> Path:
    """Capture, save, and return one runnable ``.tlspec`` path.

    Parameters
    ----------
    model:
        Module to trace.
    capture_input:
        Tensor input used for capture.
    path:
        Destination path for the runnable artifact.
    save_kwargs:
        Additional keyword arguments forwarded to ``Trace.save``.

    Returns
    -------
    Path
        Saved runnable artifact path.
    """

    trace = tl.trace(model, capture_input, capture=_CAPTURE)
    trace.save(path, level="runnable", **save_kwargs)
    return path


def _run(path: Path, x: torch.Tensor) -> tl.RunResult:
    """Load and run one sparse runnable artifact.

    Parameters
    ----------
    path:
        Runnable artifact path.
    x:
        Runtime tensor input.

    Returns
    -------
    tl.RunResult
        Sparse replay result with divergence returned in the report.
    """

    return tl.load(path).run(inputs=x.clone(), seed=0, on_divergence="return_diverged")


def _state_clone(model: nn.Module) -> dict[str, torch.Tensor]:
    """Return a detached clone of a module state dict.

    Parameters
    ----------
    model:
        Module whose state should be cloned.

    Returns
    -------
    dict[str, torch.Tensor]
        Detached tensor-only state clone.
    """

    return {name: value.detach().clone() for name, value in model.state_dict().items()}


class ParamMutateRestore(nn.Module):
    """Mutate a parameter, consume it, then restore its bytes before return."""

    def __init__(self) -> None:
        """Initialize the module with one parameter."""

        super().__init__()
        self.w = nn.Parameter(torch.tensor([2.0, 3.0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a value that depends on a transient parameter mutation.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output using the transient parameter value.
        """

        with torch.no_grad():
            self.w.add_(10.0)
        out = x * self.w
        with torch.no_grad():
            self.w.sub_(10.0)
        return out


class BufferHostMutateRestore(nn.Module):
    """Mutate a registered buffer through a host alias, consume it, then restore it."""

    npb: np.ndarray

    def __init__(self) -> None:
        """Initialize the module with one persistent buffer and a host alias."""

        super().__init__()
        self.register_buffer("b", torch.tensor([2.0, 3.0]))
        self.npb = self.b.detach().numpy()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return a value that depends on a transient host-side buffer mutation.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output using the transient buffer value.
        """

        self.npb[0] += 10.0
        out = x * self.b
        self.npb[0] -= 10.0
        return out


class PureJournaledBufferAddSub(nn.Module):
    """Use two journaled buffer ops and return the transient buffer value."""

    def __init__(self) -> None:
        """Initialize the module with one persistent buffer."""

        super().__init__()
        self.register_buffer("b", torch.tensor([2.0, 3.0]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return an output that depends on a pure journaled intermediate buffer value.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Output using the journaled buffer value.
        """

        self.b.add_(10.0)
        out = x * self.b
        self.b.sub_(10.0)
        return out


class PlainBatchNormRunningStat(nn.Module):
    """Plain BatchNorm whose running stats are pure journaled buffer writes."""

    def __init__(self) -> None:
        """Initialize one affine-free BatchNorm layer."""

        super().__init__()
        self.bn = nn.BatchNorm1d(4, affine=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run BatchNorm in the module's current train/eval mode.

        Parameters
        ----------
        x:
            Input batch.

        Returns
        -------
        torch.Tensor
            BatchNorm output.
        """

        return self.bn(x)


def _fresh_journaled_output(
    model_factory: type[nn.Module],
    state_dict: dict[str, torch.Tensor],
    x: torch.Tensor,
    *,
    train: bool = False,
) -> torch.Tensor:
    """Run a fresh module from captured state for the oracle output.

    Parameters
    ----------
    model_factory:
        Module class to instantiate.
    state_dict:
        Captured state dict clone.
    x:
        Input tensor.
    train:
        Whether to put the fresh module in train mode.

    Returns
    -------
    torch.Tensor
        Fresh forward output.
    """

    fresh = model_factory()
    fresh.load_state_dict(state_dict)
    fresh.train(mode=train)
    with torch.no_grad():
        return fresh(x.clone())


@pytest.mark.smoke
def test_param_mutate_consume_restore_is_unverifiable(tmp_path: Path) -> None:
    """A transient parameter mutation consumed by a traced op must fail closed."""

    capture_x = torch.tensor([2.0, 4.0])
    path = _save(ParamMutateRestore(), capture_x, tmp_path / "param.tlspec", include_weights=True)

    result = _run(path, capture_x)

    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert result.report.numeric_attestation is not NumericAttestationStatus.ATTESTED


@pytest.mark.smoke
def test_buffer_host_mutate_consume_restore_is_unverifiable(tmp_path: Path) -> None:
    """A transient host-side buffer mutation consumed by a traced op must fail closed."""

    capture_x = torch.tensor([2.0, 4.0])
    path = _save(
        BufferHostMutateRestore(),
        capture_x,
        tmp_path / "buffer.tlspec",
        include_weights=True,
    )

    result = _run(path, capture_x)

    assert result.report.path_faithfulness is PathFaithfulness.UNVERIFIABLE
    assert result.report.numeric_attestation is not NumericAttestationStatus.ATTESTED


@pytest.mark.smoke
def test_plain_batchnorm_running_stat_stays_verified(tmp_path: Path) -> None:
    """Pure journaled BatchNorm running-stat writes stay verified and value-correct."""

    torch.manual_seed(0)
    model = PlainBatchNormRunningStat()
    model.train()
    state = _state_clone(model)
    capture_x = torch.randn(8, 4)
    path = _save(model, capture_x, tmp_path / "bn.tlspec", include_weights=True)

    result = _run(path, capture_x)
    expected = _fresh_journaled_output(PlainBatchNormRunningStat, state, capture_x, train=True)

    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    torch.testing.assert_close(result.output, expected)


@pytest.mark.smoke
def test_pure_journaled_buffer_add_sub_include_activations_does_not_raise(
    tmp_path: Path,
) -> None:
    """Buffer-slot activation payloads do not raise on a correct journaled replay."""

    capture_x = torch.tensor([2.0, 4.0])
    model = PureJournaledBufferAddSub()
    state = _state_clone(model)
    path = _save(
        model,
        capture_x,
        tmp_path / "journaled_acts.tlspec",
        include_weights=True,
        include_activations=True,
    )

    result = _run(path, capture_x)
    expected = _fresh_journaled_output(PureJournaledBufferAddSub, state, capture_x)

    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert result.report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE
    torch.testing.assert_close(result.output, expected)
