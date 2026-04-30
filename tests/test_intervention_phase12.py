"""Phase 12 append rerun tests."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens import RunState
from torchlens.intervention.errors import (
    AppendBatchDependenceError,
    AppendMismatchError,
    BatchNormTrainModeWarning,
)


class _LinearRelu(nn.Module):
    """Small deterministic model for append tests."""

    def __init__(self) -> None:
        """Initialize the linear layer."""

        super().__init__()
        self.linear = nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a linear layer followed by relu and add.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Model output.
        """

        return torch.relu(self.linear(x)) + 1


class _AddOne(nn.Module):
    """Parameter-free shape-preserving model."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add one to the input tensor.

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


class _BranchModel(nn.Module):
    """Model whose topology changes with input sign."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Choose relu or tanh from input sign.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Branch output.
        """

        if bool(torch.sum(x) > 0):
            return torch.relu(x)
        return torch.tanh(x)


class _BatchNormModel(nn.Module):
    """Model with train-mode BatchNorm."""

    def __init__(self) -> None:
        """Initialize BatchNorm."""

        super().__init__()
        self.bn = nn.BatchNorm1d(3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply BatchNorm.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Batch-normalized tensor.
        """

        return self.bn(x)


def _capture(model: nn.Module, x: torch.Tensor) -> tl.ModelLog:
    """Capture an intervention-ready log.

    Parameters
    ----------
    model:
        Model to capture.
    x:
        Input tensor.

    Returns
    -------
    tl.ModelLog
        Captured log.
    """

    return tl.log_forward_pass(model, x, vis_opt="none", intervention_ready=True)


def _first_batch_activation(log: tl.ModelLog) -> torch.Tensor:
    """Return the first saved activation with a batch dimension.

    Parameters
    ----------
    log:
        Model log to inspect.

    Returns
    -------
    torch.Tensor
        First activation with at least one dimension.
    """

    for layer in log.layer_list:
        if isinstance(layer.activation, torch.Tensor) and layer.activation.ndim > 0:
            return layer.activation
    raise AssertionError("no batch activation found")


@pytest.mark.smoke
def test_append_success_grows_batch_and_sets_state() -> None:
    """Compatible append concatenates saved activations and records state."""

    torch.manual_seed(0)
    model = _LinearRelu()
    model.eval()
    log = _capture(model, torch.randn(2, 3))
    original_history_len = len(log.operation_history)

    result = log.rerun(model, torch.randn(3, 3), append=True)

    assert result is log
    assert log.is_appended is True
    assert log._append_sequence_id == 1
    assert log.run_state is RunState.APPENDED
    assert log.last_run_ctx["engine"] == "append"
    assert log.operation_history[-1]["op"] == "append"
    assert len(log.operation_history) == original_history_len + 1
    assert _first_batch_activation(log).shape[0] == 5


def test_append_topology_mismatch_raises() -> None:
    """Append rejects changed topology/site labels."""

    log = _capture(_BranchModel(), torch.ones(2, 3))

    with pytest.raises(AppendMismatchError):
        log.rerun(_BranchModel(), -torch.ones(2, 3), append=True)


def test_append_shape_mismatch_raises() -> None:
    """Append rejects shape changes outside the batch dimension."""

    model = _AddOne()
    log = _capture(model, torch.ones(2, 3))

    with pytest.raises(AppendMismatchError):
        log.rerun(model, torch.ones(2, 4), append=True)


def test_append_dtype_mismatch_raises() -> None:
    """Append rejects dtype changes."""

    model = _AddOne()
    log = _capture(model, torch.ones(2, 3, dtype=torch.float32))

    with pytest.raises(AppendMismatchError):
        log.rerun(model, torch.ones(2, 3, dtype=torch.float64), append=True)


def test_append_batch_dependent_helper_rejected_after_clean_rerun() -> None:
    """Append rejects helpers declared batch-dependent."""

    model = _LinearRelu()
    model.eval()
    x = torch.randn(2, 3)
    log = _capture(model, x)
    log.attach_hooks(tl.func("relu"), tl.resample_ablate(source=torch.zeros(2, 3), seed=1))
    log.rerun(model, x)

    with pytest.raises(AppendBatchDependenceError):
        log.rerun(model, torch.randn(2, 3), append=True)


def test_append_batchnorm_train_mode_warns() -> None:
    """Append warns but does not reject train-mode BatchNorm."""

    model = _BatchNormModel()
    model.train()
    log = _capture(model, torch.randn(2, 3))

    with pytest.warns(BatchNormTrainModeWarning):
        log.rerun(model, torch.randn(2, 3), append=True)


def test_append_recipe_stale_rejected_before_capture() -> None:
    """Append requires propagated activations to match the current recipe."""

    model = _LinearRelu()
    model.eval()
    log = _capture(model, torch.randn(2, 3))
    log.attach_hooks(tl.func("relu"), tl.zero_ablate())

    with pytest.raises(AppendMismatchError, match="recipe is stale"):
        log.rerun(model, torch.randn(2, 3), append=True)


def test_append_state_round_trips_bundle_and_tlspec(tmp_path: Path) -> None:
    """Append provenance survives ordinary bundles and tlspec metadata."""

    model = _LinearRelu()
    model.eval()
    log = _capture(model, torch.randn(2, 3))
    log.attach_hooks(tl.func("relu"), tl.zero_ablate())
    log.rerun(model, torch.randn(2, 3))
    log.rerun(model, torch.randn(1, 3), append=True)

    bundle_path = tmp_path / "append_bundle"
    tl.save(log, bundle_path)
    loaded = tl.load(bundle_path)
    assert loaded.is_appended is True
    assert loaded._append_sequence_id == log._append_sequence_id
    assert loaded.operation_history[-1]["op"] == "append"

    spec_path = tmp_path / "append.tlspec"
    log.save_intervention(spec_path, level="audit")
    spec = tl.load_intervention_spec(spec_path)
    append_state = spec.metadata["append_state"]
    assert append_state["is_appended"] is True
    assert append_state["append_sequence_id"] == log._append_sequence_id
    assert append_state["operation_history"][-1]["op"] == "append"
