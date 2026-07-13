"""Non-tensor Python model-input contract for sparse runnable execution.

These tests lock the honesty rule that a sparse runnable ``run`` may never report
``VERIFIED``/``ATTESTED`` after silently replaying the recorded taken path for a
*changed* non-tensor Python input (bool/int/float). A changed control input can
make the recorded path numerically wrong, so it must diverge (or, under
``return_diverged``, be reported diverged/poisoned and never attested) while an
unchanged input still verifies and attests exactly as before.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.errors import PathDivergenceError
from torchlens.options import CaptureOptions
from torchlens.runnable import (
    DivergencePolicy,
    NumericAttestationStatus,
    PathFaithfulness,
)


class BoolBranch(nn.Module):
    """Select addition or subtraction from a non-tensor Python bool."""

    def forward(self, left: torch.Tensor, right: torch.Tensor, add_values: bool) -> torch.Tensor:
        """Add or subtract according to the runtime Python bool."""

        if add_values:
            return left + right
        return left - right


class IntBranch(nn.Module):
    """Select an arithmetic arm from a non-tensor Python int."""

    def forward(self, value: torch.Tensor, mode: int) -> torch.Tensor:
        """Route on an integer mode selector."""

        if mode == 0:
            return value + 1
        return value * 10


class FloatBranch(nn.Module):
    """Select an arithmetic arm from a non-tensor Python float."""

    def forward(self, value: torch.Tensor, scale: float) -> torch.Tensor:
        """Route on a float threshold."""

        if scale > 1.0:
            return value * 2
        return value - 5


class MixedSingleTensorModel(nn.Module):
    """One tensor argument beside one non-tensor Python argument."""

    def forward(self, value: torch.Tensor, flag: bool) -> torch.Tensor:
        """Shift the tensor up or down from the Python bool."""

        if flag:
            return value + 100
        return value - 100


def _save_runnable(model: nn.Module, capture_inputs: list, path: Path) -> Path:
    """Capture and save a runnable artifact carrying activation attestation."""

    trace = tl.trace(
        model,
        list(capture_inputs),
        capture=CaptureOptions(
            intervention_ready=True,
            capture_container_structure=True,
            cache=False,
        ),
    )
    trace.save(path, level="runnable", include_activations=True)
    return path


@pytest.mark.smoke
def test_changed_bool_input_diverges_and_is_not_attested(tmp_path: Path) -> None:
    """A changed bool input must raise under the default rollback policy."""

    model = BoolBranch()
    left = torch.tensor([1.0, 4.0])
    right = torch.tensor([2.0, 2.0])
    path = _save_runnable(model, [left, right, True], tmp_path / "bool.tlspec")

    with pytest.raises(PathDivergenceError):
        tl.load(path).run(inputs=[left, right, False])

    diverged = tl.load(path).run(
        inputs=[left, right, False],
        on_divergence=DivergencePolicy.RETURN_DIVERGED,
    )
    assert diverged.report.path_faithfulness is PathFaithfulness.DIVERGED
    assert diverged.report.numeric_attestation is not NumericAttestationStatus.ATTESTED
    assert diverged.report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE
    assert diverged.report.poisoned
    # The honest report must never masquerade the silent-wrong result as correct.
    fresh = model(left, right, False)
    assert torch.equal(fresh, left - right)


@pytest.mark.smoke
def test_unchanged_bool_input_still_verifies_and_attests(tmp_path: Path) -> None:
    """An identical bool input must verify, attest, and return the right output."""

    model = BoolBranch()
    left = torch.tensor([1.0, 4.0])
    right = torch.tensor([2.0, 2.0])
    path = _save_runnable(model, [left, right, True], tmp_path / "bool.tlspec")

    result = tl.load(path).run(inputs=[left, right, True])
    assert result.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert result.report.numeric_attestation is NumericAttestationStatus.ATTESTED
    assert not result.report.poisoned
    assert all(check.passed for check in result.report.contract_checks)
    torch.testing.assert_close(result.output, left + right)


def test_changed_int_input_diverges_unchanged_verifies(tmp_path: Path) -> None:
    """Int control inputs follow the same changed-diverges/unchanged-verifies rule."""

    model = IntBranch()
    value = torch.tensor([1.0, 2.0])
    path = _save_runnable(model, [value, 0], tmp_path / "int.tlspec")

    same = tl.load(path).run(inputs=[value, 0])
    assert same.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert same.report.numeric_attestation is NumericAttestationStatus.ATTESTED
    torch.testing.assert_close(same.output, value + 1)

    with pytest.raises(PathDivergenceError):
        tl.load(path).run(inputs=[value, 1])
    diverged = tl.load(path).run(inputs=[value, 1], on_divergence=DivergencePolicy.RETURN_DIVERGED)
    assert diverged.report.path_faithfulness is PathFaithfulness.DIVERGED
    assert diverged.report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE


def test_changed_float_input_diverges_unchanged_verifies(tmp_path: Path) -> None:
    """Float control inputs follow the same changed-diverges/unchanged-verifies rule."""

    model = FloatBranch()
    value = torch.tensor([3.0, 4.0])
    path = _save_runnable(model, [value, 2.0], tmp_path / "float.tlspec")

    same = tl.load(path).run(inputs=[value, 2.0])
    assert same.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert same.report.numeric_attestation is NumericAttestationStatus.ATTESTED
    torch.testing.assert_close(same.output, value * 2)

    with pytest.raises(PathDivergenceError):
        tl.load(path).run(inputs=[value, 0.5])
    diverged = tl.load(path).run(
        inputs=[value, 0.5], on_divergence=DivergencePolicy.RETURN_DIVERGED
    )
    assert diverged.report.path_faithfulness is PathFaithfulness.DIVERGED
    assert diverged.report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE


def test_mixed_tensor_and_python_arg_binds_without_crashing(tmp_path: Path) -> None:
    """One tensor arg plus one Python arg binds on original inputs and diverges on a change.

    Regression for the binder shortcut that treated the whole ``[tensor, python]``
    runtime argument list as the single tensor site, producing a spurious
    input-tree mismatch even for the original inputs.
    """

    model = MixedSingleTensorModel()
    value = torch.tensor([5.0])
    path = _save_runnable(model, [value, True], tmp_path / "mixed.tlspec")

    original = tl.load(path).run(inputs=[value, True])
    assert original.report.path_faithfulness is PathFaithfulness.VERIFIED
    assert original.report.numeric_attestation is NumericAttestationStatus.ATTESTED
    torch.testing.assert_close(original.output, value + 100)

    with pytest.raises(PathDivergenceError):
        tl.load(path).run(inputs=[value, False])
    diverged = tl.load(path).run(
        inputs=[value, False], on_divergence=DivergencePolicy.RETURN_DIVERGED
    )
    assert diverged.report.path_faithfulness is PathFaithfulness.DIVERGED
    assert diverged.report.numeric_attestation is NumericAttestationStatus.NOT_APPLICABLE
