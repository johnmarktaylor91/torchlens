"""Tests for meaningful modes and divergence classification."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from menagerie.crawler.constants import RunMode
from menagerie.crawler.modes import (
    classify_observed_mode_receipts,
    classify_train_eval_divergence,
    detect_meaningful_modes,
    output_signature,
    output_value_sha256,
)


class _BatchNormModel(torch.nn.Module):
    """Fixture with statistical mode behavior."""

    def __init__(self) -> None:
        """Construct one BatchNorm layer."""

        super().__init__()
        self.norm = torch.nn.BatchNorm1d(3)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Apply mode-sensitive normalization.

        Parameters
        ----------
        value:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Normalized tensor.
        """

        return self.norm(value)


class _ShapeBranchModel(torch.nn.Module):
    """Fixture with structural train/eval behavior."""

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        """Return mode-dependent output shapes.

        Parameters
        ----------
        value:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Full tensor in train mode and one column in eval mode.
        """

        return value if self.training else value[:, :1]


@dataclass(frozen=True)
class _StablePayload:
    """Supported structured output used by the stable-encoding fixture."""

    label: str
    values: tuple[int, ...]


class _AddressBearingOutput:
    """Unsupported output whose default repr contains a process address."""


def test_modes_classify_none_statistical_and_structural() -> None:
    """Captured outputs distinguish equality, value drift, and shape drift."""

    value = torch.tensor([[1.0, 2.0, 4.0], [3.0, 6.0, 8.0]])
    assert classify_train_eval_divergence(value, value.clone()).classification == "none"

    statistical = _BatchNormModel()
    statistical.train()
    train_output = statistical(value)
    statistical.eval()
    eval_output = statistical(value)
    assert classify_train_eval_divergence(train_output, eval_output).classification == "statistical"
    assert detect_meaningful_modes(statistical) == (RunMode.TRAIN, RunMode.EVAL)

    structural = _ShapeBranchModel()
    structural.train()
    train_output = structural(value)
    structural.eval()
    eval_output = structural(value)
    assert classify_train_eval_divergence(train_output, eval_output).classification == "structural"
    assert detect_meaningful_modes(structural) == (RunMode.TRAIN, RunMode.EVAL)


def test_independent_mode_receipts_recover_all_divergence_classes() -> None:
    """Per-process structure and value digests mechanically recover mode divergence."""

    train = torch.tensor([[1.0, 2.0]])
    equal = train.clone()
    drifted = torch.tensor([[2.0, 3.0]])
    reshaped = torch.tensor([[1.0], [2.0]])

    def receipt(value: torch.Tensor) -> dict[str, object]:
        """Build the observation subset retained by an isolated mode receipt.

        Parameters
        ----------
        value:
            Captured mode output.

        Returns
        -------
        dict[str, object]
            Structure and value-digest observation.
        """

        return {
            "output_signature": output_signature(value),
            "output_value_sha256": output_value_sha256(value),
        }

    equal_result = classify_observed_mode_receipts(receipt(train), receipt(equal))
    drifted_result = classify_observed_mode_receipts(receipt(train), receipt(drifted))
    reshaped_result = classify_observed_mode_receipts(receipt(train), receipt(reshaped))
    assert equal_result is not None and equal_result.classification == "none"
    assert drifted_result is not None and drifted_result.classification == "statistical"
    assert reshaped_result is not None and reshaped_result.classification == "structural"


def test_output_value_digest_is_stable_for_supported_structures_and_endianness() -> None:
    """Mapping order, dataclasses, and machine byte order do not perturb the digest."""

    little = np.array([1, 2, 3], dtype="<i4")
    big = np.array([1, 2, 3], dtype=">i4")
    left = {"array": little, "payload": _StablePayload("x", (1, 2))}
    right = {"payload": _StablePayload("x", (1, 2)), "array": big}
    assert output_value_sha256(left) == output_value_sha256(right)


def test_output_value_digest_rejects_object_arrays_and_unsupported_leaves() -> None:
    """No address-bearing repr or Python object-array fallback enters a stable digest."""

    assert output_value_sha256(_AddressBearingOutput()) is None
    assert output_value_sha256(np.array([_AddressBearingOutput()], dtype=object)) is None
