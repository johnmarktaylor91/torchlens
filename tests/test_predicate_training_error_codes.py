"""Provocations for the R65a/R65c stable refusal codes (fixwave-6).

R65a: ``PredicateError.__init__`` historically swallowed the structured
payload channel (bare ``super().__init__(message)``), so ``exc.fields`` was
empty at every raise site and no site could carry ``code=``. R65c:
``TrainingModeConfigError`` inherited the channel cleanly but no raise site
used it. Every test here provokes one stamped code through the public API
and branches on ``exc.fields["code"]``, never message text.
"""

from __future__ import annotations

import pytest
import torch
from torch import nn

import torchlens as tl
from torchlens.errors import TrainingModeConfigError
from torchlens.fastlog import CaptureSpec
from torchlens.fastlog.exceptions import PredicateError


def _model() -> nn.Module:
    """Return the minimal two-op provocation model."""

    return nn.Sequential(nn.Linear(4, 2), nn.ReLU())


def _x(requires_grad: bool = False) -> torch.Tensor:
    """Return a tiny forward input."""

    return torch.randn(3, 4, requires_grad=requires_grad)


def test_save_predicate_invalid_return_carries_predicate_return_invalid() -> None:
    """A save predicate returning an out-of-contract value stamps the code."""

    with pytest.raises(PredicateError) as exc_info:
        tl.record(_model(), _x(), save=lambda ctx: "bogus", on_predicate_error="fail-fast")
    assert exc_info.value.fields["code"] == "predicate_return_invalid"
    assert exc_info.value.fields["remedy"]


def test_halt_predicate_non_bool_return_carries_predicate_return_invalid() -> None:
    """A halt predicate returning a non-bool stamps the same family code."""

    with pytest.raises(PredicateError) as exc_info:
        tl.trace(_model(), _x(), halt=lambda ctx: 1)
    assert exc_info.value.fields["code"] == "predicate_return_invalid"


def test_invalid_default_capture_decision_carries_predicate_default_invalid() -> None:
    """A default_op value outside bool/CaptureSpec stamps the default code."""

    with pytest.raises(PredicateError) as exc_info:
        tl.record(_model(), _x(), default_op="bogus", on_predicate_error="fail-fast")
    assert exc_info.value.fields["code"] == "predicate_default_invalid"


def test_accumulated_predicate_failures_carry_predicate_evaluation_failed() -> None:
    """The end-of-recording accumulated raise stamps the evaluation code."""

    def raising_predicate(ctx: object) -> bool:
        raise ValueError("boom")

    with pytest.raises(PredicateError) as exc_info:
        tl.record(_model(), _x(), save=raising_predicate, on_predicate_error="accumulate")
    assert exc_info.value.fields["code"] == "predicate_evaluation_failed"
    assert exc_info.value.failures


def test_keep_grad_integer_dtype_carries_predicate_storage_conflict() -> None:
    """keep_grad=True with an integer payload dtype stamps the storage code."""

    with pytest.raises(PredicateError) as exc_info:
        tl.record(
            _model(),
            _x(),
            default_op=CaptureSpec(keep_grad=True, dtype=torch.int32),
            on_predicate_error="fail-fast",
        )
    assert exc_info.value.fields["code"] == "predicate_storage_conflict"


def test_capture_spec_unknown_save_mode_carries_save_mode_invalid() -> None:
    """CaptureSpec construction rejects unknown save modes with the code."""

    from torchlens.errors import InvalidArgumentError

    with pytest.raises(InvalidArgumentError) as exc_info:
        CaptureSpec(save_mode="bogus")
    assert exc_info.value.fields["code"] == "save_mode_invalid"


def test_record_followed_by_carries_followed_by_unsupported() -> None:
    """record() refuses followed_by retroactive capture with the code."""

    with pytest.raises(PredicateError) as exc_info:
        tl.record(_model(), _x(), save=tl.func("linear") & tl.followed_by(tl.func("relu")))
    assert exc_info.value.fields["code"] == "followed_by_unsupported"


def test_followed_by_metadata_only_carries_lookback_payload_policy_conflict() -> None:
    """followed_by under the default metadata_only lookback policy refuses."""

    with pytest.raises(PredicateError) as exc_info:
        tl.trace(
            _model(),
            _x(),
            save=tl.func("linear") & tl.followed_by(tl.func("relu")),
            lookback=4,
        )
    assert exc_info.value.fields["code"] == "lookback_payload_policy_conflict"


def test_inference_only_with_backward_ready_carries_inference_only_conflict() -> None:
    """inference_only=True plus a backward flag stamps the conflict code."""

    with pytest.raises(TrainingModeConfigError) as exc_info:
        tl.trace(
            _model(),
            _x(requires_grad=True),
            capture=tl.options.CaptureOptions(inference_only=True, backward_ready=True),
        )
    assert exc_info.value.fields["code"] == "inference_only_conflict"
    assert exc_info.value.fields["remedy"]


@pytest.mark.smoke
def test_backward_ready_with_detach_carries_backward_ready_conflict() -> None:
    """backward_ready=True plus explicit detaching stamps the conflict code."""

    with pytest.raises(TrainingModeConfigError) as exc_info:
        tl.trace(
            _model(),
            _x(requires_grad=True),
            capture=tl.options.CaptureOptions(backward_ready=True, detach_saved_activations=True),
        )
    assert exc_info.value.fields["code"] == "backward_ready_conflict"


def test_integer_activation_transform_carries_transform_not_differentiable() -> None:
    """A non-grad-dtype transform under backward_ready stamps the code."""

    with pytest.raises(TrainingModeConfigError) as exc_info:
        tl.trace(
            _model(),
            _x(requires_grad=True),
            capture=tl.options.CaptureOptions(backward_ready=True),
            save=tl.options.SaveOptions(activation_transform=lambda t: t.to(torch.int64)),
        )
    assert exc_info.value.fields["code"] == "transform_not_differentiable"
