"""R65: two runnable-load parse-time exception classes were plain ValueError
subclasses invisible to ``except TorchLensError`` handling, with no branchable
``fields["code"]``. They now subclass ``TorchLensError`` (keeping ``ValueError``
via MRO) and carry their code on ``fields``.
"""

from __future__ import annotations

from torchlens._io.runnable_load import (
    ContextFieldInvalidError,
    DescriptorStructuralBoundError,
)
from torchlens.errors._base import TorchLensError
from torchlens.runnable import RunnableErrorCode


def test_context_field_invalid_is_torchlens_error_with_code() -> None:
    err = ContextFieldInvalidError("default_device", "not a device string")
    assert isinstance(err, TorchLensError)
    assert isinstance(err, ValueError)  # legacy catchers still work
    assert err.fields["code"] == "context_field_invalid"
    assert err.fields["field"] == "default_device"
    assert err.field == "default_device"


def test_descriptor_structural_bound_is_torchlens_error_with_code() -> None:
    err = DescriptorStructuralBoundError(
        RunnableErrorCode.STATE_SHAPE_MISMATCH, "tensor_slots[0]", "shape too large"
    )
    assert isinstance(err, TorchLensError)
    assert isinstance(err, ValueError)
    assert err.fields["code"] == RunnableErrorCode.STATE_SHAPE_MISMATCH.value
    assert err.code is RunnableErrorCode.STATE_SHAPE_MISMATCH
    assert err.field == "tensor_slots[0]"
