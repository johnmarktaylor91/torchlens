"""Capture helpers shared by the validation-tripwire hardening test files.

``tests/test_validation_hardening.py`` and its sibling
``tests/test_validation_bool_exemption.py`` both build traces the way
``validate_forward_pass`` does and run the public path with provenance
warnings silenced.
"""

import warnings

import torch
import torch.nn as nn

import torchlens as tl
from torchlens.options import CaptureOptions
from torchlens.validation import last_validation_failure, validate_forward_pass


def _capture(model: nn.Module, x: torch.Tensor, seed: int = 0):
    """Capture a trace the way validate_forward_pass does, plus ground truth."""

    torch.manual_seed(seed)
    ground_truth = model(x)
    torch.manual_seed(seed)
    trace = tl.trace(
        model,
        x,
        capture=CaptureOptions(layers_to_save="all", save_args=True, seed=seed),
    )
    return trace, ground_truth


def _quiet_validate(model: nn.Module, x) -> bool:
    """Public-path validation with provenance warnings silenced."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return bool(validate_forward_pass(model, x))


def _assert_degenerate_forward(model: nn.Module, x: torch.Tensor) -> None:
    """Assert validation fails as degenerate: every final output element is non-finite.

    Used where the NaN or inf is intrinsic to the model: no affine input change
    ``c + s * x`` (c in [-25, 25], s in {1, 0.1, 0.01}) makes the output finite.
    """

    assert not validate_forward_pass(model, x)
    failure = last_validation_failure()
    assert failure is not None and failure.check_name == "degenerate_forward", failure


def _with_identity_weights(model, fc1_bias: float, fc2_bias: float):
    """Pin ``fc1`` and ``fc2`` to the identity with constant biases (a finite control)."""

    with torch.no_grad():
        for layer, bias in ((model.fc1, fc1_bias), (model.fc2, fc2_bias)):
            layer.weight.copy_(torch.eye(layer.weight.shape[0]))
            layer.bias.fill_(bias)
    return model
