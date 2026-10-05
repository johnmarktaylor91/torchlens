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
from torchlens.validation import validate_forward_pass


def _capture(model: nn.Module, x: torch.Tensor, seed: int = 0):
    """Capture a trace the way validate_forward_pass does, plus ground truth."""

    torch.manual_seed(seed)
    ground_truth = model(x)
    torch.manual_seed(seed)
    trace = tl.trace(
        model,
        x,
        capture=CaptureOptions(layers_to_save="all", save_arg_values=True, random_seed=seed),
    )
    return trace, ground_truth


def _quiet_validate(model: nn.Module, x) -> bool:
    """Public-path validation with provenance warnings silenced."""

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return bool(validate_forward_pass(model, x))
