"""The public-command implementations never keep a patched user_funcs name.

``torchlens._user_public_impls`` calls ``trace`` and
``_run_model_and_save_specified_outs`` through ``torchlens.user_funcs``. It
used to hold copies of both, refreshed by every public entry call: a test that
patched either name and then called a public entry left the patched function
in the implementation module after its patch was undone, and a later test that
called the implementation directly ran it (the corrupting runner of
``test_validate_forward_pass_deepcopy_fallback_tripwire_still_fails`` failed
unrelated validation tests on the same xdist worker).
"""

from __future__ import annotations

from typing import Any

import pytest
import torch
import torch.nn as nn

import torchlens._user_public_impls as user_public_impls
import torchlens.user_funcs as user_funcs
from torchlens.validation import validate_forward_pass


class _AddOne(nn.Module):
    """Stateless model with one traced op."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add one."""

        return x + 1


def test_undone_runner_patch_does_not_reach_a_later_direct_validation() -> None:
    """A runner patched during a public call is gone once the patch is undone."""

    real_runner = user_funcs._run_model_and_save_specified_outs
    calls: list[int] = []

    def counting_runner(*args: Any, **kwargs: Any) -> Any:
        """Count calls, then run the real capture."""

        calls.append(1)
        return real_runner(*args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(user_funcs, "_run_model_and_save_specified_outs", counting_runner)
        assert validate_forward_pass(_AddOne(), torch.randn(2, 3)) is True
    patched_calls = len(calls)
    assert patched_calls > 0

    assert user_public_impls._validate_forward_pass_torch(_AddOne(), torch.randn(2, 3)) is True
    assert len(calls) == patched_calls


def test_undone_trace_patch_does_not_reach_a_later_direct_metadata_call() -> None:
    """A ``trace`` patched during a public call is gone once the patch is undone."""

    real_trace = user_funcs.trace
    calls: list[int] = []

    def counting_trace(*args: Any, **kwargs: Any) -> Any:
        """Count calls, then run the real trace."""

        calls.append(1)
        return real_trace(*args, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(user_funcs, "trace", counting_trace)
        user_funcs.validate_forward_pass(_AddOne(), torch.randn(2, 3))
        user_public_impls.log_model_metadata(_AddOne(), torch.randn(2, 3))
    patched_calls = len(calls)
    assert patched_calls > 0

    user_public_impls.log_model_metadata(_AddOne(), torch.randn(2, 3))
    assert len(calls) == patched_calls
