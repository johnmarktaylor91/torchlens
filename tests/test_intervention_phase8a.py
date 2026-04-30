"""Phase 8a mutator-method tests."""

from __future__ import annotations

from typing import Any

import pytest
import torch

import torchlens as tl
from torchlens import RunState
from torchlens.intervention.errors import SpecMutationError


class ReluAdd(torch.nn.Module):
    """Small model with a stable relu intervention site."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a relu followed by an add.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            ReLU activation plus one.
        """

        return torch.relu(x) + 1


def _identity_hook(activation: torch.Tensor, *, hook: Any) -> torch.Tensor:
    """Return the activation unchanged.

    Parameters
    ----------
    activation:
        Activation passed to the hook.
    hook:
        Hook context supplied by TorchLens.

    Returns
    -------
    torch.Tensor
        Original activation.
    """

    del hook
    return activation


def _capture() -> Any:
    """Capture an intervention-ready log for Phase 8a tests.

    Returns
    -------
    Any
        Intervention-ready model log.
    """

    return tl.log_forward_pass(
        ReluAdd(),
        torch.randn(2, 3),
        vis_opt="none",
        intervention_ready=True,
    )


def test_set_tensor_marks_spec_stale_and_returns_self() -> None:
    """``set(site, tensor)`` updates recipe state without propagation."""

    log = _capture()
    initial_revision = log._spec_revision
    replacement = torch.zeros(2, 3)

    result = log.set(tl.func("relu"), replacement)

    assert result is log
    assert log._spec_revision == initial_revision + 1
    assert log.run_state is RunState.SPEC_STALE
    assert log._activation_recipe_revision == initial_revision
    assert len(log._intervention_spec.target_value_specs) == 1
    value_spec = log._intervention_spec.target_value_specs[0]
    assert value_spec.value is replacement
    assert value_spec.metadata == {}
    assert not log._recipe_is_clean()


def test_set_callable_tags_one_shot_metadata() -> None:
    """``set(site, fn)`` tags the spec entry as a one-shot callable set."""

    log = _capture()

    def replacement_fn(activation: torch.Tensor) -> torch.Tensor:
        """Return a zero activation matching the original.

        Parameters
        ----------
        activation:
            Matched activation.

        Returns
        -------
        torch.Tensor
            Zero replacement.
        """

        return activation * 0

    log.set(tl.func("relu"), replacement_fn)

    value_spec = log._intervention_spec.target_value_specs[-1]
    assert value_spec.value is replacement_fn
    assert value_spec.metadata["created_by"] == "set_callable_one_shot"


def test_attach_clear_and_detach_hooks_are_sticky_mutators() -> None:
    """Sticky hook mutators return self, increment revisions, and mark stale."""

    log = _capture()
    initial_revision = log._spec_revision

    assert log.attach_hooks({tl.func("relu"): _identity_hook}) is log
    assert log._spec_revision == initial_revision + 1
    assert log.run_state is RunState.SPEC_STALE
    assert len(log._intervention_spec.hook_specs) == 1

    assert log.clear_hooks() is log
    assert log._spec_revision == initial_revision + 2
    assert log._intervention_spec.hook_specs == []

    log.attach_hooks({tl.func("relu"): _identity_hook})
    assert log._spec_revision == initial_revision + 3
    assert len(log._intervention_spec.hook_specs) == 1

    assert log.detach_hooks(tl.func("relu")) is log
    assert log._spec_revision == initial_revision + 4
    assert log._intervention_spec.hook_specs == []


def test_detach_hooks_no_site_is_noop_unless_strict() -> None:
    """``detach_hooks()`` is a non-mutating no-op unless strict mode is requested."""

    log = _capture()
    initial_revision = log._spec_revision

    assert log.detach_hooks() is log
    assert log._spec_revision == initial_revision

    with pytest.raises(SpecMutationError, match="requires a site or handle"):
        log.detach_hooks(strict=True)


@pytest.mark.smoke
def test_rerun_advances_activation_recipe_revision_after_set() -> None:
    """Successful rerun advances the activation recipe revision."""

    x = torch.randn(2, 3)
    log = tl.log_forward_pass(ReluAdd(), x, vis_opt="none", intervention_ready=True)

    log.set(tl.func("relu"), torch.zeros(2, 3))
    assert log._activation_recipe_revision == 0

    result = log.rerun(ReluAdd(), x)

    assert result is log
    assert log.run_state is RunState.RERUN_PROPAGATED
    assert log._activation_recipe_revision == log._spec_revision
    assert log._recipe_is_clean()


def test_fork_and_auto_do_are_implemented_by_phase8b() -> None:
    """Phase 8b implements fork and auto dispatch paths."""

    log = _capture()

    fork = log.fork("candidate")
    assert fork is not log
    assert fork.name == "candidate"
    assert fork.parent_run() is log

    result = log.do({tl.func("relu"): _identity_hook}, confirm_mutation=True)
    assert result is log
    assert log.run_state is RunState.REPLAY_PROPAGATED
