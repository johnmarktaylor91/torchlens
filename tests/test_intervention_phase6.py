"""Phase 6 saved-DAG replay engine tests."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any, cast

import pytest
import torch

import torchlens as tl
from torchlens.intervention.errors import (
    ControlFlowDivergenceWarning,
    MultiMatchWarning,
    ReplayPreconditionError,
)
from torchlens.io import TraceState

replay_mod = importlib.import_module("torchlens.intervention.replay")


def _zero_hook(out: torch.Tensor, *, ctx: Any) -> torch.Tensor:
    """Return a zero ablation for hook tests.

    Parameters
    ----------
    out:
        Activation passed to the hook.
    hook:
        Hook context supplied by TorchLens.

    Returns
    -------
    torch.Tensor
        Zeroed out.
    """

    return out * 0


def _identity_hook(out: torch.Tensor, *, ctx: Any) -> torch.Tensor:
    """Return an unchanged out.

    Parameters
    ----------
    out:
        Activation passed to the hook.
    hook:
        Hook context supplied by TorchLens.

    Returns
    -------
    torch.Tensor
        Original out.
    """

    return out


class ResidualRelu(torch.nn.Module):
    """Small residual model with a relu cone feeding the output."""

    def __init__(self) -> None:
        """Initialize the linear layer."""

        super().__init__()
        self.linear = torch.nn.Linear(3, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the residual relu graph.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Residual output.
        """

        h = self.linear(x)
        return torch.relu(h) + h


class SplitModel(torch.nn.Module):
    """Model that records a multi-output torch call."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a multi-output ``torch.max`` call.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Values plus indices coerced to float.
        """

        values, indices = torch.max(x, dim=1)
        return values + indices.float()


class RecurrentRelu(torch.nn.Module):
    """Model with repeated relu operations."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run three recurrent relu-style operations.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Final tensor.
        """

        for _idx in range(3):
            x = torch.relu(x)
        return x


class ReplayPatchModel(torch.nn.Module):
    """Tiny model for differentiable replay patching tests."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a relu cone with a scalar output.

        Parameters
        ----------
        x:
            Input tensor.

        Returns
        -------
        torch.Tensor
            Scalar output depending on the relu activation.
        """

        hidden = torch.relu(x)
        return (hidden * 3.0).sum()


def _interventions(model: torch.nn.Module, x: torch.Tensor) -> Any:
    """Capture an intervention-ready model log.

    Parameters
    ----------
    model:
        Model to log.
    x:
        Input tensor.

    Returns
    -------
    Any
        TorchLens model log.
    """

    return tl.trace(model, x, capture=tl.options.CaptureOptions(replayable=True))


def _first_func(log: Any, func_name: str) -> Any:
    """Return the first site with a function name.

    Parameters
    ----------
    log:
        Model log.
    func_name:
        Function name to find.

    Returns
    -------
    Any
        Matching layer pass.
    """

    return next(layer for layer in log._layer_list if layer.func_name == func_name)


def test_differentiable_replay_patching_captures_backward_grads() -> None:
    """Differentiable replay patches one run into another and captures grads."""

    clean_x = torch.tensor([-1.0, 0.5, 2.0], requires_grad=True)
    corrupt_x = torch.tensor([2.0, -4.0, 1.0], requires_grad=True)
    clean = tl.trace(
        ReplayPatchModel(),
        clean_x,
        capture=tl.options.CaptureOptions(save_grads="all", replayable=True, backward_ready=True),
        save="all",
    )
    corrupt = tl.trace(
        ReplayPatchModel(),
        corrupt_x,
        capture=tl.options.CaptureOptions(save_grads="all", replayable=True, backward_ready=True),
        save="all",
    )
    clean_relu = _first_func(clean, "relu")
    patch = clean_relu.out

    def patch_hook(out: torch.Tensor, *, ctx: Any) -> torch.Tensor:
        """Replace the replayed activation with a tensor from another run.

        Parameters
        ----------
        out:
            Replay-computed activation.
        hook:
            TorchLens hook context.

        Returns
        -------
        torch.Tensor
            Activation from the clean source run.
        """

        del out, ctx
        return patch

    replayed = corrupt.push(
        replay=tl.options.ReplayOptions(hooks={tl.func("relu"): patch_hook}, differentiable=True)
    )
    replay_relu = _first_func(replayed, "relu")
    replay_output = replayed[replayed.output_layers[0]].out

    reference_patch = patch.detach().clone().requires_grad_()
    reference_output = (reference_patch * 3.0).sum()
    reference_output.backward()
    replay_output.backward()

    assert replayed is not corrupt
    assert corrupt.autograd.has_backward_pass is False
    assert replay_relu.layer_label in replayed._replay_frontier
    assert replayed._replay_frontier[replay_relu.layer_label].is_leaf
    assert replayed._replay_frontier[replay_relu.layer_label].requires_grad
    assert torch.equal(replay_output, reference_output.detach())
    assert reference_patch.grad is not None
    assert replay_relu.grad is not None
    frontier_grad = replayed._replay_frontier[replay_relu.layer_label].grad
    assert frontier_grad is not None
    replay_relu_grad = replay_relu.grad
    reference_grad = reference_patch.grad
    assert replay_relu_grad is not None
    assert reference_grad is not None
    assert torch.equal(replay_relu_grad, reference_grad)
    assert torch.equal(frontier_grad, reference_grad)
    assert clean_x.grad is None
    assert corrupt_x.grad is None
    last_run = cast(dict[str, Any], replayed.last_run)
    assert last_run["differentiable"] is True


def test_differentiable_replay_forks_isolated_records() -> None:
    """Differentiable replay forks COW records isolated from the source.

    The M11 COW fork replaced the deep-cone/shallow-rest forkcopier split:
    every fork Op is a fresh facade over shared frozen storage, so the
    history record now carries the replay cone for provenance only, and the
    isolation contract holds uniformly inside and outside the cone.
    """

    x = torch.tensor([-1.0, 0.5, 2.0], requires_grad=True)
    log = tl.trace(
        ReplayPatchModel(),
        x,
        capture=tl.options.CaptureOptions(save_grads="all", replayable=True, backward_ready=True),
        save="all",
    )

    replayed = log.push(
        replay=tl.options.ReplayOptions(
            hooks={tl.func("relu"): _identity_hook}, differentiable=True
        )
    )

    fork_record = next(
        record
        for record in reversed(log.state_history)
        if isinstance(record, dict) and record.get("op") == "fork"
    )
    fork_payload = cast(dict[str, Any], fork_record)
    last_run = cast(dict[str, Any], replayed.last_run)
    source_cone_labels = set(fork_payload["source_cone_labels"])
    replay_cone_labels = set(last_run["cone"])
    all_labels = {site.layer_label for site in log._layer_list}
    outside_cone_label = next(iter(all_labels - replay_cone_labels))

    assert source_cone_labels == replay_cone_labels
    assert source_cone_labels < all_labels
    assert replayed[outside_cone_label] is not log[outside_cone_label]
    assert replayed[outside_cone_label].source_trace is replayed
    assert replayed[outside_cone_label].edits is not log[outside_cone_label].edits


def test_replay_from_preserves_origin_out_and_recomputes_children() -> None:
    """``replay_from`` treats the origin as already mutated."""

    torch.manual_seed(1)
    log = _interventions(ResidualRelu(), torch.randn(2, 3))
    relu_site = _first_func(log, "relu")
    replacement = torch.full_like(relu_site.out, 2.0)
    relu_site._internal_set("out", replacement)

    log.push_from(relu_site)

    assert torch.equal(relu_site.out, replacement)
    assert log.state is TraceState.REPLAY_PROPAGATED
    assert relu_site.layer_label in log.last_run["origins"]


def test_replay_executes_multi_output_func_call_group_once() -> None:
    """Replay executes one saved function call for all output siblings."""

    log = _interventions(SplitModel(), torch.randn(4, 5))
    max_sites = [layer for layer in log._layer_list if layer.func_name == "max"]
    original_func: Callable[..., Any] = max_sites[0].func
    calls = {"count": 0}

    def counted_max(*args: Any, **kwargs: Any) -> Any:
        """Count replay executions and delegate to the captured function."""

        calls["count"] += 1
        return original_func(*args, **kwargs)

    for site in max_sites:
        site._internal_set("func", counted_max)

    with pytest.warns(MultiMatchWarning, match="matched 2 sites"):
        log.push(replay=tl.options.ReplayOptions(hooks={tl.func("max"): _identity_hook}))

    assert calls["count"] == 1


def test_replay_warns_on_saved_edge_divergence() -> None:
    """Replay emits a control-flow divergence warning for edge mismatch."""

    log = _interventions(ResidualRelu(), torch.randn(2, 3))
    relu_site = _first_func(log, "relu")
    relu_site.parents = ()

    with pytest.warns(ControlFlowDivergenceWarning):
        log.push(replay=tl.options.ReplayOptions(hooks={tl.func("relu"): _identity_hook}))


def test_replay_rejects_non_intervention_ready_logs() -> None:
    """Replay requires intervention-ready metadata."""

    log = tl.trace(ResidualRelu(), torch.randn(2, 3))

    with pytest.raises(ReplayPreconditionError):
        log.push(replay=tl.options.ReplayOptions(hooks={tl.func("relu"): _identity_hook}))
