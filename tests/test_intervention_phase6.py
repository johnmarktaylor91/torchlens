"""Phase 6 saved-DAG replay engine tests."""

from __future__ import annotations

from collections.abc import Callable
import importlib
from typing import Any, cast

import pytest
import torch

import torchlens as tl
from torchlens import TraceState
from torchlens.intervention.errors import ControlFlowDivergenceWarning, ReplayPreconditionError
from torchlens.intervention.replay import cone_of_effect

replay_mod = importlib.import_module("torchlens.intervention.replay")


def _zero_hook(out: torch.Tensor, *, hook: Any) -> torch.Tensor:
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


def _identity_hook(out: torch.Tensor, *, hook: Any) -> torch.Tensor:
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

    return tl.trace(model, x, intervention_ready=True)


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

    return next(layer for layer in log.layer_list if layer.func_name == func_name)


def test_replay_hook_updates_downstream_cone_and_run_state() -> None:
    """Replay applies hooks, mutates downstream output, and records run state."""

    torch.manual_seed(0)
    log = _interventions(ResidualRelu(), torch.randn(2, 3))
    original_output = log[log.output_layers[0]].out.clone()

    log.replay(hooks={tl.func("relu"): _zero_hook})

    relu_site = _first_func(log, "relu")
    assert torch.equal(relu_site.out, torch.zeros_like(relu_site.out))
    assert not torch.equal(log[log.output_layers[0]].out, original_output)
    assert log.state is TraceState.REPLAY_PROPAGATED
    assert log.last_run["engine"] == "replay"
    assert relu_site.interventions[-1].engine == "replay"


def test_differentiable_replay_patching_captures_backward_grads() -> None:
    """Differentiable replay patches one run into another and captures grads."""

    clean_x = torch.tensor([-1.0, 0.5, 2.0], requires_grad=True)
    corrupt_x = torch.tensor([2.0, -4.0, 1.0], requires_grad=True)
    clean = tl.trace(
        ReplayPatchModel(),
        clean_x,
        layers_to_save="all",
        save_grads="all",
        intervention_ready=True,
        backward_ready=True,
    )
    corrupt = tl.trace(
        ReplayPatchModel(),
        corrupt_x,
        layers_to_save="all",
        save_grads="all",
        intervention_ready=True,
        backward_ready=True,
    )
    clean_relu = _first_func(clean, "relu")
    patch = clean_relu.out

    def patch_hook(out: torch.Tensor, *, hook: Any) -> torch.Tensor:
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

        del out, hook
        return patch

    replayed = corrupt.replay(hooks={tl.func("relu"): patch_hook}, differentiable=True)
    replay_relu = _first_func(replayed, "relu")
    replay_output = replayed[replayed.output_layers[0]].out

    reference_patch = patch.detach().clone().requires_grad_()
    reference_output = (reference_patch * 3.0).sum()
    reference_output.backward()
    replay_output.backward()

    assert replayed is not corrupt
    assert corrupt.has_backward_pass is False
    assert replay_relu.layer_label in replayed.replay_frontier
    assert replayed.replay_frontier[replay_relu.layer_label].is_leaf
    assert replayed.replay_frontier[replay_relu.layer_label].requires_grad
    assert torch.equal(replay_output, reference_output.detach())
    assert reference_patch.grad is not None
    assert replay_relu.grad is not None
    frontier_grad = replayed.replay_frontier[replay_relu.layer_label].grad
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


def test_differentiable_replay_deep_copies_only_replay_cone_ops() -> None:
    """Differentiable replay uses a shallow untouched-op fork."""

    x = torch.tensor([-1.0, 0.5, 2.0], requires_grad=True)
    log = tl.trace(
        ReplayPatchModel(),
        x,
        layers_to_save="all",
        save_grads="all",
        intervention_ready=True,
        backward_ready=True,
    )

    replayed = log.replay(hooks={tl.func("relu"): _identity_hook}, differentiable=True)

    fork_record = next(
        record
        for record in reversed(log.state_history)
        if isinstance(record, dict) and record.get("op") == "fork"
    )
    fork_payload = cast(dict[str, Any], fork_record)
    last_run = cast(dict[str, Any], replayed.last_run)
    deep_copy_labels = set(fork_payload["deep_copy_layer_labels"])
    replay_cone_labels = set(last_run["cone"])
    all_labels = {site.layer_label for site in log.layer_list}
    outside_cone_label = next(iter(all_labels - replay_cone_labels))

    assert deep_copy_labels == replay_cone_labels
    assert deep_copy_labels < all_labels
    assert replayed[outside_cone_label] is not log[outside_cone_label]
    assert replayed[outside_cone_label].source_trace is replayed
    assert replayed[outside_cone_label].interventions is not log[outside_cone_label].interventions


def test_replay_failure_rolls_back_partial_out_updates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replay failures leave outs and intervention records unchanged."""

    torch.manual_seed(0)
    log = _interventions(ResidualRelu(), torch.randn(2, 3))
    original_tensors = {
        layer.layer_label: layer.out.clone()
        for layer in log.layer_list
        if isinstance(layer.out, torch.Tensor)
    }
    original_records = {layer.layer_label: list(layer.interventions) for layer in log.layer_list}
    original_state = log.state
    real_execute = replay_mod._execute_replay_func_strict
    execute_count = 0

    def flaky_execute(site: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
        """Raise after the replay has computed an upstream update."""

        nonlocal execute_count
        execute_count += 1
        if execute_count == 2:
            raise RuntimeError("replay boom")
        return real_execute(site, args, kwargs)

    monkeypatch.setattr(replay_mod, "_execute_replay_func_strict", flaky_execute)

    with pytest.raises(RuntimeError, match="replay boom"):
        log.replay(hooks={tl.func("relu"): _zero_hook})

    assert log.state is original_state
    for layer in log.layer_list:
        if layer.layer_label in original_tensors:
            assert torch.equal(layer.out, original_tensors[layer.layer_label])
        assert list(layer.interventions) == original_records[layer.layer_label]


def test_replay_from_preserves_origin_out_and_recomputes_children() -> None:
    """``replay_from`` treats the origin as already mutated."""

    torch.manual_seed(1)
    log = _interventions(ResidualRelu(), torch.randn(2, 3))
    relu_site = _first_func(log, "relu")
    replacement = torch.full_like(relu_site.out, 2.0)
    relu_site._internal_set("out", replacement)

    log.replay_from(relu_site)

    assert torch.equal(relu_site.out, replacement)
    assert log.state is TraceState.REPLAY_PROPAGATED
    assert relu_site.layer_label in log.last_run["origins"]


def test_cone_of_effect_follows_bfs_children_and_stops_at_outputs() -> None:
    """Cone traversal walks children and terminates at output leaves."""

    log = _interventions(ResidualRelu(), torch.randn(2, 3))
    relu_site = _first_func(log, "relu")

    cone = cone_of_effect(log, [relu_site])
    cone_labels = [site.layer_label for site in cone]

    assert relu_site.layer_label in cone_labels
    assert log.output_layers[0] in cone_labels
    assert cone_labels == sorted(cone_labels, key=lambda label: log[label].step_index)


def test_cone_of_effect_follows_output_versions_per_child() -> None:
    """Cone traversal follows child-version edges even when children is stale."""

    log = _interventions(ResidualRelu(), torch.randn(2, 3))
    origin = _first_func(log, "linear")
    child_label = origin.children[0]
    origin.children.remove(child_label)
    origin.out_versions_by_child[child_label] = origin.out

    cone_labels = [site.layer_label for site in cone_of_effect(log, [origin])]

    assert child_label in cone_labels


def test_cone_of_effect_includes_all_func_call_id_siblings_once() -> None:
    """Cone traversal expands same-call multi-output siblings."""

    log = _interventions(SplitModel(), torch.randn(4, 5))
    max_sites = [layer for layer in log.layer_list if layer.func_name == "max"]
    assert len(max_sites) == 2

    cone_labels = [site.layer_label for site in cone_of_effect(log, [max_sites[0]])]

    assert {site.layer_label for site in max_sites}.issubset(cone_labels)
    assert len(cone_labels) == len(set(cone_labels))


def test_replay_executes_multi_output_func_call_group_once() -> None:
    """Replay executes one saved function call for all output siblings."""

    log = _interventions(SplitModel(), torch.randn(4, 5))
    max_sites = [layer for layer in log.layer_list if layer.func_name == "max"]
    original_func: Callable[..., Any] = max_sites[0].func
    calls = {"count": 0}

    def counted_max(*args: Any, **kwargs: Any) -> Any:
        """Count replay executions and delegate to the captured function."""

        calls["count"] += 1
        return original_func(*args, **kwargs)

    for site in max_sites:
        site._internal_set("func", counted_max)

    log.replay(hooks={tl.func("max"): _identity_hook})

    assert calls["count"] == 1


def test_cone_of_effect_handles_cycles() -> None:
    """Cone traversal tracks visited sites and avoids graph cycles."""

    log = _interventions(ResidualRelu(), torch.randn(2, 3))
    relu_site = _first_func(log, "relu")
    output = log[log.output_layers[0]]
    output.children.append(relu_site.layer_label)

    cone = cone_of_effect(log, [relu_site])

    assert len(cone) == len({site.layer_label for site in cone})


def test_replay_warns_on_saved_edge_divergence() -> None:
    """Replay emits a control-flow divergence warning for edge mismatch."""

    log = _interventions(ResidualRelu(), torch.randn(2, 3))
    relu_site = _first_func(log, "relu")
    relu_site.parents.clear()

    with pytest.warns(ControlFlowDivergenceWarning):
        log.replay(hooks={tl.func("relu"): _identity_hook})


def test_replay_rejects_non_intervention_ready_logs() -> None:
    """Replay requires intervention-ready metadata."""

    log = tl.trace(ResidualRelu(), torch.randn(2, 3))

    with pytest.raises(ReplayPreconditionError):
        log.replay(hooks={tl.func("relu"): _identity_hook})
