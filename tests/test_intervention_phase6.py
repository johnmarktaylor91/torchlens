"""Phase 6 saved-DAG replay engine tests."""

from __future__ import annotations

from collections.abc import Callable
import importlib
from typing import Any

import pytest
import torch

import torchlens as tl
from torchlens import RunState
from torchlens.intervention.errors import ControlFlowDivergenceWarning, ReplayPreconditionError
from torchlens.intervention.replay import cone_of_effect

replay_mod = importlib.import_module("torchlens.intervention.replay")


def _zero_hook(activation: torch.Tensor, *, hook: Any) -> torch.Tensor:
    """Return a zero ablation for hook tests.

    Parameters
    ----------
    activation:
        Activation passed to the hook.
    hook:
        Hook context supplied by TorchLens.

    Returns
    -------
    torch.Tensor
        Zeroed activation.
    """

    return activation * 0


def _identity_hook(activation: torch.Tensor, *, hook: Any) -> torch.Tensor:
    """Return an unchanged activation.

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

    return activation


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


def _intervention_log(model: torch.nn.Module, x: torch.Tensor) -> Any:
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

    return tl.log_forward_pass(model, x, vis_opt="none", intervention_ready=True)


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
    log = _intervention_log(ResidualRelu(), torch.randn(2, 3))
    original_output = log[log.output_layers[0]].activation.clone()

    log.replay(hooks={tl.func("relu"): _zero_hook})

    relu_site = _first_func(log, "relu")
    assert torch.equal(relu_site.activation, torch.zeros_like(relu_site.activation))
    assert not torch.equal(log[log.output_layers[0]].activation, original_output)
    assert log.run_state is RunState.REPLAY_PROPAGATED
    assert log.last_run_ctx["engine"] == "replay"
    assert relu_site.intervention_log[-1].engine == "replay"


def test_replay_failure_rolls_back_partial_activation_updates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replay failures leave activations and intervention records unchanged."""

    torch.manual_seed(0)
    log = _intervention_log(ResidualRelu(), torch.randn(2, 3))
    original_tensors = {
        layer.layer_label: layer.activation.clone()
        for layer in log.layer_list
        if isinstance(layer.activation, torch.Tensor)
    }
    original_records = {layer.layer_label: list(layer.intervention_log) for layer in log.layer_list}
    original_state = log.run_state
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

    assert log.run_state is original_state
    for layer in log.layer_list:
        if layer.layer_label in original_tensors:
            assert torch.equal(layer.activation, original_tensors[layer.layer_label])
        assert list(layer.intervention_log) == original_records[layer.layer_label]


def test_replay_from_preserves_origin_activation_and_recomputes_children() -> None:
    """``replay_from`` treats the origin as already mutated."""

    torch.manual_seed(1)
    log = _intervention_log(ResidualRelu(), torch.randn(2, 3))
    relu_site = _first_func(log, "relu")
    replacement = torch.full_like(relu_site.activation, 2.0)
    relu_site._internal_set("activation", replacement)

    log.replay_from(relu_site)

    assert torch.equal(relu_site.activation, replacement)
    assert log.run_state is RunState.REPLAY_PROPAGATED
    assert relu_site.layer_label in log.last_run_ctx["origins"]


def test_cone_of_effect_follows_bfs_children_and_stops_at_outputs() -> None:
    """Cone traversal walks children and terminates at output leaves."""

    log = _intervention_log(ResidualRelu(), torch.randn(2, 3))
    relu_site = _first_func(log, "relu")

    cone = cone_of_effect(log, [relu_site])
    cone_labels = [site.layer_label for site in cone]

    assert relu_site.layer_label in cone_labels
    assert log.output_layers[0] in cone_labels
    assert cone_labels == sorted(cone_labels, key=lambda label: log[label].operation_num)


def test_cone_of_effect_follows_children_tensor_versions() -> None:
    """Cone traversal follows child-version edges even when child_layers is stale."""

    log = _intervention_log(ResidualRelu(), torch.randn(2, 3))
    origin = _first_func(log, "linear")
    child_label = origin.child_layers[0]
    origin.child_layers.remove(child_label)
    origin.children_tensor_versions[child_label] = origin.activation

    cone_labels = [site.layer_label for site in cone_of_effect(log, [origin])]

    assert child_label in cone_labels


def test_cone_of_effect_includes_all_func_call_id_siblings_once() -> None:
    """Cone traversal expands same-call multi-output siblings."""

    log = _intervention_log(SplitModel(), torch.randn(4, 5))
    max_sites = [layer for layer in log.layer_list if layer.func_name == "max"]
    assert len(max_sites) == 2

    cone_labels = [site.layer_label for site in cone_of_effect(log, [max_sites[0]])]

    assert {site.layer_label for site in max_sites}.issubset(cone_labels)
    assert len(cone_labels) == len(set(cone_labels))


def test_replay_executes_multi_output_func_call_group_once() -> None:
    """Replay executes one saved function call for all output siblings."""

    log = _intervention_log(SplitModel(), torch.randn(4, 5))
    max_sites = [layer for layer in log.layer_list if layer.func_name == "max"]
    original_func: Callable[..., Any] = max_sites[0].func_applied
    calls = {"count": 0}

    def counted_max(*args: Any, **kwargs: Any) -> Any:
        """Count replay executions and delegate to the captured function."""

        calls["count"] += 1
        return original_func(*args, **kwargs)

    for site in max_sites:
        site._internal_set("func_applied", counted_max)

    log.replay(hooks={tl.func("max"): _identity_hook})

    assert calls["count"] == 1


def test_cone_of_effect_handles_cycles() -> None:
    """Cone traversal tracks visited sites and avoids graph cycles."""

    log = _intervention_log(ResidualRelu(), torch.randn(2, 3))
    relu_site = _first_func(log, "relu")
    output = log[log.output_layers[0]]
    output.child_layers.append(relu_site.layer_label)

    cone = cone_of_effect(log, [relu_site])

    assert len(cone) == len({site.layer_label for site in cone})


def test_replay_warns_on_saved_edge_divergence() -> None:
    """Replay emits a control-flow divergence warning for edge mismatch."""

    log = _intervention_log(ResidualRelu(), torch.randn(2, 3))
    relu_site = _first_func(log, "relu")
    relu_site.parent_layers.clear()

    with pytest.warns(ControlFlowDivergenceWarning):
        log.replay(hooks={tl.func("relu"): _identity_hook})


def test_replay_rejects_non_intervention_ready_logs() -> None:
    """Replay requires intervention-ready metadata."""

    log = tl.log_forward_pass(ResidualRelu(), torch.randn(2, 3), vis_opt="none")

    with pytest.raises(ReplayPreconditionError):
        log.replay(hooks={tl.func("relu"): _identity_hook})
